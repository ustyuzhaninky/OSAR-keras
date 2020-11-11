# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author: Konstantin Ustyuzhanin
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import losses


class BolzmannGate(tf.keras.layers.Layer):
    '''A cell implementing an online bolzmann restricted machine layer in TensorFlow 2.

    '''

    def __init__(self,
                 units,
                 learning_rate=0.01,
                 n_passes=1,
                 return_hidden=True,
                 use_bias=True,
                 visible_activation='sigmoid',
                 hidden_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 activity_regularizer='l2',
                 #  kernel_constraint=None,
                 kernel_constraint=constraints.MinMaxNorm(
                     min_value=-1.0, max_value=1.0, rate=1.0, axis=-1
                 ),
                 #  bias_constraint=None,
                 bias_constraint=constraints.MinMaxNorm(
                     min_value=-1.0, max_value=1.0, rate=1.0, axis=-1
                 ),
                 optimizer='Adam',
                 ** kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(BolzmannGate, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = units
        self.learning_rate = learning_rate
        self.return_hidden = return_hidden
        self.visible_activation = activations.get(visible_activation)
        self.hidden_activation = activations.get(hidden_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.optimizer = optimizers.get(optimizer)
        self.optimizer.learning_rate = self.learning_rate
        self.n_passes = n_passes

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)
    
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `BolzmannGate` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `BolzmannGate` '
                             'should be defined. Found `None`.')

        last_dim = tensor_shape.dimension_value(input_shape[-1])
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
            name='w',
            shape=(last_dim, self.units),
            constraint=self.kernel_constraint,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.m, self.r = None, None

        if self.use_bias:
            self.bias_hidden = self.add_weight(
                name='b_h',
                shape=(self.units,),
                constraint=self.bias_constraint,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True)
            self.bias_visible = self.add_weight(
                name='b_v',
                shape=(last_dim,),
                constraint=self.bias_constraint,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True)

            self.m_h, self.m_v, self.r_h, self.r_v = None, None, None, None
        else:
            self.bias_hidden = None
            self.bias_visible = None
        self.built = True
    
    def call(self, inputs):
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            # Broadcasting is required for the inputs.
            inputs = tf.cast(inputs, dtype=self._compute_dtype)

        hidden_units = self.prop_up(inputs)
        visible_units = self.prop_down(hidden_units)
        randoms_uniform_values = tf.random.uniform(
            (1, self.units))
        sample_hidden_units = tf.cast(
            randoms_uniform_values < hidden_units, dtype=tf.float32)
        self.states = sample_hidden_units

        if self.return_hidden:
            outputs = self.states
        else:
            outputs = visible_units

        if rank is not None and rank > 2:
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)

        self.pcd_k_step(inputs)

        return outputs

    def pcd_k_step(self, inputs):
        hidden_units = self.prop_up(inputs)
        visible_units = self.prop_down(hidden_units)
        randoms_uniform_values = tf.random.uniform(
            (1, self.units))
        sample_hidden_units = tf.cast(
            randoms_uniform_values < hidden_units, dtype=tf.float32)

        # Positive gradient
        # Outer product. N is the batch size length.
        # From http://stackoverflow.com/questions/35213787/tensorflow-batch-outer-product
        positive_grad = tf.matmul(
            tf.transpose(sample_hidden_units), inputs,
            a_is_sparse=K.is_sparse(sample_hidden_units), b_is_sparse=K.is_sparse(inputs))

        # Negative gradient
        # Gibbs sampling
        sample_hidden_units_gibbs_step = sample_hidden_units
        for t in range(self.n_passes):
            compute_visible_units = self.prop_down(
                sample_hidden_units_gibbs_step)
            compute_hidden_units_gibbs_step = self.prop_up(
                compute_visible_units)
            random_uniform_values = tf.random.uniform(
                (1, self.units))
            sample_hidden_units_gibbs_step = tf.cast(
                random_uniform_values < compute_hidden_units_gibbs_step, dtype=tf.float32)

        # compute_visible_units = self.prop_down(
        #     sample_hidden_units_gibbs_step)
        # compute_hidden_units_gibbs_step = self.prop_up(
        #     compute_visible_units)
        # random_uniform_values = tf.random.uniform(
        #     (1, self.units))
        # sample_hidden_units_gibbs_step = tf.cast(
        #     random_uniform_values < compute_hidden_units_gibbs_step, dtype=tf.float32)

        negative_grad = tf.matmul(tf.transpose(sample_hidden_units_gibbs_step),
                                  compute_visible_units,
                                  a_is_sparse=K.is_sparse(
            sample_hidden_units_gibbs_step),
            b_is_sparse=K.is_sparse(compute_visible_units))
        
        grad_kernel = tf.transpose(
            positive_grad - negative_grad)
        if self.use_bias:
            grad_visible = tf.reduce_mean(
                inputs - compute_visible_units, 0)
            grad_hidden = tf.reduce_mean(
                sample_hidden_units - sample_hidden_units_gibbs_step, 0)
            self.bias_visible.assign_add(-self.learning_rate*grad_visible)
            self.bias_hidden.assign_add(-self.learning_rate*grad_hidden)
        self.kernel.assign_add(self.learning_rate*grad_kernel)

    def _gaussian_neighborhood(self, input_vector, sigma=0.2):
        """ calculates gaussian distance """
        return (1 / tf.sqrt(2*np.pi*sigma**2)) * tf.exp(-1*input_vector**2 / (2*sigma**2))
    
    def prop_up(self, v, q=1):
        """ upwards mean-field propagation """
        if K.is_sparse(v):
            outputs = tf.matmul(v, self.kernel, a_is_sparse=True)
        else:
            outputs = tf.matmul(v, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias_hidden)
        return self.hidden_activation(q*outputs)

    def prop_down(self, h, q=1, sig=True):
        """ downwards mean-field propagation """
        if K.is_sparse(h):
            outputs = tf.matmul(h, tf.transpose(self.kernel), a_is_sparse=True)
        else:
            outputs = tf.matmul(h, tf.transpose(self.kernel))
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias_visible)
        return self.visible_activation(q*outputs)

    def get_config(self):
        config = {
            'units': self.units,
            'learning_rate': self.learning_rate,
            'n_passes': self.n_passes,
            'return_hidden': self.return_hidden,
            'visible_activation': activations.serialize(self.visible_activation),
            'hidden_activation': activations.serialize(self.hidden_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'optimizer':  optimizers.serialize(self.optimizer)
        }

        base_config = super(OnlineBolzmannCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

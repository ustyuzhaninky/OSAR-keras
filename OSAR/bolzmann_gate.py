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
from tensorflow.python.framework import dtypes

__all__ = ['BolzmannGate', 'TargetingGate']

class BolzmannGate(tf.keras.layers.Layer):
    '''A cell implementing an online bolzmann restricted machine layer in TensorFlow 2.

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            learning_rate: float > 0.0. Learning rate (default - 0.01).
            n_passes: int > 0. Number of passes (default - 1).
            return_hidden: bool. Whether to return hiddenn or visible state (default - True).
            use_bias: Boolean. Whether to use bias (dafault - True).
            visible_activation: keras.activations. Activation function for visible units (default - 'softmax').
            hidden_activation: keras.activations. Activation function for hidden units (default - 'softmax').
            kernel_initializer: keras.initializer. Initializer for attention weights (default - glorot_uniform).
            kernel_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            kernel_constraint: keras.constraint. Constraint for attention weights (default - None).
            bias_initializer: keras.initializer. Initializer for biases (default - zeros).
            bias_regularizer: keras.regularizer. Regularizer for biases (default - l2).
            bias_constraint: keras.constraint. Constraint for attention weights (default - None),

        # Inputs Shape
            1: 2D tensor with shape: `(batch_size, features)`,
            2: 2D tensor with shape: `(batch_size, 1)`.
        # Output Shape
            2D tensor with shape: `(batch_size, n_actions)` or, if return_sequences == True:
            3D tensor with shape: `(batch_size, units, n_actions)`.
        # References
            - [Restricted Bolzmann Machine](http://www.scholarpedia.org/article/Boltzmann_machine)

    '''

    def __init__(self,
                 units,
                 learning_rate=0.01,
                 n_passes=1,
                 return_hidden=True,
                 use_bias=True,
                 visible_activation='softmax',
                 hidden_activation='softmax',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 activity_regularizer='l2',
                 kernel_constraint=None,
                 bias_constraint=None,
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
        self.n_passes = n_passes

        self.supports_masking = True
        # self.input_spec = InputSpec(min_ndim=2)
    
    def compute_output_shape(self, input_shape):
        if self.return_hidden:
            return input_shape[0], self.units
        else:
            return input_shape

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
        # self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
            name='w',
            shape=(last_dim, self.units),
            constraint=self.kernel_constraint,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=False)
        self.m, self.r = None, None

        if self.use_bias:
            self.bias_hidden = self.add_weight(
                name='b_h',
                shape=(self.units,),
                constraint=self.bias_constraint,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=False)
            self.bias_visible = self.add_weight(
                name='b_v',
                shape=(last_dim,),
                constraint=self.bias_constraint,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=False)

            self.m_h, self.m_v, self.r_h, self.r_v = None, None, None, None
        else:
            self.bias_hidden = None
            self.bias_visible = None
        self.built = True
    
    def compute_mask(self, inputs, mask=None):
        mask = super(BolzmannGate, self).compute_mask(inputs, mask)
        return mask

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
            outputs = self.hidden_activation(self.states)
        else:
            outputs = self.visible_activation(visible_units)

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
            self.add_update(K.update(self.bias_visible, -self.learning_rate*grad_visible))
            self.add_update(K.update(self.bias_hidden, -self.learning_rate*grad_hidden))
        self.add_update(K.update(self.kernel, -self.learning_rate*grad_kernel))

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
            outputs = K.bias_add(outputs, self.bias_hidden)
        return self.hidden_activation(q*outputs)

    def prop_down(self, h, q=1, sig=True):
        """ downwards mean-field propagation """
        if K.is_sparse(h):
            outputs = tf.matmul(h, tf.transpose(self.kernel), a_is_sparse=True)
        else:
            outputs = tf.matmul(h, tf.transpose(self.kernel))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias_visible)
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
        }

        base_config = super(BolzmannGate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TargetingGate(BolzmannGate):
    '''A reward-trageting gate: a bolzmann gate modification that trains according to the given reward ratio.

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            learning_rate: float > 0.0. Learning rate (default - 0.01).
            n_passes: int > 0. Number of passes (default - 1).
            return_hidden: bool. Whether to return hiddenn or visible state (default - True).
            use_bias: Boolean. Whether to use bias (dafault - True).
            visible_activation: keras.activations. Activation function for visible units (default - 'softmax').
            hidden_activation: keras.activations. Activation function for hidden units (default - 'softmax').
            kernel_initializer: keras.initializer. Initializer for attention weights (default - glorot_uniform).
            kernel_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            kernel_constraint: keras.constraint. Constraint for attention weights (default - None).
            bias_initializer: keras.initializer. Initializer for biases (default - zeros).
            bias_regularizer: keras.regularizer. Regularizer for biases (default - l2).
            bias_constraint: keras.constraint. Constraint for attention weights (default - None),

        # Inputs Shape
            1: 2D tensor with shape: `(batch_size, features)`,
            2: 2D tensor with shape: `(batch_size, 1)`.
        # Output Shape
            2D tensor with shape: `(batch_size, n_actions)` or, if return_sequences == True:
            3D tensor with shape: `(batch_size, units, n_actions)`.
        # References
            - [Restricted Bolzmann Machine](http://www.scholarpedia.org/article/Boltzmann_machine)

    '''

    def __init__(self,
                 units,
                 learning_rate=0.01,
                 n_passes=1,
                 return_hidden=True,
                 use_bias=True,
                 visible_activation='softmax',
                 hidden_activation='softmax',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 activity_regularizer='l2',
                 kernel_constraint=None,
                 bias_constraint=None,
                 ** kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(TargetingGate, self).__init__(
            units,
            learning_rate=learning_rate,
            n_passes=n_passes,
            return_hidden=return_hidden,
            use_bias=use_bias,
            visible_activation=visible_activation,
            hidden_activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            ** kwargs)

        # self.input_spec = InputSpec(shape=[(None, None), (None, None)], min_ndim=2)

    def compute_output_shape(self, input_shape):
        input_shape, reward_shape = input_shape[0], input_shape[1]
        if self.return_hidden:
            return input_shape[0], self.units
        else:
            return input_shape
    
    def compute_mask(self, inputs, mask=None):
        inputs, rewards = inputs[0], inputs[1]
        mask = super(TargetingGate, self).compute_mask(inputs, mask)
        return mask

    def call(self, inputs):
        inputs, rewards = inputs[0], inputs[1]
        if rewards is None:
            rewards = 0.0
        rewards = K.sum(rewards)
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
            outputs = self.hidden_activation(self.states)
        else:
            outputs = self.visible_activation(visible_units)

        if rank is not None and rank > 2:
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)

        self.pcd_k_step(inputs, rewards)

        return outputs

    def pcd_k_step(self, inputs, rawards):
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
            self.add_update(K.update(self.bias_visible, -
                                     self.learning_rate*rawards*grad_visible))
            self.add_update(K.update(self.bias_hidden, -
                                     self.learning_rate*rawards*grad_hidden))
        self.add_update(
            K.update(self.kernel, -self.learning_rate*rawards*grad_kernel))
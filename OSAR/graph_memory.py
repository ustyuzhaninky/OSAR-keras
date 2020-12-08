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
from tensorflow.python.keras import backend as K

__all__ = ['GraphMemory', 'HadamarMixture',]


class GraphMemory(tf.keras.layers.Layer):
    """Graph memory layer.

    # Arguments
        shift: int: Phase shift of the graph - where the target should be placed (default - 1)).
        axis: int: Index of a shifting axis or timesteps axis (default - 1).
        learning_rate: float > 0.0. Learning rate (default - 0.01).
        kernel_initializer: keras.initializer. Initializer for filter kernel weights (default - glorot_uniform).
        use_bias: Boolean. Whether to use bias (dafault - True).
        bias_initializer: keras.initializer. Initializer for filter bias weights (default - glorot_uniform).

    # Input shape
        ND tensor with shape: `(batch_size, ..., output_dim)`.

    # Output shape
        ND tensor with shape: `(batch_size, ..., output_dim)` - target, identical to the shape of original input.

    # References
        - None

    """

    def __init__(self,
                 shift=1,
                 axis=0,
                 learning_rate=0.01,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer='l2',
                 kernel_constraint=None,
                 use_bias=True, 
                 bias_initializer='glorot_uniform',
                 bias_regularizer='l2',
                 bias_constraint=None,
                 **kwargs,):
        super(GraphMemory, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True
        self.shift = shift
        self.axis = axis
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.space = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        batch_size = K.cast(input_shape[0], 'int32')
        encoding_len = K.cast(input_shape[-1], 'int32')

        self.space = self.add_weight(
            shape=(batch_size,) + input_shape[1:] + input_shape[1:],
            initializer='zeros',
            trainable=False,
            name='space',
        )
        self.kernel = self.add_weight(
            shape=(batch_size,) + input_shape[1:] + input_shape[1:],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            name='kernel',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=self.space.shape[1:],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                name='bias',
            )

        self.I = K.eye(encoding_len, dtype=tf.float32)
        super(GraphMemory, self).build(input_shape)

    def retrieve(self, inputs):
        operator_1 = tf.linalg.LinearOperatorFullMatrix(inputs)
        operator_2 = tf.linalg.LinearOperatorFullMatrix(self.I)
        operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        kronecker = operator.to_dense() * 1 / tf.math.sqrt(2.)
        
        kronecker = tf.reshape(kronecker, self.space.shape)
        updated_space = self.space + self.kernel*kronecker
        if self.use_bias:
            updated_space = K.bias_add(updated_space, self.bias)
        # updated_space = K.sigmoid(updated_space)
        
        # Slice space in search of a target
        indexes = 'QWERTYUIOPASDFGHJKLZXCVBNM'[:len(inputs.shape)-1]
        target = tf.einsum(
            f'{"b"+indexes+indexes},{"b"+indexes}->{"b"+indexes}', updated_space, inputs)
        
        return target, updated_space

    def call(self, inputs, training=False, **kwargs):
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        encoding_len = K.cast(K.shape(inputs)[-1], 'int32')
        
        # Adding updates to space
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.kernel)
            tape.watch(self.bias)
            target, updated_space = self.retrieve(inputs)

            # calculating losses
            loss = tf.losses.Huber()(
                target - tf.roll(inputs, shift=self.shift, axis=self.axis), 0)

        # Updating filter and bias so the targeting ops will yield next value
        grad_kernel = tape.gradient(loss, self.kernel)
        if self.use_bias:
            grad_bias = tape.gradient(loss, self.bias)

        # Implementing self-annealing updates if not training:
        if not training:
            if self.use_bias:
                self.add_update(K.update(self.bias, 
                                         self.bias-self.learning_rate*grad_bias))
            self.add_update(
                K.update(self.kernel, self.kernel-self.learning_rate*grad_kernel))

        self.add_update(K.update(self.space, updated_space))

        return target

    def get_config(self):
        config = {
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'self.learning_rate': self.learning_rate,
            'shift': self.shift,
            'axis': self.axis,
            'use_bias': self.use_bias,
        }
        base_config = super(GraphMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HadamarMixture(tf.keras.layers.Layer):
    """Hadamar Mixture layer.

    # Arguments
        memory_len: int > 0. Maximum memory length.

    # Input shape
        ND tensor with shape: `(batch_size, ..., output_dim)`, first input vector.
        ND tensor with shape: `(batch_size, ..., output_dim)`, second input vector.

    # Output shape
        ND tensor with shape: `(batch_size, ..., output_dim)` - target, identical to the shape of original input.

    # References
        - None

    """
    
    def __init__(self, **kwargs):
        super(HadamarMixture, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        batch_size = K.cast(inputs[0].shape[0], 'int32')
        encoding_len = K.cast(inputs[0].shape[-1], 'int32')
        input_1, input_2 = inputs

        operator_1 = tf.linalg.LinearOperatorFullMatrix(input_1)
        operator_2 = tf.linalg.LinearOperatorFullMatrix(input_2)
        operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        kronecker = operator.to_dense() * 1 / tf.math.sqrt(2.)

        return kronecker

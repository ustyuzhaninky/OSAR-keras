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

__all__ = ['QueueMemory', ]


class QueueMemory(tf.keras.layers.Layer):
    """Queue memory with floating-point priority index.

    # Arguments
        memory_len: int > 0. Maximum memory length.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.

    # Input shape
        2D tensor with shape: `(batch_size, feature_dim)` - represents last state.
        2D tensor with shape: `(batch_size, space_dim)` - represents flattened space dimension.

    # Output shape
        3D tensor with shape: `(batch_size, 1, output_dim)` - most important member of the queue.
        3D tensor with shape: `(batch_size, 1, 1)` - importance.

    # References
        - None

    """

    def __init__(
        self,
        memory_len: int,
        kernel_initializer='glorot_uniform',
        kernel_regularizer='l2',
        kernel_constraint=None,
        **kwargs):
        super(QueueMemory, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True

        self.memory_len = memory_len
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.index = None
        self.memory = None

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.memory_len, input_shape[-1]

    def build(self, input_shape):
        batch_size = K.cast(input_shape[0][0], 'int32')
        timesteps_dim = 1
        feature_dim = K.cast(input_shape[0][-1], 'int32')
        space_dim = K.cast(input_shape[1][-1], 'int32')

        self.memory = self.add_weight(
            shape=(batch_size, self.memory_len, feature_dim),
            initializer='zeros',
            trainable=False,
            name=f'{self.name}_memory',
        )
        self.index = self.add_weight(
            shape=(batch_size, self.memory_len, 1),
            initializer='zeros',
            trainable=False,
            name=f'{self.name}_index',
        )

        super(QueueMemory, self).build(input_shape)

    # @tf.function
    def call(self, inputs, **kwargs):
        inputs, space = inputs[0], inputs[1]
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        timesteps_dim = 1
        feature_dim = K.cast(K.shape(inputs)[-1], 'int32')
        
        priority = tf.nn.softmax_cross_entropy_with_logits(
            inputs,
            inputs,
        )

        # Build new memory and index
        new_memory = K.concatenate([self.memory, K.expand_dims(inputs, axis=1)], axis=1)
        new_priority = K.concatenate([self.index, K.expand_dims(K.expand_dims(priority, axis=1))], axis=1)
        new_memory = tf.slice(                                     # (batch_size, self.memory_len, output_dim)
            new_memory,
            (0, timesteps_dim, 0),
            (batch_size, self.memory_len, inputs.shape[-1]),
        )
        new_priority = tf.slice(                                     # (batch_size, self.memory_len, output_dim)
            new_priority,
            (0, timesteps_dim, 0),
            (batch_size, self.memory_len, 1),
        )

        indexes = K.reshape(new_priority, (new_priority.shape[1],))
        indexes = tf.argsort(indexes, axis=0, direction='ASCENDING')
        new_priority = tf.sort(new_priority, axis=-1, direction='ASCENDING')
        new_memory = tf.gather(new_memory, indexes, axis=1)
        self.add_update(K.update(self.index, new_priority))
        self.add_update(K.update(self.memory, new_memory))

        return self.memory[:, -1, :], self.index[:, -1, :]

    def get_config(self):
        config = {
            'memory_len': self.memory_len,
        }
        base_config = super(QueueMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

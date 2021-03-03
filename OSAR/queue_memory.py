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

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        1D tensor with shape: `(batch_size, sequence_length, 1)` represents queue priority value.

    # Output shape
        3D tensor with shape: `(batch_size, memory_length, output_dim)`.

    # References
        - None

    """

    def __init__(self, memory_len, **kwargs):
        super(QueueMemory, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True

        self.memory_len = memory_len
        self.index = None
        self.memory = None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.memory_len, input_shape[0][-1]

    def build(self, input_shape):
        batch_size = K.cast(input_shape[0][0], 'int32')
        self.memory = self.add_weight(
            shape=(batch_size, self.memory_len, input_shape[0][-1]),
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

    def call(self, inputs, **kwargs):
        inputs, priority = inputs
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')
        features = K.cast(K.shape(inputs)[-1], 'int32')

        # Build new memory and index
        new_memory = K.concatenate([self.memory, inputs], axis=1)
        new_priority = K.concatenate([self.index, priority], axis=1)
        new_memory = tf.slice(                                     # (batch_size, self.memory_len, output_dim)
            new_memory,
            (0, seq_len, 0),
            (batch_size, self.memory_len, inputs.shape[-1]),
        )
        new_priority = tf.slice(                                     # (batch_size, self.memory_len, output_dim)
            new_priority,
            (0, seq_len, 0),
            (batch_size, self.memory_len, 1),
        )

        indexes = K.reshape(new_priority, (new_priority.shape[1],))
        indexes = tf.argsort(indexes, axis=0, direction='ASCENDING')
        new_priority = tf.sort(new_priority, axis=-1, direction='ASCENDING')
        new_memory = tf.gather(new_memory, indexes, axis=1)
        self.add_update(K.update(self.index, new_priority))
        self.add_update(K.update(self.memory, new_memory))

        return self.memory

    def get_config(self):
        config = {
            'memory_len': self.memory_len,
        }
        base_config = super(QueueMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

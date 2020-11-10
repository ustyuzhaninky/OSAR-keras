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
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import  backend as K
from tensorflow.python.ops import nn
from .tfxl import Memory

__all__ = ['CompressiveAvgPoolMemory']


class CompressiveAvgPoolMemory(Memory):
    """Positional embeddings.

    # Arguments
        batch_size: int > 0. Maximum batch size.
        memory_len: int > 0. Maximum memory length.
        conv_memory_len: int > 0. Maximum length of convolution memory.
        output_dim: int > 0. Dimension of outputs.
        compression_rate: int > 0. Rate of compression for old memories.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        1D tensor with shape: `(batch_size,)` represents length of memory.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length + memory_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
        - [Compressive Transformer](https://arxiv.org/abs/1911.05507.pdf)
    """

    def __init__(self, batch_size, memory_len, conv_memory_len, output_dim, compression_rate, **kwargs):
        super(CompressiveAvgPoolMemory, self).__init__(
            batch_size,
            memory_len,
            conv_memory_len,
            output_dim,
            **kwargs
            )
        self.supports_masking = True
        self.stateful = True

        self.batch_size = batch_size
        self.memory_len = memory_len
        self.conv_memory_len = conv_memory_len
        self.output_dim = output_dim
        self.compression_rate = compression_rate

        self.memory = None

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_len +
                   self.conv_memory_len, self.output_dim),
            initializer='zeros',
            trainable=False,
            name='memory',
        )
        super(CompressiveAvgPoolMemory, self).build(input_shape)
    
    def _compress(self, inputs):
        return nn.relu(nn.avg_pool1d(
            inputs,
            self.compression_rate,
            self.compression_rate,
            'SAME',
            name='compressor'
        ))

    def call(self, inputs, **kwargs):
        
        inputs, memory_length = inputs

        if len(inputs.shape) < 2:
            raise ValueError(
                'The dimension of the input vector'
                ' should be at least 3D: `(time_steps, ..., features)`')

        if len(inputs.shape) < 2:
            raise ValueError('The dimensions of the first tensor of the inputs to `_transformerShift` '
                            f'should be at least 2D: `(timesteps, ..., features)`. Found `{inputs.shape}`.')
        if tensor_shape.dimension_value(inputs.shape[-1]) is None:
            raise ValueError('The last dimension of the first tensor of the inputs'
                            'should be defined. Found `None`.')
        
        memory_length = K.cast(memory_length[0][0], 'int32')
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')

        # Build new memory
        pad = K.tile(inputs[0:1, ...], (self.batch_size - batch_size, 1, 1))
        # (self.batch_size, seq_len, output_dim)
        padded = K.concatenate([inputs, pad], axis=0)
        # (self.batch_size, self.memory_len + seq_len, ...)
        new_memory = K.concatenate([self.memory, padded], axis=1)
        new_memory = tf.slice(                                     # (self.batch_size, self.memory_len, output_dim)
            new_memory,
            (0, seq_len, 0),
            (self.batch_size, self.memory_len + \
             self.conv_memory_len, self.output_dim),
        )  

        # Compressing fallout memory
        old_memory = tf.slice(                                     # (self.batch_size, self.memory_len, output_dim)
            new_memory,
            (0, 0, 0),
            (self.batch_size, 1, self.output_dim),
        )
        old_memory = self._compress(old_memory)                    # (batch_size, memory_length, output_dim)
        new_memory = K.concatenate([self.memory, padded], axis=1)
        
        conv_memory = tf.slice(new_memory,
                               (0, 1, 0), (batch_size, self.conv_memory_len-1, self.output_dim))
        new_conv = K.concatenate([old_memory, conv_memory], axis=1)
        short_memory = tf.slice(new_memory,
                                (0, self.conv_memory_len, 0), (batch_size, self.memory_len, self.output_dim))
        new_memory = K.concatenate([new_conv, short_memory], axis=1)
        self.add_update(K.update(self.memory, new_memory))
        
        # Build output
        old_memory = tf.slice(                                     # (batch_size, memory_length, output_dim)
            new_memory,
            (0, K.maximum(0, self.memory_len + \
                          self.conv_memory_len - seq_len - memory_length), 0),
            (batch_size, K.minimum(self.memory_len + \
                                   self.conv_memory_len, memory_length), self.output_dim),
        )

        return old_memory

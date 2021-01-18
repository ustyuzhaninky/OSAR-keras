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
from tensorflow.keras import  backend as K
from tensorflow.python.ops import nn

__all__ = ['HelixMemory']


class HelixMemory(tf.keras.layers.Layer):
    """Helix memory unit.

    # Arguments
        batch_size: int > 0. Maximum batch size.
        memory_len: int > 0. Maximum memory length.
        n_turns: int >= `compression_rate`+1. Number of helix turns.
        compression_rate: int > 0. Rate of compression for old memories.
            WARNING: `sequence_length` should be at least `n_turns` times 
            divisible by `compression_rate`.
        mode: ['avg', 'max', 'conv',] - mode of compression (default - 'avg').
            - 'avg': Average 1d pooling;
            - 'max': Max pooling 1d;
            - 'conv': 1d convolution with a filter.
            WARNING: with setting `mode='conv'` is trainable.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, compression_rate^n_turns + memory_len, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
        - [Compressive Transformer](https://arxiv.org/abs/1911.05507.pdf)
    """

    def __init__(
        self,
        batch_size,
        memory_len,
        n_turns,
        compression_rate=2,
        mode='avg',
        initializer='glorot_uniform',
        regularizer=None,
        constraint=None,
        **kwargs):

        kwargs['batch_size'] = batch_size
        super(HelixMemory, self).__init__(
            **kwargs
            )
        
        if n_turns <= compression_rate:
            raise AttributeError('Value of `n_tuns` should be at least `compression_rate`+1')
        if not mode.lower() in ['avg', 'max', 'conv']:
            raise AttributeError(f'Mode type `{mode}` is not supported.')
        
        self.supports_masking = True
        self.stateful = True

        self.batch_size = batch_size
        self.memory_len = memory_len
        self.n_turns = n_turns + 1
        self.compression_rate = compression_rate
        self.mode = mode.lower()

        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        self.k = 0

        self.memory = None

    def build(self, input_shape):
        output_dim = input_shape[-1]
        n_conv = sum(pow(self.compression_rate, i)
                     for i in range(1, self.n_turns))
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_len +
                   n_conv, output_dim),
            initializer='zeros',
            trainable=False,
            name=f'{self.name}-memory',
        )
        if self.mode == 'conv':
            _out_channels = tf.cast(
                pow(self.compression_rate, self.n_turns-1), tf.int32)
            self.filters = self.add_weight(
                name=f'{self.name}-filter',
                shape=[self.compression_rate,
                    n_conv, _out_channels],
                dtype=tf.float32,
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )
        self.built = True
        super(HelixMemory, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return self.memory.shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def _compress(self, inputs):
        
        padded_input = K.tile(inputs, [1, 1, inputs.shape[1]])

        if self.mode == 'conv':
            if self.k == 0:
                k_filter_channels_in = inputs.shape[1]
                k_filter_channels_out = int(
                    inputs.shape[1] / self.compression_rate)
                filter = tf.slice(
                    self.filters,
                    (0, self.filters.shape[1] - k_filter_channels_in, 0),
                    (self.compression_rate, k_filter_channels_in, k_filter_channels_out),
                    name='filter_slice_0'
                )
            else:
                k_filter_channels_in_start = sum(pow(self.compression_rate, i)
                                        for i in range(1, self.n_turns - self.k))
                k_filter_channels_in_end = pow(self.compression_rate, self.n_turns - self.k)
                k_filter_channels_out = pow(self.compression_rate, self.n_turns - self.k - 1)
                filter = tf.slice(
                    self.filters,
                    (0, k_filter_channels_in_start, k_filter_channels_out),
                    (self.compression_rate, k_filter_channels_in_end, k_filter_channels_out),
                    name='filter_slice_1'
                )
            compressed = nn.conv1d(padded_input,
                                filter,
                                stride=self.compression_rate,
                                padding='VALID', name='compressed_conv1d')
        elif self.mode == 'avg':
            compressed = nn.avg_pool1d(padded_input, self.compression_rate, self.compression_rate, padding='VALID')
        elif self.mode == 'max':
            compressed = nn.max_pool1d(
                padded_input, self.compression_rate, self.compression_rate, padding='VALID')
        compressed = compressed[..., :inputs.shape[-1]]
        if len(compressed.shape) < 3:
            compressed = K.expand_dims(compressed, axis=-1)
        return compressed
    
    def _helix(self, inputs):
        output_dim = inputs.shape[-1]
        # turn_start = sum(pow(self.compression_rate, i)
                        #  for i in range(1, self.n_turns - self.k))
        turn_end = pow(self.compression_rate, self.n_turns - self.k)
        turn_start = inputs.shape[1] - turn_end
        # Turn extraction, compression, slice and build
        helix_k_turn_old = tf.slice(
            inputs,
            (0, turn_start, 0),
            (self.batch_size, turn_end, output_dim),
            name='helix_turns_new'
        )

        compression = self._compress(helix_k_turn_old)
        compression_lenght = compression.shape[1]

        other_helix = tf.slice(
            inputs,
            (0, 0, 0),
            (self.batch_size, turn_start, output_dim),
            name='other_helix'
        )

        new_other_helix = K.concatenate(
            [other_helix, compression],
            axis=1,
        )
        
        
        helix_k_turn_prep = tf.slice(
            helix_k_turn_old,
            (0, compression_lenght, 0),
            (self.batch_size,
             helix_k_turn_old.shape[1] - compression_lenght, output_dim),
            name='helix_turns_new'
        )

        helix_k_turn_new = K.concatenate(
            [helix_k_turn_prep, compression],
            axis=1,
        )

        return new_other_helix, helix_k_turn_new

    def call(self, inputs, **kwargs):
        self.k = 0
        if len(inputs.shape) < 3:
            raise ValueError(
                'The dimension of the input vector'
                ' should be at least 3D: `(batch_size, timesteps, features)`')

        if tensor_shape.dimension_value(inputs.shape[-1]) is None:
            raise ValueError('The last dimension of the first tensor of the inputs'
                            'should be defined. Found `None`.')
        
        batch_size = inputs.shape[0]
        output_dim = inputs.shape[-1]
        seq_len = inputs.shape[1]

        long_mem_end = sum(pow(self.compression_rate, i) for i in range(1, self.n_turns))
        short_mem_start = pow(self.compression_rate, self.n_turns)
        # Build new memory
        
        new_memory = K.concatenate(
            [self.memory, inputs], axis=1)
        
        # Separating short and long-term memories
        short_memory = tf.slice(
            new_memory,
            (0, short_mem_start,
             0),
            (self.batch_size, self.memory_len,
             output_dim),
            name='short_memory'
        )
        long_memory = tf.slice(
            new_memory,
            (0, 0, 0),
            (self.batch_size, short_mem_start, output_dim),
            name='long_memory'
        )
        
        # Shrinking fallout part for the zero turn of the helix
        fallout = tf.slice(
            short_memory,
            (0, 0, 0),
            (self.batch_size, seq_len, output_dim),
            name='fallout'
        )
        
        sh_fallout = self._compress(fallout)

        long_memory = K.concatenate(
            (long_memory, sh_fallout),
            axis=1,
        )
        
        new_helix = long_memory

        def body(new_helix):
            self.k += 1
            new_helix, helix_part = self._helix(new_helix)
            # Building the helix

            return new_helix, helix_part

        for i in range(1, self.n_turns):
            # Updating the helix
            new_helix, helix_part = body(new_helix)
            # Re-joining the updated helix turn with the rest of the memory
            if i==1:
                new_mem = K.concatenate(
                    [
                        helix_part,
                        short_memory,
                    ], axis=1)
            elif i==self.n_turns-1:
                new_mem = K.concatenate(
                    [
                        helix_part,
                        new_mem,
                    ], axis=1)
            else:
                new_mem = K.concatenate(
                    [
                        helix_part,
                        new_mem,
                    ], axis=1)

        self.k = 0
        self.add_update(K.update(self.memory, new_mem))
        return new_mem

    def get_config(self):
        config = {
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
            'constraint': tf.keras.constraints.serialize(self.constraint),
            'compression_rate': self.compression_rate,
            'mode': self.mode.lower(),
            'memory_len': self.memory_len,
            'n_turns': self.n_turns,
            'batch_size': self.batch_size,
        }
        base_config = super(HelixMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

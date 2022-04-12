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
        memory_len,
        n_turns,
        compression_rate=2,
        mode='avg',
        initializer='glorot_uniform',
        regularizer='l2',
        constraint=None,
        **kwargs):

        super(HelixMemory, self).__init__(
            **kwargs
            )
        
        if n_turns <= compression_rate:
            raise AttributeError('Value of `n_tuns` should be at least `compression_rate`+1')
        if not mode.lower() in ['avg', 'max', 'conv']:
            raise AttributeError(f'Mode type `{mode}` is not supported.')
        
        self.supports_masking = True
        self.stateful = True
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
            shape=(self.memory_len +
                   n_conv, output_dim),
            initializer='glorot_uniform',
            trainable=False,
            name=f'{self.name}-memory',
        )
        if self.mode == 'conv':
            _out_channels = tf.cast(
                pow(self.compression_rate, self.n_turns-1), tf.int32)
            self.filters = self.add_weight(
                name=f'{self.name}-filter',
                shape=[self.compression_rate,
                       output_dim, output_dim
                    ],
                dtype=tf.float32,
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )
        self.built = True
        super(HelixMemory, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.memory.shape[0], self.memory.shape[1]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def _compress(self, inputs):
        
        # padded_input = K.tile(inputs, [1, 1, inputs.shape[1]])
        output_dim = inputs.shape[-1]
        rate = inputs.shape[1] if inputs.shape[1] < self.compression_rate else self.compression_rate
        if inputs.shape[1] < self.compression_rate:
            inputs = K.tile(
                inputs, (1, self.compression_rate // inputs.shape[1], 1))
        if self.mode == 'conv':
            
            compressed = nn.conv1d(inputs,
                                   self.filters,
                                   stride=rate,
                                   padding='VALID', name='compressed_conv1d')
        elif self.mode == 'avg':
            compressed = nn.avg_pool1d(
                inputs, rate, rate, padding='VALID')
        elif self.mode == 'max':
            compressed = nn.max_pool1d(
                inputs, rate, rate, padding='VALID')
        return compressed
    
    def _helix(self, inputs):
        
        output_dim = inputs.shape[-1]
        n_long_mem = sum(pow(self.compression_rate, i)
                         for i in range(1, self.n_turns + 1 - self.k))
        turn_lenght = pow(self.compression_rate, self.n_turns - self.k)
        add_lenght = inputs.shape[0] - n_long_mem 
        # turn_start = inputs.shape[1] - turn_lenght - add_lenght
        
        # Turn extraction, compression, slice and build
        helix_k_turn_old = inputs[-turn_lenght-add_lenght:-add_lenght, :]

        compression = self._compress(tf.expand_dims(helix_k_turn_old, axis=0))[0]
        compression_lenght = compression.shape[1]
        
        other_helix = inputs[:-turn_lenght-add_lenght, :]
        new_other_helix = K.concatenate(
            [other_helix, compression],
            axis=0,
        )

        helix_k_turn_prep = inputs[-turn_lenght:, :]

        return new_other_helix, helix_k_turn_prep

    def call(self, inputs, **kwargs):
        self.k = 0
        # if len(inputs.shape) < 3:
        #     raise ValueError(
        #         'The dimension of the input vector'
        #         ' should be at least 3D: `(batch_size, timesteps, features)`')

        if tensor_shape.dimension_value(inputs.shape[-1]) is None:
            raise ValueError('The last dimension of the first tensor of the inputs'
                            'should be defined. Found `None`.')
        
        # batch_size = inputs.shape[0]
        # output_dim = inputs.shape[2]
        # long_mem_end = sum(pow(self.compression_rate, i) for i in range(1, self.n_turns))
        # short_mem_start = pow(self.compression_rate, self.n_turns)

        return tf.map_fn(self._memory_pass, inputs)
        
    def _memory_pass(self, inputs):
        seq_len = inputs.shape[0]

        # Build new memory
        new_memory = K.concatenate(
            [self.memory, inputs], axis=0)
        
        # Separating short and long-term memories

        short_memory = new_memory[-self.memory_len:, :]
        long_memory = new_memory[:-self.memory_len, :]
        # Shrinking fallout part for the zero turn of the helix
        long_memory = long_memory[:-seq_len, :]
        fallout = short_memory[-seq_len:, :]
        sh_fallout = self._compress(tf.expand_dims(fallout, axis=0))
        sh_fallout = K.reshape(sh_fallout, (sh_fallout.shape[0]*sh_fallout.shape[1], sh_fallout.shape[-1]))
            
        long_memory = K.concatenate(
            (long_memory, sh_fallout),
            axis=0,
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
                    ], axis=0)
            elif i==self.n_turns-1:
                new_mem = K.concatenate(
                    [
                        helix_part,
                        new_mem,
                    ], axis=0)
            else:
                new_mem = K.concatenate(
                    [
                        helix_part,
                        new_mem,
                    ], axis=0)

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
        }
        base_config = super(HelixMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

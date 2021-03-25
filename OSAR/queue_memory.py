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
        epsilon_probability: 0 < float < 1. Bias for closeness of two independent events.
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
        epsilon_probability: float = 0.51,
        kernel_initializer = 'glorot_uniform',
        kernel_regularizer = None,
        kernel_constraint = None,
        **kwargs):
        super(QueueMemory, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True

        self.memory_len = memory_len
        self.epsilon_probability = epsilon_probability
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
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
            initializer='glorot_uniform',
            trainable=False,
            name=f'{self.name}_memory',
        )
        self.index = self.add_weight(
            shape=(batch_size, self.memory_len, 1),
            initializer='glorot_uniform',
            trainable=False,
            name=f'{self.name}_index',
        )

        super(QueueMemory, self).build(input_shape)

    def _compute_compatability(self, logit, source):
        """Computes a compatability for a single-shot """
        logit_substract = tf.tile(tf.expand_dims(
            logit, axis=0), (1, source.shape[1], 1))
        levels = tf.norm(tf.norm(source - logit_substract, axis=0), axis=-1)
        levels = 0.5 - K.hard_sigmoid(levels)
        return levels

    def _remove_by_index(self, index):

        indices = tf.concat([
            tf.range(index[0]),
            tf.range(index[0]+1, self.memory_len)
        ], axis=0)
        return tf.gather(self.index, indices, axis=1), tf.gather(self.memory, indices, axis=1)
    
    def _replace_by_index(self, source, index, value):
        left_indices = tf.range(index)
        right_indices = tf.range(index+1, source.shape[1])

        gathered_indices_left = tf.gather(source, left_indices, axis=1)
        gathered_indices_right = tf.gather(source, right_indices, axis=1)
        prepped_value = tf.expand_dims(value, axis=1)
        return tf.concat([gathered_indices_left, prepped_value, gathered_indices_right], axis=1)

    # @tf.function
    def call(self, inputs, frozen=False, **kwargs):
        states, maximum_route = inputs[0][:, -1, :], inputs[1]
        batch_size = K.cast(K.shape(inputs[0])[0], 'int32')
        timesteps_dim = 1
        feature_dim = K.cast(K.shape(inputs[0])[-1], 'int32')
        # TODO: Add support batches longer than 1: now only
        # one-element batches must be used.
        reward_sum = K.cumsum(inputs[0][..., -1], axis=1)[:, -1]
        

        # Searching for similarities
        compatability_levels_by_queue = self._compute_compatability(
            states, self.memory)
        compatability_levels_by_space = self._compute_compatability(
            states, maximum_route)
        
        if not frozen:
            # Applying by-similarity update
            filter_queue = tf.where(compatability_levels_by_queue >= self.epsilon_probability)
            if filter_queue.shape[0] != 0:
                max_index = tf.math.argmax(compatability_levels_by_queue, axis=0)
                updated_memory_by_queue = self.memory[:, max_index] * (
                    1 - compatability_levels_by_queue[max_index]) + states * compatability_levels_by_queue[max_index]
                updated_index_by_queue = self.index[:, max_index] * (
                    1 - compatability_levels_by_queue[max_index]) + reward_sum * compatability_levels_by_queue[max_index]
                self.add_update(K.update(self.memory, self._replace_by_index(
                    self.memory, max_index, updated_memory_by_queue)))
                self.add_update(K.update(self.index, self._replace_by_index(
                    self.index, max_index, updated_index_by_queue)))
            
            filter_space = tf.where(compatability_levels_by_space >= self.epsilon_probability)
            if filter_space.shape[0] != 0:
                max_index = tf.math.argmax(compatability_levels_by_space, axis=0)
                updated_memory_by_space = maximum_route[:, max_index] * compatability_levels_by_space[max_index] + states * (
                    1 - compatability_levels_by_space[max_index])
                updated_index_by_space = maximum_route[:, max_index, -1] * compatability_levels_by_space[max_index] + reward_sum * (
                    1 - compatability_levels_by_space[max_index])
                
                min_reward_index = tf.argmin(self.index, axis=1)[0]
                reduced_index, reduced_memory = self._remove_by_index(
                    min_reward_index)
                
                new_memory = K.concatenate(
                    [reduced_memory, K.expand_dims(updated_memory_by_space, axis=1)], axis=1)
                new_priority = K.concatenate(
                    [reduced_index, K.expand_dims(K.expand_dims(updated_index_by_space, axis=1), axis=1)], axis=1)
                self.add_update(K.update(self.index, new_priority))
                self.add_update(K.update(self.memory, new_memory))
            
            if (filter_queue.shape[0] == 0) & (filter_space.shape[0] == 0):
                min_reward_index = tf.argmin(self.index, axis=1)[0]
                reduced_index, reduced_memory = self._remove_by_index(min_reward_index)

                # Build new memory and index with a unique element
                new_memory = K.concatenate(
                    [reduced_memory, K.expand_dims(states, axis=1)], axis=1)
                new_priority = K.concatenate(
                    [reduced_index, K.expand_dims(K.expand_dims(reward_sum, axis=1))], axis=1)
                self.add_update(K.update(self.index, new_priority))
                self.add_update(K.update(self.memory, new_memory))

        max_index = tf.argmax(self.index, axis=1)[0]
        return tf.gather(self.memory, max_index, axis=1), tf.gather(self.index, max_index, axis=1)

    def get_config(self):
        config = {
            'memory_len': self.memory_len,
            'epsilon_probability': self.epsilon_probability,
        }
        base_config = super(QueueMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

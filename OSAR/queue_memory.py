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
        2D tensor with shape: `(space_dim,)` - represents flattened space dimension.

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

        feature_dim = K.cast(input_shape[0][-1], 'int32')

        self.memory = self.add_weight(
            shape=(self.memory_len, feature_dim),
            initializer='glorot_uniform',
            trainable=False,
            name=f'{self.name}_memory',
        )
        self.index = self.add_weight(
            shape=(self.memory_len, 1),
            initializer='glorot_uniform',
            trainable=False,
            name=f'{self.name}_index',
        )

        super(QueueMemory, self).build(input_shape)

    def _compute_compatability(self, logit, source):
        """Computes a compatability for a single-shot """
        levels = source - logit
        levels = 0.5 - K.hard_sigmoid(levels)
        return levels

    def _remove_add_by_index(self, index, new_index, new_memory):
        new_index = tf.where(index, new_index, self.index)
        new_memory = tf.where(index, new_memory, self.memory)
        self.add_update(K.update(self.index, new_index))
        self.add_update(K.update(self.memory, new_memory))
    
    def _replace_by_index(self, source, index, value):
        update = tf.where(tf.tile(tf.expand_dims(index, axis=-1), (1, source.shape[-1])), source, value)
        self.add_update(K.update(source, update))

    # @tf.function(autograph=True)
    def call(self, inputs, frozen=False, **kwargs):
        states, maximum_routes = inputs[0], inputs[1]
        targets, importances = tf.map_fn(
            self._update,
            (states, maximum_routes),
            dtype=(states.dtype, maximum_routes.dtype)
        )
        return targets, importances
    
    def _update(self, inputs):
        states, maximum_routes = K.expand_dims(inputs[0], axis=0), K.expand_dims(inputs[1], axis=0)
        rewards = states[:, :, -1]
        states = states[:, -1, :]
        reward_sum =K.cumsum(rewards, axis=0)[:, -1]
        # Searching for similarities
        compatability_levels_by_queue = self._compute_compatability(
            states, self.memory)
        compatability_levels_by_space = self._compute_compatability(
            states, maximum_routes)
        
        # Applying by-similarity update
        filter_queue = tf.where(compatability_levels_by_queue >= self.epsilon_probability)
        if filter_queue.shape[0] != 0:
            max_boolean_mask = compatability_levels_by_queue[..., -1]==tf.reduce_max(compatability_levels_by_queue[..., -1], axis=0)
            max_index = tf.math.argmax(compatability_levels_by_queue[..., -1], axis=0)
            compatability_levels_by_queue = tf.gather(
                compatability_levels_by_queue,
                max_index, axis=0)[..., -1]
            index_slice = tf.gather(self.index, max_index, axis=0)
            memory_slice = tf.gather(self.memory, max_index, axis=0)

            updated_memory_by_queue = memory_slice * (
                1 - compatability_levels_by_queue) + states * compatability_levels_by_queue
            updated_index_by_queue = index_slice * (
                1 - compatability_levels_by_queue) + reward_sum * compatability_levels_by_queue
            
            self._replace_by_index(self.memory, max_boolean_mask, updated_memory_by_queue)
            if len(updated_index_by_queue.shape) < 2:
                updated_index_by_queue = K.expand_dims(updated_index_by_queue, axis=-1)
            if updated_index_by_queue.shape[-1] != 1:
                updated_index_by_queue = tf.transpose(updated_index_by_queue, perm=[1, 0])
            self._replace_by_index(self.index, max_boolean_mask, updated_index_by_queue)
            
        filter_space = tf.where(compatability_levels_by_space >= self.epsilon_probability)
        if filter_space.shape[0] != 0:
            # max_boolean_mask = compatability_levels_by_space[..., -1]==tf.reduce_max(compatability_levels_by_space[..., -1], axis=1)
            max_index = tf.math.argmax(compatability_levels_by_space[..., -1], axis=0)
            compatability_levels_by_space = tf.reduce_max(tf.norm(tf.gather(
                compatability_levels_by_space, max_index, axis=0), axis=0), axis=0)
            maximum_routes = tf.reduce_max(tf.gather(maximum_routes, max_index, axis=0), axis=0)
            r_maximum_routes = tf.reduce_max(tf.gather(maximum_routes[..., -1], max_index, axis=0), axis=0)
            
            updated_memory_by_space = tf.expand_dims(tf.reduce_max(maximum_routes * compatability_levels_by_space + states * (
                1 - compatability_levels_by_space), axis=0), axis=0)
            updated_index_by_space = r_maximum_routes * compatability_levels_by_space[..., -1] + reward_sum * (
                1 - compatability_levels_by_space[..., -1])
            min_reward_index = self.index==tf.reduce_max(-self.index, axis=0)[0]
            self._remove_add_by_index(min_reward_index, updated_index_by_space, updated_memory_by_space)
            
        # Build new memory and index with a unique element
        if (filter_queue.shape[0] == 0) & (filter_space.shape[0] == 0):
            min_reward_index = self.index==tf.reduce_max(-self.index, axis=0)[0]
            self._remove_add_by_index(min_reward_index, reward_sum, states)

        max_index = tf.argmax(self.index, axis=0)[0]
        queue_leader = tf.gather(self.memory, max_index, axis=0)
        leader_importance = tf.gather(self.index, max_index, axis=0)
        return K.expand_dims(queue_leader, axis=0), K.expand_dims(leader_importance, axis=0)

    def get_config(self):
        config = {
            'memory_len': self.memory_len,
            'epsilon_probability': self.epsilon_probability,
        }
        base_config = super(QueueMemory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# coding=utf-8
# Copyright 2021 Konstantin Ustyuzhanin.
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

import collections
from typing import Optional, Text, cast, Iterable

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import special_math_ops
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
from tensorflow.keras import backend as K

from . import EventSpace
from . import QueueMemory

__all__ = ['SympatheticCircuit',]


class SympatheticCircuit(tf.keras.layers.Layer):
    """Sympathetic circuit that joins queue and graph memories.
    
    # Arguments
        units: int >= 0. Dimension of hidden units.
        shape: Iterable of 3 Integers. Dimensions of context (states, rewards, actions).
        memory_len: int > 0 Number of memory units.
        dropout: 0.0 <= float <= 1.0. Dropout rate for hidden units.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    # Output shape
        2D tensor with shape: `(batch_size, sequence_length, 1)` - distance to target.
        2D tensor with shape: `(batch_size, sequence_length, 1)` - importance ratio.
        3D tensor with shape: `(batch_size, sequence_length, feature_dim)` - corrected context.

    # References
        - None yet
    """

    def __init__(self,
                 units:int,
                 shape:int,
                 memory_len,
                 dropout=0.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='truncated_normal',
                 bias_regularizer=None,
                 bias_constraint=None,
                 return_probabilities=False,
                 **kwargs):
        super(SympatheticCircuit, self).__init__(**kwargs)

        self.units = units
        self.shape = shape
        self.memory_len = memory_len
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint


    def build(self, input_shape):
        batch_dim = tf.cast(input_shape[0], tf.int32)
        timesteps_dim = tf.cast(input_shape[1], tf.int32)
        feature_dim = tf.cast(input_shape[-1], tf.int32)

        # if len(input_shape) != 3:
        #     raise ValueError(
        #         f'Input requires a vector of shape (batch_size, timesteps, state_features), not {input_shape}.')

        self.event_space = EventSpace(
            self.units,
            self.shape,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            return_space=True
            )
        self.event_space.build((batch_dim, timesteps_dim, feature_dim))
        self.event_space.built = True
        
        self.queue = QueueMemory(
            self.memory_len,
            kernel_regularizer=self.kernel_regularizer,
            )
        self.queue.build(
            [(batch_dim, feature_dim), (batch_dim, timesteps_dim*feature_dim)])
        self.queue.built = True

        self.built = True
    
    def compute_output_shape(self, input_shape):
        batch_dim = tf.cast(input_shape[0], tf.int32)
        timesteps_dim = tf.cast(input_shape[1], tf.int32)
        feature_dim = tf.cast(input_shape[-1], tf.int32)
        return (batch_dim, timesteps_dim, 1), (batch_dim, timesteps_dim, 1), (batch_dim, timesteps_dim, feature_dim)

    #@tf.function
    def call(self, inputs):
        batch_dim = tf.cast(tf.shape(inputs)[0], tf.int32)
        timesteps_dim = tf.cast(tf.shape(inputs)[1], tf.int32)
        feature_dim = tf.cast(tf.shape(inputs)[-1], tf.int32)

        last_step = inputs[:, -1, :]
        output, space = self.event_space(inputs)
        space = K.reshape(space, (batch_dim, timesteps_dim,
                                  timesteps_dim, feature_dim, feature_dim))
        space = K.max(K.max(space, axis=1), axis=2)
        flat_space = tf.keras.layers.Flatten()(space)
        target, importance = self.queue([last_step, flat_space])
        target = tf.expand_dims(space[:, -1, :], axis=1) / 2 + target / 2

        distance = tf.norm(
            inputs - tf.tile(target, (batch_dim, timesteps_dim, 1)),
            axis=-1,
            ord='euclidean',
            keepdims=True)
        importance = tf.tile(tf.expand_dims(
            importance, axis=1), (batch_dim, timesteps_dim, 1))
        
        return distance, importance, output

    def get_config(self):
        config = {
            'untis': self.units,
            'shape': self.shape,
            'memory_len': self.memory_len,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(SympatheticCircuit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


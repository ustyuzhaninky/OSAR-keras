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
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import Optional, Text, cast, Iterable
from OSAR.cap_layers import softmax

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tf_agents.utils import common

from . import Capsule, SequenceEncoder1D

__all__ = ['EventSpace', 'KroneckerMixture',]


class EventSpace(tf.keras.layers.Layer):
    """Graph memory.

        # Arguments
            units: number of time units.
            shape: Iterable. An tuple or list of shapes for splitting feature dimension of the input.
                Example: An event space with shape `(dim_x, value_dim, dim_y)` will yiled space matrix of
                size (timesteps, dim_x, dim_y) if value_dim=1 (recommended), where timesteps - is second
                dimension of the input tensor.
            attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.
            use_bias: Boolean. Whether to use bias (dafault - True).
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            kernel_constraint: Constraint function applied to the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            return_space: Boolean. Whether to return space variable or no (default - False).
        
        # Input shape
            ND tensor with shape: `(batch_size, timesteps, output_dim)`.

        # Output shape
            ND tensor with shape: `(batch_size, timesteps, output_dim)`
                OR
            ND tensor with shape: `(batch_size, timesteps, output_dim)`
                AND
            ND tensor with shape: `(batch_size, timesteps*timesteps, output_dim*output_dim)`

        # References
           - None
    """
    
    def __init__(
        self,
        units:int,
        shape:Iterable,
        attention_dropout=0.0,
        use_bias:bool=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer='glorot_normal',
        bias_regularizer=None,
        bias_constraint=None,
        return_space:bool=False,
        **kwargs,):
        super(EventSpace, self).__init__(**kwargs)

        self.units = units
        self.shape = shape
        self.attention_dropout = attention_dropout

        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.return_space = return_space

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self._space = None
        self._space_shape = ()

    @property
    def space(self) -> tf.Variable:
        
        if not self.built:
            raise ValueError("The `space` property could be called only from an initialized layer. "
                "Call EventSpace.build before calling EventSpace.space.")
        
        if len(self._space_shape) != 4:
            raise ValueError("The shape of `space` property has not been initialized properly or EventSpace.build has not been called. "
                "Call EventSpace.build before calling EventSpace.space.")

        if self._space is None:
            self._space = tf.Variable(
                initial_value=tf.keras.initializers.GlorotNormal()(shape=self._space_shape),
                shape=tf.TensorShape(self._space_shape), trainable=False)
            return self._space
            # initialize space if it is empty with GlorotNormal white noise of size (timesteps, timesteps, features, features) or return self._space
        
        return self._space

    @space.setter
    # Automatic tracking catches "self._space" which adds an extra weight and
    # breaks HDF5 checkpoints.
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def space(self, space):
        # self._space = space
        self.add_update(K.update(self._space, space))

    def compute_output_shape(self, input_shape):
        batch_size = tf.cast(input_shape[0], tf.int32)
        output_dim = tf.cast(input_shape[-1], tf.int32)
        return batch_size, self.units, output_dim

    def build(self, input_shape):
        batch_size = tf.cast(input_shape[0], tf.int32)
        seq_dim = tf.cast(input_shape[1], tf.int32)
        output_dim = tf.cast(input_shape[-1], tf.int32)
        s_shape = seq_dim * self.shape[0] * self.shape[-1]

        self._space_shape = (input_shape[1], input_shape[1], input_shape[-1], input_shape[-1])
        
        self.caps = Capsule(
            self.units,
            self.units,
            name=f'{self.name}_Capsule',
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        self.caps.build((batch_size, seq_dim, output_dim))
        self.caps.built = True

        # self.space = self.add_weight(
        #     shape=(seq_dim*seq_dim, output_dim*output_dim),
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     name=f'{self.name}-space',
        #     trainable=False,
        # )

        self.encoder = SequenceEncoder1D(
            output_dim, seq_dim,
            activation='relu',
            name=f'{self.name}_encoder',
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
            )
        self.encoder.build(
            (batch_size, self.units, self.units))
        self.encoder.built = True

        # self.scaler = SequenceEncoder1D(
        #     output_dim*output_dim, seq_dim*seq_dim,
        #     activation='relu',
        #     name=f'{self.name}_scaler'
        #     )
        # self.scaler.build(
        #     (batch_size, self.units, self.units))
        # self.scaler.built = True
                                  
        self.built = True
    
    def _likelihood_values(self, inputs):
        seq_dim = tf.cast(tf.shape(inputs)[1], tf.int32)
        output_dim = tf.cast(tf.shape(inputs)[-1], tf.int32)

        cross_levels = tf.einsum('ijk,jjkk->ijk',
                                inputs,
                                self.space)
        cross_levels = tf.raw_ops.LeakyRelu(features=cross_levels)
            
        cross_levels = self.caps(cross_levels)
        cross_levels = self.encoder(cross_levels)

        return cross_levels

    def _update_space(self, levels, inputs):
        seq_dim = tf.cast(tf.shape(inputs)[1], tf.int32)
        output_dim = tf.cast(tf.shape(inputs)[-1], tf.int32)

        ideal_rewards = K.max(
                K.sum(self.space[:, :, -1, -1],
                axis=0), axis=-1)
        kr_prod = KroneckerMixture()([levels, inputs])
        updated_space = tf.reshape(K.tanh(kr_prod), (seq_dim, seq_dim, output_dim, output_dim))#self.scaler(kr_prod))
        updated_space = self.space * \
                (1 - ideal_rewards) + ideal_rewards * updated_space
            
        self.add_update(K.update(self.space, updated_space))
        return K.expand_dims(updated_space, axis=0)

    # @tf.function(autograph=True)
    def call(self, inputs, frozen=False, **kwargs):
        batch_dim = inputs.shape[0]

        if not frozen:
            cross_levels = self._likelihood_values(inputs)
            spaces = tf.concat(tf.nest.map_structure(
                self._update_space,
                tf.split(cross_levels, batch_dim, axis=0),
                tf.split(inputs, batch_dim, axis=0)), axis=0)
        else:
            cross_levels = self._likelihood_values(inputs)
            spaces = tf.tile(K.expand_dims(self.space, axis=0), (batch_dim, 1, 1, 1, 1))
        
        if self.return_space:
            return cross_levels, spaces
        else:
            return cross_levels
    
    def get_config(self):
        config = {
            'untis': self.units,
            'shape': self.shape,
            'attention_dropout': self.attention_dropout,
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(EventSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KroneckerMixture(tf.keras.layers.Layer):
    """Kronecker Mixture layer.

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
        super(KroneckerMixture, self).__init__(**kwargs)

        self.supports_masking = True
        self.stateful = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        input_1, input_2 = inputs
        
        operator_1 = tf.linalg.LinearOperatorFullMatrix(input_1)
        operator_2 = tf.linalg.LinearOperatorFullMatrix(input_2)
        operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        kronecker = operator.to_dense() * 1 / tf.math.sqrt(2.)

        return kronecker

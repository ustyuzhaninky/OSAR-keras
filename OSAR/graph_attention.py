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
from tensorflow.keras import backend as K
from tensorflow.python.ops import nn

__all__ = ['GraphAttention']


class GraphAttention(tf.keras.layers.Layer):
    """Graph Attention layer (Zhao, et al.).

    # Arguments
        units: int >= 0. Dimensions of all tensors.
        use_bias: Boolean. Whether to use bias term.
        attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, compression_rate^n_turns + memory_len, output_dim)`.

    # References
        - [Graph Attention Network](http://arxiv.org/abs/2009.02040.pdf)
    """

    def __init__(self,
                 units,
                 activation='sigmoid',
                 use_bias=True,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    
    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                'The dimension of the input vector'
                ' should be at least 3D: `(batch_size, timesteps, features)`')

        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the first tensor of the inputs'
                             'should be defined. Found `None`.')

        batch_dim, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        
        self.kernel = self.add_weight(
            shape=(2 * feature_dim,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel',
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias',
                trainable=True,
            )

    def call(self, inputs, **kwargs):
        
        tf.debugging.assert_equal(tf.rank(
            inputs), 3,
            message='The dimension of the input vector'
             ' should be at least 3D: `(batch_size, timesteps, features)`')

        batch_dim, seq_len, feature_dim = inputs.shape[0], inputs.shape[1], inputs.shape[2]

        concat_inputs = K.concatenate([inputs, inputs], axis=-1)

        inputs_linear = tf.einsum('ijk,k->ijk', concat_inputs, self.kernel)
        if self.use_bias:
            inputs_linear = nn.bias_add(inputs_linear, self.bias)
        if self.attention_dropout > 0:
            inputs_linear = nn.dropout_v2(
                inputs_linear, self.attention_dropout)
        score = tf.raw_ops.LeakyRelu(features=inputs_linear)
        
        importance_levels = K.softmax(score)
        sum_importances = tf.einsum('nij,nik->nik', importance_levels, inputs)

        h = self.activation(sum_importances)

        attention_weights = tf.nn.softmax(score, axis=1)

        return h, attention_weights

    def get_config(self):
        config = {
            'units': self.units,
            'attention_dropout': self.attention_dropout,
            'use_bias': self.use_bias,
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

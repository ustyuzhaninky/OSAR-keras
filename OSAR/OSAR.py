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

import sys
import string

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.eager import context
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.python.framework import auto_control_deps
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.framework import errors
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.keras import utils

from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward

from .compressive_memory import CompressiveAvgPoolMemory
from .gates import TransferGate
from .gates import AttentionGate
from .context_embedding import ContextEmbedding

__all__ = ['OSARLayer',]

class OSARLayer(tf.keras.layers.Layer):
    """OSAR - Object-Stimulated Active Repeater

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            memory_size: int > 0 Number of units in the memory.
            conv_memory_size: int > 0. Number of units in the compressed memory.
            n_actions: int > 1. Number of actions in the output vector.
            num_heads: int > 0. Number of attention `heads`.
            dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
            heads: int > 0. Number of heads for multi-head attention
            use_bias: Boolean. Whether to use bias (dafault - True).
            compression_rate: int. Compression rate of the transfomer memories.
            gate_initializer: keras.initializer. Initializer for attention weights (default - glorot_normal).
            gate_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            gate_constraint: keras.constraint. Constraint for attention weights (default - None).
            state_initializer: keras.initializer. Initializer for state matrices (default - glorot_uniform).
            bias_initializer: keras.initializer. Initializer for biases (default - zeros).
            bias_regularizer: keras.regularizer. Regularizer for biases (default - l2).
            bias_constraint: keras.constraint. Constraint for attention weights (default - None),
            return_sequences: bool If the layer should return sequences (default - False).

        # Inputs Shape
            1: 2D tensor with shape: `(batch_size, features)`,
            2: 2D tensor with shape: `(batch_size, 1)`.
        # Output Shape
            2D tensor with shape: `(batch_size, n_actions)` or, if return_sequences == True:
            3D tensor with shape: `(batch_size, units, n_actions)`.
        # References
            - None yet

        """
    
    def __init__(self,
                 units,
                 memory_size,
                 conv_memory_size,
                 n_actions,
                 dropout=0.2,
                 heads=10,
                 use_bias=True,
                 compression_rate=4,
                 gate_initializer='glorot_uniform',
                 gate_regularizer='l2',
                 gate_constraint=None,
                 state_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer='l2',
                 bias_constraint=None,
                 return_sequences=False,
                 **kwargs):
        
        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self._return_runtime = kwargs.pop('return_runtime', False)

        super(OSARLayer, self).__init__(
            dynamic=False,
            **kwargs)
        # self.run_eagerly = True

        self.units = units
        self.n_actions = n_actions
        self.conv_memory_size = conv_memory_size
        self.dropout = dropout
        self.heads = heads
        self.use_bias = use_bias
        self.memory_size = memory_size
        self.compression_rate = compression_rate
        self.state_initializer = initializers.get(state_initializer)

        self.gate_initializer = initializers.get(gate_initializer)
        self.gate_regularizer = regularizers.get(gate_regularizer)
        self.gate_constraint = constraints.get(gate_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.return_sequences = return_sequences

    def build(self, input_shape):
        batch_size = input_shape[0]
        input_features = input_shape[-1] - 1
        real_input_shape = tf.TensorShape((batch_size, input_shape[-1]-1))
        reward_shape = tf.TensorShape((batch_size, 1))
        
        timesteps = self.units + self.conv_memory_size

        self.cont_embedding = ContextEmbedding(mask_zero=False)
        self.cont_embedding.build(real_input_shape)
        self.cont_embedding.built = True

        self.generator_attention = AttentionGate(
            units=self.units,
            embed_dim=self.units,
            hidden_dim=self.units,
            num_token=self.units,
            num_head=self.heads,
            memory_len=self.memory_size,
            conv_memory_len=self.conv_memory_size,
            compression_rate=self.compression_rate,
            dropout=0.2,
            attention_dropout=0.2,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.generator_attention.build(real_input_shape)
        self.generator_attention.built = True
        
        self.slow_selector = TransferGate(
            self.n_actions,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.slow_selector.build((batch_size, input_features, input_features))
        self.slow_selector.built = True

        # Memory States

        # self.expected_reward = self.add_weight(
        #     shape=(self.conv_memory_size+self.units,),
        #     initializer=self.state_initializer,
        #     name='expected_reward',
        #     trainable=False
        # )
        # self.internal_reward = self.add_weight(
        #     shape=(self.conv_memory_size+self.units,),
        #     initializer=self.state_initializer,
        #     name='internal_reward',
        #     trainable=False
        # )
        # self.relevance_memory = CompressiveAvgPoolMemory(
        #     batch_size,
        #     self.units,
        #     self.conv_memory_size,
        #     1,
        #     self.compression_rate
        # )
        # self.relevance_memory.build(
        #     [(batch_size, self.conv_memory_size+self.units, 1), (batch_size, 1)])
        # self.relevance_memory.built = True       
        
        # Dense gates     

        super(OSARLayer, self).build(input_shape)#real_input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape[0], self.units, self.n_actions
        else:
            return input_shape[0], self.n_actions
    
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]
    
    def get_config(self):
        config = {
            'units': self.units,
            'n_actions': self.n_actions,
            'memory_size': self.memory_size,
            'conv_memory_size': self.conv_memory_size,
            'dropout': self.dropout,
            'heads': self.heads,
            'use_bias': self.use_bias,
            'compression_rate': self.compression_rate,
            'state_initializer': initializers.serialize(self.state_initializer),
            'gate_initializer': initializers.serialize(self.gate_initializer),
            'gate_regularizer': regularizers.serialize(self.gate_regularizer),
            'gate_constraint': constraints.serialize(self.gate_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'return_sequences': self.return_sequences,
        }
        base_config = super(OSARLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=False, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        input_features = inputs.shape[-1] - 1

        object_input = tf.slice(
            inputs, [0, 0], [inputs.shape[0], inputs.shape[-1]-1])
        reward_input = tf.slice(inputs, [0, inputs.shape[-1]-1],
                                [inputs.shape[0], 1])
        
        object_input = math_ops.cast(object_input, dtype=tf.float32)
        reward_input = math_ops.cast(
            reward_input, dtype=tf.float32)  # inputs[1] # (1,)
        n_digits = tensor_shape.dimension_value(object_input.shape[-1])
        batch_size = tensor_shape.dimension_value(object_input.shape[0])
        
        # Context generator
        token_embedding = self.cont_embedding(object_input)
        mem_view = self.generator_attention(token_embedding)
        action_a = self.slow_selector(mem_view)
    
        return action_a[:, -1]
        
        # # Unpacking state matrices
        # object_queries = K.expand_dims(tf.cast(self.O_state, tf.float32), 0)
        # object_queries = K.array_ops.tile(
        #     object_queries,
        #     [batch_size, 1, 1])                                                  # (batch_size, timesteps, n_digits)
        # action_queries = K.expand_dims(tf.cast(self.A_state, tf.float32), 0)
        # action_queries = K.array_ops.tile(
        #     action_queries,
        #     [batch_size, 1, 1])                                                  # (batch_size, timesteps, n_actions)

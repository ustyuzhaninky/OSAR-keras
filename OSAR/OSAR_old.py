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

from OSAR.multi_head_attention import MultiHeadAttention

__all__ = ['OSARLayer']

def _transformerShift(inputs, memory,
                     conv_units, short_units,
                     compression_op,
                     ):
    """Compressive Transformer left-shift operation with fixed short and convolution memories

       # Arguments
            inputs: A +2D Tensor with shape: `(timesteps, ..., features)`.
            memory: A +2D tensor with shape: `(conv_units+short_units, ..., features)`.
            conv_units: int > 0, Number of convolution memory units in the layer.
            short_units: int > 0, Number of short-term memory units in the layer.
            compression_op: predefined compression operation, preferabley a tf.keras.layer,
            for instance, keras.layers.Conv1D.

        # Inputs Shape
            +2D tensor with shape: `(timesteps, ..., features)`,
            +2D tensor with shape: `(conv_units+short_units, ..., features)`.
        # Output Shape
            +2D tensor with shape: `(conv_units+short_units, ..., features)`.
        # References
            - ["Compressive Transformers for Long-Range Sequence Modelling" by Rae et. al.](https://arxiv.org/abs/1911.05507)

    """

    if len(inputs.shape) < 2:
        raise ValueError(
            'The dimension of the input vector'
            ' should be at least 2D: `(time_steps, ..., features)`')
    if len(memory.shape) < 2:
        raise ValueError(
            'The dimension of the memory vector'
            ' should be at least 2D: `(conv_units+short_units, ..., features)`')

    if len(inputs.shape) < 2:
        raise ValueError('The dimensions of the first tensor of the inputs to `_transformerShift` '
                         f'should be at least 2D: `(timesteps, ..., features)`. Found `{inputs.shape}`.')
    if len(memory.shape) < 2:
        raise ValueError('The dimensions of the second tensor of the memory to `_transformerShift` '
                         f'should be at least 2D: `(conv_units+short_units, ..., features)`. Found `{memory.shape}`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is None:
        raise ValueError('The last dimension of the first tensor of the inputs to `_transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(memory.shape[-1]) is None:
        raise ValueError('The last dimension of the second tensor of the inputs (memory) to `_transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is not \
            tensor_shape.dimension_value(memory.shape[-1]):
        raise ValueError('The last dimension of both input tensors to `_transformerShift` '
                         f'should match. Found `{tensor_shape.dimension_value(inputs.shape[-1])}` and '
                         f'`{tensor_shape.dimension_value(memory.shape[-1])}`.')

    timesteps = inputs.shape[0]

    conv_mem = memory[:, :conv_units]
    short_mem = memory[:, -short_units:]
    if short_mem.shape[1] > timesteps:
        old_memory = short_mem[:, :timesteps] #K.expand_dims(short_mem[:, :timesteps], axis=1)
    else:
        old_memory = short_mem #K.expand_dims(short_mem[:, :timesteps], axis=1)

    # Compressing the oldest memories
    compression = compression_op(old_memory)[0]
    # Update Short-Term Memory
    short_mem = K.concatenate(
        (short_mem[:, timesteps:], K.expand_dims(inputs, axis=1)), axis=1)

    # Update Compressed Memory
    conv_mem = K.concatenate(
        (conv_mem[:, 1:], K.expand_dims(compression, axis=1)), axis=1)

    return K.concatenate((conv_mem, short_mem), axis=1)

class OSARLayer(tf.keras.layers.Layer):
    """OSAR - Object-Stimulated Active Repeater

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            memory_size: int > 1. Size of the dictionary memory. Big numbers allows to store more combinations, but requires more space and speed.
            conv_units: int > 0. Number of convolution units in the layer.
            n_actions: int > 1. Number of actions in the output vector.
            input_features: int > 1. Number of features (classes) in the input vector.
            num_heads: int > 0. Number of attention `heads`.
            dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
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
                 conv_units,
                 n_actions,
                 input_features,
                 num_heads=10,
                 dropout=0.0,
                 use_bias=True,
                 compression_rate=4,
                 gate_initializer='glorot_normal',
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
            # dynamic=True,
            **kwargs)
        # self.run_eagerly = True

        self.units = units
        self.n_actions = n_actions
        self.input_features = input_features
        self.conv_units = conv_units
        self.num_heads = num_heads
        self.dropout = dropout
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

        self._initialize_gates()
        self._init_states()
    
    def _initialize_gates(self):

        # Probability gate
        self.p_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.input_features,
            use_bias=self.use_bias,
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='P-Gate',
        )

        # New action gate
        self.an_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.input_features,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            output_shape=(self.n_actions,),
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='An-Gate',
        )

        # Stimuli-Reward gate
        self.SR_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.n_actions,
            use_bias=self.use_bias,
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='SR-Gate',
        )

        # Forecast gate
        self.f_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.n_actions,
            value_dim=self.input_features,
            use_bias=self.use_bias,
            output_shape=(self.input_features,),
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='F-Gate',
        )

        # Backprop decision gate
        self.d_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.input_features,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            output_shape=(self.n_actions,),
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='D-Gate',
        )

        # Action overhaul gate
        self.ao_gate = MultiHeadAttention(
            self.num_heads,
            key_dim=self.n_actions,
            use_bias=self.use_bias,
            attention_axes=1,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='AO-Gate',
        )

        # Dense gates     

        self.wPshort_gate_kernel = self.add_weight(
            shape=[1, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wPshort_gate_kernel',
            trainable=True
        )
        self.wPshort_gate_bias = self.add_weight(
            shape=[1,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='wPshort_gate_kernel',
            trainable=True
        )

        self.wPlong_gate_kernel = self.add_weight(
            shape=[1, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wPlong_gate_kernel',
            trainable=True
        )
        self.wPlong_gate_bias = self.add_weight(
            shape=[1, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='wPlong_gate_kernel',
            trainable=True
        )

        self.wR_gate_kernel = self.add_weight(
            shape=[self.n_actions, self.n_actions],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wR_gate_kernel',
            trainable=True
        )
        self.wR_gate_bias = self.add_weight(
            shape=[self.n_actions,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='wR_gate_kernel',
            trainable=True
        )

        self.wS_gate_kernel = self.add_weight(
            shape=[self.n_actions, self.n_actions],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wS_gate_kernel',
            trainable=True
        )
        self.wS_gate_bias = self.add_weight(
            shape=[self.n_actions, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='wS_gate_kernel',
            trainable=True
        )

        self.wrs_gate_kernel = self.add_weight(
            shape=[self.conv_units+self.units, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wrs_gate_kernel',
            trainable=True
        )

        self.wAmount_gate_kernel = self.add_weight(
            shape=[1, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wAmount_gate_kernel',
            trainable=True
        )

        self.wStep_gate_kernel = self.add_weight(
            shape=[self.conv_units+self.units, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wStep_gate_kernel',
            trainable=True
        )

        self.wBoost_gate_kernel = self.add_weight(
            shape=[self.conv_units+self.units, 1],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wBoost_gate_kernel',
            trainable=True
        )

        self.wStimuli_gate_kernel = self.add_weight(
            shape=[self.n_actions, self.n_actions],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='wStimuli_gate_kernel',
            trainable=True
        )

        # compressors
        self.o_compressor_kernel = self.add_weight(
            shape=[self.compression_rate,
                   self.input_features, self.input_features],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='o_compressor_kernel',
            trainable=True
        )
        self.o_compressor_bias = self.add_weight(
            shape=[self.input_features,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='o_compressor_kernel',
            trainable=True
        )

        self.a_compressor_kernel = self.add_weight(
            shape=[self.compression_rate, self.n_actions, self.n_actions],
            initializer=self.gate_initializer,
            regularizer=self.gate_regularizer,
            constraint=self.gate_constraint,
            name='a_compressor_kernel',
            trainable=True
        )
        self.a_compressor_bias = self.add_weight(
            shape=[self.n_actions,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='a_compressor_kernel',
            trainable=True
        )

    def _wPshort_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wPshort_gate_kernel,
            self.wPshort_gate_bias,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )

    def _wPlong_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wPlong_gate_kernel,
            self.wPlong_gate_bias,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wR_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wR_gate_kernel,
            self.wR_gate_bias,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wS_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wS_gate_kernel,
            self.wS_gate_bias,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wrs_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wrs_gate_kernel,
            None,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wAmount_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wAmount_gate_kernel,
            None,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wStep_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wStep_gate_kernel,
            None,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wBoost_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wBoost_gate_kernel,
            None,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _wStimuli_gate(self, inputs):
        return core_ops.dense(
            inputs,
            self.wStimuli_gate_kernel,
            None,
            activation=activations.softmax,
            dtype=self._compute_dtype_object
        )
    
    def _o_compressor(self, inputs):
        return nn.bias_add(nn.relu(nn.conv1d_v2(
            inputs,
            self.o_compressor_kernel,
            self.compression_rate,
            'SAME'
        )), self.o_compressor_bias)

    def _a_compressor(self, inputs):
        return nn.bias_add(nn.relu(nn.conv1d_v2(
            inputs,
            self.a_compressor_kernel,
            self.compression_rate,
            'SAME'
        )), self.a_compressor_bias)
        

    def _init_states(self):
        self.A_state = self.state_initializer(
            shape=(self.conv_units+self.units, self.n_actions,),
            dtype=tf.float32,)
        self.O_state = self.state_initializer(
            shape=(self.conv_units+self.units, self.input_features,),
            dtype=tf.float32,)
        self.S_state = self.state_initializer(
            shape=(self.conv_units+self.units,
                   self.input_features, self.n_actions,),
            dtype=tf.float32,)
        self.expected_reward = self.state_initializer(
            shape=(self.conv_units+self.units,),
            dtype=tf.float32,)
        self.internal_reward = self.state_initializer(
            shape=(self.conv_units+self.units,),
            dtype=tf.float32,)

        self.relevance_memory = initializers.get('glorot_uniform')(
            shape=(self.memory_size,),
            dtype=tf.float32,)
        self.object_keys = self.state_initializer(
            shape=(self.memory_size, self.input_features,),
            dtype=tf.float32,)
        self.action_keys = self.state_initializer(
            shape=(self.memory_size, self.n_actions,),
            dtype=tf.float32,)

    def build(self, input_shape):

        batch_dim = input_shape[0]
        
        real_input_shape = tf.TensorShape((batch_dim, input_shape[-1]-1))
        reward_shape = tf.TensorShape((batch_dim, 1))
        
        timesteps = self.units+self.conv_units
        self.p_gate.build([(batch_dim, timesteps, self.input_features),
                           (batch_dim, self.memory_size, self.input_features)
                           ])
        self.an_gate.build([(batch_dim, timesteps, self.input_features),
                            (batch_dim, self.memory_size, self.input_features),
                           (batch_dim, self.memory_size, self.n_actions)
                           ])
        self.SR_gate.build([
            (batch_dim, self.input_features),
            (batch_dim, timesteps, self.input_features, self.n_actions),
            (batch_dim, timesteps, self.input_features, self.n_actions)
        ])
        self.f_gate.build([
            (batch_dim, timesteps, self.n_actions),
            (batch_dim, self.memory_size, self.n_actions),
            (batch_dim, self.memory_size, self.input_features),
        ])
        self.d_gate.build([
            (batch_dim, timesteps, self.n_actions),
            (batch_dim, self.memory_size, self.input_features),
            (batch_dim, self.memory_size, self.n_actions),
        ])
        self.ao_gate.build([
            (batch_dim, timesteps, self.input_features),
            (batch_dim, timesteps, self.n_actions),
        ])
        # self.wPshort_gate.build((1,))
        # self.wPlong_gate.build((1,))
        # self.wR_gate.build((batch_dim, timesteps, self.n_actions))
        # self.wS_gate.build((batch_dim, timesteps, self.n_actions))
        # self.wrs_gate.build((batch_dim, timesteps))
        # self.wAmount_gate.build((batch_dim, 1))
        # self.wStep_gate.build((batch_dim, timesteps))
        # self.wBoost_gate.build((batch_dim, timesteps))
        # self.wStimuli_gate.build(
            # (batch_dim, timesteps, self.input_features, self.n_actions))
        # self.o_compressor.build((batch_dim, 1, self.input_features))
        # self.a_compressor.build((batch_dim, 1, self.n_actions))

        super(OSARLayer, self).build(real_input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape[0][0], self.n_actions
        else:
            return input_shape[0][0], self.units, self.n_actions
    
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]
    
    def get_config(self):
        config = {
            'units': self.units,
            'n_actions': self.n_actions,
            'input_features': self.input_features,
            'conv_units': self.conv_units,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'memory_size': self.memory_size,
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
    
    # @tf.function
    def _transformer_shift_objects(self, new_object, object_memory):
        return _transformerShift(
            new_object, object_memory,
            self.conv_units, self.units,
            self._o_compressor)

    # @tf.function
    def _transformer_shift_actions(self, new_action, action_memory):
        return _transformerShift(
            new_action, action_memory,
            self.conv_units, self.units,
            self._a_compressor)

    
    def call(self, inputs):
        
        # if len(inputs) != 2:
        #     raise ValueError('OSAR model must recive two inputs during the call.')

        real_input = tf.slice(inputs, [0, 0], [inputs.shape[0], inputs.shape[1]-1])
        reward = tf.slice(inputs, [0, inputs.shape[1]-1],
                          [inputs.shape[0], 1])
        return self._operations(real_input, reward)

    def _operations(self, inputs, rewards):
        # if not len(inputs.shape) is 2:
        #     raise ValueError(
        #         'The dimension of the inputs vector should be 2: `(input_shape, reward)`')
        # [0]  # (batch_dim, timesteps n_digits)
        object_input = math_ops.cast(inputs, dtype=tf.float32)
        reward_input = math_ops.cast(
            rewards, dtype=tf.float32)  # inputs[1] # (1,)

        n_digits = tensor_shape.dimension_value(object_input.shape[-1])
        batch_dim = tensor_shape.dimension_value(object_input.shape[0])

        # Unpacking state matrices
        # object_queries = self.O_state

        object_queries = K.expand_dims(self.O_state, 0)
        object_queries = K.array_ops.tile(
            object_queries,
            [batch_dim, 1, 1])  # (batch_dim, self.units+self.conv_units, n_digits)
        # object_keys = self.object_keys
        object_keys = K.expand_dims(self.object_keys, 0)

        object_keys = K.array_ops.tile(
            object_keys,
            [batch_dim, 1, 1])  # (batch_dim, Tk, n_digits)

        # (batch_dim, self.units+self.conv_units, n_actions)
        # action_queries = self.A_state
        action_queries = K.expand_dims(self.A_state, 0)
        action_queries = K.array_ops.tile(
            action_queries,
            [batch_dim, 1, 1])

        # action_keys = self.action_keys
        action_keys = K.expand_dims(self.action_keys, 0)
        action_keys = K.array_ops.tile(
            action_keys,
            [batch_dim, 1, 1])  # (batch_dim, Tk, n_actions)

        # Context generator
        p_object = self.p_gate(
            object_queries, object_keys)  # (batch_dim, timesteps n_digits), (batch_dim, Tk, n_digits) -> (batch_dim, timesteps n_digits)

        shifted_object_sequence = self._transformer_shift_objects(
            object_input, object_queries)  # (batch_dim, timesteps n_digits)
        # (batch_dim, timesteps n_digits), (batch_dim, timesteps n_digits) -> (batch_dim, timesteps n_digits)
        object_query_corrected = math_ops.multiply(
            p_object, shifted_object_sequence)

        # (batch_dim, timesteps n_digits), (batch_dim, Tk, n_digits), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps n_actions)
        action_by_object = self.an_gate(
            object_query_corrected, action_keys, key=object_keys)

        # Sympathetic circuit
        # (batch_dim, timesteps)
        steps = K.array_ops.tile(
            K.expand_dims(K.init_ops.constant_op.constant(list(range(self.units+self.conv_units)),
                                                          dtype=tf.float32), axis=0), K.init_ops.constant_op.constant([batch_dim, 1]))

        # (batch_dim, timesteps), (batch_dim, timesteps) -> (batch_dim, timesteps)
        old_reward = self.internal_reward
        am_r = self._wAmount_gate(math_ops.exp(
            K.expand_dims(reward_input, axis=0)))
        st_r = self._wStep_gate(math_ops.exp(steps))
        bs_r = self._wBoost_gate(reward_input * math_ops.exp(steps))
        self.internal_reward = self.internal_reward + bs_r - st_r - am_r
        if len(self.internal_reward.shape) > 2:
            self.internal_reward = self.internal_reward[0]

        # (batch_dim, timesteps n_digits), (batch_dim, timesteps n_actions) -> (batch_dim, n_digits, n_actions)
        corrected_strategy = self.ao_gate(
            action_queries, action_keys)
        reward_matrix = K.softmax(special_math_ops.einsum('ijk,ijn->ikn',
                                                          object_query_corrected, corrected_strategy) / math_ops.sqrt(0.5 * self.n_actions * n_digits))

        # (batch_dim, timesteps, n_digits), (batch_dim, timesteps, n_actions) -> (batch_dim, timesteps, n_digits, n_actions)
        potential_reward = K.softmax(math_ops.tanh(
            special_math_ops.einsum('ijk,ijn->ijkn',
                                    object_query_corrected, corrected_strategy)))

        # (batch_dim, timesteps, n_digits, n_actions) * (batch_dim, timesteps) -> (batch_dim, timesteps, n_digits, n_actions)

        delta_stimuli = special_math_ops.einsum(
            'ijkn,ij->ijkn', potential_reward, self.internal_reward)

        # ws(timesteps, n_digits, n_actions) * (batch_dim, timesteps, n_digits, n_actions) -> (batch_dim, timesteps, n_digits, n_actions)
        new_state = self._wStimuli_gate(delta_stimuli)

        # (batch_dim, n_digits, n_actions), (batch_dim, timesteps, n_digits, n_actions) -> (batch_dim, timesteps)
        reward_intersection = special_math_ops.einsum(
            'ikn,ijkn->ij', reward_matrix, new_state)
        # w(1,) * (batch_dim, timesteps) + (batch_dim, timesteps) -> (batch_dim, timesteps)
        reward_forecast = self._wrs_gate(
            reward_intersection) + self.internal_reward

        reward_keys = special_math_ops.einsum(
            'ikn,ipk->ipn', reward_matrix, object_keys)
        stimuli_keys = special_math_ops.einsum(
            'ijkn,ipn->ipn', new_state, action_keys)

        # (batch_dim, n_digits), (batch_dim, timesteps, n_digits, n_actions), (batch_dim, timesteps, n_digits, n_actions) -> (batch_dim, n_actions)
        rewarded_actions = self.SR_gate(
            K.expand_dims(action_by_object[:, -1, ...], axis=1), reward_keys, key=stimuli_keys,)

        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_actions), (batch_dim, Tk, n_digits) -> (batch_dim, n_digits)
        object_forecast = self.f_gate(
            rewarded_actions, object_keys, key=action_keys)
        # (batch_dim, timesteps n_digits) -> (batch_dim, timesteps n_digits)
        object_forecast_seq = self._transformer_shift_objects(
            object_forecast[:, -1, ...], shifted_object_sequence)
        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_digits), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps, n_actions)
        simulated_action = self.d_gate(
            object_forecast_seq, action_keys, key=object_keys)

        # Repeater
        # (batch_dim, timesteps n_actions), ((batch_dim, v) -  (batch_dim, timesteps)) ->
        # w(1,), (batch_dim, timesteps) -> (batch_dim, timesteps n_actions)
        reward_ratio_action = self._wR_gate(special_math_ops.einsum('ijk,ij->ijk', action_by_object,
                                                                   K.softmax(K.abs(self.internal_reward - self.expected_reward))))

        #  (batch_dim, timesteps n_actions), (batch_dim, timesteps n_actions) -> (batch_dim, timesteps, n_actions)
        simulated_action = self._wS_gate(simulated_action)
        selected_action = reward_ratio_action + \
            special_math_ops.einsum('ij,ijn->ijn',
                                    K.softmax(
                                        K.abs(self.internal_reward - self.expected_reward)),
                                    simulated_action)

        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps, n_actions)
        new_strategy = self._transformer_shift_actions(
            selected_action[:, -1, ...], corrected_strategy)  # (batch_dim, timesteps n_actions)

        # Packing and updating

        new_obj = object_query_corrected[:, -1, ...]
        new_act = new_strategy[:, -1, ...]

        O_c1 = K.dot(object_queries[:, self.conv_units:],
                     K.transpose(new_obj))
        O_c2 = K.dot(object_queries[:, :self.units], K.transpose(new_obj))
        E_c1 = (1 / self.units) * K.expand_dims(special_math_ops.einsum(
            'ijk->i', (object_queries[:, self.conv_units:] - O_c1)**2), axis=-1)  # (timesteps,)
        E_c2 = (1 / self.units) * K.expand_dims(special_math_ops.einsum(
            'ijk->i', (object_queries[:, :self.units] - O_c2)**2), axis=-1)  # (timesteps,)
        P_short = self._wPshort_gate(E_c1)
        P_long = self._wPlong_gate(E_c2)
        self.internal_reward = self.internal_reward[0]
        reward_diff = reward_forecast[-1, -1] - self.internal_reward[-1, ]

        # @tf.function
        def _filter_replacer_op(condition, array, replacer):
            condition = K.equal(
                array,
                K.array_ops.boolean_mask(
                    array,
                    K.reshape(condition, (condition.shape[0],)), axis=0))

            return K.array_ops.where(condition, replacer, array)

        # @tf.function
        def cond_replacer_true(value, matrix, reward_diff=reward_diff):

            min_v = K.min(K.math_ops.neg(
                K.transpose(self.relevance_memory)), axis=0)

            condition = K.transpose(K.expand_dims(
                K.equal(K.transpose(self.relevance_memory), min_v), axis=0))
            cut = K.array_ops.boolean_mask(K.transpose(K.expand_dims(
                self.relevance_memory, axis=0)), condition, axis=0)

            self.relevance_memory = K.reshape(
                K.array_ops.where(
                    condition, K.array_ops.tile(
                        [[reward_diff]],  (matrix.shape[0], 1)),
                    K.expand_dims(self.relevance_memory, axis=-1)),
                self.relevance_memory.shape)

            return _filter_replacer_op(condition, matrix, K.array_ops.tile(value, (matrix.shape[0], 1)))

        # @tf.function
        def cond_replacer_false(value, matrix, reward_diff=reward_diff):

            cov_filter = K.math_ops.abs(
                K.dot(matrix, K.transpose(value)))
            max_v = K.max(
                K.transpose(cov_filter), axis=0)

            condition = K.transpose(
                K.equal(K.transpose(cov_filter), max_v))

            cut = K.array_ops.boolean_mask(
                K.transpose(K.expand_dims(
                    self.relevance_memory, axis=0)), condition, axis=0)

            self.relevance_memory = K.reshape(K.array_ops.where(
                condition, K.array_ops.tile(
                    [(cut + reward_diff) / 2], (matrix.shape[0], 1)),
                K.expand_dims(self.relevance_memory, axis=-1)),
                self.relevance_memory.shape)

            # condition = K.equal(
            #     matrix,
            #     K.array_ops.boolean_mask(
            #         matrix,
            #         K.reshape(condition, (condition.shape[0],)), axis=0))

            # return K.array_ops.where(condition, K.tile(value, (matrix.shape[0], 1)), matrix)

            return _filter_replacer_op(condition, matrix, K.array_ops.tile(value, (matrix.shape[0], 1)))

        condition = K.math_ops.logical_and(
            K.less(P_short, 0.51), K.less(P_long, 0.51))

        object_keys = K.switch(condition,
                               cond_replacer_true(
                                   new_obj, object_keys[-1, ...]),
                               cond_replacer_true(
                                   new_obj, object_keys[-1, ...]),
                               )
        action_keys = K.switch(condition,
                               cond_replacer_true(
                                   new_act, action_keys[-1, ...]),
                               cond_replacer_true(
                                   new_act, action_keys[-1, ...]),
                               )

        # if condition:
        #     object_keys = cond_replacer_true(new_obj, object_keys[-1, ...])
        #     action_keys = cond_replacer_true(new_act, action_keys[-1, ...])
        # else:
        #     object_keys = cond_replacer_true(new_obj, object_keys[-1, ...])
        #     action_keys = cond_replacer_true(new_act, action_keys[-1, ...])

        # object_keys = _filter_replacer_op(condition,
        #                                   cond_replacer_false(
        #                                       new_obj, object_keys[-1, ...]),
        #                                   )

        # object_keys = _filter_replacer_op(condition,
        #                                   cond_replacer_false(
        #                                       new_obj, object_keys[-1, ...]),
        #                                   )

        # object_keys = K.array_ops.where(
        #     condition, cond_replacer_true(new_obj, object_keys[-1, ...]),
        #     cond_replacer_false(new_obj, object_keys[-1, ...]))
        # action_keys = K.array_ops.where(
        #     condition, cond_replacer_true(new_act, action_keys[-1, ...]),
        #     cond_replacer_false(new_act, action_keys[-1, ...]))
        update_reward_diff = old_reward[-1] - self.internal_reward[-2]
        object_keys = cond_replacer_false(
            new_obj, object_keys, update_reward_diff)
        action_keys = cond_replacer_false(
            new_act, action_keys, update_reward_diff)

        self.expected_reward = reward_forecast[-1, ...]
        self.S_state = self.S_state + K.sum(new_state, axis=0)
        self.O_state = object_query_corrected[-1]
        self.A_state = corrected_strategy[-1]
        self.object_keys = object_keys
        self.action_keys = action_keys

        # self._update_relevance_matrix(
        #     self.internal_reward, object_query_corrected[:, -2, ...], new_strategy[:, -2, ...])  # t-1 case
        # self._update_relevance_matrix(
        #     self.expected_reward, new_obj, new_act)  # t case

        if self.return_sequences:
            model = tf.keras.Model(
                inputs=[inputs, rewards], outputs=new_strategy)
        else:
            model = tf.keras.Model(inputs=[inputs, rewards], outputs=new_act)

        return model


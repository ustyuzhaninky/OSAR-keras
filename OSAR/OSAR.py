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
from tensorflow import nn
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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import special_math_ops

from OSAR.multi_head_attention import MultiHeadAttention 

__all__ = ['OSAR', 'transformerShift']

def transformerShift(inputs, memory, conv_units, short_units,
                     compression_rate:int=2, 
                     method:str='conv',
                     use_bias:bool=True,
                     kernel=None,
                     bias=None,
                     activation=None,
                     ):
    """Compressive Transformer left-shift operation with fixed short and convolution memories
       
       # Arguments
            inputs: A +3D Tensor with shape: `(timesteps, ..., features)`.
            memory: A +3D tensor with shape: `(conv_units+short_units, ..., features)`.
            conv_units: int > 0, Number of convolution memory units in the layer.
            short_units: int > 0, Number of short-term memory units in the layer.
            compression_rate: int >= 1 (default - 2). Compression rate for the selected compression method.
            method: str [`conv`, `mean`, `max`], Defines a memory compression function (default - conv).
                Available compression functions:
                - `conv`: 1D convolution over memory sequence.
                - `mean`: 1D Mean keras.layers.AveragePooling1D.
                - `max`: 1D Max Pooling via keras.MaxPooling1D.

            Only for method=`conv`:
                use_bias: Boolean (default - True), whether the layer uses a bias vector.
                kernel: A 3D Tensor with shape: `(compression_rate, features, features)`.
                bias: A Tensor of shape `(features,)`
                        
            activation: (default - None) Activation functon applied to the output of the compression function ( see keras.activations).

        # Inputs Shape
            +3D tensor with shape: `(batch_size, timesteps, ..., features)`,
            +3D tensor with shape: `(conv_units+short_units, ..., features)`.
        # Output Shape
            +3D tensor with shape: `(batch_size, conv_units+short_units, ..., features)`.
        # References
            - ["Compressive Transformers for Long-Range Sequence Modelling" by Rae et. al.](https://arxiv.org/abs/1911.05507)
    
    """

    if method not in ['conv', 'mean', 'max']:
        raise ValueError(
            f'Compression method `{method}` is not supported.')
    if len(inputs.shape) < 3:
        raise ValueError(
            'The dimension of the input vector'
            ' should be at least 3D: `(batch_size, time_steps, ..., features)`')
    if len(memory.shape) < 2:
        raise ValueError(
            'The dimension of the memory vector'
            ' should be at least 2D: `(conv_units+short_units, ..., features)`')

    if len(inputs.shape) < 3:
        raise ValueError('The dimensions of the first tensor of the inputs to `transformerShift` '
                         f'should be at least 3D: `(batch_size, timesteps, ..., features)`. Found `{inputs.shape}`.')
    if len(memory.shape) < 2:
        raise ValueError('The dimensions of the second tensor of the memory to `transformerShift` '
                         f'should be at least 2D: `(conv_units+short_units, ..., features)`. Found `{memory.shape}`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is None:
        raise ValueError('The last dimension of the first tensor of the inputs to `transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(memory.shape[-1]) is None:
        raise ValueError('The last dimension of the second tensor of the inputs (memory) to `transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is not \
            tensor_shape.dimension_value(memory.shape[-1]):
        raise ValueError('The last dimension of both input tensors to `transformerShift` '
                         f'should match. Found `{tensor_shape.dimension_value(inputs.shape[-1])}` and '
                         f'`{tensor_shape.dimension_value(memory.shape[-1])}`.')
    
    if compression_rate < 1:
        raise ValueError('Compression rate cannot be less than 1.')
    if method in ['max', 'mean']:
        compression_rate = np.int(np.floor(compression_rate))

    activation = activations.get(activation)

    timesteps = inputs.shape[1]

    memories = []
    batch_memory = tf.reshape(memory, (1,)+memory.shape)

    for i, batch in enumerate(inputs):
        # Forgetting the oldest
        conv_mem = batch_memory[:, :conv_units]
        short_mem = batch_memory[:, conv_units:]
        old_memory = short_mem[:, :timesteps, ...]

        # Compressing the oldest memories
        if method is 'conv':
            if kernel.shape[1:-1] != batch.shape[1:]:
                raise ValueError(
                    'The dimension of the kernel should be 3D: `(compression_rate, features, features)` '
                    'if shape of the input is 3D: `(batch_size, time_steps, features)`. '
                    f'Found `{kernel.shape[1:-1]}`!=`{batch.shape[1:]}`')
            compression = nn.conv1d(old_memory,
                                    filters=kernel,
                                    stride=compression_rate,
                                    padding='SAME',
                                    name='Compression')
            if use_bias:
                if bias.shape[-1] != compression.shape[-1]:
                    raise ValueError(
                        'The dimension of the bias'
                        ' should be equal to the number of features. '
                        f'Found `{bias.shape[-1]}`!=`{kernel.shape[-1]}`')
                compression = nn.bias_add(
                    compression, bias)
        elif method is 'mean':
            compression = nn.avg_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        elif method is 'max':
            compression = nn.max_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        else:
            compression = nn.avg_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        if compression.shape[-1] > conv_mem.shape[-1]:
            compression = nn.avg_pool1d(compression, compression_rate,
                                        strides=compression_rate, padding='SAME')
        if activation is not None:
            compression = activation(compression)

        # Update Short-Term Memory
        short_mem = K.concatenate([short_mem[:, self.units:], tf.reshape(batch, (1,)+batch.shape)], axis=1)

        # Update Compressed Memory
        conv_mem = K.concatenate([conv_mem[:, 1:, ...], compression], axis=1)

        batch_memory = K.concatenate([conv_mem, short_mem], axis=1)
        memories.append(batch_memory)

    return K.concatenate(memories, axis=0)

class OSAR(Layer):

    """OSAR - Object-Stimulated Active Repeater

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            memory_size: int > 1. Size of the dictionary memory. Big numbers allows to store more combinations, but requires more space and speed.
            conv_units: int > 0. Number of convolution units in the layer.
            n_actions: int > 1. Number of actions in the output vector.
            num_heads: int > 0. Number of attention `heads`.
            dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
            use_bias: Boolean. Whether to use bias (dafault - True).
            compression_rate: int. Compression rate of the transfomer memories.
            gate_initializer: keras.initializer. Initializer for attention weights (default - glorot_normal).
            gate_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            gate_constraint: keras.constraint. Constraint for attention weights (default - None).
            state_initializer: keras.initializer. Initializer for state matrices (default - glorot_uniform).
            state_constraint: keras.constraint. Constraint for state matrices (default - None).
            bias_initializer: keras.initializer. Initializer for biases (default - zeros).
            bias_regularizer: keras.regularizer. Regularizer for biases (default - l2).
            bias_constraint: keras.constraint. Constraint for attention weights (default - None),
            return_sequences: bool If the layer should return sequences (default - False).
        
        # Input Shape
            3D tensor with shape: `(batch_size, sequence_length, units)`.
        # Output Shape
            
        # References
            - None yet

        """
    
    def __init__(self,
                 units,
                 memory_size,
                 conv_units,
                 n_actions,
                 num_heads=10,
                 dropout=0.0,
                 use_bias=True,
                 compression_rate=4,
                 gate_initializer='glorot_normal',
                 gate_regularizer='l2',
                 gate_constraint=None,
                 state_initializer='glorot_uniform',
                 state_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer='l2',
                 bias_constraint=None,
                 return_sequences = False,
                 **kwargs):
        
        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self._return_runtime = kwargs.pop('return_runtime', False)

        super(OSAR, self).__init__(
            dynamic=True,
            **kwargs)

        self.units = units
        self.n_actions = n_actions
        self.conv_units = conv_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.memory_size = memory_size
        self.compression_rate = compression_rate
        self.state_constraint = constraints.get(state_constraint)
        self.state_initializer = initializers.get(state_initializer)

        self.gate_initializer = initializers.get(gate_initializer)
        self.gate_regularizer = regularizers.get(gate_regularizer)
        self.gate_constraint = constraints.get(gate_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.return_sequences = return_sequences

    def build(self, input_shape):
        input_shape = input_shape[0]
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `OSAR` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `OSAR` '
                             'should be defined. Found `None`.')
        n_digits = tensor_shape.dimension_value(input_shape[-1])
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        # if tensor_shape.dimension_value(input_shape[1]) != self.units:
        #     raise ValueError('The lenght of timesteps in the input vector '
        #                      'should be equal to number of short-term memory units '
        #                      f'in the layer. Got { tensor_shape.dimension_value(input_shape[1])}!={self.units}')
        
        self.input_spec = InputSpec(min_ndim=2, axes={-1: n_digits})
        self.units = self.units

        # Probability gate
        self.p_gate = MultiHeadAttention(
            self.num_heads,
            n_digits,
            use_bias=self.use_bias,
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
            n_digits,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
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
            self.n_actions,
            use_bias=self.use_bias,
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
            self.n_actions,
            value_dim=n_digits,
            use_bias=self.use_bias,
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
            n_digits,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
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
            self.n_actions,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='AO-Gate',
        )

        self.o_compression_kernel = self.add_weight(
            name='o_compression_kernel',
            shape=(self.compression_rate, n_digits, n_digits),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.o_compression_bias = self.add_weight(
            name='o_compression_bias',
            shape=(n_digits,),
            constraint=self.bias_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        self.a_compression_kernel = self.add_weight(
            name='a_compression_kernel',
            shape=(self.compression_rate, self.n_actions, self.n_actions),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.a_compression_bias = self.add_weight(
            name='a_compression_bias',
            shape=(self.n_actions,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.w_boost = self.add_weight(
            name='w_boost',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.w_step = self.add_weight(
            name='w_step',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.w_amount = self.add_weight(
            name='w_amount',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.w_stimuli = self.add_weight(
            name='w_stimuli',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.w_rs = self.add_weight(
            name='w_rs',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.W_S = self.add_weight(
            name='W_S',
            shape=(self.units, self.n_actions),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.b_S = self.add_weight(
            name='b_S',
            shape=(self.units, self.n_actions),
            constraint=self.bias_constraint,
            regularizer=self.bias_regularizer,
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        self.W_R = self.add_weight(
            name='W_R',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.W_Pshort = self.add_weight(
            name='W_Pshort',
            shape=(self.units,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.W_Plong = self.add_weight(
            name='W_Plong',
            shape=(self.units,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.W_Pnew = self.add_weight(
            name='W_Pnew',
            shape=(self.units,),
            constraint=self.gate_constraint,
            regularizer=self.gate_regularizer,
            initializer=self.gate_initializer,
            dtype=self.dtype,
            trainable=True)
        self.b_Pshort = self.add_weight(
            name='b_Pshort',
            shape=(1,),
            constraint=self.bias_constraint,
            regularizer=self.bias_regularizer,
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        self.b_Plong = self.add_weight(
            name='b_Plong',
            shape=(1,),
            constraint=self.bias_constraint,
            regularizer=self.bias_regularizer,
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        self.b_Pnew = self.add_weight(
            name='b_Pnew',
            shape=(1,),
            constraint=self.gate_constraint,
            regularizer=self.bias_regularizer,
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)

        self.A_state = self.add_weight(
            name='A_state',
            shape=(self.conv_units+self.units, self.n_actions,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.O_state = self.add_weight(
            name='O_state',
            shape=(self.conv_units+self.units, n_digits,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.S_state = self.add_weight(
            name='S_state',
            shape=(self.conv_units+self.units, n_digits, self.n_actions,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.expected_reward = self.add_weight(
            name='expected_reward',
            shape=(self.conv_units+self.units,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.internal_reward = self.add_weight(
            name='internal_reward',
            shape=(self.conv_units+self.units,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        
        self.relevance_memory = self.add_weight(
            name='relevance_memory',
            shape=(self.memory_size,),
            constraint=self.state_constraint,
            initializer=initializers.get('zeros'),
            dtype=self.dtype,
            trainable=False)
        self.object_keys = self.add_weight(
            name='object_keys',
            shape=(self.memory_size, n_digits,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.action_keys = self.add_weight(
            name='action_keys',
            shape=(self.memory_size, self.n_actions,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)

        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0] + (self.n_actions,))

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if not (inputs.shape is 2):
            raise ValueError(
                'The dimension of the inputs vector should be 2: `(input_shape, reward)`')
        object_input = inputs[0]  # (batch_dim, timesteps n_digits)
        reward_input = inputs[1] # (1,)

        n_digits = tensor_shape.dimension_value(object_input[-1])
        batch_dim = tensor_shape.dimension_value(object_input[0])
        self.units = tensor_shape.dimension_value(object_input[1])

        # Unpacking state matrices
        object_queries = tf.tile(
            tf.reshape(self.O_state, (1,)+self.O_state.shape),
            (batch_dim,) + self.O_state.shape) # (batch_dim, timesteps n_digits)
        object_keys = tf.tile(
            tf.reshape(self.object_keys, (1,)+self.object_keys.shape),
            (batch_dim,) + self.object_keys.shape)  # (batch_dim, Tk, n_digits)

        # (self.units, n_actions)
        action_queries = tf.tile(
            tf.reshape(self.A_state, (1,)+self.A_state.shape),
            (batch_dim,) + self.A_state.shape)

        action_keys = tf.tile(
            tf.reshape(self.action_keys, (1,)+self.action_keys.shape),
            (batch_dim,) + self.action_keys.shape
        )  # (Tk, n_actions)
        # action_values = self.O_state[:, 2*int(
        #     self.A_state.shape[1] / 3):3*int(self.A_state.shape[1] / 3), :]

        # Context generator
        p_object = self.p_gate(
            [object_queries, object_keys])  # (batch_dim, timesteps n_digits), (batch_dim, Tk, n_digits) -> (batch_dim, timesteps n_digits)
        
        shifted_object_sequence = self._transformer_shift_objects(
            object_input, object_queries)  # (batch_dim, timesteps n_digits)
        
        # (batch_dim, timesteps n_digits), (batch_dim, timesteps n_digits) -> (batch_dim, timesteps n_digits)
        object_query_corrected = math_ops.multiply(
            p_object, shifted_object_sequence)
        
        # (batch_dim, timesteps n_digits), (batch_dim, Tk, n_digits), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps n_actions)
        action_by_object = self.an_gate(
            [object_query_corrected, object_keys, action_keys])

        # Sympathetic circuit
        # (batch_dim, timesteps)
        steps = tf.tile(tf.constant(list(range(self.units)), dtype=float), tf.constant([1, batch_dim]))
        
        # (batch_dim, timesteps), (batch_dim, timesteps) -> (batch_dim, timesteps)
        old_reward = self.internal_reward
        self.internal_reward.assign(self.internal_reward + self.w_boost * reward_input * math_ops.exp(steps) - \
            self.w_step * math_ops.exp(steps) - \
            self.w_amount * math_ops.exp(reward_input))
        
        # (batch_dim, timesteps n_digits), (batch_dim, timesteps n_actions) -> (batch_dim, timesteps, n_digits, n_actions)
        corrected_strategy = self.ao_gate(action_queries, action_keys)
        reward_matrix = K.softmax(tf.einsum('ijk,ijn->ijkn',
            object_queries, corrected_strategy) / math_ops.sqrt(0.5 * self.n_actions * n_digits))
        
        # (batch_dim, timesteps n_digits), (batch_dim, timesteps n_actions) -> (batch_dim, timesteps n_digits, n_actions)
        potential_reward = K.softmax(math_ops.tanh(
            tf.einsum('ijk,ijn->ijkn',
                      object_query_corrected, corrected_strategy)))

        # (batch_dim, timesteps n_digits, n_actions) * (batch_dim, timesteps) -> (batch_dim, timesteps n_digits, n_actions)
        delta_stimuli = potential_reward * self.internal_reward 
        # tf.einsum('ijkn,ij->jkn', potential_reward, self.internal_reward)
        
        # ws(n_digits, n_actions) * (batch_dim, timesteps n_digits, n_actions) -> (batch_dim, timesteps n_digits, n_actions)
        new_state = self.w_stimuli * delta_stimuli
        
        # (batch_dim, timesteps n_digits, n_actions), (timesteps, n_digits, n_actions) -> (batch_dim, self.units)
        reward_intersection = tf.einsum('ijkn,ijkn->ij', reward_matrix, new_state)
        # w(1,) * (batch_dim, timesteps) + (batch_dim, timesteps) -> (batch_dim, timesteps)
        reward_forecast = self.w_rs * reward_intersection + self.internal_reward

        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_digits), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps n_actions)
        rewarded_actions = self.SR_gate(
            [action_by_object, new_state, reward_matrix])

        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_actions), (batch_dim, Tk, n_digits) -> (batch_dim, timesteps n_digits)
        object_forecast = self.f_gate([rewarded_actions, action_keys, object_keys])
        # (batch_dim, timesteps n_digits) -> (batch_dim, timesteps n_digits)
        object_forecast_seq = self._transformer_shift_objects(
            object_forecast, shifted_object_sequence)
        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_digits), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps n_actions)
        simulated_action = self.d_gate(
            [object_forecast_seq, object_keys, action_keys])
        
        # Repeater
        # (batch_dim, timesteps n_actions), ((batch_dim, self.units) -  (batch_dim, self.units)) ->
        # w(1,), (batch_dim, self.units) -> (batch_dim, timesteps n_actions)
        reward_ratio_action = self.W_R * tf.einsum('ijk,ij->ijk', action_by_object,
                                       K.softmax(K.abs(self.internal_reward - self.expected_reward)))
        
        #  (batch_dim, timesteps n_actions), (batch_dim, timesteps n_actions) -> (batch_dim, timesteps n_actions)
        selected_action = reward_ratio_action + \
            K.softmax(K.abs(self.internal_reward - self.expected_reward)) * \
                K.softmax(K.dot(self.W_S, simulated_action)+self.b_S)
        
        # (batch_dim, timesteps n_actions), (batch_dim, Tk, n_actions) -> (batch_dim, timesteps n_actions)
        
        new_strategy = self._transformer_shift_actions(
            selected_action, corrected_strategy)  # (batch_dim, timesteps n_actions)

        # Packing and updating
        
        new_obj = object_query_corrected[:, -1, ...]
        new_act = new_strategy[:, -1, ...]
        O_c1 = K.dot(object_queries[self.conv_units:], tf.transpose(new_obj))
        O_c2 = K.dot(object_queries[:self.units], tf.transpose(new_obj))
        E_c1 = (1 / self.units) * tf.einsum(
            'ik->', (object_queries[self.conv_units:] - O_c1)**2) # (timesteps,)
        E_c2 = (1 / self.units) * tf.einsum(
            'ik->', (object_queries[self.conv_units:] - O_c2)**2)  # (timesteps,)
        P_short = tf.math.softmax(K.dot(E_c1, self.W_Pshort) + self.b_Pshort)
        P_long = tf.math.softmax(K.dot(E_c2, self.W_Plong) + self.b_Plong)

        if (P_short < 0.51) & (P_long < 0.51):
            object_keys, action_keys = self._min_ABdict_replace_op(
                object_keys[-1, ...],
                action_keys[-1, ...], new_obj, new_act, reward_forecast[-1, ...] - self.internal_reward)
        else:
            object_keys, action_keys = self._mean_ABdict_mix_op(
                object_keys[-1, ...],
                action_keys[-1, ...], new_obj, new_act, reward_forecast[-1, ...] - self.internal_reward)
        
        object_keys, action_keys = self._mean_ABdict_mix_op(
            object_keys[-2, ...],
            action_keys[-2, ...], new_obj, new_act, old_reward - self.internal_reward)
            
        self.expected_reward.assign(reward_forecast[-1, ...])
        self.S_state.assign(tf.cumsum(self.S_state, axis=0) + tf.reduce_sum(new_state, axis=0))
        self.O_state.assign(object_query_corrected)
        self.A_state.assign(corrected_strategy)
        self.object_keys.assign(object_keys)
        self.action_keys.assign(action_keys)
        
        # self._update_relevance_matrix(
        #     self.internal_reward, object_query_corrected[:, -2, ...], new_strategy[:, -2, ...])  # t-1 case
        # self._update_relevance_matrix(
        #     self.expected_reward, new_obj, new_act)  # t case
        if self.return_sequences:
            return new_strategy
        else:
            return new_strategy[..., -1, :]
    
    def _min_ABdict_replace_op(self, memory_matrix_A, memory_matrix_B, vector_A, vector_B, new_relevance):
        '''Replaces most irrelevant member of AB dictionary'''

        min_idx = tf.argmin(self.relevance_memory, axis=0) # minumum of relevance element

        zero_mask_A = tf.concat([memory_matrix_A[:min_idx], 
                                 tf.zeros_like(memory_matrix_A[min_idx]),
                                 memory_matrix_A[min_idx+1:]], axis=0)
        add_mask_A = tf.concat([memory_matrix_A[:min_idx], 
                                vector_A,
                                memory_matrix_A[min_idx+1:]], axis=0)
        
        zero_mask_B = tf.concat([memory_matrix_B[:min_idx],
                                 tf.zeros_like(memory_matrix_B[min_idx]),
                                 memory_matrix_B[min_idx+1:]], axis=0)
        add_mask_B = tf.concat([memory_matrix_B[:min_idx],
                                vector_B,
                                memory_matrix_B[min_idx+1:]], axis=0)
        
        self.relevance_memory[min_idx].assign(new_relevance)
        
        return tf.math.add(memory_matrix_A * zero_mask_A, add_mask_A), \
            tf.math.add(memory_matrix_B * zero_mask_B, add_mask_B)

    def _mean_ABdict_mix_op(self, memory_matrix_A, memory_matrix_B, vector_A, vector_B, new_relevance):
        '''Mixes new AB pair with the most relevant member of AB dictionary with mean of the two'''
        cov_filter = tf.math.abs(K.dot(memory_matrix_A, tf.transpose(vector_A)))
        max_idx = tf.argmax(cov_filter, axis=0)

        add_mask_A = tf.concat([memory_matrix_A[:max_idx],
                                vector_A,
                                memory_matrix_A[max_idx+1:]], axis=0)

        add_mask_B = tf.concat([memory_matrix_B[:max_idx],
                                vector_B,
                                memory_matrix_B[max_idx+1:]], axis=0)

        self.relevance_memory[max_idx].add(new_relevance)
        self.relevance_memory[max_idx].assign(self.relevance_memory[max_idx] / 2)

        return tf.math.add(memory_matrix_A, add_mask_A) / 2, \
            tf.math.add(memory_matrix_B, add_mask_B) / 2

    def _transformer_shift_objects(self, new_object, object_memory):
        return transformerShift(
            new_object, object_memory,
            conv_units=self.conv_units, short_units=self.units,
            compression_rate=self.compression_rate,
            kernel=self.o_compression_kernel, bias=self.o_compression_bias,)

    def _transformer_shift_actions(self, new_action, action_memory):
        return transformerShift(
            new_action, action_memory,
            conv_units=self.conv_units, short_units=self.units,
            compression_rate=self.compression_rate,
            kernel=self.a_compression_kernel, bias=self.a_compression_bias,)

    def get_config(self):
        config = {
            'units': self.units,
            'conv_units': self.conv_units,
            'n_actions': self.n_actions,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'memory_size': self.memory_size,
            'compression_rate': self.compression_rate,
            'state_constraint': constraints.serialize(self.state_constraint),
            'state_initializer': initializers.serialize(self.state_initializer),

            'gate_initializer': initializers.serialize(self.gate_initializer),
            'gate_regularizer': regularizers.serialize(self.gate_regularizer),
            'gate_constraint': constraints.serialize(self.gate_constraint),

            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(OSAR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

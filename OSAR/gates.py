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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import special_math_ops
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow import keras
from tensorflow.keras import backend as K
from .tfxl import FeedForward as LinearGate
from .tfxl import AdaptiveEmbedding, AdaptiveSoftmax, PositionalEmbedding, \
    Scale, LayerNormalization, RelativePartialMultiHeadSelfAttention, FeedForward, Memory
from . import HelixMemory
from . import GraphAttention

__all__ = ['LinearGate', 'AttentionGate', 'TransferGate', 'SequenceEncoder1D', 'FiniteDifference']


class TransferGate(tf.keras.layers.Dense):
    """Copy of Dense layerd with softmax activation and noise chanel"""

    def __init__(self,
               units,
               activation='softmax',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               noise_chanel_generator='glorot_uniform',
               **kwargs):
        super(TransferGate, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.noise_chanel_generator = tf.keras.initializers.get(
            noise_chanel_generator)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            f'{self.name}-kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.noisy_kernel = self.add_weight(
            f'{self.name}-noisy-kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                f'{self.name}-bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            self.noisy_bias = self.add_weight(
                f'{self.name}-noisy-bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
            self.noisy_bias = None
        self.built = True

    def call(self, inputs):
        noise = self.noise_chanel_generator((1,))
        return self.activation(
            core_ops.dense(
                inputs,
                self.noisy_kernel,
                self.noisy_bias,
                None,
                dtype=self._compute_dtype_object) +
            core_ops.dense(
                inputs,
                self.kernel * noise,
                self.bias * noise,
                None,
                dtype=self._compute_dtype_object))

    def get_config(self):
        config = super(TransferGate, self).get_config()
        config.update({
            'noise_chanel_generator':
                tf.keras.initializers.serialize(self.noise_chanel_generator),
        })
        return config


class FiniteDifference(tf.keras.layers.Layer):
    """Calculates finite difference across given axis.
    """

    def __init__(self,
                 order,
                 **kwargs):
        super(FiniteDifference, self).__init__(**kwargs)

        self.order = int(order) if not isinstance(
            order, int) else order
        self.supports_masking = True

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `FiniteDifference` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)

        self.input_memory = self.add_weight(
            shape=(self.order,)+input_shape[1:],
            initializer=tf.keras.initializers.get('zeros'),
            trainable=False,
            name=f'{self.name}-input-memory'
        )

    def call(self, inputs, **kwargs):

        batch_size = inputs.shape[0]

        mem = tf.tile(
            tf.expand_dims(self.input_memory, axis=0),
            [batch_size,]+[1 for i in range(len(self.input_memory.shape))])
        inputs_reshaped = tf.tile(
            K.reshape(inputs, (batch_size, 1,)+inputs.shape[1:]),
            [1, self.order, ]+[1 for i in range(len(inputs.shape[1:]))]
        )
        diff = inputs_reshaped - mem

        return K.sum(diff, axis=1) / (2*self.order)

    def get_config(self):
        config = super(FiniteDifference, self).get_config()
        config.update({
            'order': self.order,
        })
        return config

class SequenceEncoder1D(tf.keras.layers.Dense):
    """Encodes and decodes sequences into one another.

        # Arguments
            units: int >= 0. Dimension of hidden units.
            activation: Activation function to use
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            kernel_constraint: Constraint function applied to the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, units, units)`.

        # References
            - 
    """

    def __init__(self,
                 feature_units,
                 timesteps_units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SequenceEncoder1D, self).__init__(
            feature_units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer, **kwargs)

        self.feature_units = int(feature_units) if not isinstance(
            feature_units, int) else feature_units
        self.timesteps_units = int(timesteps_units) if not isinstance(
            timesteps_units, int) else timesteps_units

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `SequenceEncoder1D` layer with non-floating point '
                            'dtype %s' % (dtype,))
        if not len(input_shape):
            raise ValueError('Layer input should be represented by 3D tensors of shape'
                             '`(batch_size, sequence_length, input_dim)`. '
                             f'Found: `{input_shape}`')
        input_shape = tensor_shape.TensorShape(input_shape)
        timesteps = tensor_shape.dimension_value(input_shape[1])
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            f'{self.name}-kernel',
            shape=[timesteps, last_dim,
                   self.timesteps_units, self.feature_units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                f'{self.name}-bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        h = special_math_ops.einsum('ijk,jkmn->imn', inputs, self.kernel)
        if self.use_bias:
            h = K.bias_add(h, self.bias)
        if self.activation is not None:
            h = self.activation(h)
        return h
    
    def get_config(self):
        config = super(SequenceEncoder1D, self).get_config()
        config.pop('units')
        config.update({
            'feature_units': self.feature_units,
            'timesteps_units': self.timesteps_units,
        })
        return config

class AttentionGate(tf.keras.layers.Layer):
    """Attention-based gate.
    
    # Arguments
        units: int >= 0. Dimension of hidden units.
        embed_dim: int > 0. Dimension of the dense embedding.
        hidden_dim: int > 0. Number of units in the Feed-forward part.
        num_token: int > 0. Size of the vocabulary.
        num_head: int > 0. Number of heads in the attention unit.
        memory_len: int > 0 Number of memory units.
        conv_memory_len: int > 0. Number of compressed units.
        compression_rate: int > 0. Rate of memory compression.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """
    
    def __init__(self,
                 units,
                 dropout=0.0,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.units = units
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        return (batch_size, sequence_length, self.units)

    def build(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        output_dim = input_shape[-1]

        self.attention_layer_features = GraphAttention(
            self.units,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            attention_dropout=self.attention_dropout
            )
        self.attention_layer_features.build((batch_size, sequence_length, output_dim))
        self.attention_layer_features.built = True

        self.attention_layer_timesteps = GraphAttention(self.units)
        self.attention_layer_timesteps.build(
            (batch_size, output_dim, sequence_length))
        self.attention_layer_timesteps.built = True

        self.permute = tf.keras.layers.Permute((2, 1))
        self.permute.build(
            (batch_size, sequence_length, output_dim))
        self.permute.built = True

        self.permute_1 = tf.keras.layers.Permute((2, 1))
        self.permute_1.build(
            (batch_size, sequence_length, self.units))
        self.permute_1.built = True

        # self.permute_2 = tf.keras.layers.Permute((2, 1))
        # self.permute_2.build(
        #     (batch_size, output_dim, self.units))
        # self.permute_2.built = True

        self.seq2seq = tf.keras.layers.GRU(
            self.units,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            activation='sigmoid'
            )
        self.seq2seq.build(
            (batch_size, 2*output_dim, 2*sequence_length))
        self.seq2seq.built = True

        super(AttentionGate, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        
        attention_result_ts = self.attention_layer_timesteps(
            self.permute(inputs))
        attention_result_features = self.attention_layer_features(inputs)
        attention_result_features = self.permute_1(attention_result_features)
        attention_result = tf.keras.layers.Concatenate(axis=-1)(
            [
                attention_result_features,
                K.reshape(attention_result_ts, attention_result_features.shape),
            ])
        attention_result = self.seq2seq(attention_result)

        return attention_result
    
    def get_config(self):
        config = {
            "units": self.units,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(AttentionGate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

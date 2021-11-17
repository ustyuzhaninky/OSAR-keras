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
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.keras import backend as K
from . import GraphAttention

__all__ = ['AttentionGate', 'TransferGate',
           'SequenceEncoder1D', 'Transformer']


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
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)
        noise = self.noise_chanel_generator((1,))

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, sparse_tensor.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    self.kernel * noise, ids, weights, combiner='sum')
                noisy_outputs =  embedding_ops.embedding_lookup_sparse_v2(
                    self.noisy_kernel, ids, weights, combiner='sum')
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel * noise)
                noisy_outputs = gen_math_ops.MatMul(a=inputs, b=self.noisy_kernel)
                # Broadcast kernel to inputs.
        else:
            noisy_outputs = standard_ops.tensordot(inputs, self.noisy_kernel, [[rank - 1], [0]])
            outputs = standard_ops.tensordot(inputs, self.kernel * noise, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)
                noisy_outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(
                noisy_outputs, self.noisy_bias) + nn_ops.bias_add(
                    outputs, self.bias * noise)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(TransferGate, self).get_config()
        config.update({
            'noise_chanel_generator':
                tf.keras.initializers.serialize(self.noise_chanel_generator),
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
                 bias_initializer='glorot_uniform',
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
                shape=[self.timesteps_units, self.feature_units],
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
        if self.activation != None:
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
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
        attention_dropout: 0.0 <= float <= 1.0. Dropout rate for attention units.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length input_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    # References
        - [Graph Networks](http://arxiv.org/abs/2009.05602.pdf)
    """
    
    def __init__(self,
                 units,
                 dropout=0.0,
                 attention_dropout=0.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='glorot_uniform',
                 bias_regularizer=None,
                 bias_constraint=None,
                 return_probabilities=False,
                 **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.units = units
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.return_probabilities = return_probabilities
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        output_dim = input_shape[2]
        return (batch_size, sequence_length, output_dim)

    def build(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        output_dim = input_shape[-1]

        self.attention_layer_features = GraphAttention(
            self.units,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            attention_dropout=self.attention_dropout
            )
        self.attention_layer_features.build((batch_size, sequence_length, output_dim))
        self.attention_layer_features.built = True

        self.attention_layer_timesteps = GraphAttention(
            self.units,
            use_bias=self.use_bias,
            attention_dropout=self.attention_dropout,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,)
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

        self.permute_2 = tf.keras.layers.Permute((2, 1))
        self.permute_2.build(
            (batch_size, 2*output_dim, 2*sequence_length))
        self.permute_2.built = True

        self.seq2seq = tf.keras.layers.GRU(
            output_dim,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dropout=self.dropout,
            recurrent_dropout=0,
            return_sequences=True,
            reset_after=True,
            unroll=False,
            name=f'{self.name}-seq2seq'
            )
        self.seq2seq.build(
            (batch_size, sequence_length, output_dim))
        self.seq2seq.built = True

        super(AttentionGate, self).build(input_shape)
    
    # @tf.function(autograph=True)
    def call(self, inputs, **kwargs):
        
        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]
        output_dim = inputs.shape[-1]

        attention_result_ts, attention_weights_ts = self.attention_layer_timesteps(
            self.permute(inputs))
        attention_result_features, attention_weights_features = self.attention_layer_features(inputs)
        
        attention_result = tf.keras.layers.Concatenate(axis=-1)(
            [
                attention_result_features,
                self.permute_2(attention_result_ts),
            ])[:, :, -output_dim:]
        attention_weights_ts = attention_weights_ts[:, :, -sequence_length:]
        attention_weights_features = attention_weights_features[:, :, -output_dim:]
        
        attention_weights = self.permute(
            attention_weights_ts) + attention_weights_features
        
        attention_result = self.seq2seq(attention_result)
        if self.return_probabilities:
            return attention_result, attention_weights
        else:
            return attention_result
    
    def get_config(self):
        config = {
            "units": self.units,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_bias": self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionGate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

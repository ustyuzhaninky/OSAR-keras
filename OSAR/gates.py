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
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow import keras
from tensorflow.keras import backend as K
from .tfxl import FeedForward as LinearGate
from .tfxl import AdaptiveEmbedding, AdaptiveSoftmax, PositionalEmbedding, \
    Scale, LayerNormalization, RelativePartialMultiHeadSelfAttention, FeedForward, Memory
from . import CompressiveAvgPoolMemory

__all__ = ['LinearGate', 'AttentionGate', 'TransferGate']


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
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.noisy_kernel = self.add_weight(
            'noisy_kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            self.noisy_bias = self.add_weight(
                'noisy_bias',
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
                 embed_dim,
                 hidden_dim,
                 num_token,
                 num_head,
                 memory_len,
                 conv_memory_len,
                 compression_rate,
                 dropout=0.0,
                 attention_dropout=0.0,
                 cutoffs=None,
                 div_val=1,
                 force_projection=None,
                 bind_embeddings=True,
                 bind_projections=True,
                 clamp_len=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.units = units
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_token = num_token
        self.num_head = num_head
        self.memory_len = memory_len
        self.conv_memory_len = conv_memory_len
        self.compression_rate = compression_rate
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.cutoffs = cutoffs
        self.div_val = div_val
        self.force_projection = force_projection
        self.bind_embeddings = bind_embeddings
        self.bind_projections = bind_projections
        self.clamp_len = clamp_len

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

        self.adaptive_embedding = AdaptiveEmbedding(
            input_dim=self.num_token,
            output_dim=self.units,
            embed_dim=self.embed_dim,
            cutoffs=self.cutoffs,
            div_val=self.div_val,
            mask_zero=True,
            force_projection=self.force_projection,
            return_embeddings=True,
            return_projections=True,
            name='Embed-Token',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            embeddings_regularizer=self.kernel_regularizer,
            embeddings_constraint=self.kernel_constraint,
        )
        self.adaptive_embedding.build((batch_size, sequence_length))
        self.adaptive_embedding.built = True

        self.scale = Scale(scale=np.sqrt(self.units), name='Embed-Token-Scaled')
        self.scale.build((batch_size, sequence_length, self.units))
        self.scale.built = True

        self.last_memory = CompressiveAvgPoolMemory(
            batch_size,
            self.memory_len,
            self.conv_memory_len,
            self.units,
            self.compression_rate
        )
        self.last_memory.build(
            [(batch_size, sequence_length, self.units), (batch_size, 1,)])
        self.last_memory.built = True
        
        self.position_embed = PositionalEmbedding(
            output_dim=self.units,
            clamp_len=self.clamp_len,
            name='Embed-Position',
        )
        self.position_embed.build(
            [(batch_size, sequence_length, self.units), (batch_size, self.conv_memory_len+self.memory_len, self.units)])
        self.position_embed.built = True

        if 0.0 < self.dropout < 1.0:
            self.em_dropout = tf.keras.layers.Dropout(
                rate=self.dropout, name='Embed-Token-Dropped')
            self.em_dropout.build((batch_size, sequence_length, self.units))
            self.em_dropout.built = True

            self.pos_dropout = tf.keras.layers.Dropout(
                rate=self.dropout, name='Embed-Position-Dropped')
            self.pos_dropout.build(
                (batch_size, sequence_length+self.conv_memory_len+self.memory_len, self.units))
            self.pos_dropout.built = True

        self.bias_context = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_context',
            trainable=True,
        )
        self.bias_relative = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_relative',
            trainable=True,
        )

        self.attention = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=True,
            attention_dropout=self.attention_dropout,
            name='Attention-1',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.attention.build(
            [
                (batch_size, sequence_length, self.units),
                (batch_size, sequence_length +
                 self.conv_memory_len+self.memory_len, self.units),
                (batch_size, self.conv_memory_len+self.memory_len, self.units),
                (self.units,),
                (self.units,)
            ])
        self.attention.built = True

        if 0.0 < self.dropout < 1.0:
            self.block_dropout = tf.keras.layers.Dropout(
                rate=self.dropout, name='Feed-Forward-Dropped')
            self.block_dropout.build((batch_size, sequence_length, self.units))
            self.block_dropout.built = True
        
        self.att_rescale = tf.keras.layers.Add(name='Attention-Res-1')
        self.att_rescale.build([
            (batch_size, sequence_length, self.units),
            (batch_size, sequence_length, self.units)
        ])
        self.att_rescale.built = True

        self.att_norm = LayerNormalization(
            name='Attention-Norm-1',
            gamma_regularizer=self.kernel_regularizer,
            beta_regularizer=self.kernel_regularizer,
            gamma_constraint=self.kernel_constraint,
            beta_constraint=self.kernel_constraint,
            )
        self.att_norm.build((batch_size, sequence_length, self.units))
        self.att_norm.built = True

        self.feed_forward = FeedForward(
            units=self.hidden_dim,
            dropout_rate=self.dropout,
            name='FeedForward-1',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            )
        self.feed_forward.build((batch_size, sequence_length, self.units))
        self.feed_forward.built = True

        if 0.0 < self.dropout < 1.0:
            self.forward_dropout = tf.keras.layers.Dropout(
                rate=self.dropout, name='Embed-Token-Dropped')
            self.forward_dropout.build((batch_size, sequence_length, self.units))
            self.forward_dropout.built = True
        
        self.forward_rescale = tf.keras.layers.Add(name='FeedForward-Res-1')
        self.forward_rescale.build([
            (batch_size, sequence_length, self.units),
            (batch_size, sequence_length, self.units)
        ])
        self.forward_rescale.built = True

        self.forward_norm = LayerNormalization(
            name='FeedForward-Norm-1',
            gamma_regularizer=self.kernel_regularizer,
            beta_regularizer=self.kernel_regularizer,
            gamma_constraint=self.kernel_constraint,
            beta_constraint=self.kernel_constraint,)
        self.forward_norm.build((batch_size, sequence_length, self.units))
        self.forward_norm.built = True

        self.softmax = AdaptiveSoftmax(
            input_dim=self.units,
            output_dim=self.num_token,
            embed_dim=self.embed_dim,
            cutoffs=self.cutoffs,
            div_val=self.div_val,
            force_projection=self.force_projection,
            bind_embeddings=self.bind_embeddings,
            bind_projections=self.bind_projections,
            name='Softmax',
        )
        self.softmax.build([(batch_size, sequence_length, self.num_token)])
        self.softmax.built = True

        super(AttentionGate, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        
        batch_size = inputs.shape[0]
        input_features = inputs.shape[-1]

        memory_lenght = tf.tile(
            tf.expand_dims(tf.expand_dims(
                self.conv_memory_len+self.memory_len, axis=0), axis=0), [batch_size, 1])
        x = self.adaptive_embedding(inputs)
        token_embed, embedding_weights = x[0], x[1:]
        token_embed = self.scale(token_embed)
        last_memory = self.last_memory([token_embed, memory_lenght])
        position_embed = self.position_embed([token_embed, last_memory])
        
        if 0.0 < self.dropout < 1.0:
            token_embed = self.em_dropout(token_embed)
            position_embed = self.pos_dropout(position_embed)
        
        # context_bias, relative_bias = self.relative_bias(last_memory)
        x = self.attention([token_embed, position_embed,
                            last_memory, self.bias_context, self.bias_relative])
        
        if 0.0 < self.dropout < 1.0:
            x = self.block_dropout(x)
        
        x = self.att_rescale([x, token_embed])
        x = self.att_norm(x)
        x = self.feed_forward(x)

        if 0.0 < self.dropout < 1.0:
            x = self.forward_dropout(x)
        x = self.forward_rescale([x, token_embed])
        x = self.forward_norm(x)
        
        return self.softmax(x + embedding_weights)
    
    def get_config(self):
        config = {
            "units": self.units,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_token": self.num_token,
            "num_head": self.num_head,
            "memory_len": self.memory_len,
            "conv_memory_len": self.conv_memory_len,
            "compression_rate": self.compression_rate,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "cutoffs": self.cutoffs,
            "div_val": self.div_val,
            "force_projection": self.force_projection,
            "bind_embeddings": self.bind_embeddings,
            "bind_projections": self.bind_projections,
            "clamp_len": self.clamp_len,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(AttentionGate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

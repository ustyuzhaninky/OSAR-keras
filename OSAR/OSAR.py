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
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.framework import errors
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.keras.layers.ops import core as core_ops

from .tfxl import AdaptiveEmbedding, AdaptiveSoftmax
from .tfxl import LayerNormalization
from .tfxl import FeedForward, Memory

from .compressive_memory import CompressiveAvgPoolMemory
from .gates import TransferGate
from .gates import FiniteDifference
from .gates import AttentionGate
from .gates import SequenceEncoder1D
from .context_embedding import ContextEmbedding
from .bolzmann_gate import BolzmannGate, TargetingGate
from .cap_layers import Capsule
from .queue_memory import QueueMemory
from .graph_memory import GraphMemory

__all__ = ['OSARLayer',]

class OSARLayer(tf.keras.layers.Layer):
    """OSAR - Object-Stimulated Active Repeater

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            memory_size: int > 0 Number of units in the memory.
            conv_memory_size: int > 0. Number of units in the compressed memory.
            n_actions: int > 1. Number of actions in the output vector.
            action_spec: environment action specifications (env.action_spec()).
            num_heads: int > 0. Number of attention `heads`.
            dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
            heads: int > 0. Number of heads for multi-head attention.
            vmax: float, the value distribution support is [-vmax, vmax].
            use_bias: Boolean. Whether to use bias (dafault - True).
            compression_rate: int. Compression rate of the transfomer memories.
            gate_initializer: keras.initializer. Initializer for attention weights (default - glorot_normal).
            gate_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            gate_constraint: keras.constraint. Constraint for attention weights (default - None).
            state_initializer: keras.initializer. Initializer for state matrices (default - glorot_normal).
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
                 action_spec,
                 dropout=0.2,
                 heads=10,
                 vmax=10.,
                 use_bias=True,
                 compression_rate=4,
                 gate_initializer='glorot_normal',
                 gate_regularizer='l2',
                 gate_constraint=None,
                 state_initializer='zeros',
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
        self.action_spec = action_spec
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

        self.vmax = float(vmax)
        self._support = tf.linspace(-self.vmax, self.vmax, self.units)

    def build(self, input_shape):
        batch_size = input_shape[0]
        input_features = input_shape[-1] - 1
        real_input_shape = tf.TensorShape((batch_size, input_shape[-1]-1))
        reward_shape = tf.TensorShape((batch_size, 1))
        
        timesteps = self.memory_size + self.conv_memory_size
        
        self.action_mem_view = self.add_weight(
            'action_mem_view',
            shape=(batch_size, self.n_actions),
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False
        )
        K.variable(np.random.rand(
            batch_size, self.n_actions), dtype=tf.float32)
        self.cont_embedding = ContextEmbedding(mask_zero=False)
        self.cont_embedding.build(real_input_shape)
        self.cont_embedding.built = True

        self.generator_attention_objects = AttentionGate(
            units=self.units,
            embed_dim=self.units,
            hidden_dim=self.units,
            num_token=input_features,
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
        self.generator_attention_objects.build(real_input_shape)
        self.generator_attention_objects.built = True

        self.generator_attention_actions = AttentionGate(
            units=self.units,
            embed_dim=self.units,
            hidden_dim=self.units,
            num_token=self.n_actions,
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
        self.generator_attention_actions.build(real_input_shape)
        self.generator_attention_actions.built = True

        self.reward_memory = Memory(
            batch_size,
            self.units,
            self.units,
            1,
        )
        self.reward_memory.build(
            [(batch_size, 1, 1), (batch_size, 1,)])
        self.reward_memory.built = True

        self.object_encoder = SequenceEncoder1D(
            self.units,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.object_encoder.build(
            (batch_size, input_features, input_features))
        self.object_encoder.built = True

        self.action_encoder = SequenceEncoder1D(
            self.units,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.action_encoder.build(
            (batch_size, self.units, self.n_actions))
        self.action_encoder.built = True

        self.reward_encoder = SequenceEncoder1D(
            self.units,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.reward_encoder.build(
            (batch_size, timesteps, 1))
        self.reward_encoder.built = True

        self.context_encoder = SequenceEncoder1D(
            self.units,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.context_encoder.build(
            (batch_size, self.units, 3*self.units))
        self.context_encoder.built = True

        self.action_mem_view_encoder = SequenceEncoder1D(
            self.n_actions,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.action_mem_view_encoder.build(
            (batch_size, self.n_actions, self.n_actions))
        self.action_mem_view_encoder.built = True

        self.object_caps = Capsule(self.units, self.units, 3, True)
        self.object_caps.build(
            (batch_size, self.units, self.units))
        self.object_caps.built = True

        self.distib_caps = Capsule(self.units, self.units, 3, True)
        self.distib_caps.build(
            (batch_size, self.units, 3*self.units))
        self.distib_caps.built = True

        self.context_queue = QueueMemory(self.units)
        self.context_queue.build(
            [
                (batch_size, self.units, self.units),
                (batch_size, self.units, 1),
            ]
        )
        self.context_queue.built = True

        self.event_space = GraphMemory()
        self.event_space.build(
                (batch_size, self.units, self.units),
        )
        self.event_space.built = True

        self.object_decoder = SequenceEncoder1D(
            input_features,
            input_features,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.object_decoder.build(
            (batch_size, self.units, self.units))
        self.object_decoder.built = True

        self.action_decoder = SequenceEncoder1D(
            self.n_actions,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.action_decoder.build(
            (batch_size, self.units, self.units))
        self.action_decoder.built = True

        self.reward_decoder = SequenceEncoder1D(
            1,
            timesteps,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.reward_decoder.build(
            (batch_size, self.units, self.units))
        self.reward_decoder.built = True

        self.context_decoder = SequenceEncoder1D(
            3*self.units,
            self.units,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,)
        self.context_decoder.build(
            (batch_size, self.units, self.units))
        self.context_decoder.built = True

        self.gru = tf.keras.layers.GRU(
            self.units,
            use_bias=True,
            recurrent_dropout=self.dropout,
            dropout=self.dropout,
            kernel_regularizer=self.gate_regularizer,
            bias_regularizer=self.gate_regularizer,
            activity_regularizer=self.gate_regularizer,
            return_sequences=False,
        )
        self.gru.build(
                (batch_size, self.units, 3*self.units),
        )
        self.gru.built = True

        self.action_selector = TransferGate(
            self.n_actions,
            activation='softmax',
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.action_selector.build(
            (batch_size, self.units),
        )
        self.action_selector.built = True

        super(OSARLayer, self).build(input_shape)
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
            'action_spec': self.action_spec,
            'memory_size': self.memory_size,
            'conv_memory_size': self.conv_memory_size,
            'dropout': self.dropout,
            'heads': self.heads,
            'use_bias': self.use_bias,
            'vmax': self.vmax,
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

    def _get_vi_ratio(self, vector, matrix):
        matrix_entropy = K.sum(K.softmax(matrix, axis=-1) *
                               K.log(K.softmax(matrix, axis=-1)), axis=-1)
        vector_entropy = K.sum(K.softmax(vector, axis=-1) * \
                               K.log(K.softmax(matrix, axis=-1)), axis=-1)
        intersection = K.dot(matrix, K.transpose(vector))#tf.sets.intersection(matrix, vector)

        intersection_entropy = K.sum(K.softmax(intersection, axis=-1) *
                                     K.log(K.softmax(intersection, axis=-1)), axis=-1)
        return K.expand_dims(2*intersection_entropy - vector_entropy - matrix_entropy, axis=-1)

    def call(self, inputs, training=False, *args, **kwargs):
        inputs = tf.convert_to_tensor(inputs)

        input_features = inputs.shape[-1] - 1
        timesteps = self.memory_size + self.conv_memory_size

        object_input = tf.slice(
            inputs, [0, 0], [inputs.shape[0], inputs.shape[-1]-1])
        reward_input = tf.slice(inputs, [0, inputs.shape[-1]-1],
                                [inputs.shape[0], 1])
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.variables)
            object_input = math_ops.cast(object_input, dtype=tf.float32)
            reward_input = math_ops.cast(
                reward_input, dtype=tf.float32)  # inputs[1] # (1,)
            n_digits = tensor_shape.dimension_value(object_input.shape[-1])
            batch_size = tensor_shape.dimension_value(object_input.shape[0])
            
            memory_lenght = tf.tile(
                tf.expand_dims(tf.expand_dims(
                    self.conv_memory_size+self.memory_size, axis=0), axis=0), [batch_size, 1])

            updated_reward = self.reward_memory([K.expand_dims(reward_input, axis=-1), memory_lenght])

            # Context generator
            token_embedding = self.cont_embedding(object_input)
            
            object_context = self.generator_attention_objects(token_embedding)
            
            # Decomposing context into separate features and encoding in R_{self.units} space
            object_context = tf.keras.layers.Reshape(
                (input_features, input_features))(object_context)
            
            object_context_en = self.object_encoder(object_context)
            
            action_mem = self.generator_attention_actions(
                self.action_mem_view, axis=1)
            action_mem = self.action_mem_view_encoder(action_mem)
            if len(action_mem.shape) < 3:
                action_context = K.expand_dims(action_mem, axis=0)
            else:
                action_context = action_mem

            action_context_en = self.action_encoder(action_context)
            reward_context_en = self.reward_encoder(updated_reward)
            
            context = K.concatenate(
                [object_context_en, action_context_en, reward_context_en], axis=1)
            context = K.permute_dimensions(context, (0, 2, 1))
              # tf.concat(self.variables
            # object_grad = tape.gradient(
            #     object_context, object_input)#, axis=-1)
            # action_grad = tf.concat(tape.gradient(
            #     action_context, self.variables), axis=-1)
            # reward_grad = tf.concat(tape.gradient(
            #     reward_context, self.variables), axis=-1)
            
            # grad_context = K.concatenate(
            #     [object_grad, action_grad, reward_grad], axis=-1)
            
            grad_context_shrinked = self.context_encoder(context)#grad_context)
            decomposed_objects = K.softmax(
                self.object_caps(grad_context_shrinked))
            # Updating Importance Queue
            estimate = self._get_vi_ratio(
                decomposed_objects[:, -1, ...], decomposed_objects)
            queue = self.context_queue([object_context_en, estimate])

            p_distribution = self.distib_caps(context)#grad_context)
            target = self.event_space(p_distribution)

            target_probabilities = K.softmax(target)
            target_q_values = tf.reduce_sum(tf.reduce_sum(
                self._support * target_probabilities, axis=-1), axis=-1)
            
            queue_probability = K.softmax(queue[-1])
            queue_q_values = tf.reduce_sum(tf.reduce_sum(
                self._support * queue_probability, axis=-1), axis=-1)

            if target_q_values <= queue_q_values:
                target = queue#[-1]
        
        # Repeater
        # target_grad = tape.gradient(
        #     target, self.variables)  # [object_input, self.action_mem_view[..., -1, :], reward_input])
        # Repacking context gradients
        # object_grad = tape.gradient(
        #     object_context, self.variables)#object_input)
        # action_grad = tape.gradient(
        #     action_context, self.variables)#self.action_mem_view[..., -1, :])
        # reward_grad = tape.gradient(
        #     reward_context, self.variables)#reward_input)
        
        # grad_context = K.concatenate(
        #     [object_grad, action_grad, reward_grad], axis=-1)

        # objects_grad = grad_context[..., :self.units]
        # objects_grad = objects_grad + object_context[0, ...] - target
        # action_grad = target_grad[..., self.units:2*self.units]
        # reward_grad = grad_context[..., self.units*2:3*self.units]
        # target_grad_updated = K.concatenate(
        #     [objects_grad, action_grad, reward_grad], axis=-1)
        objects = context[:, :, :self.units]
        objects = objects - target
        action = context[:, :, self.units:2*self.units]
        reward = context[:, :, self.units*2:3*self.units]
        target_updated = K.concatenate(
            [objects, action, reward], axis=-1)

        # Registering additional losses:
        ## Decoder-encoder stabilisation:
        objects = self.object_decoder(object_context_en)
        actions_l = self.action_decoder(action_context_en)
        rewards = self.reward_decoder(reward_context_en)
        long_context = K.concatenate(
            [object_context_en, action_context_en, reward_context_en], axis=-1)
        shrinken_context = self.context_encoder(long_context)
        context = self.context_decoder(shrinken_context)
        self.add_loss(tf.keras.losses.Huber()(object_context - objects, 0))
        self.add_loss(tf.keras.losses.Huber()(action_context - actions_l, 0))
        self.add_loss(tf.keras.losses.Huber()(updated_reward - rewards, 0))
        self.add_loss(tf.keras.losses.Huber()(
            context - long_context, 0))
        self.add_loss(-K.sum(updated_reward, axis=1))

        # Constructing GRU input and generating next strategy step:
        strategy_step = self.gru(target_updated)#target_grad_updated)
        action = K.expand_dims(self.action_selector(strategy_step)[-1], axis=0)
        self.add_update(K.update(self.action_mem_view, action))

        return action

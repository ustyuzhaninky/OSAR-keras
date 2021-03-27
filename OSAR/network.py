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
from .helix_memory import HelixMemory
from .gates import AttentionGate
import gin

from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import encoding_network, lstm_encoding_network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils
from tf_agents.networks import categorical_q_network
from tf_agents.networks import utils as network_utils

from . import ContextGenerator
from . import Memory
from . import SympatheticCircuit

__all__ = ['OSARNetwork']

"""Sample Keras Network with OSAR, based on Q-network:
    https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/networks/q_network.py#L43-L149 
Implements a network that will generate the following layers:
    [optional]: preprocessing_layers  # preprocessing_layers
    [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
    [optional]: Conv2D # conv_layer_params
    Flatten
    [optional]: Dense  # input_fc_layer_params
    [optional]: OSAR   # 
    [optional]: Dense  # output_fc_layer_params
    Dense -> 1         # Value output
"""

def validate_specs(action_spec, observation_spec):
  """Validates the spec contains a single action."""
  del observation_spec  # not currently validated

  flat_action_spec = tf.nest.flatten(action_spec)
  if len(flat_action_spec) > 1:
    raise ValueError(
        'Network only supports action_specs with a single action.')

  if flat_action_spec[0].shape not in [(), (1,)]:
    raise ValueError(
        'Network only supports action_specs with shape in [(), (1,)])')

@gin.configurable
class OSARNetwork(q_network.QNetwork):
    """OSAR network."""

    """Recurrent value network. Reduces to 1 value output per batch item."""

    def __init__(self,
                input_tensor_spec,
                action_spec,
                batch_size,
                memory_len=10,
                frozen=False,
                n_turns=3,
                preprocessing_layers=None,
                preprocessing_combiner=None,
                conv_layer_params=None,
                activation_fn=tf.nn.relu,
                input_fc_layer_params=(10, 10),
                input_dropout_layer_params=None,
                fc_layer_params=(10,),
                output_fc_layer_params=(10, 10),
                conv_type='2d',
                dtype=tf.float32,
                 name='OSARNetwork'):
        """Creates an instance of `OSARNetwork`.
        The logits output by __call__ will ultimately have a shape of
        `[batch_size, num_actions]`, where `num_actions` is computed as
        `action_spec.maximum - action_spec.minimum + 1`. Each value is a logit for
        a particular action at a particular atom (see above).
        As an example, if
        `action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 4)`.
        Args:
        input_tensor_spec: A `tensor_spec.TensorSpec` specifying the observation
            spec.
        action_spec: A `tensor_spec.BoundedTensorSpec` representing the actions.
        frozen: Whether the network auto-tuning should be disabled (for target network).
        preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
        preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
        conv_layer_params: Optional list of convolution layer parameters for
            observations, where each item is a length-three tuple indicating
            (num_units, kernel_size, stride).
        fc_layer_params: Optional list of fully connected parameters for
            observations, where each item is the number of units in the layer.
        activation_fn: Activation function, e.g. tf.nn.relu or tf.nn.leaky_relu.
        name: A string representing the name of the network.
        Raises:
        TypeError: `action_spec` is not a `BoundedTensorSpec`.
        """
        del input_dropout_layer_params

        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        encoder_input_tensor_spec = input_tensor_spec

        kernel_initializer = tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_encoder = encoding_network.EncodingNetwork(
            encoder_input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=input_fc_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            conv_type=conv_type,
            dtype=dtype)
        
        generator = ContextGenerator(
            units=fc_layer_params[0],
            batch_size=batch_size,
            memory_len=memory_len,
            n_turns=n_turns,
            n_states=fc_layer_params[0],
            dropout=0.2,
            attention_dropout=0.2,
            kernel_regularizer='l2',
            bias_regularizer='l2',
            )

        gru = tf.keras.layers.GRU(fc_layer_params[0],
                                   kernel_regularizer='l2',
                                   dropout=0.2,
                                   recurrent_dropout=0,
                                   bias_regularizer='l2',
                                   reset_after=True,
                                   unroll=False,
                                   name='gru'
                                   )

        circuit = SympatheticCircuit(
            fc_layer_params[0],
            (input_fc_layer_params[-1], 1, num_actions),
            memory_len,
            kernel_regularizer='l2',
            dropout=0.2,
            bias_regularizer='l2',
        )

        units = num_actions
        q_value_layer = tf.keras.layers.Dense(
            units,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype)

        super(OSARNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            name=name)
        
        self._encoder = input_encoder
        self._context_generator = generator
        self._circuit = circuit
        self._repeater = gru
        self._q_value_layer = q_value_layer
        self._action_memory = self.add_weight(
            shape=(batch_size, 1, num_actions),
            initializer=tf.keras.initializers.get('glorot_uniform'),
            trainable=False,
            name='action-memory'
        )
        self.frozen = frozen

    # @tf.function(autograph=True)
    def call(self,
             observation,
             reward=tf.constant([0.0], dtype=tf.float32),
             step_type=None,
             network_state=(),
             training=False):
        batch_dim = tf.shape(observation)[0]

        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training)
        
        action_memory = self._action_memory
        
        state = tf.expand_dims(state, axis=0)
        reward = tf.expand_dims(tf.expand_dims(reward, axis=-1), axis=-1)
        context = K.concatenate(
            [state,
             action_memory,
             reward], axis=-1)
        context = self._context_generator(context, training=training)
        distance, importance, context_updated = self._circuit(context, training=training, frozen=self.frozen)
        context = K.concatenate([distance, importance, context_updated], axis=-1)
        
        action = self._repeater(context)

        q_value = self._q_value_layer(action, training=training)
        
        new_memory = q_value
        self.add_update(K.update(self._action_memory,
                                 K.expand_dims(new_memory, axis=0)))
        
        logits = q_value

        return logits, network_state

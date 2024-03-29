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

from tensorflow.keras import backend as K
from tf_agents.networks import q_network
from tf_agents.networks import Network
from tf_agents.networks import encoding_network
import gin

from .. import ContextGenerator
from .. import SympatheticCircuit

__all__ = ['OSARQNetwork']

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
class OSARQNetwork(q_network.QNetwork):
    """OSAR Q-network."""

    def __init__(self,
                input_tensor_spec,
                action_spec,
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
                batch_squash=True,
                dtype=tf.float32,
                 name='OSARQNetwork'):
        """Creates an instance of `OSARQNetwork`.
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
        batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
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
            batch_squash=batch_squash,
            dtype=dtype)
        
        generator = ContextGenerator(
            units=fc_layer_params[0],
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

        q_value_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype)

        super(OSARQNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            name=name)
        
        self._encoder = input_encoder
        self._context_generator = generator
        self._circuit = circuit
        self._repeater = gru
        self._q_value_layer = q_value_layer
        self._action_size = num_actions
        self._action_memory = None
        self._batch_size = None
        self.frozen = frozen

    @property
    def action_memory(self):
        if self._action_memory is None:
            action_memory = tf.keras.initializers.GlorotNormal()(shape=(self.batch_size, self._action_size,))
            return action_memory
            # return action_memory if tf.nest.is_nested(self._action_size) else [action_memory]
        return self._action_memory

    @action_memory.setter
    # Automatic tracking catches "self._action_memory" which adds an extra weight and
    # breaks HDF5 checkpoints.
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def action_memory(self, action_memory):
        self._action_memory = action_memory
    
    @property
    def batch_size(self):
        if self._batch_size is None:
            return 1
        return self._batch_size
    
    @batch_size.setter
    # Automatic tracking catches "self._batch_size" which adds an extra weight and
    # breaks HDF5 checkpoints.
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    # @tf.function(autograph=True)
    def call(self,
             observation,
             reward=tf.constant([0.0], dtype=tf.float32),
             step_type=None,
             network_state=(),
             training=False):
        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training)
        
        batch_size = 1 
        if self.batch_size is None:
            self.batch_size = 1
        if observation.shape[0] != None:
            self.action_memory = None
            self.batch_size = observation.shape[0]
            batch_size = observation.shape[0]
        
        reward = tf.expand_dims(reward, axis=-1)
        if reward.shape[0] != batch_size:
            reward = tf.tile(reward, (batch_size, 1))

        context = K.concatenate(
            [state,
             self.action_memory,
             reward], axis=-1)
        
        context = self._context_generator(context, training=training)

        distances, importances, context_updated = self._circuit(context, training=training, frozen=self.frozen)
        context = K.concatenate([distances, importances, context_updated], axis=-1)

        action = self._repeater(context)

        logits = self._q_value_layer(action, training=training)
        
        self.action_memory = logits
        return logits, network_state

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

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import encoding_network
from tf_agents.networks import Network
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import common

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import util as tf_util  # TF internal
# pylint: enable=g-direct-tensorflow-import

from .. import QueueMemory
from .. import EventSpace

__all__ = ['ArtDiscriminator']

@gin.configurable
class ArtDiscriminator(Network):
    """ART Discriminator Network."""

    def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               reward_tensor_spec=None,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(16, 16),
               memory_len=10,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               kernel_regularizer='l1_l2',
               bias_regularizer='l1_l2',
               batch_squash=True,
               dtype=tf.float32,
               name='ArtDiscriminator'):
        """Creates an instance of `ArtDiscriminator`.

        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
                input of the agent.
            output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
                the output of the agent.
            preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
                representing preprocessing for the different observations.
                All of these layers must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`.
            preprocessing_combiner: (Optional.) A keras layer that takes a flat list
                of tensors and combines them. Good options include
                `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
                This layer must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`.
            conv_layer_params: Optional list of convolution layers parameters, where
                each item is a length-three tuple indicating (filters, kernel_size,
                stride).
            fc_layer_params: Optional list of fully_connected parameters, where each
                item is the number of units in the layer.
            memory_len: (Optional.) Lenght of the short-term part (number of cells)
                of the Helix Memory (OSAR.HelixMemory), where each cell is a step in
                the time series of vectors [state, action, reward], integer.
            dropout_layer_params: Optional list of dropout layer parameters, each item
                is the fraction of input units to drop or a dictionary of parameters
                according to the keras.Dropout documentation. The additional parameter
                `permanent`, if set to True, allows to apply dropout at inference for
                approximated Bayesian inference. The dropout layers are interleaved with
                the fully connected layers; there is a dropout layer after each fully
                connected layer, except if the entry in the list is None. This list must
                have the same length of gru_layer_params, or be None.
            activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
            kernel_initializer: Initializer to use for the kernels of the conv and
                dense layers. If none is provided a default glorot_normal.
            kernel_regularizer: Regularizer to use for the kernels of the main layers.
                If none is provided a default 'l1_l2'.
            bias_regularizer: Regularizer to use for the biases of the main layers.
                If none is provided a default 'l1_l2'.
            batch_squash: If True the outer_ranks of the observation are squashed into
                the batch dimension. This allow encoding networks to be used with
                observations with shape [BxTx...].
            dtype: The dtype to use by the convolution and fully connected layers.
            discrete_projection_net: Callable that generates a discrete projection
                network to be called with some hidden state and the outer_rank of the
                state.
            continuous_projection_net: Callable that generates a continuous projection
                network to be called with some hidden state and the outer_rank of the
                state.
            name: A string representing name of the network.
        Raises:
            ValueError: If `input_tensor_spec` contains more than one observation.
        """
        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        if not reward_tensor_spec:
            reward_tensor_spec = tensor_spec.BoundedTensorSpec(shape=[], dtype=dtype, minimum=dtype.min, maximum=dtype.max)

        encoders = [
        encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype),
        encoding_network.EncodingNetwork(
            output_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=None,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype),
        encoding_network.EncodingNetwork(
            reward_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=None,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype),
        ]
        output_tensor_spec = tensor_spec.BoundedTensorSpec(shape=[], dtype=dtype, minimum=0, maximum=1)

        try:
            num_actions = output_tensor_spec.shape[-1]
        except:
            num_actions = 1

        event_space = EventSpace(
            fc_layer_params[0],
            (input_tensor_spec.shape[-1], num_actions, 1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            return_space=True
            )
        queue = QueueMemory(
            memory_len,
            kernel_regularizer=kernel_regularizer,
            )

        decoder = tf.keras.layers.Dense(
            1,
            # activation=tf.keras.activations.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype)

        self._encoders = encoders
        self._batch_size = None
        self._event_space = event_space
        self._queue = queue
        self._decoder = decoder

        super(ArtDiscriminator, self).__init__(
                input_tensor_spec=input_tensor_spec,
                state_spec=(),
                name=name)

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

    def find_max_path(self, space):
        max_index_features = tf.argmax(tf.reduce_sum(space[:, :, :, :, -1], axis=1), axis=-2)
        max_index_timesteps = tf.argmax(tf.reduce_sum(space[:, :, :, :, -1], axis=1), axis=2)
        max_path = tf.gather(
            space,
            tf.squeeze(max_index_features),
            axis=-1,
            batch_dims=0,
        )
        max_path = tf.gather(
            max_path,
            tf.squeeze(max_index_timesteps),
            axis=-1,
            batch_dims=0,
        )
        return tf.reduce_max(tf.reduce_max(max_path, axis=1,), axis=-1)

    def create_variables(self, input_tensor_spec=None, **kwargs):
        """Force creation of the network's variables.
        Return output specs.
        Args:
        input_tensor_spec: (Optional).  Override or provide an input tensor spec
            when creating variables.
        **kwargs: Other arguments to `network.call()`, e.g. `training=True`.
        Returns:
        Output specs - a nested spec calculated from the outputs (excluding any
        batch dimensions).  If any of the output elements is a tfp `Distribution`,
        the associated spec entry returned is a `DistributionSpec`.
        Raises:
        ValueError: If no `input_tensor_spec` is provided, and the network did
            not provide one during construction.
        """

        self._input_tensor_spec = input_tensor_spec[0]
        random_inputs = []
        for spec in input_tensor_spec:
            random_inputs.append(tensor_spec.sample_spec_nest(
                spec, outer_dims=(1,)))
        initial_state = self.get_initial_state(batch_size=1)
        step_type = tf.fill((1,), time_step.StepType.FIRST)
        outputs = self.__call__(
            random_inputs[0],
            actions=random_inputs[1],
            rewards=random_inputs[2],
            step_type=step_type,
            network_state=initial_state,
            **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        """A modified wrapper around `Network.call`.
        """
        if self.input_tensor_spec is not None:
            nest_utils.assert_matching_dtypes_and_inner_shapes(
                inputs,
                self.input_tensor_spec,
                allow_extra_fields=True,
                caller=self,
                tensors_name="`inputs`",
                specs_name="`input_tensor_spec`")

        call_argspec = tf_util.tf_inspect.getargspec(self.call)

        # Convert *args, **kwargs to a canonical kwarg representation.
        normalized_kwargs = tf_util.tf_inspect.getcallargs(
            self.call, inputs, *args, **kwargs)
        # TODO(b/156315434): Rename network_state to just state.
        network_state = normalized_kwargs.get("network_state", None)
        normalized_kwargs.pop("self", None)

        if common.safe_has_state(network_state):
            nest_utils.assert_matching_dtypes_and_inner_shapes(
                network_state,
                self.state_spec,
                allow_extra_fields=True,
                caller=self,
                tensors_name="`network_state`",
                specs_name="`state_spec`")

        if "step_type" not in call_argspec.args and not call_argspec.keywords:
            normalized_kwargs.pop("step_type", None)

        # network_state can be a (), None, Tensor or NestedTensors.
        if (not tf.is_tensor(network_state)
            and network_state in (None, ())
            and "network_state" not in call_argspec.args
            and not call_argspec.keywords):
            normalized_kwargs.pop("network_state", None)

        outputs, true_targets, importances, potential_rewards, new_state = super(
            Network, self).__call__(**normalized_kwargs)  # pytype: disable=attribute-error  # typed-keras

        nest_utils.assert_matching_dtypes_and_inner_shapes(
            new_state,
            self.state_spec,
            allow_extra_fields=True,
            caller=self,
            tensors_name="`new_state`",
            specs_name="`state_spec`")

        return outputs, true_targets, importances, potential_rewards, new_state


    # @tf.function
    def call(self,
           observations,
           actions,
           rewards,
           step_type=None,
           network_state=(),
           training=False,
           mask=None):
        state, network_state = self._encoders[0](
            observations,
            step_type=step_type,
            network_state=network_state,
            training=training)
        action, network_action_state = self._encoders[1](
            actions,
            step_type=step_type,
            network_state=network_state,
            training=training)
        reward, network_reward_state = self._encoders[2](
            rewards,
            step_type=step_type,
            network_state=network_state,
            training=training)

        state = tf.concat(
            [state,
             action,
             reward], axis=-1)
        network_state = tf.concat(
            [network_state,
             network_action_state,
             network_reward_state], axis=-1)

        state = tf.expand_dims(state, axis=1)

        output, spaces = self._event_space(state)

        max_spaces = tf.concat(tf.nest.map_structure(
            self.find_max_path,
            tf.split(spaces, spaces.shape[0], axis=0)
        ), axis=0)
        targets, importances = self._queue([output, max_spaces])

        ideal_targets = max_spaces[..., :state.shape[-1]]
        true_targets = (targets + ideal_targets) / 2
        distances = tf.map_fn(
            lambda x:  tf.expand_dims(tf.norm(x,  ord='euclidean'), axis=-1),
            state[..., :state.shape[-1]] - true_targets,
            # dtype=true_targets.dtype,
        )
        true_targets = tf.reduce_mean(distances * true_targets, axis=1)

        potential_rewards = (1 - distances) * tf.expand_dims(tf.reduce_sum(reward, axis=-1), axis=-1) +  tf.expand_dims(
            max_spaces[..., -1], axis=1) * distances

        context = tf.concat(
            [
                output,
                importances,
                state,
                tf.expand_dims(max_spaces, axis=1)
            ], axis=-1)
        context = tf.keras.layers.Flatten()(context)
        output_value = self._decoder(context)
        importance = tf.expand_dims(tf.reduce_max(importances, axis=-1), axis=-1)

        return output_value, true_targets, importance, potential_rewards, network_state

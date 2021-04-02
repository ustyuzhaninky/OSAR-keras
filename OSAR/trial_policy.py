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
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import Optional, Text, cast

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import shifted_categorical
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

@gin.configurable
class TrialPolicy(tf_policy.TFPolicy):
    def __init__(
            self,
            time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            network: network.Network,
            expert: bool = False,
            emit_log_probability: bool = False,
            observation_and_action_constraint_splitter: Optional[
                types.Splitter] = None,
            validate_action_spec_and_network: bool = True,
            name: Optional[Text] = None):

        network_action_spec = getattr(network, 'action_spec', None)

        if network_action_spec is not None:
            action_spec = cast(tf.TypeSpec, action_spec)
            if not action_spec.is_compatible_with(network_action_spec):
                raise ValueError(
                    'action_spec must be compatible with network.action_spec; '
                    'instead got action_spec=%s, network.action_spec=%s' % (
                        action_spec, network_action_spec))

        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only scalar actions are supported now, but action spec is: {}'
                .format(action_spec))
        if validate_action_spec_and_network:
            spec = flat_action_spec[0]
        if spec.shape.rank > 0:
            raise ValueError(
                'Only scalar actions are supported now, but action spec is: {}'
                .format(action_spec))

        if spec.minimum != 0:
            raise ValueError(
                'Action specs should have minimum of 0, but saw: {0}'.format(spec))

        num_actions = spec.maximum - spec.minimum + 1
        # network_utils.check_single_floating_network_output(
        #     network.create_variables(), (num_actions,), str(network))

        # We need to maintain the flat action spec for dtype, shape and range.
        self._flat_action_spec = flat_action_spec[0]

        self._q_network = network
        super(TrialPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=network.state_spec,
            clip=False,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter),
            name=name)
        self._expert = expert
    
    def _distribution(self, time_step, policy_state):
        observation_and_action_constraint_splitter = (
            self.observation_and_action_constraint_splitter)
        network_observation = time_step.observation
        network_reward = time_step.reward

        if observation_and_action_constraint_splitter is not None:
            network_observation, mask = observation_and_action_constraint_splitter(
                network_observation)
        # if len(network_observation.shape) == 1:
        #     network_observation = tf.expand_dims(network_observation, axis=0)
        #     network_reward = tf.expand_dims(network_reward, axis=0)
        logits, policy_state = self._q_network(
            network_observation,
            reward=network_reward,
            network_state=policy_state,
            step_type=time_step.step_type)

        if observation_and_action_constraint_splitter is not None:
            # Overwrite the logits for invalid actions to logits.dtype.min.
            almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
            logits = tf.compat.v2.where(
                tf.cast(mask, tf.bool), logits, almost_neg_inf)

        if self._flat_action_spec.minimum != 0:
            distribution = shifted_categorical.ShiftedCategorical(
                logits=logits,
                dtype=self._flat_action_spec.dtype,
                shift=self._flat_action_spec.minimum)
        else:
            distribution = tfp.distributions.Categorical(
                logits=logits,
                dtype=self._flat_action_spec.dtype)
        distribution = tf.nest.pack_sequence_as(self._action_spec, [distribution])

        return policy_step.PolicyStep(distribution, policy_state)

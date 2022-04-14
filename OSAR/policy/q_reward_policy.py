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
from tf_agents.policies import q_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.networks import network
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

__all__ = ['QRewardPolicy',]


@gin.configurable
class QRewardPolicy(q_policy.QPolicy):
    def __init__(
            self,
            time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            q_network: network.Network,
            emit_log_probability: bool = False,
            observation_and_action_constraint_splitter: Optional[
                types.Splitter] = None,
            validate_action_spec_and_network: bool = True,
            name: Optional[Text] = None):

        super(QRewardPolicy, self).__init__(
            time_step_spec,
            action_spec,
            q_network=q_network,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter),
            validate_action_spec_and_network=validate_action_spec_and_network,
            name=name)
    
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

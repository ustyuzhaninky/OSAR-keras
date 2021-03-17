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

"""DISCLAIMER
    Some parts of the code are taken from td3_agent.py from:
        https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/agents/td3/td3_agent.py
    and dqn_agent.py from:
        https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/agents/dqn/dqn_agent.py

    under authorship of TF-Agents Authors.

"""


from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import Optional, Text, cast

import gin
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from . import TrialPolicy

def compute_td_targets(next_q_values: types.Tensor,
                       rewards: types.Tensor,
                       discounts: types.Tensor) -> types.Tensor:
  return tf.stop_gradient(rewards + discounts * next_q_values)


@gin.configurable
class TrialAgent(tf_agent.TFAgent):
    """A Trial Agent.
    """

    def  __init__(
            self,
            time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensor,
            network: network.Network,
            optimizer: types.Optimizer,
            observation_and_action_constraint_splitter: Optional[
                types.Splitter] = None,
            exploration_noise_std: types.Float = 0.1,
            update_period: types.Int = 1,
            td_errors_loss_fn: Optional[types.LossFn] = None,
            gamma: types.Float = 1.0,
            batch_size: types.Int = 1.0,
            epsilon_greedy: types.FloatOrReturningFloat = 0.1,
            n_step_update: int = 2,
            boltzmann_temperature: Optional[types.Int] = None,
            temperature_storage_size: types.Int = 100,
            emit_log_probability: bool = False,
            reward_scale_factor: types.Float = 1.0,
            gradient_clipping: Optional[types.Float] = None,
            debug_summaries: bool = False,
            summarize_grads_and_vars: bool = False,
            train_step_counter: Optional[tf.Variable] = None,
            name: Text = None):
        """Creates a Trial Agent.
            Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            network: A tf_agents.network.Network to be used by the agent. The
                network will be called with call(observation, step_type).
            optimizer: The default optimizer to use for the actor network.
            observation_and_action_constraint_splitter: A function used to process
                observations with action constraints. These constraints can indicate,
                for example, a mask of valid/invalid actions for a given state of the
                environment.
                The function takes in a full observation and returns a tuple consisting
                of 1) the part of the observation intended as input to the network and
                2) the constraint. An example
                `observation_and_action_constraint_splitter` could be as simple as:
                ```
                def observation_and_action_constraint_splitter(observation):
                return observation['network_input'], observation['constraint']
                ```
                *Note*: when using `observation_and_action_constraint_splitter`, make
                sure the provided `network` is compatible with the network-specific
                half of the output of the `observation_and_action_constraint_splitter`.
                In particular, `observation_and_action_constraint_splitter` will be
                called on the observation before passing to the network.
                If `observation_and_action_constraint_splitter` is None, action
                constraints are not applied.
            exploration_noise_std: Scale factor on exploration policy noise.
            update_period: Period for the optimization step on actor network.
            td_errors_loss_fn:  A function for computing the TD errors loss. If None,
                a default value of elementwise huber_loss is used.
            gamma: A discount factor for future rewards.
            batch_size: Batch size of the training inputs.
            epsilon_greedy: probability of choosing a random action in the default
                epsilon-greedy collect policy (used only if a wrapper is not provided to
                the collect_policy method).
            n_step_update: The number of steps to consider when computing TD error and
                TD loss. Defaults to single-step updates. Note that this requires the
                user to call train on Trajectory objects with a time dimension of
                `n_step_update + 1`. However, note that we do not yet support
                `n_step_update > 1` in the case of RNNs (i.e., non-empty
                `q_network.state_spec`).
            boltzmann_temperature: Temperature value to use for Boltzmann sampling of
                the actions during data collection. The closer to 0.0, the higher the
                probability of choosing the best action.
            temperature_storage_size: Number of q_value samples to store for computing
            emit_log_probability: Whether policies emit log probabilities or not.
            reward_scale_factor: Multiplicative scale for the reward.
            gradient_clipping: Norm length to clip gradients.
            debug_summaries: A bool to gather debug summaries.
            summarize_grads_and_vars: If True, gradient and network variable summaries
                will be written during training.
            train_step_counter: An optional counter to increment every time the train
                op is run.  Defaults to the global_step.
            name: The name of this agent. All variables in this module will fall
                under that name. Defaults to the class name.
        """
    
        tf.Module.__init__(self, name=name)

        self._check_action_spec(action_spec)

        if epsilon_greedy is not None and boltzmann_temperature is not None:
            raise ValueError(
                'Configured both epsilon_greedy value {} and temperature {}, '
                'however only one of them can be used for exploration.'.format(
                    epsilon_greedy, boltzmann_temperature))

        net_observation_spec = time_step_spec.observation
        if observation_and_action_constraint_splitter:
            net_observation_spec, _ = observation_and_action_constraint_splitter(
                net_observation_spec)
        network.create_variables(net_observation_spec)

        self._network = network

        self._target_network = common.maybe_copy_target_network_with_checks(
            self._network, None, input_spec=net_observation_spec,
            name='TargetNetwork')

        self._check_network_output(self._network, 'network')
        self._check_network_output(self._target_network, 'target_network')

        self._optimizer = optimizer

        self._observation_and_action_constraint_splitter = (
            observation_and_action_constraint_splitter)

        # Creating temperature storage for system's picked memories
        num_actions = action_spec.maximum - action_spec.minimum + 1
        features = time_step_spec.observation.shape[0]
        
        self._boltzmann_temperature = boltzmann_temperature
        self._epsilon_greedy = epsilon_greedy
        self._exploration_noise_std = exploration_noise_std
        self._td_errors_loss_fn = (
            td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        self._n_step_update = n_step_update
        self._update_target = self._get_target_updater(
            1.0, 1)

        policy, collect_policy = self._setup_policy(time_step_spec, action_spec,
                                                    boltzmann_temperature,
                                                    emit_log_probability)

        # if network.state_spec and n_step_update != 1:
        #     raise NotImplementedError(
        #         'DqnAgent does not currently support n-step updates with stateful '
        #         'networks (i.e., RNNs), but n_step_update = {}'.format(n_step_update))
        
        # train_sequence_length = (
        #     n_step_update if not network.state_spec else None)
        train_sequence_length = 3

        super(TrialAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            validate_args=False
        )

        if network.state_spec:
            # AsNStepTransition does not support emitting [B, T, ...] tensors,
            # which we need for DQN-RNN.
            self._as_transition = data_converter.AsTransition(
                self.data_context, squeeze_time_dim=True)
        else:
            # This reduces the n-step return and removes the extra time dimension,
            # allowing the rest of the computations to be independent of the
            # n-step parameter.
            self._as_transition = data_converter.AsNStepTransition(
                self.data_context, gamma=gamma, n=n_step_update)

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)

        # TODO(oars): Get DQN working with more than one dim in the actions.
        if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 0:
            raise ValueError(
                'Only scalar actions are supported now, but action spec is: {}'
                .format(action_spec))

        spec = flat_action_spec[0]

        # TODO(b/119321125): Disable this once index_with_actions supports
        # negative-valued actions.
        if spec.minimum != 0:
            raise ValueError(
                'Action specs should have minimum of 0, but saw: {0}'.format(spec))

        self._num_actions = spec.maximum - spec.minimum + 1

    def _check_network_output(self, net, label):
        """Check outputs of q_net and target_q_net against expected shape.
        Subclasses that require different q_network outputs should override
        this function.
        Args:
        net: A `Network`.
        label: A label to print in case of a mismatch.
        """
        network_utils.check_single_floating_network_output(
            net.create_variables(),
            expected_output_shape=(self._num_actions,),
            label=label)

    def _setup_policy(self, time_step_spec, action_spec,
                      boltzmann_temperature, emit_log_probability):

        policy = TrialPolicy(
            time_step_spec,
            action_spec,
            q_network=self._network,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))

        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(
                policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
            policy = greedy_policy.GreedyPolicy(policy)
        
        target_policy = TrialPolicy(
            time_step_spec,
            action_spec,
            q_network=self._target_network,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))
        self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)

        return policy, collect_policy
    
    def _compute_next_q_values(self, next_time_steps, info, training=False):
        del info

        network_observation = next_time_steps.observation
        network_reward = tf.expand_dims(next_time_steps.reward, axis=-1)

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        action_spec = cast(tensor_spec.BoundedTensorSpec, self._action_spec)
        multi_dim_actions = action_spec.shape.rank > 0

        next_target_q_values, _ = self._target_network(
            network_observation, step_type=next_time_steps.step_type)
        batch_size = (
            next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self._policy.get_initial_state(batch_size)
        # Find the greedy actions using our greedy policy. This ensures that action
        # constraints are respected and helps centralize the greedy logic.
        greedy_actions = self._target_greedy_policy.action(
            next_time_steps, dummy_state).action

        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(
            next_target_q_values,
            greedy_actions,
            multi_dim_actions=multi_dim_actions)
        
    def _compute_q_values(self, time_steps, actions, training=False):
        network_observation = time_steps.observation
        network_reward = time_steps.reward

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)
        
        q_values, _ = self._network(network_observation,
                                    reward=network_reward,
                                    step_type=time_steps.step_type,
                                    training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        action_spec = cast(tensor_spec.BoundedTensorSpec, self._action_spec)
        multi_dim_actions = action_spec.shape.rank > 0
        
        return common.index_with_actions(
            q_values,
            tf.cast(actions, tf.int32),
            multi_dim_actions=multi_dim_actions)

    def _initialize(self):
        common.soft_variables_update(
            self._network.variables, self._target_network.variables, tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.
        For each weight w_s in the q network, and its corresponding
        weight w_t in the target_network, a soft update is:
        w_t = (1 - tau) * w_t + tau * w_s
        Args:
        tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
        period: Step interval at which the target network is updated.
        Returns:
        A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_targets'):

            def update():
                return common.soft_variables_update(
                    self._network.variables,
                    self._target_network.variables,
                    tau,
                    tau_non_trainable=1.0)

            return common.Periodically(update, period, 'periodic_update_targets')

    # Use @common.function in graph mode or for speeding up.
    @common.function
    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
        
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self._network.trainable_weights
        non_trainable_weights = self._network.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = list(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                            self._gradient_clipping)

        if self._summarize_grads_and_vars:
            grads_and_vars_with_non_trainable = (
                grads_and_vars + [(None, v) for v in non_trainable_weights])
            eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)
            self._optimizer.apply_gradients(grads_and_vars)
            self.train_step_counter.assign_add(1)

        return loss_info

    def _loss(self,
              experience,
              td_errors_loss_fn=common.element_wise_huber_loss,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None,
              training=False):

        self._check_trajectory_dimensions(experience)

        squeeze_time_dim = not self._network.state_spec
        if self._n_step_update == 1:
            time_steps, policy_steps, next_time_steps = (
                trajectory.experience_to_transitions(experience, squeeze_time_dim))
            actions = policy_steps.action
        else:
            # To compute n-step returns, we need the first time steps, the first
            # actions, and the last time steps. Therefore we extract the first and
            # last transitions from our Trajectory.
            first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
            last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
            time_steps, policy_steps, _ = (
                trajectory.experience_to_transitions(
                    first_two_steps, squeeze_time_dim))
            actions = policy_steps.action
            _, _, next_time_steps = (
                trajectory.experience_to_transitions(
                    last_two_steps, squeeze_time_dim))

        with tf.name_scope('loss'):
            q_values = self._compute_q_values(
                time_steps, actions, training=training)
            
            next_q_values = self._compute_next_q_values(
                time_steps, policy_steps.info, training=training)

            # This applies to any value of n_step_update and also in the RNN-DQN case.
            # In the RNN-DQN case, inputs and outputs contain a time dimension.
            td_targets = compute_td_targets(
                next_q_values,
                rewards=reward_scale_factor * next_time_steps.reward,
                discounts=gamma * next_time_steps.discount)

            valid_mask = tf.cast(time_steps.is_last(), tf.float32)

            td_error = valid_mask * (q_values - td_targets)

            td_loss = valid_mask * td_errors_loss_fn(q_values, td_targets)

            if nest_utils.is_batched_nested_tensors(
                time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            # Note: We use an element wise loss above to ensure each element is always
            #   weighted by 1/N where N is the batch size, even when some of the
            #   weights are zero due to boundary transitions. Weighting by 1/K where K
            #   is the actual number of non-zero weight would artificially increase
            #   their contribution in the loss. Think about what would happen as
            #   the number of boundary samples increases.

            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights,
                regularization_loss=self._network.losses)
            total_loss = agg_loss.total_loss
            
            losses_dict = {'td_loss': agg_loss.weighted,
                            'reg_loss': agg_loss.regularization,
                            'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                        step=self.train_step_counter,
                                        name_scope='Losses/')

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self._network.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common.generate_tensor_summaries('td_error', td_error,
                                                self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                self.train_step_counter)
                common.generate_tensor_summaries('next_q_values', next_q_values,
                                                self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                self.train_step_counter)

            return tf_agent.LossInfo(total_loss, dqn_agent.DqnLossInfo(td_loss=td_loss,
                                                            td_error=td_error))

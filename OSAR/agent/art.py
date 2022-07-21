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

"""Module, defining Adversarial Reward Tracing Agent for Keras

Classes:
    - ArtAgent: Main class of ArtAgent

Other:
    - PnuLossInfo: Loss Info object

Dependancy:
    - OSAR.policy.ActorRewardPolicy


DISCLAIMER
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
from typing import Callable, Optional, Text

import gin
import numpy as np
from six.moves import zip
import tensorflow as tf # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.utils import object_identity
from tf_agents.agents.sac import sac_agent
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec

from OSAR.policy import ActorRewardPolicy

__all__ = ['ArtAgent']

PnuLossInfo = collections.namedtuple(
    'PnuLossInfo', ('discriminator_loss', 'actor_loss', 'alpha_loss'))

@gin.configurable
class ArtAgent(tf_agent.TFAgent):
    """A Positive--Unlabelled agent for work with ART network and discriminator.

    Based on SAC agents from tf_agents.agents.sac_agent.SacAgent.
    """

    def __init__(self,
                time_step_spec: ts.TimeStep,
                action_spec: types.NestedTensorSpec,
                actor_network: network.Network,
                discriminator_network: network.Network,
                actor_optimizer: types.Optimizer,
                discriminator_optimizer: types.Optimizer,
                alpha_optimizer: Optional[types.Optimizer] = None,
                actor_policy_ctor: Callable[
                    ..., tf_policy.TFPolicy] = ActorRewardPolicy,
                initial_alpha: Optional[types.Float] = 0.1,
                initial_etha: Optional[types.Float] = 0.1,
                initial_beta: Optional[types.Float] = 0.5,
                use_log_alpha_in_alpha_loss: bool = True,
                loss_fn: types.LossFn = common.element_wise_huber_loss,
                gamma: Optional[types.Float] = 1.0,
                reward_scale_factor: Optional[types.Float] = 1.0,
                gradient_clipping: Optional[types.Float] = None,
                debug_summaries: bool = False,
                summarize_grads_and_vars: bool = False,
                train_step_counter: Optional[tf.Variable] = None,
                name: Optional[Text] = None):
        """A Positive--Negative-Unlabelled agent for work with ART network and discriminator.
        Args:
        time_step_spec: A `TimeStep` spec of the expected time_steps.
        action_spec: A nest of BoundedTensorSpec representing the actions.
        actor_network: A function actor_network(observation, action_spec) that
            returns action distribution.
        discriminator_network: A function discriminator_network((observations, actions, rewards)) that
            returns binary estimation (q_values) for each observation, action and reward.
        actor_optimizer: The optimizer to use for the actor network.
        discriminator_optimizer: The optimizer to use for the discriminator network.
        alpha_optimizer: (Optional.) The optimizer to use for the alpha hyperparameter.
        actor_policy_ctor: The policy class to use.
        initial_alpha: Initial value for initial_alpha.
        initial_etha: Initial value for etha hyperparameter (0 < etha <= 1).
        initial_beta: Initial value for beta hyperparameter (beta > 0).
        use_log_alpha_in_alpha_loss: A boolean, whether using log_alpha or alpha
            in alpha loss. Certain implementations of SAC use log_alpha as log
            values are generally nicer to work with.
        loss_fn: A function for computing the elementwise discriminator errors loss.
        gamma: A discount factor for future rewards.
        reward_scale_factor: Multiplicative scale for the reward.
        gradient_clipping: Norm length to clip gradients.
        summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
        train_step_counter: An optional counter to increment every time the train
            op is run.  Defaults to the global_step.
        name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name=name)

        # self._check_action_spec(action_spec)

        net_observation_spec = time_step_spec.observation
        reward_spec = time_step_spec.reward
        if not reward_spec:
            reward_spec = tensor_spec.BoundedTensorSpec(
                shape=[], dtype=tf.float32, minimum=tf.float32.min, maximum=tf.float32.max)
        discriminator_spec = [net_observation_spec, action_spec, reward_spec]

        if actor_network:
            actor_network.create_variables(net_observation_spec)
        self._actor_network = actor_network

        policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False)

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True)

        if discriminator_network:
            discriminator_network.create_variables(discriminator_spec)
        self._discriminator_network = discriminator_network

        if not alpha_optimizer:
            alpha_optimizer = discriminator_optimizer.copy()

        self._alpha = common.create_variable(
            'initial_alpha',
            initial_value=initial_alpha,
            dtype=tf.float32,
            trainable=True)

        self._etha = common.create_variable(
            'etha',
            initial_value=initial_etha,
            dtype=tf.float32,
            trainable=False)
        self._beta = common.create_variable(
            'beta',
            initial_value=initial_beta,
            dtype=tf.float32,
            trainable=False)

        entropy = self._get_default_entropy(action_spec)

        self._use_log_alpha_in_alpha_loss = use_log_alpha_in_alpha_loss
        self._actor_optimizer = actor_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._loss_fn = loss_fn
        self._reward_scale_factor = reward_scale_factor
        self._gamma = gamma
        self._entropy = entropy
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

        train_sequence_length = 2 if not discriminator_network.state_spec else None

        super(ArtAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
        )

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=(train_sequence_length == 2))

    # def _check_action_spec(self, action_spec):
    #     flat_action_spec = tf.nest.flatten(action_spec)
    #     for spec in flat_action_spec:
    #         if spec.dtype.is_integer:
    #             raise NotImplementedError(
    #                 'ArtAgent does not currently support discrete actions. '
    #                 'Action spec: {}'.format(action_spec))

    def _get_default_entropy(self, action_spec):
        # If target_entropy was not passed, set it to -dim(A)/2.0
        # Note that the original default entropy target is -dim(A) in the SAC paper.
        # However this formulation has also been used in practice by the original
        # authors and has in our experience been more stable for gym/mujoco.
        flat_action_spec = tf.nest.flatten(action_spec)
        entropy = -np.sum([
            np.product(single_spec.shape.as_list())
            for single_spec in flat_action_spec
        ]) / 2.0
        return entropy

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.
        This method trains with the provided batched experience.
        Args:
        experience: A time-stacked trajectory object.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
        A train_op.
        Raises:
        ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        trainable_discriminator_variables = list(object_identity.ObjectIdentitySet(
            self._discriminator_network.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_discriminator_variables, ('No trainable discriminator variables to '
                                                'optimize.')
            tape.watch(trainable_discriminator_variables)
            discriminator_loss = self.discriminator_loss(
                time_steps,
                actions,
                next_time_steps,
                errors_loss_fn=self._loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(discriminator_loss, 'Discriminator loss is inf or nan.')
        discriminator_grads = tape.gradient(discriminator_loss, trainable_discriminator_variables)
        self._apply_gradients(discriminator_grads, trainable_discriminator_variables,
                              self._discriminator_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                                'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self.actor_loss(
                time_steps, weights=weights, training=True)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        alpha_variable = [self._alpha]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert alpha_variable, 'No alpha variable to optimize.'
            tape.watch(alpha_variable)
            alpha_loss = self.alpha_loss(
                time_steps, weights=weights, training=True)
        tf.debugging.check_numerics(alpha_loss, 'alpha loss is inf or nan.')
        alpha_grads = tape.gradient(alpha_loss, alpha_variable)
        self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='discriminator_loss', data=discriminator_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)

        total_loss = discriminator_loss + actor_loss + alpha_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _loss(self,
                experience: types.NestedTensor,
                weights: Optional[types.Tensor] = None,
                training: bool = False):
        """Returns the loss of the provided experience.
        This method is only used at test time!
        Args:
        experience: A time-stacked trajectory object.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        training: Whether this loss is being calculated as part of training.
        Returns:
        A `LossInfo` containing the loss for the experience.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action
        discriminator_loss = self.discriminator_loss(
            time_steps,
            actions,
            next_time_steps,
            td_errors_loss_fn=self._loss_fn,
            gamma=self._gamma,
            reward_scale_factor=self._reward_scale_factor,
            weights=weights,
            training=training)
        tf.debugging.check_numerics(discriminator_loss, 'Discriminator loss is inf or nan.')

        actor_loss = self.actor_loss(
            time_steps, weights=weights, training=training)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

        alpha_loss = self.alpha_loss(
            time_steps, weights=weights, training=training)
        tf.debugging.check_numerics(alpha_loss, 'alpha loss is inf or nan.')

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='discriminator_loss', data=discriminator_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

        total_loss = discriminator_loss + actor_loss + alpha_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _actions_and_log_probs(self, time_steps, training=False):
        """Get actions and corresponding log probabilities from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        policy_state = self._train_policy.get_initial_state(batch_size)
        if training:
            action_distribution = self._train_policy.distribution(
                time_steps, policy_state=policy_state).action
        else:
            action_distribution = self._policy.distribution(
                time_steps, policy_state=policy_state).action

        # Sample actions and log_pis from transformed distribution.
        actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
        log_pi = common.log_probability(action_distribution, actions,
                                        self.action_spec)

        return actions, log_pi

    def _calculate_risk(self,
            log_pis: types.Tensor,
            policy_probs: types.Tensor,
            expert_probs: types.Tensor,
            reward_fn: Optional[types.Tensor] = None,
            ) -> types.Tensor:
        '''Non-Negative positive unlabelled risk estimator method. This code is highly adapted from
        https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py
        and
        https://github.com/copenlu/check-worthiness-pu-learning/blob/master/nn.py

        '''

        positive = expert_probs * tf.nn.sigmoid(reward_fn)
        n_positive = tf.math.maximum(1.0, tf.reduce_sum(positive))

        unlabelled = policy_probs * tf.nn.sigmoid(reward_fn)
        n_unlabelled = tf.math.maximum(1.0, tf.reduce_sum(unlabelled))

        negative = (1 - expert_probs) * -tf.nn.sigmoid(reward_fn)
        n_negative = tf.math.maximum(1.0, tf.reduce_sum(negative))

        base_loss = log_pis
        reverse_loss = -log_pis

        positive_risk = (self._etha * positive / n_positive) * base_loss
        negative_treshold = (unlabelled / n_unlabelled - self._etha * negative / n_negative)
        negative_risk = -negative_treshold * reverse_loss,

        pu_risk = self._etha * positive / n_positive + tf.math.maximum(-self._beta, negative_risk)

        ### Here is the non-negative trick that Kiryo et al. introduced in NeurIPS 2017 for flexible estimators
        ### (i.e. neural nets)
        # loss = tf.where(negative_risk >= - self._beta, - self._gamma * negative_risk, positive_risk + negative_risk)
        loss = tf.where(negative_treshold >= - self._beta, pu_risk, negative_risk)

        labels = tf.nn.sigmoid(reward_fn) > self._beta
        logits = tf.nn.sigmoid(expert_probs) > self._beta
        positive = tf.where(labels, 1.0, 0.0)
        negative = tf.where(labels, 1.0, 0.0)
        unlabelled = tf.where(tf.logical_not(logits), 1.0, 0.0)

        return loss[0, ...], positive, negative

    def discriminator_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the discriminator loss for PU training.
        Args:
        time_steps: A batch of timesteps.
        actions: A batch of actions.
        next_time_steps: A batch of next timesteps.
        errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        training: Whether this loss is being used for training.
        Returns:
        discriminator_loss: A scalar critic loss.
        """
        with tf.name_scope('discriminator_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            actions, next_log_pis = self._actions_and_log_probs(next_time_steps,
                                                                    training=False)
            discriminator_input = next_time_steps.observation

            tf.debugging.check_numerics(discriminator_input, 'discriminator_input Values are inf or nan.')
            discriminator_q_values, targets, target_q_values, reward_fn, unused_network_state = self._discriminator_network(
                discriminator_input, actions=actions, rewards=next_time_steps.reward,
                step_type=next_time_steps.step_type, training=training)
            target_q_values = target_q_values[..., 0] - tf.exp(self._alpha) * tf.expand_dims(next_log_pis, axis=-1)

            tf.debugging.check_numerics(discriminator_q_values, 'Discriminator Values are inf or nan.')
            discriminator_probs = tf.nn.softmax(discriminator_q_values, axis=-1)
            target_probs = tf.stop_gradient(tf.nn.softmax(target_q_values, axis=-1))
            reward = tf.expand_dims(time_steps.reward, axis=-1)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(discriminator_probs)
                risk_fn, positive, negative = self._calculate_risk(
                    tf.expand_dims(next_log_pis, axis=-1), discriminator_probs, target_probs, reward_fn)
            new_d = tape.gradient([risk_fn], discriminator_probs)
            new_reward = tf.where(new_d >= 0.5, reward_fn, reward)
            # new_reward = -tf.math.log(1 - new_d)

            td_targets = tf.stop_gradient(
                tf.expand_dims(
                    reward_scale_factor, axis=-1) * new_reward + tf.expand_dims(
                        gamma, axis=-1) * tf.expand_dims(next_time_steps.discount, axis=-1) * target_probs)
            # discriminator_loss = self._loss_fn(discriminator_probs, td_targets)
            discriminator_loss = common.element_wise_squared_loss(discriminator_probs, td_targets)

            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=discriminator_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            discriminator_loss = agg_loss.total_loss

            self._discriminator_loss_debug_summaries(
                discriminator_q_values,
                positive,
                negative)

            return discriminator_loss

    def actor_loss(self,
                    time_steps: ts.TimeStep,
                    weights: Optional[types.Tensor] = None,
                    training: Optional[bool] = True) -> types.Tensor:
        """Computes the actor_loss for PU training.
        Args:
        time_steps: A batch of timesteps.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        training: Whether training should be applied.
        Returns:
        actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            actions, next_log_pis = self._actions_and_log_probs(time_steps,
                                                                     training=training)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                discriminator_input = time_steps.observation#, actions, time_steps.reward)

                discriminator_q_values, targets, target_q_values, reward_fn, unused_network_state = self._discriminator_network(
                    discriminator_input, actions=actions, rewards=time_steps.reward,
                    step_type=time_steps.step_type, training=False)
                target_q_values = target_q_values[..., 0] - tf.exp(self._alpha) * tf.expand_dims(next_log_pis, axis=-1)

                tf.debugging.check_numerics(discriminator_q_values, 'Discriminator Values are inf or nan.')
                discriminator_probs = tf.nn.softmax(discriminator_q_values, axis=-1)
                target_probs = tf.stop_gradient(tf.nn.softmax(target_q_values, axis=-1))
                reward = time_steps.reward

                risk_fn, positive, negative = self._calculate_risk(
                    tf.expand_dims(next_log_pis, axis=-1), discriminator_probs, target_probs, reward_fn)

                actions = tf.nest.flatten(actions)

            d4pg_grad = tape.gradient([discriminator_probs], actions)
            actor_loss = d4pg_grad[0] + tf.exp(self._alpha) * tf.expand_dims(next_log_pis, axis=-1) - risk_fn

            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss

            self._actor_loss_debug_summaries(actor_loss, actions, next_log_pis,
                                        discriminator_q_values, time_steps)

            return actor_loss

    def alpha_loss(self,
                    time_steps: ts.TimeStep,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the alpha_loss for EC-SAC training.
        Args:
        time_steps: A batch of timesteps.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        training: Whether this loss is being used during training.
        Returns:
        alpha_loss: A scalar alpha loss.
        """
        with tf.name_scope('alpha_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            # We do not update actor during alpha loss.
            actions, next_log_pis = self._actions_and_log_probs(
                time_steps, training=False)
            discriminator_q_values, targets, target_q_values, reward_fn, unused_network_state = self._discriminator_network(
                time_steps.observation, actions=actions, rewards=time_steps.reward,
                step_type=time_steps.step_type, training=False)

            tf.debugging.check_numerics(discriminator_q_values, 'Discriminator Values are inf or nan.')
            discriminator_probs = tf.nn.softmax(discriminator_q_values, axis=-1)
            target_probs = tf.stop_gradient(tf.nn.softmax(target_q_values, axis=-1))
            reward = time_steps.reward

            risk_fn, positive, negative = self._calculate_risk(
                tf.expand_dims(next_log_pis, axis=-1), discriminator_probs, target_probs, reward_fn)

            entropy_diff = tf.stop_gradient(-next_log_pis - self._entropy)
            if self._use_log_alpha_in_alpha_loss:
                alpha_loss = (self._alpha * entropy_diff)
            else:
                alpha_loss = (tf.exp(self._alpha) * entropy_diff)

            agg_loss = common.aggregate_losses(
                per_example_loss=alpha_loss, sample_weight=weights)
            alpha_loss = agg_loss.total_loss

            self._alpha_loss_debug_summaries(alpha_loss, entropy_diff)

            return alpha_loss

    def _discriminator_loss_debug_summaries(self, td_targets, positive_samples, negative_samples):
        if self._debug_summaries:
            tf_positive_errors = td_targets - positive_samples
            td_negative_errors = td_targets - negative_samples
            td_errors = tf.concat([tf_positive_errors, td_negative_errors], axis=0)
            common.generate_tensor_summaries('td_errors', td_errors,
                                            self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                            self.train_step_counter)
            common.generate_tensor_summaries('pred_positive_td_samples', positive_samples,
                                            self.train_step_counter)
            common.generate_tensor_summaries('pred_negative_td_samples', negative_samples,
                                            self.train_step_counter)

    def _actor_loss_debug_summaries(self, actor_loss, actions, log_pi,
                                  target_q_values, time_steps):
        if self._debug_summaries:
            common.generate_tensor_summaries('actor_loss', actor_loss,
                                            self.train_step_counter)
            try:
                for name, action in nest_utils.flatten_with_joined_paths(actions):
                    common.generate_tensor_summaries(name, action,
                                                    self.train_step_counter)
            except ValueError:
                pass  # Guard against internal SAC variants that do not directly
                # generate actions.

            common.generate_tensor_summaries('log_pi', log_pi,
                                            self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_avg',
                data=-tf.reduce_mean(input_tensor=log_pi),
                step=self.train_step_counter)
            common.generate_tensor_summaries('target_q_values', target_q_values,
                                            self.train_step_counter)
            batch_size = nest_utils.get_outer_shape(time_steps,
                                                    self._time_step_spec)[0]
            policy_state = self._train_policy.get_initial_state(batch_size)
            action_distribution = self._train_policy.distribution(
                time_steps, policy_state).action
            if isinstance(action_distribution, tfp.distributions.Normal):
                common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                                self.train_step_counter)
                common.generate_tensor_summaries('act_stddev',
                                                action_distribution.scale,
                                                self.train_step_counter)
            elif isinstance(action_distribution, tfp.distributions.Categorical):
                common.generate_tensor_summaries('act_mode', action_distribution.mode(),
                                                self.train_step_counter)
            try:
                for name, action_dist in nest_utils.flatten_with_joined_paths(
                    action_distribution):
                    common.generate_tensor_summaries('entropy_' + name,
                                                    action_dist.entropy(),
                                                    self.train_step_counter)
            except NotImplementedError:
                pass  # Some distributions do not have an analytic entropy.

    def _alpha_loss_debug_summaries(self, alpha_loss, entropy_diff):
        if self._debug_summaries:
            common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                            self.train_step_counter)
            common.generate_tensor_summaries('entropy_diff', entropy_diff,
                                            self.train_step_counter)

            tf.compat.v2.summary.scalar(
                name='alpha', data=self._alpha, step=self.train_step_counter)

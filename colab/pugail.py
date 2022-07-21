# coding=utf-8
# Copyright 2022 Konstantin Ustyuzhanin.
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
from turtle import back
from typing import Callable, Optional, Text
from typing import cast

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
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import utils

__all__ = ['PugailAgent', 'CategoricalCritic', 'CategoricalActor']

PnuLossInfo = collections.namedtuple(
    'PnuLossInfo', ('discriminator_loss', 'actor_loss', ))

def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)

def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1,
                           seed_stream_class=tfp.util.SeedStream,
                           seed=None):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

@gin.configurable
class CategoricalActor(actor_distribution_network.ActorDistributionNetwork):
     def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(200, 100),
               dropout_layer_params=None,
               reg_num=0.01,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               seed_stream_class=tfp.util.SeedStream,
               seed=None,
               batch_squash=True,
               dtype=tf.float32,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               name='CategoricalActorNetwork'):

        super(CategoricalActor, self).__init__(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            seed_stream_class=seed_stream_class,
            seed=seed,
            batch_squash=batch_squash,
            dtype=dtype,
            discrete_projection_net=discrete_projection_net,
            continuous_projection_net=continuous_projection_net,
            name=name,
        )

        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            weight_decay_params=[reg_num for _ in range(len(fc_layer_params))],
            dtype=dtype)

        def map_proj(spec):
            if tensor_spec.is_discrete(spec):
                return discrete_projection_net(spec)
            else:
                kwargs = {}
                if continuous_projection_net is _normal_projection_net:
                    kwargs['seed'] = seed
                    kwargs['seed_stream_class'] = seed_stream_class
                return continuous_projection_net(spec, **kwargs)

        projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
        output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                            projection_networks)

        self._encoder = encoder
        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec


@gin.configurable
class CategoricalCritic(critic_network.CriticNetwork):
    """Creates a critic network."""

    def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=None,
               observation_dropout_layer_params=None,
               num_atoms=51,
               q_value=20,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               activation_fn=tf.nn.relu,
               output_activation_fn=None,
               kernel_initializer=None,
               reg_num=0.01,
               last_kernel_initializer=None,
               last_layer=None,
               name='CategoricalCriticNetwork'):

        super(CategoricalCritic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            observation_conv_layer_params=observation_conv_layer_params,
            observation_fc_layer_params=observation_fc_layer_params,
            observation_dropout_layer_params=observation_dropout_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            action_dropout_layer_params=action_dropout_layer_params,
            joint_fc_layer_params=joint_fc_layer_params,
            joint_dropout_layer_params=joint_dropout_layer_params,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            kernel_initializer=kernel_initializer,
            last_kernel_initializer=last_kernel_initializer,
            last_layer=last_layer,
            name=name,
        )

        observation_spec, action_spec = input_tensor_spec

        if kernel_initializer is None:
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform')
        if last_kernel_initializer is None:
            last_kernel_initializer = tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003)

        # TODO(kbanoop): Replace mlp_layers with encoding networks.
        self._observation_layers = utils.mlp_layers(
            observation_conv_layer_params,
            observation_fc_layer_params,
            observation_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[reg_num for _ in range(len(observation_fc_layer_params))] if observation_fc_layer_params else None,
            name='observation_encoding')

        self._action_layers = utils.mlp_layers(
            None,
            action_fc_layer_params,
            action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[reg_num for _ in range(len(action_fc_layer_params))] if action_fc_layer_params else None,
            name='action_encoding')

        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[reg_num for _ in range(len(joint_fc_layer_params))] if joint_fc_layer_params else None,
            name='joint_mlp')

        if not last_layer:
            last_layer = tf.keras.layers.Dense(
                1,
                activation=output_activation_fn,
                kernel_initializer=last_kernel_initializer,
                kernel_regularizer='l1_l2',
                name='value')
        self._joint_layers.append(last_layer)

        self.output_logits = tf.keras.layers.Dense(
            num_atoms,
            kernel_initializer=kernel_initializer,
            name='output_logits'
        )

        # Generate support in numpy so that we can assign it to a constant and avoid
        # having a tensor property.
        support = np.linspace(-q_value, q_value, num_atoms,
                            dtype=np.float32)
        self.support = tf.constant(support, dtype=tf.float32)

        self._num_atoms = num_atoms
        self._q_value = q_value

    @property
    def num_atoms(self):
        return self._num_atoms

    @property
    def q_value(self):
        return self._q_value

    def call(self,
           observations,
           step_type,
           network_state=None,
           training=False,
           mask=None):
        output_levels, network_state = super(CategoricalCritic, self).call(
            observations,
            step_type,
            network_state,
            training=training,
            # mask=mask
        )
        q_logits = self.output_logits(tf.expand_dims(output_levels, axis=-1))
        q_values = common.convert_q_logits_to_values(q_logits, self.support)

        output_actions = q_values
        # return output_actions, network_state
        return q_logits, network_state

@gin.configurable
class PugailAgent(tf_agent.TFAgent):
    """A Positive--Unlabelled agent for work with generic network and discriminator.

    Based on SAC agents from tf_agents.agents.sac_agent.SacAgent.
    """

    def __init__(self,
                time_step_spec: ts.TimeStep,
                action_spec: types.NestedTensorSpec,
                actor_network: network.Network,
                discriminator_network: network.Network,
                actor_optimizer: types.Optimizer,
                discriminator_optimizer: types.Optimizer,
                etha_optimizer: Optional[types.Optimizer] = None,
                discriminator_loss_fn: Optional[types.LossFn] = common.element_wise_squared_loss,
                gamma: Optional[types.Float] = 1.0,
                reward_scale_factor: Optional[types.Float] = 1.0,
                actor_policy_ctor: Callable[
                    ..., tf_policy.TFPolicy] = actor_policy.ActorPolicy,
                initial_etha: Optional[types.Float] = 0.1,
                initial_beta: Optional[types.Float] = 0.5,
                gradient_clipping: Optional[types.Float] = None,
                debug_summaries: Optional[bool] = False,
                summarize_grads_and_vars: Optional[bool] = False,
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
        etha_optimizer: (Optional.) The optimizer to use for the etha hyperparameter.
        discriminator_loss_fn: A function for computing the elementwise discriminator errors loss.
        gamma: A discount factor for future rewards.
        reward_scale_factor: Multiplicative scale for the reward.
        actor_policy_ctor: The policy class to use.
        initial_etha: Initial value for etha hyperparameter (0 < etha <= 1).
        initial_beta: Initial value for beta hyperparameter (beta > 0).
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

        self._actor_optimizer = actor_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._discriminator_loss_fn = discriminator_loss_fn
        self._gamma = gamma
        self._dqda_clipping = None
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

        train_sequence_length = 2 if not discriminator_network.state_spec else None

        super(PugailAgent, self).__init__(
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

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)
        for spec in flat_action_spec:
            if spec.dtype.is_integer:
                raise NotImplementedError(
                    'PugailAgent does not currently support discrete actions. '
                    'Action spec: {}'.format(action_spec))

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
                errors_loss_fn=self._discriminator_loss_fn,
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

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='discriminator_loss', data=discriminator_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)

        total_loss = discriminator_loss + actor_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _loss(self,
                experience: types.NestedTensor,
                weights: Optional[types.Tensor] = None,
                training: Optional[bool] = False):
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
            td_errors_loss_fn=self._discriminator_loss_fn,
            gamma=self._gamma,
            reward_scale_factor=self._reward_scale_factor,
            weights=weights,
            training=training)
        tf.debugging.check_numerics(discriminator_loss, 'Discriminator loss is inf or nan.')

        actor_loss = self.actor_loss(
            time_steps,
            weights=weights, training=training)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')


        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='discriminator_loss', data=discriminator_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)

        total_loss = discriminator_loss + actor_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss, )

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

    def _calculate_pu(self,
            actor_logits: types.Tensor,
            discriminator_logits: types.Tensor,
            reward_function: Optional[types.Tensor] = None,
            ) -> types.Tensor:
        '''Non-Negative positive unlabelled risk estimator method. This code is highly adapted from
        https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py
        and
        https://github.com/copenlu/check-worthiness-pu-learning/blob/master/nn.py

        Here reward function
        '''

        policy_probs = tf.nn.softmax(actor_logits, axis=-1)
        disc_probs = -tf.math.log(tf.nn.sigmoid(discriminator_logits*policy_probs))
        td_estimator = -tf.math.log(
            tf.nn.sigmoid(policy_probs * reward_function + self._gamma * discriminator_logits))
        # td_estimator = tf.math.tanh(-tf.math.log(tf.nn.sigmoid(reward_function)))

        # First we need to form our labels and logits functions
        labels = td_estimator > 0
        logits = disc_probs > 0

        positive = tf.cast(logits, tf.float32) * tf.where(logits, 1.0, 0.0)
        n_positive = tf.math.maximum(1.0, tf.reduce_sum(positive))

        unlabelled = tf.cast(logits, tf.float32) * tf.where(tf.logical_not(logits), 1.0, 0.0)
        n_unlabelled = tf.math.maximum(1.0, tf.reduce_sum(unlabelled))

        negative = tf.cast(td_estimator < 0, tf.float32) * tf.where(td_estimator < 0, 1.0, 0.0)

        base_loss = self._discriminator_loss_fn(
                tf.cast(logits, tf.float32), tf.cast(labels, tf.float32))
        reverse_loss = self._discriminator_loss_fn(
                tf.cast(logits, tf.float32), tf.cast(tf.logical_not(labels), tf.float32))

        # positive_loss = tf.reduce_sum(
        positive_loss = (self._etha * positive / n_positive) * base_loss,
            # axis=-1)
        positive_loss = tf.cast(positive_loss, tf.float32)
        # negative_loss = tf.reduce_sum(
        negative_loss = (td_estimator * unlabelled / n_unlabelled - self._etha * td_estimator * positive / n_positive) * reverse_loss,
            # axis=-1)
        negative_loss = tf.cast(negative_loss, tf.float32)

        ### Here is the non-negative trick that Kiryo et al. introduced in NeurIPS 2017 for flexible estimators
        ### (i.e. neural nets)
        loss = tf.where(negative_loss <= - self._beta, - self._gamma * negative_loss, positive_loss + negative_loss)

        return loss, positive, negative

    def discriminator_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: Optional[bool] = False) -> types.Tensor:
        """Computes the discriminator loss for PNU training.
        Args:
        time_steps: A batch of timesteps.
        actions: A batch of actions.
        next_time_steps: A batch of next timesteps.
        errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
        gamma: Discount for future rewards.
        reward_scale_factor: Multiplicative factor to scale rewards.
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

            batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

            next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps,
                                                                    training=False)
            discriminator_input = (next_time_steps.observation, next_actions)
            initial_state = self._discriminator_network.get_initial_state(batch_size)
            discriminator_q_logits, unused_network_state = self._discriminator_network(
                discriminator_input,
                #network_state=initial_state,
                step_type=next_time_steps.step_type, training=training)
            discriminator_q_values = common.convert_q_logits_to_values(discriminator_q_logits, self._discriminator_network.support)

            reward = time_steps.reward

            pu_loss, pos_samples, neg_samples = self._calculate_pu(
                next_log_pis,
                discriminator_q_values,
                reward
                )
            new_d = tf.stop_gradient(pu_loss)
            # new_reward = tf.where(new_d >= 0.5, 1.0, 0.0) * reward
            new_reward = tf.math.log(1 - new_d)

            discriminator_loss = pu_loss

            if discriminator_loss.shape.rank > 1:
                # Sum over the time dimension.
                discriminator_loss = tf.reduce_sum(
                    discriminator_loss, axis=range(1, discriminator_loss.shape.rank))

            reg_loss = self._discriminator_network.losses if self._discriminator_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=discriminator_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)

            discriminator_loss = agg_loss.total_loss

            self._discriminator_loss_debug_summaries(
                new_d,
                pos_samples,
                neg_samples)

            return discriminator_loss

    def actor_loss(self,
                    time_steps: ts.TimeStep,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: Optional[bool] = True) -> types.Tensor:
        """Computes the actor_loss for PNU training.
        Args:
        time_steps: A batch of timesteps.
        gamma: Discount for future rewards.
        reward_scale_factor: Multiplicative factor to scale rewards.
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

                discriminator_input = (time_steps.observation, actions)#, next_actions, time_steps.reward)

                discriminator_q_logits, unused_network_state = self._discriminator_network(
                    discriminator_input,
                    step_type=time_steps.step_type, training=False)
                tf.debugging.check_numerics(discriminator_q_logits, 'Discriminator Values are inf or nan.')
                discriminator_q_values = common.convert_q_logits_to_values(discriminator_q_logits, self._discriminator_network.support)
                discriminator_probs = tf.nn.softmax(discriminator_q_values, axis=-1)

                reward = time_steps.reward

                pu_loss, pos_samples, neg_samples = self._calculate_pu(
                    next_log_pis,
                    discriminator_q_values,
                    reward
                    )
                new_d = tf.stop_gradient(pu_loss)
                # new_reward = tf.where(new_d >= 0.5, 1.0, 0.0) * reward
                new_reward = tf.math.log(1 - new_d)

            actor_loss = pu_loss

            if actor_loss.shape.rank > 1:
                # Sum over the time dimension.
                actor_loss = tf.reduce_sum(
                    actor_loss, axis=range(1, actor_loss.shape.rank))

            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss

            self._actor_loss_debug_summaries(actor_loss, actions, next_log_pis,
                                        new_d, time_steps)

            return actor_loss

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

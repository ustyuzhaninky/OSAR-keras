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

'''DISCLAIMER
Based on D4PG implementation of Mark Sinton (msinto93@gmail.com),
URL: https://github.com/msinto93/D4PG

'''
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

__all__ = ['NnPugailAgent', 'CategoricalCritic', 'CategoricalActor', 'EnvCritic']

PnuLossInfo = collections.namedtuple(
    'PnuLossInfo', ('discriminator_loss', 'actor_loss', 'reward_model_loss'))

def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)

def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1,
                           seed_stream_class=tfp.util.SeedStream,
                           seed=None):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p_new = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_p_new)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_p_new - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q_new = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p_new - z_q_new   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p_new = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p_new, 2)

@gin.configurable
class EnvCritic(network.Network):
    """Creates a critic network."""

    def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=None,
               observation_dropout_layer_params=None,
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
               name='EnvCriticNetwork'):

        super(EnvCritic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        observation_spec = input_tensor_spec

        if len(tf.nest.flatten(observation_spec)) > 1:
            raise ValueError('Only a single observation is supported by this network')

        observation_spec = input_tensor_spec

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
            weight_decay_params=[
                reg_num for _ in range(len(observation_fc_layer_params))] if observation_fc_layer_params else None,
            name='observation_encoding')

        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[
                reg_num for _ in range(len(joint_fc_layer_params))] if joint_fc_layer_params else None,
            name='joint_mlp')

        if not last_layer:
            last_layer = tf.keras.layers.Dense(
                1,
                activation=output_activation_fn,
                kernel_initializer=last_kernel_initializer,
                # kernel_regularizer='l1_l2',
                name='value')
        self._joint_layers.append(last_layer)

    def call(self,
           observations,
           step_type,
           network_state=None,
           training=False,
           mask=None):
        del step_type  # unused.
        observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
        for layer in self._observation_layers:
            observations = layer(observations, training=training)

        joint = observations
        for layer in self._joint_layers:
            joint = layer(joint, training=training)

        return joint, network_state
        # return tf.reshape(joint, [-1]), network_state

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
            weight_decay_params=[
                reg_num for _ in range(len(observation_fc_layer_params))] if observation_fc_layer_params else None,
            name='observation_encoding')

        self._action_layers = utils.mlp_layers(
            None,
            action_fc_layer_params,
            action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[
                reg_num for _ in range(len(action_fc_layer_params))] if action_fc_layer_params else None,
            name='action_encoding')

        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            weight_decay_params=[
                reg_num for _ in range(len(joint_fc_layer_params))] if joint_fc_layer_params else None,
            name='joint_mlp')

        if not last_layer:
            last_layer = tf.keras.layers.Dense(
                1,
                activation=output_activation_fn,
                kernel_initializer=last_kernel_initializer,
                # kernel_regularizer='l1_l2',
                name='value')
        self._joint_layers.append(last_layer)

        self.output_logits = tf.keras.layers.Dense(
            1,
            kernel_initializer=kernel_initializer,
            name='output',
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
        return output_actions, network_state
        # return q_logits, network_state

@gin.configurable
class NnPugailAgent(tf_agent.TFAgent):
    """A Positive--Unlabelled agent for work with generic network and discriminator.

    Based on SAC agents from tf_agents.agents.sac_agent.SacAgent.
    """

    def __init__(self,
                time_step_spec: ts.TimeStep,
                action_spec: types.NestedTensorSpec,
                actor_network: network.Network,
                discriminator_network: network.Network,
                reward_network: network.Network,
                actor_optimizer: types.Optimizer,
                discriminator_optimizer: types.Optimizer,
                reward_model_optimizer: types.Optimizer,
                etha_optimizer: Optional[types.Optimizer] = None,
                target_update_tau: Optional[types.Float] = 1.0,
                target_update_period: Optional[types.Int] = 1,
                loss_fn: Optional[types.LossFn] = common.element_wise_huber_loss,
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
        reward_network: A function reward_network(observation,) that
            returns predicted reward distribution.
        actor_optimizer: The optimizer to use for the actor network.
        discriminator_optimizer: The optimizer to use for the discriminator network.
        reward_model_optimizer: The optimizer to use for the reward model.
        etha_optimizer: (Optional.) The optimizer to use for the etha hyperparameter.
        target_update_tau: Factor for soft update of the target networks.
        target_update_period: Period for soft update of the target networks.
        loss_fn: A function for computing the elementwise errors loss.
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

        if reward_network:
            reward_network.create_variables(net_observation_spec)
        self._reward_network = reward_network

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

        self._target_actor_network = self._actor_network.copy(name='TargetActorNetwork')
        self._target_actor_network.create_variables(time_step_spec.observation)

        self._target_discriminator_network = self._discriminator_network.copy(name='TargetDiscriminatorNetwork')
        self._target_discriminator_network.create_variables(time_step_spec.observation)

        self._target_reward_network = self._reward_network.copy(name='TargetRewardNetwork')
        self._target_reward_network.create_variables(time_step_spec.observation)


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
        self._reward_model_optimizer = reward_model_optimizer
        self._td_loss_fn = loss_fn
        self._gamma = gamma
        self._d4pg_clipping = None
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._target_update_period = target_update_period
        self._target_update_tau = target_update_tau
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)

        train_sequence_length = 2 if not discriminator_network.state_spec else None

        super(NnPugailAgent, self).__init__(
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
                    'NnPugailAgent does not currently support discrete actions. '
                    'Action spec: {}'.format(action_spec))

    def _initialize(self):
        """Returns an op to initialize the agent.
        Copies weights from the Q networks to the target Q network.
        """

        common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

        common.soft_variables_update(
            self._discriminator_network.variables,
            self._target_discriminator_network.variables,
            tau=1.0)

        common.soft_variables_update(
            self._reward_network.variables,
            self._target_reward_network.variables,
            tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.
        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws
        Args:
        tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
        period: Step interval at which the target network is updated.
        Returns:
        A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_target'):

            def update():
                """Update target network."""
                actor_update = common.soft_variables_update(
                    self._actor_network.variables,
                    self._target_actor_network.variables,
                    tau,
                    tau_non_trainable=1.0)

                discriminator_update = common.soft_variables_update(
                    self._discriminator_network.variables,
                    self._target_discriminator_network.variables,
                    tau,
                    tau_non_trainable=1.0)

                reward_update = common.soft_variables_update(
                    self._reward_network.variables,
                    self._target_reward_network.variables,
                    tau,
                    tau_non_trainable=1.0)

                return actor_update, discriminator_update, reward_update

            return common.Periodically(update, period, 'update_targets')

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
                errors_loss_fn=self._td_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(discriminator_loss, 'Discriminator loss is inf or nan.')
        discriminator_grads = tape.gradient(discriminator_loss, trainable_discriminator_variables)
        self._apply_gradients(discriminator_grads, trainable_discriminator_variables,
                              self._discriminator_optimizer)


        trainable_reward_network_variables = list(object_identity.ObjectIdentitySet(
                    self._reward_network.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_reward_network_variables, ('No trainable discriminator variables to '
                                                'optimize.')
            tape.watch(trainable_reward_network_variables)
            reward_model_loss = self.reward_model_loss(
                time_steps,
                actions,
                next_time_steps,
                errors_loss_fn=self._td_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(reward_model_loss, 'Reward Model loss is inf or nan.')
        reward_model_grads = tape.gradient(reward_model_loss, trainable_reward_network_variables)
        self._apply_gradients(reward_model_grads, trainable_reward_network_variables,
                              self._reward_model_optimizer)


        trainable_reward_network_variables = list(object_identity.ObjectIdentitySet(
                    self._reward_network.trainable_variables))

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
            tf.compat.v2.summary.scalar(
                name='reward_model_loss', data=reward_model_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)
        self._update_target()

        total_loss = discriminator_loss + actor_loss + reward_model_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss, reward_model_loss=reward_model_loss)

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
            td_errors_loss_fn=self._td_loss_fn,
            gamma=self._gamma,
            reward_scale_factor=self._reward_scale_factor,
            weights=weights,
            training=training)
        tf.debugging.check_numerics(discriminator_loss, 'Discriminator loss is inf or nan.')

        actor_loss = self.actor_loss(
            time_steps,
            weights=weights, training=training)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

        reward_model_loss = self.reward_model_loss(
                time_steps,
                actions,
                next_time_steps,
                errors_loss_fn=self._td_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
        tf.debugging.check_numerics(reward_model_loss, 'Reward Model loss is inf or nan.')

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='discriminator_loss', data=discriminator_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='reward_model_loss', data=reward_model_loss, step=self.train_step_counter)

        total_loss = discriminator_loss + actor_loss + reward_model_loss

        extra = PnuLossInfo(
            discriminator_loss=discriminator_loss, actor_loss=actor_loss, reward_model_loss=reward_model_loss)

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

        # First we need to form our labels and logits functions
        # labels = discriminator_probs > 0.5
        # if not diff_reward_fn:
        #     reward_fn = tf.math.tanh(-tf.math.log(tf.nn.sigmoid(expert_probs)))
        #     labels = reward_fn
        # else:
        labels = tf.nn.sigmoid(reward_fn)

        positive = expert_probs * labels
        n_positive = tf.math.maximum(1.0, tf.reduce_sum(positive))

        unlabelled = policy_probs * labels
        n_unlabelled = tf.math.maximum(1.0, tf.reduce_sum(unlabelled))

        negative = (1 - expert_probs) * -labels
        n_negative = tf.math.maximum(1.0, tf.reduce_sum(negative))

        # positive = tf.cast(reward_fn, tf.float32) * tf.where(labels, 1.0, 0.0)
        # n_positive = tf.math.maximum(1.0, tf.reduce_sum(positive))

        # unlabelled = tf.cast(reward_fn, tf.float32) * tf.where(tf.logical_not(labels), 1.0, 0.0)
        # n_unlabelled = tf.math.maximum(1.0, tf.reduce_sum(unlabelled))

        # negative = tf.cast(td_estimator < 0, tf.float32) * tf.where(td_estimator < 0, 1.0, 0.0)

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

        return loss[0, ...], positive, negative

    def reward_model_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: Optional[bool] = False) -> types.Tensor:
        """Computes the reward model loss.
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
        reward_model_loss: A scalar reward (a.k.a. critic) loss.
        """
        with tf.name_scope('reward_model_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

            reward_fn, _ = self._reward_network(
                time_steps.observation,
                step_type=time_steps.step_type, training=training)

            initial_state = self._target_actor_network.get_initial_state(batch_size)
            target_reward_fn, _ = self._target_reward_network(
                next_time_steps.observation,
                network_state=initial_state,
                step_type=time_steps.step_type, training=False)

            reward = tf.expand_dims(time_steps.reward, axis=-1)

            tf.debugging.check_numerics(target_reward_fn, 'target_reward_fn is inf or nan.')
            td_targets = tf.stop_gradient(
                reward_scale_factor * reward +
                gamma * tf.expand_dims(next_time_steps.discount, axis=-1) * target_reward_fn)
            reward_model_loss = self._td_loss_fn(td_targets, reward_fn)

            if reward_model_loss.shape.rank > 1:
                # Sum over the time dimension.
                reward_model_loss = tf.reduce_sum(
                    reward_model_loss, axis=range(1, reward_model_loss.shape.rank))

            reg_loss = self._reward_network.losses if self._reward_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=reward_model_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)

            reward_model_loss = agg_loss.total_loss

            self._reward_model_loss_debug_summaries(
                reward_fn,
                td_targets)

            return reward_model_loss

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

            next_actions, next_log_pis = self._actions_and_log_probs(time_steps,
                                                                    training=False)
            discriminator_input = (time_steps.observation, next_actions)
            initial_state = self._discriminator_network.get_initial_state(batch_size)
            discriminator_q_logits, unused_network_state = self._discriminator_network(
                discriminator_input,
                #network_state=initial_state,
                step_type=time_steps.step_type, training=training)
            discriminator_probs = tf.expand_dims(tf.nn.softmax(discriminator_q_logits, axis=-1), axis=-1)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(discriminator_probs)

                reward_fn, _ = self._reward_network(
                    time_steps.observation,
                    step_type=time_steps.step_type, training=False)
                reward = tf.expand_dims(time_steps.reward, axis=-1)

                initial_state = self._target_actor_network.get_initial_state(batch_size)
                target_action_distribution, unused_target_network_state = self._target_actor_network(
                    next_time_steps.observation, network_state=initial_state,
                    step_type=next_time_steps.step_type,
                    training=False)
                target_actions = tf.nest.map_structure(lambda d: d.sample(), target_action_distribution)

                target_discriminator_input = (next_time_steps.observation, target_actions)
                initial_state = self._target_discriminator_network.get_initial_state(batch_size)
                target_discriminator_q_logits, _ = self._target_discriminator_network(
                    target_discriminator_input,
                    network_state=initial_state,
                    step_type=next_time_steps.step_type, training=False)
                target_discriminator_probs = tf.expand_dims(tf.nn.softmax(discriminator_q_logits, axis=-1), axis=-1)

                td_targets = tf.stop_gradient(
                        reward_scale_factor * reward_fn +
                        gamma * tf.expand_dims(next_time_steps.discount, axis=-1) * target_discriminator_probs)
                pu_loss, pos_samples, neg_samples = self._calculate_pu(
                    next_log_pis,
                    discriminator_probs,
                    td_targets,
                    reward_fn
                    )
            new_d = tape.gradient([pu_loss], discriminator_probs)
            new_reward = tf.where(new_d >= 0.5, reward_fn, reward)
            # new_reward = -tf.math.log(1 - new_d)

            ### A PU update
            discriminator_loss = new_d

            ### A classic update is below
            # td_targets = tf.stop_gradient(
            #     reward_scale_factor * new_reward +
            #     gamma * tf.expand_dims(next_time_steps.discount, axis=-1) * (
            #         target_discriminator_probs - 0.01 * tf.expand_dims(next_log_pis, axis=-1)))

            # discriminator_loss = errors_loss_fn(
            #     td_targets,
            #     new_d,
            # )

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

            batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)

                discriminator_input = (time_steps.observation, actions)

                discriminator_q_logits, unused_network_state = self._discriminator_network(
                    discriminator_input,
                    step_type=time_steps.step_type, training=False)
                discriminator_probs = tf.expand_dims(tf.nn.softmax(discriminator_q_logits, axis=-1), axis=-1)

                reward_fn, _ = self._reward_network(
                    time_steps.observation,
                    step_type=time_steps.step_type, training=False)

                reward = tf.expand_dims(time_steps.reward, axis=-1)

                initial_state = self._target_actor_network.get_initial_state(batch_size)
                target_action_distribution, unused_target_network_state = self._target_actor_network(
                    time_steps.observation, network_state=initial_state,
                    step_type=time_steps.step_type,
                    training=False)
                target_actions = tf.nest.map_structure(lambda d: d.sample(), target_action_distribution)

                target_discriminator_input = (time_steps.observation, target_actions)
                initial_state = self._target_discriminator_network.get_initial_state(batch_size)
                target_discriminator_q_logits, _ = self._target_discriminator_network(
                    target_discriminator_input,
                    network_state=initial_state,
                    step_type=time_steps.step_type, training=False)
                target_discriminator_probs = tf.expand_dims(tf.nn.softmax(discriminator_q_logits, axis=-1), axis=-1)

                pu_loss, pos_samples, neg_samples = self._calculate_pu(
                    next_log_pis,
                    discriminator_probs,
                    target_discriminator_probs,
                    reward_fn
                    )
                new_d = pu_loss#tape.gradient([pu_loss], discriminator_probs)
                # new_reward = tf.where(new_d >= 0.5, reward_fn, reward)
                new_reward = (0.5 - pu_loss) * reward
                # new_reward = tf.math.log(1 - new_d)
                actions = tf.nest.flatten(actions)

                td_targets = reward_scale_factor * new_reward + gamma * tf.expand_dims(
                    time_steps.discount, axis=-1) * discriminator_probs
                td_targets = tf.nest.flatten(td_targets)

            d4pgs = discriminator_probs
            #tape.gradient([discriminator_probs], actions)

            # actor_losses = []
            # for d4pg, action in zip(d4pgs, actions):
            #     if self._d4pg_clipping is not None:
            #         d4pg = tf.clip_by_value(d4pg, -1 * self._d4pg_clipping,
            #             self._d4pg_clipping)
            #     loss = common.element_wise_squared_loss(
            #         tf.stop_gradient(d4pg + action), action)
            #     if nest_utils.is_batched_nested_tensors(
            #         time_steps, self.time_step_spec, num_outer_dims=2):
            #         loss = tf.reduce_sum(loss, axis=1)
            #     if weights is not None:
            #         loss *= weights
            #     loss = tf.reduce_mean(loss)
            #     actor_losses.append(loss)

            # actor_loss = tf.add_n(actor_losses)

            actor_loss = d4pgs[0]

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
                                        discriminator_q_logits, time_steps)

            return actor_loss

    def _reward_model_loss_debug_summaries(self, predicted_reward, real_reward):
        if self._debug_summaries:
            reward_errors = predicted_reward - real_reward
            reward_error_mean = tf.reduce_mean(predicted_reward) - tf.reduce_mean(real_reward)
            reward_error_stddev = tfp.stats.stddev(predicted_reward) - tfp.stats.stddev(real_reward)

            common.generate_tensor_summaries('reward_error_mean', reward_error_mean,
                                             self.train_step_counter)
            common.generate_tensor_summaries('reward_error_stddev',
                                             reward_error_stddev,
                                             self.train_step_counter)

            tf.compat.v2.summary.scalar(
                name='reward_errors',
                data=-tf.reduce_mean(input_tensor=reward_errors),
                step=self.train_step_counter)

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

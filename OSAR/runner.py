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

import os
import base64
import imageio
import IPython
import matplotlib
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import gin
import reverb
import tempfile
import tqdm

from typing import Any, Callable, Dict, Optional, Sequence, Text, List

import tensorflow as tf

from tf_agents.environments import suite_pybullet
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents import agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train import triggers
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import  boltzmann_policy
from tf_agents.policies import q_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.metrics import py_metrics
from tf_agents.trajectories import trajectory
from tf_agents import trajectories
from tf_agents.utils import common
from tf_agents.typing import types
from tf_agents.specs import tensor_spec
import pandas as pd
from tf_agents.drivers import py_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts

from . import OSARNetwork

__all__ = ['Experiment', 'Runner']

@gin.configurable
class RewardClipWrapper(PyEnvironmentBaseWrapper):
  """Clips reward to values of -1, 0, 1 and stores both."""

  def __init__(self, env: py_environment.PyEnvironment):
    """Creates a reward clip wrapper.
    Args:
      env: Environment to wrap.
    """
    super(RewardClipWrapper, self).__init__(env)

    self.reward = []
    self.clipped_reward = []

  def _step(self, action):

    time_step = self._env.step(action)
    clipped_reward = np.sign(time_step.reward)

    self.reward.append(time_step.reward)
    self.clipped_reward.append(clipped_reward)

    clipped_reward = np.asarray(clipped_reward,
                              dtype=np.asarray(time_step.reward).dtype)
    return ts.TimeStep(time_step.step_type, clipped_reward, time_step.discount,
                       time_step.observation)


@gin.configurable
class Experiment:
    def __init__(
            self,
            # Agent Options
            cache_dir: Text = '',
            agent_specs: Dict=None,
            agent_generator: Callable[[Dict], agents.TFAgent] = None,
            # Environment Options
            env_name: Text = 'Alien-ram-v0',
            num_iterations: int = 20000,
            initial_collect_steps: int = 100,
            collect_steps_per_iteration: int = 1,
            # Dataset options
            replay_buffer_max_length: int = 10000,
            num_eval_episodes: int = 10,
            eval_interval: int = 1000,
            n_prefetch: int = 50,
            n_step_update: int = 2,
            policy_save_interval: int = 100,
            saved_model_dir = ''
            ):

        # Common parameters
        self._num_iterations = num_iterations
        self._cache_dir = cache_dir
        self._initial_collect_steps = initial_collect_steps
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._replay_buffer_max_length = replay_buffer_max_length
        self._num_eval_episodes = num_eval_episodes
        self._eval_interval = eval_interval
        self._n_prefetch = n_prefetch
        self._n_step_update = n_step_update
        self._env_name = env_name

        # Setting up training and evaluation environments
        train_py_env = suite_pybullet.load(
            env_name)  # suite_gym.load(env_name)
        eval_py_env = suite_pybullet.load(
            env_name)  # suite_gym.load(env_name)
        
        # TODO: Doesn't work well with clipped reward on q-learning
        self._train_env = RewardClipWrapper(train_py_env)
        self._eval_env = RewardClipWrapper(eval_py_env)

        # TODO: Implement gym support when Actor supports `TFPyEnvironment`s
        # self._train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        # self._eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        
        # Setting up the agent
        # self._train_env.observation_spec()
        agent_specs['n_step_update'] = n_step_update
        agent_specs['observation_spec'] = tensor_spec.from_spec(self._train_env.observation_spec())
        # self._train_env.action_spec()
        agent_specs['action_spec'] =  tensor_spec.from_spec(self._train_env.action_spec())
        # self._train_env.time_step_spec()
        agent_specs['time_step_spec'] = tensor_spec.from_spec(self._train_env.time_step_spec())

        # Registering policies
        train_step_counter = train_utils.create_train_step()
        agent_specs['train_step_counter'] = train_step_counter
        self._agent = agent_generator(**agent_specs)
        self.episode_step_counter = agent_specs['train_step_counter']

        tf_eval_policy = self._agent.policy
        self._eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_eval_policy, use_tf_function=True)

        tf_collect_policy = self._agent.collect_policy
        self._collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_collect_policy, use_tf_function=True)

        search_q_network = q_network.QNetwork(
            agent_specs['observation_spec'],
            action_spec=agent_specs['action_spec'])
        search_policy = q_policy.QPolicy(
            agent_specs['time_step_spec'],
            agent_specs['action_spec'],
            q_network=search_q_network)
        self._search_policy =  boltzmann_policy.BoltzmannPolicy(
            search_policy, temperature=0.4)
        self._search_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_collect_policy, use_tf_function=True)

        sequence_length = 2

        # # Setting up the replay buffer
        table_name = 'prioritized_table'
        table = reverb.Table(
            table_name,
            max_size=self._replay_buffer_max_length,
            # sampler=reverb.selectors.Uniform(),
            sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1))

        reverb_server = reverb.Server([table])
        reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
            self._agent.collect_data_spec,
            sequence_length=sequence_length,
            table_name=table_name,
            local_server=reverb_server)
        dataset = reverb_replay.as_dataset(
            sample_batch_size=1,
            # sample_batch_size=self._train_env.batch_size,
            num_steps=sequence_length,
            ).prefetch(self._n_prefetch)
        self._reverb_server = reverb_server

        # Setting up the replay buffer

        def experience_dataset_fn(): return dataset

        # Initial random observation
        tempdir = tempfile.gettempdir()
        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            reverb_replay.py_client,
            table_name,
            sequence_length=sequence_length,
            stride_length=1)
        self._rb_observer = rb_observer
            
        # uniform_observer = UniformTrajectoryObserver(replay_buffer)

        initial_collect_actor = actor.Actor(
            self._train_env,
            self._search_policy,
            train_step_counter,
            steps_per_run=initial_collect_steps,
            # observers=[uniform_observer],
            observers=[rb_observer]
        )
        initial_collect_actor.run()

        learner.TRAIN_DIR = cache_dir
        env_step_metric = py_metrics.EnvironmentSteps()
        self._collect_actor = actor.Actor(
            self._train_env,
            self._collect_policy,
            train_step_counter,
            steps_per_run=1,
            metrics=actor.collect_metrics(self._num_iterations),
            summary_dir=cache_dir,
            observers=[
                # uniform_observer,
                rb_observer,
                env_step_metric])
        
        self._eval_actor = actor.Actor(
            self._eval_env,
            self._eval_policy,
            train_step_counter,
            episodes_per_run=num_eval_episodes,
            metrics=actor.eval_metrics(self._num_eval_episodes),
            summary_dir=os.path.join(cache_dir, 'eval'),
        )

        # Triggers to save the agent's policy checkpoints.
        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                cache_dir,
                self._agent,
                train_step_counter,
                interval=policy_save_interval),
            triggers.StepPerSecondLogTrigger(
                train_step_counter, interval=self._eval_interval),    # For evaluation
            triggers.StepPerSecondLogTrigger(
                train_step_counter, interval=1),    # For training
        ]
        
        self._agent_learner = learner.Learner(
            cache_dir,
            train_step_counter,
            self._agent,
            experience_dataset_fn,
            checkpoint_interval=self._eval_interval,
            summary_interval=self._eval_interval,
            run_optimizer_variable_init=False,
            triggers=learning_triggers)

    def get_eval_metrics(self):
        self._eval_actor.run()
        results = {}
        for metric in self._eval_actor.metrics:
            results[metric.name] = metric.result()
        return results
    
    def __call__(self, progress: bool = True):
        
        # Forcing to create a file writer.
        logdir = os.path.join(self._cache_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'scalars', 'metrics'))
        
        with file_writer.set_as_default():
        # with self._agent_learner.train_summary_writer.as_default():
            returns = self.call(progress)
            file_writer.flush()
            # self._agent_learner.train_summary_writer.flush()
        
        return returns

    def call(self, progress: bool=True):
        # Reset the train step
        self._agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.get_eval_metrics()["AverageReturn"]
        returns = [avg_return]

        # Forcing to create a file writer.
        logdir = os.path.join(self._cache_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'scalars', 'metrics'))

        if progress:
            with tqdm.trange(self._num_iterations) as t:
                for _ in t:
                        
                    # Training.
                    self._collect_actor.run()
                    loss_info = self._agent_learner.run(iterations=1)

                    # Evaluating.
                    step = self._agent_learner.train_step_numpy

                    step = self._agent.train_step_counter.numpy()
                    t.set_description(f'Episode {step}')
                    t.set_postfix(
                        train_loss=loss_info.loss.numpy(), avg_return=avg_return)
                        
                    if step % self._eval_interval == 0:
                        metrics = self.get_eval_metrics()
                        avg_return = metrics["AverageReturn"]
                        t.set_postfix(
                            train_loss=loss_info.loss.numpy(), avg_return=avg_return)
                        returns.append(metrics["AverageReturn"])
                    tf.summary.scalar('reward/AverageReturn',
                                      metrics["AverageReturn"],
                                      step)
        else:
            for _ in range(self._num_iterations):
                # Training.
                self._collect_actor.run()
                loss_info = self._agent_learner.run(iterations=1)

                # Evaluating.
                step = self._agent_learner.train_step_numpy

                step = self._agent.train_step_counter.numpy()
                        
                if step % self._eval_interval == 0:
                    metrics = self.get_eval_metrics()
                    avg_return = metrics["AverageReturn"]
                    returns.append(metrics["AverageReturn"])
                tf.summary.scalar('reward/AverageReturn',
                                  metrics["AverageReturn"],
                                  step)

        self._rb_observer.close()
        self._reverb_server.stop()
        
        return np.array(returns)
    
    def save(self):
        # TODO: Implement saving the agent
        
        dataset_folder = os.path.join(learner.TRAIN_DIR, 'datasets')

        dataset = pd.DataFrame(np.concatenate([np.array([self._train_env.reward]).T,
                                              np.array([self._train_env.clipped_reward]).T], axis=-1),
                     columns=[
            'Uclipped Return Train', 'Clipped Return Train'],
            )
        
        dataset.to_csv(os.path.join(dataset_folder, f'{self._env_name}-steps.csv'))

    
    def evaluate(self, filename:str, max_episodes: types.Int=5, fps: types.Int=30):
        env = self._eval_env
        tf_eval_policy = self._agent.policy
        eager_eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_eval_policy, use_tf_function=True)

        if isinstance(env, py_environment.PyEnvironment):
            self._driver = py_driver.PyDriver(
                env,
                eager_eval_policy,
                [],
                max_steps=1,
                max_episodes=1)
        elif isinstance(env, tf_environment.TFEnvironment):
            raise ValueError("Doesn't support TFEnvironments yet.")
        else:
            raise ValueError("Unknown environment type.")

        time_step = env.reset()
        end_term = time_step.is_last()

        policy_state = self._agent.policy.get_initial_state(
            env.batch_size or 1)
        
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(max_episodes):
                video.append_data(tf.cast(env.render(), tf.uint8).numpy())
                while not end_term:
                    time_step, policy_state = self._driver.run(
                        time_step, policy_state)
                    end_term = time_step.is_last()
                    video.append_data(
                        tf.cast(env.render(), tf.uint8).numpy())


@gin.configurable
class Runner:
    def __init__(
        self,
        model_name: Text,
        logpath: Text,
        list_configs:List,
        ):

        self._model_name = model_name
        
        # Configuring logging and progress-saving:
        self._logpath = os.path.abspath(os.path.normpath(logpath))
        if not os.path.exists(self._logpath):
            raise ValueError(f'Log path `{self._logpath}` does not exist.')
        

        # Configuring prerequisites for execution:
        self._list_configs = list_configs
    
    def run(self, progress: bool = False, experiment_progress: bool = True):
        n_games = len(self._list_configs)
        
        if not os.path.exists(os.path.join(self._logpath, 'logs', self._model_name)):
            os.makedirs(os.path.join(self._logpath, 'logs', self._model_name))
        if not os.path.exists(os.path.join(self._logpath, 'logs',
                                self._model_name, 'datasets')):
            os.makedirs(os.path.join(self._logpath, 'logs',
                                self._model_name, 'datasets'))
        if not os.path.exists(os.path.join(self._logpath, 'logs',
                                    self._model_name, 'videos')):
            os.makedirs(os.path.join(self._logpath, 'logs',
                                    self._model_name, 'videos'))
        if not os.path.exists(os.path.join(self._logpath, 'logs',
                                           self._model_name, 'cache')):
            os.makedirs(os.path.join(self._logpath, 'logs',
                                     self._model_name, 'cache'))

        self._experiments, self._env_names, self._ds_returns, self._ds_trajectories = [], [], [], []
        self._ds_games = pd.DataFrame(
            [], columns=['Max. Return', 'Mean Return'])

        if progress:
            with tqdm.trange(n_games) as t:
                for i in t:
                    item = self._list_configs[i]
                    self._env_names.append(item.get('env_name'))
                    cache_dir = os.path.join(
                            self._logpath, 'logs', self._model_name)
                    item['cache_dir'] = cache_dir
                    experiment = Experiment(
                        **item
                    )
                    returns = experiment(
                        progress=experiment_progress)
                    t.set_description(f'Game {i}/{n_games}')
                    if len(returns) > 0:
                        t.set_postfix(
                            max_return=np.max(returns), avg_return=sum(returns) / len(returns))
                    
                    returns = returns.reshape(
                        (1, len(returns)))
                    # Creating a gif for a game
                        
                    filename = os.path.join(
                        self._logpath, 'logs', self._model_name, 'videos', f"{item.get('env_name')}-trained.gif")
                    experiment.evaluate(filename)

                    experiment.save()

                    del experiment
                    # Saving results of the game
                    if len(returns) > 0:
                        self._ds_returns.append(pd.DataFrame(
                            np.array(returns).T, columns=['average_return']))
                        add_df = pd.DataFrame(np.array([np.max(returns, axis=-1),
                                                        np.mean(returns, axis=-1)]).T,
                                                  columns=[
                                'Max. Return', 'Mean Return'],
                                index=[self._env_names[i]])
                        self._ds_games = pd.concat([self._ds_games,
                                                    add_df,
                                                    ],
                                                    axis=0,
                                                    sort=False
                                                    )
                        
        else:
            for i in range(n_games):
                item = self._list_configs[i]
                self._env_names.append(item.get('env_name'))
                cache_dir = os.path.join(
                            self._logpath, 'logs', self._model_name)
                item['cache_dir'] = cache_dir
                experiment = Experiment(
                    **item
                    )
                returns = experiment(
                    progress=experiment_progress)

                returns = returns.reshape(
                    (1, len(returns)))

                # Creating a gif for a game
                filename = os.path.join(
                    self._logpath, 'logs', self._model_name, 'videos', f"{item.get('env_name')}-trained.gif")
                experiment.evaluate(filename)

                experiment.save()

                del experiment
                # Saving results of the game
                if len(returns) > 0:
                    self._ds_returns.append(pd.DataFrame(
                        np.array(returns).T, columns=['average_return']))
                    add_df = pd.DataFrame(np.array([np.max(returns, axis=-1),
                                                    np.mean(returns, axis=-1)]).T,
                                            columns=[
                                'Max. Return', 'Mean Return'],
                                index=[self._env_names[i]])
                    self._ds_games = pd.concat([self._ds_games,
                                                add_df,
                                                ],
                                                axis=0,
                                                sort=False
                    )  

        # Saving results
        open(os.path.join(
            self._logpath, 'logs', self._model_name, 'datasets', 'Games.csv'), 'wt').close()
        self._ds_games.to_csv(os.path.join(
            self._logpath, 'logs', self._model_name, 'datasets', 'Games.csv'))
        
        for i in range(n_games):
            ds_avg_path = os.path.join(
                self._logpath, 'logs', self._model_name, 'datasets', f'{self._env_names[i]}-avgs.csv')
            open(ds_avg_path, 'wt+').close()
            self._ds_returns[i].to_csv(ds_avg_path)
        
        tf.print('Execution Finished.')

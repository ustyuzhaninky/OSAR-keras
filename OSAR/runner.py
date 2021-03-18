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
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import gin
import tqdm

from typing import Any, Callable, Dict, Optional, Sequence, Text, List

import tensorflow as tf

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
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.typing import types
import pandas as pd

from . import OSARNetwork

__all__ = ['Experiment', 'Runner']


@gin.configurable
class Experiment:
    def __init__(
            self,
            # Agent Options
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
            num_parallel_dataset_calls: int = 3,
            n_prefetch: int = 3,
            n_step_update: int = 2
            ):

        # Common parameters
        self._num_iterations = num_iterations
        self._initial_collect_steps = initial_collect_steps
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._replay_buffer_max_length = replay_buffer_max_length
        self._num_eval_episodes = num_eval_episodes
        self._eval_interval = eval_interval
        self._num_parallel_dataset_calls = num_parallel_dataset_calls
        self._n_prefetch = n_prefetch
        self._n_step_update = n_step_update

        # Setting up training and evaluation environments
        env = suite_gym.load(env_name)
        env.reset()
        train_py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)
        self._train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self._eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        
        # Setting up the agent
        agent_specs['observation_spec'] = self._train_env.observation_spec()
        agent_specs['action_spec'] = self._train_env.action_spec()
        agent_specs['time_step_spec'] = self._train_env.time_step_spec()

        self._agent = agent_generator(**agent_specs)
        self._eval_policy = tf.function(self._agent.policy.action)
        self._collect_policy = tf.function(self._agent.collect_policy.action)
        self._train = tf.function(self._agent.train)

        # Setting up the replay buffer
        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._train_env.batch_size,
            max_length=self._replay_buffer_max_length)
        self._dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=self._num_parallel_dataset_calls, sample_batch_size=agent_specs.get('batch_size', 1),
            num_steps=self._n_step_update + 1).prefetch(self._n_prefetch)

    def _compute_avg_return(self,
                            environment: tf_py_environment.TFPyEnvironment,
                            policy,#: py_tf_policy,
                            num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                # environment.render()
                action_step = policy(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return[0]

    def _collect_step(
                self,
                environment: tf_py_environment.TFPyEnvironment,
                policy,#: py_tf_policy
                ):
        time_step = environment.current_time_step()
        action_step = policy(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self._replay_buffer.add_batch(traj)
    
    def _collect_data(self, collect_steps):
        for _ in range(collect_steps):
            # self._train_env.render()
            time_step = self._train_env.current_time_step()
            action_step = self._collect_policy(time_step)
            next_time_step = self._train_env.step(action_step.action)
            traj = trajectory.from_transition(
                time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            self._replay_buffer.add_batch(traj)

    def __call__(self, cache_dir, progress: bool = True):
        writer = tf.summary.create_file_writer(cache_dir)    
        with writer.as_default():
            returns, trajectory, losses = self.call(progress)
            writer.flush()
        return returns, trajectory, losses

    def call(self, progress: bool=True):
        # Reset the train step
        self._agent.train_step_counter.assign(0)
        
        # Evaluate the agent's policy once before training.
        avg_return = self._compute_avg_return(
            self._eval_env, self._eval_policy, self._num_eval_episodes)
        returns, trajectory, losses = [], [], []
        if progress:
            with tqdm.trange(self._num_iterations) as t:
                for i in t:
                    self._collect_data(self._initial_collect_steps)
                    # experience, _ = next(iter(self._replay_buffer.as_dataset(
                    #     num_parallel_calls=self._num_parallel_dataset_calls, sample_batch_size=1,
                    #     num_steps=self._n_step_update + 1).prefetch(self._n_prefetch)))
                    experience, _ = next(iter(self._dataset))
                    train_loss = self._train(experience).loss
                    
                    step = self._agent.train_step_counter.numpy()
                    t.set_description(f'Episode {i}')
                    t.set_postfix(
                        train_loss=train_loss.numpy(), avg_return=avg_return.numpy())
                    trajectory.append(experience.reward[0],)
                    losses.append(train_loss.numpy())
                    
                    if step % self._eval_interval == 0:
                        avg_return = self._compute_avg_return(
                            self._eval_env, self._eval_policy, self._num_eval_episodes)
                        returns.append(avg_return.numpy())
        else:
            for i in range(self._num_iterations):
                self._collect_data(self._initial_collect_steps)
                experience, _ = next(iter(self._replay_buffer.as_dataset(
                    num_parallel_calls=self._num_parallel_dataset_calls, sample_batch_size=1,
                    num_steps=self._n_step_update + 1).prefetch(self._n_prefetch)))
                train_loss = self._train(experience).loss
                   
                step = self._agent.train_step_counter.numpy()
                trajectory.append(experience.reward,)
                losses.append(train_loss.numpy())

                if step % self._eval_interval == 0:
                    avg_return = self._compute_avg_return(
                        self._eval_env, self._eval_policy, self._num_eval_episodes)
                    returns.append(avg_return.numpy())

        return np.array(returns), np.array(trajectory), np.array(losses)
    
    def save(self, cache_dir):
        # TODO: Implement saving the agent
        return
    
    def evaluate(self, filename:str, max_episodes: types.Int=5, fps: types.Int=30):
        
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(max_episodes):
                time_step = self._eval_env.reset()
                video.append_data(tf.cast(self._eval_env.render(), tf.uint8).numpy()[0])
                while not time_step.is_last():
                    action_step = self._eval_policy(time_step)
                    time_step = self._eval_env.step(action_step.action)
                    video.append_data(
                        tf.cast(self._eval_env.render(), tf.uint8).numpy()[0])


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
                    experiment = Experiment(
                        **item
                    )
                    cache_dir = os.path.join(
                            self._logpath, 'logs', self._model_name)
                    returns, trajectory, losses = experiment(
                        cache_dir,
                        progress=experiment_progress)
                    t.set_description(f'Game {i}/{n_games}')
                    if len(returns) > 0:
                        t.set_postfix(
                            max_return=np.max(returns), avg_return=sum(returns) / len(returns))
                    if (len(trajectory) != 0) & (len(losses) != 0):
                        returns = returns.reshape(
                            (1, len(returns)))
                        trajectory = np.mean(trajectory, axis=-1).reshape(
                            (1, np.array(trajectory).shape[0]))
                        losses = losses.reshape(
                            (1, np.array(losses).shape[0]))

                        # Creating a gif for a game
                        
                        filename = os.path.join(
                            self._logpath, 'logs', self._model_name, 'videos', f"{item.get('env_name')}-trained.gif")
                        experiment.evaluate(filename)

                        experiment.save(os.path.join(self._logpath, 'logs',
                                                    self._model_name, 'cache'))
                                                    
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
                        self._ds_trajectories.append(pd.DataFrame(
                            np.concatenate([trajectory, losses], axis=0).T, columns=['reward', 'loss']))

                        
        else:
            for i in range(n_games):
                item = self._list_configs[i]
                self._env_names.append(item.get('env_name'))
                experiment = Experiment(
                    **item
                )
                cache_dir = os.path.join(
                    self._logpath, 'logs', self._model_name)
                returns, trajectory, losses = experiment(
                    cache_dir,
                    progress=experiment_progress)
                returns = returns.reshape(
                            (1, len(returns)))
                trajectory = np.mean(trajectory, axis=-1).reshape(
                            (1, np.array(trajectory).shape[0]))
                losses = losses.reshape(
                    (1, np.array(losses).shape[0]))

                # Creating a gif for a game
                filename = os.path.join(
                    self._logpath, 'logs', self._model_name, 'videos', f"{item.get('env_name')}-trained.gif")
                experiment.evaluate(filename)

                experiment.save(os.path.join(self._logpath, 'logs',
                                             self._model_name, 'cache'))

                # Saving results of the game
                if (len(trajectory) != 0) & (len(losses) != 0):
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
                    self._ds_trajectories.append(pd.DataFrame(
                        np.concatenate([trajectory, losses], axis=0).T, columns=['reward', 'loss']))
                
                

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
            ds_tr_path = os.path.join(
                self._logpath, 'logs', self._model_name, 'datasets', f'{self._env_names[i]}-traj.csv')
            open(ds_tr_path, 'wt+').close()
            self._ds_trajectories[i].to_csv(ds_tr_path)
        
        tf.print('Execution Finished.')

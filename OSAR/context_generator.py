# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import special_math_ops
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
from tensorflow.keras import backend as K
from .helix_memory import HelixMemory
from .gates import AttentionGate

__all__ = ['ContextGenerator',]

class ContextGenerator(tf.keras.layers.Layer):
    """Context generator with value function restoration.
    
    # Arguments
        units: int >= 0. Dimension of hidden units.
        embed_dim: int > 0. Dimension of the dense embedding.
        hidden_dim: int > 0. Number of units in the Feed-forward part.
        num_token: int > 0. Size of the vocabulary.
        num_head: int > 0. Number of heads in the attention unit.
        memory_len: int > 0 Number of memory units.
        n_turns: int > 0. Number of compressed units.
        compression_rate: int > 0. Rate of memory compression.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - None yet
    """

    def __init__(self,
                 units,
                 batch_size,
                 memory_len,
                 n_turns,
                 compression_rate=2,
                 dropout=0.0,
                 attention_dropout=0.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 return_probabilities=False,
                 **kwargs):
        super(ContextGenerator, self).__init__(**kwargs)

        self.units = units
        self.batch_size = batch_size
        self.memory_len = memory_len
        self.n_turns = n_turns
        self.compression_rate = compression_rate
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.n_conv = sum(pow(self.compression_rate, i)
                          for i in range(1, self.n_turns+1))

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f'`ContextGenerator` requires three inputs, not {len(input_shape)}.')
        batch_dim = input_shape[0][0]

        if len(input_shape[0]) != 3:
            raise ValueError(
                f'Input 1 requires a vector of shape (batch_size, timesteps, state_features), not {input_shape[0]}.')
        if len(input_shape[1]) != 3:
            raise ValueError(
                f'Input 2 requires a vector of shape (batch_size, timesteps, action_features), not {input_shape[1]}.')
        if len(input_shape[2]) != 3:
            raise ValueError(
                f'Input 3 requires a vector of shape (batch_size, timesteps, reward_features), not {input_shape[2]}.')

        if len(input_shape[1]) != 3:
            raise ValueError(
                f'`ContextGenerator` requires three inputs, not {len(input_shape)}.')

        timesteps_dim = input_shape[0][1]
        if input_shape[1][1] != timesteps_dim:
            raise ValueError(f'Input 1 got timesteps shape with size {timesteps_dim}'
                             f' while Input 2 got {input_shape[0][1]}: [{timesteps_dim}]!=[{input_shape[0][1]}]')
        if input_shape[1][1] != timesteps_dim:
            raise ValueError(f'Input 1 got timesteps shape with size {timesteps_dim}'
                             f' while Input 3 got {input_shape[0][2]}: [{timesteps_dim}]!=[{input_shape[0][2]}]')
        features_input_A_dim = input_shape[0][-1]
        features_input_B_dim = input_shape[1][-1]
        features_input_C_dim = input_shape[2][-1]

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.concat.build(
            [
                (batch_dim, timesteps_dim, features_input_A_dim,),
                (batch_dim, timesteps_dim, features_input_B_dim,),
                (batch_dim, timesteps_dim, features_input_C_dim,),
            ])
        self.concat.built = True

        concat_features = features_input_A_dim+features_input_B_dim+features_input_C_dim

        self.memory = HelixMemory(
            self.batch_size,
            self.memory_len,
            self.n_turns,
            self.compression_rate,
            mode='conv',
            regularizer=self.kernel_regularizer,
            name=f'{self.name}-HelixMemory'
        )
        self.memory.build(
            (batch_dim, timesteps_dim, concat_features))
        self.memory.built = True
        
        self.attention = AttentionGate(
            self.units,
            self.dropout,
            self.attention_dropout,
            self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            return_probabilities=True,
            name=f'{self.name}-AttentionGate'
        )

        self.attention.build(
            (batch_dim, self.memory_len+self.n_conv, concat_features))
        self.attention.built = True

        self.kernel = self.add_weight(
            f'{self.name}-kernel',
            shape=[self.memory_len+self.n_conv, self.memory_len+self.n_conv],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            # constraint=tf.keras.constraints.MinMaxNorm(-1.0, 1.0),
            dtype=self.dtype,
            trainable=True)

        self.built = True
    
    @tf.function
    def call(self, inputs):
        if len(inputs) != 3:
            raise ValueError(
                f'`ContextGenerator` requires three inputs, not {len(inputs)}.')
        batch_dim = tf.shape(inputs[0])[0]
        
        tf.debugging.assert_equal(tf.rank(
            inputs[0]), 3,
            message=f'Input 1 requires a vector of shape (batch_size, timesteps, state_features), not {inputs[0].shape}.')
        
        tf.debugging.assert_equal(tf.rank(
            inputs[1]), 3,
            message=f'Input 2 requires a vector of shape (batch_size, timesteps, action_features), not {inputs[1].shape}.')

        tf.debugging.assert_equal(tf.rank(
            inputs[2]), 3,
            message=f'Input 3 requires a vector of shape (batch_size, timesteps, reward_features), not {inputs[2].shape}.')

        timesteps_dim = inputs[0].shape[1]
        
        tf.debugging.assert_equal(tf.shape(
            inputs[1])[1], timesteps_dim,
            message=f'Input 1 got timesteps shape with size {timesteps_dim}'
            f' while Input 2 got {inputs[0].shape[1]}: [{timesteps_dim}]!=[{inputs[0].shape[1]}]')

        tf.debugging.assert_equal(tf.shape(
            inputs[2])[1], timesteps_dim,
            message=f'Input 1 got timesteps shape with size {timesteps_dim}'
            f' while Input 3 got {inputs[0].shape[2]}: [{timesteps_dim}]!=[{inputs[0].shape[2]}]')

        features_input_A_dim = tf.shape(inputs[0])[-1]
        features_input_B_dim = tf.shape(inputs[1])[-1]
        features_input_C_dim = tf.shape(inputs[2])[-1]

        context = self.concat(inputs)
        
        context_mem = self.memory(context)
        
        att_mem, probabilities = self.attention(context_mem)

        # Calculating decomposition gate
        states, rewards, actions = att_mem[..., :features_input_A_dim], att_mem[
            ..., features_input_A_dim:features_input_A_dim+features_input_B_dim], att_mem[..., -features_input_C_dim:]
        
        new_rewards = rewards[..., -1] #if tf.rank(rewards) == 3 else rewards
        non_zero_reward_indexes = tf.where(rewards > 0)

        def loop_fn(i, tg_idx, new_rewards):
            
            target_batch_index = tf.cast(tg_idx[0], tf.int32)
            target_step_index = tf.cast(tg_idx[-1], tf.int32)

            Nz = tf.rank(
                tf.where(new_rewards[target_batch_index, :target_step_index] == 0.0))
            target = states[target_batch_index, target_step_index]

            if target_step_index > 0:

                j = tf.constant(0, dtype=tf.int32)

                def replacerIterator(j, new_rewards):
                    rewards_shape = tf.shape(new_rewards)
                    C = new_rewards[target_batch_index, target_step_index-j]
                    target = new_rewards[target_batch_index,
                                         target_step_index]
                    if i == 0:
                        row_states_ls = states[target_batch_index,
                                               target_step_index-j-1]
                        row_states_rs = states[target_batch_index,
                                               target_step_index]
                    else:
                        row_states_ls = states[target_batch_index,
                                               target_step_index-j-1]
                        row_states_rs = states[target_batch_index,
                                               target_step_index]

                    diffs_ls = target - row_states_ls
                    distance_ls = tf.norm(diffs_ls)
                    diffs_rs = target - row_states_rs
                    distance_rs = tf.norm(diffs_rs)
                    d_state = tf.keras.losses.huber(distance_ls, distance_rs)
                    
                    P = self.kernel[target_step_index-j -
                                    1, target_step_index]
                    new_item = C * P / (Nz * d_state + P)

                    if target_batch_index == 0:
                        new_item = tf.concat(
                            [[new_item], new_rewards[target_batch_index+1:, target_step_index-j-1]], axis=0)
                    elif target_batch_index < tf.shape(rewards)[0] - 1:
                        new_item = tf.concat(
                            [new_rewards[:target_batch_index, target_step_index-j-1],
                             [new_item],
                             new_rewards[target_batch_index+1:, target_step_index-1-j]], axis=0)
                    else:
                        new_item = tf.concat(
                            [new_rewards[:target_batch_index, target_step_index-j-1], [new_item]], axis=0)

                    new_item = tf.expand_dims(new_item, axis=-1)

                    if j == Nz:
                        new_rewards = tf.concat(
                            [new_item, new_rewards[:, target_step_index-j:]], axis=1)
                    else:
                        new_rewards = tf.concat(
                            [new_rewards[:, :target_step_index-j-1], new_item, new_rewards[:, target_step_index-j:]], axis=1)
                    
                    return j+1, new_rewards # tf.reshape(new_rewards, rewards_shape)
                j, new_rewards = tf.while_loop(
                    lambda j, new_rewards: j < Nz,
                    replacerIterator,
                    [j, new_rewards],
                )

                return (i+1, non_zero_reward_indexes[i+1], new_rewards)
            else:
                return (i+1, non_zero_reward_indexes[i+1], new_rewards)
            
        def false_fn(new_rewards):
            i = tf.constant(0, dtype=tf.int32)
            tf_idx = non_zero_reward_indexes[i]
            
            i, tf_idx, new_rewards = tf.while_loop(
                lambda i, tf_idx, new_rewards: tf.less(
                    i, tf.cast(tf.shape(non_zero_reward_indexes)[0], tf.int32)-1),
                loop_fn,
                [i, tf_idx, new_rewards]
            )

            return new_rewards

        new_rewards = tf.einsum('in,nn->in', new_rewards, self.kernel)
        
        if non_zero_reward_indexes.shape[0] is not None and non_zero_reward_indexes.shape[0] != 0:
            new_rewards = false_fn(new_rewards)
            new_rewards = K.reshape(new_rewards, (rewards.shape[0], rewards.shape[1]))
        new_rewards = K.expand_dims(new_rewards, axis=-1)

        redacted_memory = K.concatenate(
            [states, new_rewards, actions],
            axis=-1
        )

        return redacted_memory
    
    def get_config(self):
        config = {
            "units": self.units,
            'memory_len': self.memory_len,
            'n_turns': self.n_turns,
            'batch_size': self.batch_size,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_bias": self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(ContextGenerator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
from tensorflow.python.framework import tensor_shape
from tensorflow import keras
from tensorflow.keras import backend as K

__all__ = ['ContextEmbedding']


class ContextEmbedding(keras.layers.Layer):
    '''Context Embedding. Turns float probability features into sorted dense vectors of positive integers (indexes).

    # Arguments
        mask_zero=False,

    # Input shape
        2D tensor with shape: `(batch_size, input_dim)`.

    # Output shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # References
        - None yet

    '''

    def __init__(self,
                 mask_zero=False,
                 **kwargs):
        super(ContextEmbedding, self).__init__(**kwargs)
        
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
    
    def build(self, input_shape):
        super(ContextEmbedding, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            output_mask = None
        else:
            output_mask = None#K.not_equal(inputs, 0)
        
        return output_mask

    def call(self, inputs, **kwargs):
        k = inputs.shape[-1]
        out, id_out = K.eval(tf.math.top_k(inputs, k=k, sorted=False))

        if self.mask_zero:
            id_out = id_out + 1

        if K.dtype(id_out) != 'int32':
            id_out = K.cast(id_out, 'int32')
        return id_out
    
    def get_config(self):
        config = {
            "mask_zero":
                self.mask_zero,
            "supports_masking":
                self.supports_masking
            }
        base_config = super(ContextEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import os
import tempfile
import tensorflow as tf
from unittest import TestCase
import numpy as np
from tensorflow import keras
from nosferatu.neocortex.OSAR import transformerShift

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

np.random.seed(234)

class TestTransformerShift(TestCase):

    def test_ops_base(self):
        """Test of the baseline functionality"""
        tf.config.experimental_run_functions_eagerly(True)
        
        filters = 1
        features = 1
        method = 'conv'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.constant([[[1.0], [1.0]]], dtype=float)
        kernel = tf.constant(np.random.random(
            (compression_rate, features, filters)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)

        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
        assert ans[0, -1, 0] != np.array(
            [[[0.0]]]), f"Doing nothing: : `{ans[0, -1, 0]}` == `{np.array([0.0])}`"
        assert ans[0, -1, 0] == data[0, 0], f"Doing nothing: `{ans[0, -1, ...]}` != `{data[0, 0],}`"
    
    def test_ops_shifting(self):
        """Test of the shifting"""
        tf.config.experimental_run_functions_eagerly(True)

        filters = 1
        features = 1
        method = 'conv'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.concat([[[[1.0]]], tf.zeros((9, 1, 1), dtype=float)], axis=0)
        kernel = tf.constant(np.random.random(
            (compression_rate, features, filters)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)
        
        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
        # print(ans.shape)
        # print(ans)
        for i in range(1, 10):
            idx = conv_units+short_units - i
            assert ans[i, idx-1, 0] == 1.0, f"Stopped shifting on iteration `{i}`"
            assert ans[i-1, idx, 0] == ans[i, idx-1, 0], f"Stopped shifting on iteration `{i}`"
    
    def test_ops_filters(self):
        """Test for number of filters"""
        tf.config.experimental_run_functions_eagerly(True)

        # filters = 1
        features = 1
        method = 'conv'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.concat([[[[1.0]]], tf.zeros((9, 1, 1), dtype=float)], axis=0)

        for i in range(100, 2):
            kernel = tf.constant(np.random.random(
                    (compression_rate, features, i)), dtype=float)
            bias = tf.zeros((i,), dtype=float)
            ans = transformerShift(data, memory, conv_units, short_units,
                                compression_rate=compression_rate,
                                method=method,
                                use_bias=use_bias,
                                kernel=kernel,
                                activation=activation,
                                bias=bias,)
            assert ans.shape[-1] == 1
    
    def test_ops_features(self):
        """Test for number of features"""

        tf.config.experimental_run_functions_eagerly(True)

        filters = 3
        features = 3
        method = 'conv'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.constant([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=float)
        kernel = tf.constant(np.random.random(
            (compression_rate, features, filters)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)
        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
        assert ans[0, -1, 0] != 0.0, f"Doing nothing: : `{ans[0, -1, 0]}` == `{np.array([0.0, 0.0, 0.0])}`"
        assert ans[0, -1, 0] == data[0, 0, 0], f"Doing nothing: `{ans[0, -1, 0]}` != `{data[0, 0, 0],}`"

    def test_ops_max(self):
        """Test of the baseline functionality for max compression"""
        tf.config.experimental_run_functions_eagerly(True)

        filters = 32
        features = 1
        method = 'max'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.constant([[[1.0], [1.0]]], dtype=float)
        kernel = tf.constant(np.random.random(
            (compression_rate, features, filters)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)

        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
        assert ans[0, -1, 0] != np.array(
            [[[0.0]]]), f"Doing nothing: : `{ans[0, -1, 0]}` == `{np.array([0.0])}`"
        assert ans[0, -1, 0] == data[0,
                                     0], f"Doing nothing: `{ans[0, -1, ...]}` != `{data[0, 0],}`"

        data = tf.concat([[[[1.0]]], tf.zeros((9, 1, 1), dtype=float)], axis=0)

        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)

    def test_ops_mean(self):
        """Test of the baseline functionality for mean compression"""
        tf.config.experimental_run_functions_eagerly(True)

        filters = 32
        features = 1
        method = 'mean'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.constant([[[1.0], [1.0]]], dtype=float)
        kernel = tf.constant(np.random.random(
            (compression_rate, features, filters)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)

        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
        assert ans[0, -1, 0] != np.array(
            [[[0.0]]]), f"Doing nothing: : `{ans[0, -1, 0]}` == `{np.array([0.0])}`"
        assert ans[0, -1, 0] == data[0,
                                     0], f"Doing nothing: `{ans[0, -1, ...]}` != `{data[0, 0],}`"

        data = tf.concat([[[[1.0]]], tf.zeros((9, 1, 1), dtype=float)], axis=0)

        ans = transformerShift(data, memory, conv_units, short_units,
                               compression_rate=compression_rate,
                               method=method,
                               use_bias=use_bias,
                               kernel=kernel,
                               activation=activation,
                               bias=bias,)
    
    def test_ops_compression_rate(self):
        """Test of the baseline functionality mean compression rate"""
        tf.config.experimental_run_functions_eagerly(True)

        filters = 3
        features = 3
        method = 'conv'
        use_bias = True
        compression_rate = 2
        activation = 'relu'
        conv_units = 10
        short_units = 10

        memory = tf.zeros((conv_units+short_units, features), dtype=float)
        data = tf.constant(np.random.random((100, 10, features)), dtype=float)
        bias = tf.zeros((filters,), dtype=float)

        for i in range(100, 2):
            kernel = tf.constant(np.random.random(
                (i, features, filters)), dtype=float)
            bias = tf.zeros((i,), dtype=float)
            ans = transformerShift(data, memory, conv_units, short_units,
                                   compression_rate=i,
                                   method=method,
                                   use_bias=use_bias,
                                   kernel=kernel,
                                   activation=activation,
                                   bias=bias,)
            assert ans.shape[-1] == 1

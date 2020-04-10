# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for octave convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class OctaveConv1DTest(keras_parameterized.TestCase):

    def test_octave_conv1D_padding_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'padding': 'same'
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
            y = layer(keras.backend.variable(np.ones((1, 10, 2))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([1, 10, 2], [1, 5, 1]))

    def test_octave_conv1D_dilation_rate_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'dilation_rate': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
            y = layer(keras.backend.variable(np.ones((1, 20, 4))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([1, 20, 2], [1, 10, 1]))

    def test_octave_conv1D_strides_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'strides': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
            y = layer(keras.backend.variable(np.ones((1, 20, 4))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([1, 10, 2], [1, 5, 1]))

    def test_octave_conv1D_regularizers(self):
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
            layer.build((None, 10, 4))
            self.assertEqual(len(layer.losses), 4)
            layer(keras.backend.variable(np.ones((1, 10, 4))))
            self.assertEqual(len(layer.losses), 8)

    def test_octave_conv1D_constraints(self):
        k_constraint = lambda x: x
        b_constraint = lambda x: x

        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_constraint': k_constraint,
            'bias_constraint': b_constraint,
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
            layer.build((None, 10, 4))
            # list of 2 kernels: one for self.conv_high_to_high and the other
            # for self.conv_high_to_low
            self.assertEqual(len(layer.kernel), 2)
            self.assertEqual(len(layer.bias), 2)
            self.assertEqual(layer.kernel[0].constraint, k_constraint)
            self.assertEqual(layer.bias[0].constraint, b_constraint)

    def test_octave_conv1D_recreate_conv(self):
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv1D(filters=1,
                                 kernel_size=3,
                                 low_freq_ratio=0.5,
                                 strides=1,
                                 dilation_rate=2,
                                 padding='same')
            inpt1 = keras.backend.variable(np.random.normal(size=[1, 4, 2]))
            inpt2 = keras.backend.variable(np.random.normal(size=[1, 6, 2]))
            outp1_shape = layer(inpt1).shape
            _ = layer(inpt2).shape
            self.assertEqual(outp1_shape, layer(inpt1).shape)


@keras_parameterized.run_all_keras_modes
class OctaveConv2DTest(keras_parameterized.TestCase):

    def test_octave_conv2D_padding_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'padding': 'same'
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 2], [2, 14, 14, 1]))

    def test_octave_conv2D_dilation_rate_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'dilation_rate': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 2], [2, 14, 14, 1]))

    def test_octave_conv2D_strides_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'strides': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 14, 14, 2], [2, 7, 7, 1]))

    def test_octave_conv2D_regularizers(self):
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            layer.build((None, 10, 10, 4))
            self.assertEqual(len(layer.losses), 4)
            layer(keras.backend.variable(np.ones((1, 10, 10, 4))))
            self.assertEqual(len(layer.losses), 8)

    def test_octave_conv2D_constraints(self):
        k_constraint = lambda x: x
        b_constraint = lambda x: x

        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_constraint': k_constraint,
            'bias_constraint': b_constraint,
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            layer.build((None, 10, 10, 4))
            # list of 2 kernels: one for self.conv_high_to_high and the other
            # for self.conv_high_to_low
            self.assertEqual(len(layer.kernel), 2)
            self.assertEqual(len(layer.bias), 2)
            self.assertEqual(layer.kernel[0].constraint, k_constraint)
            self.assertEqual(layer.bias[0].constraint, b_constraint)

    def test_octave_conv2D_recreate_conv(self):
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(filters=1,
                                 kernel_size=3,
                                 low_freq_ratio=0.5,
                                 strides=1,
                                 dilation_rate=2,
                                 padding='same')
            inpt1 = keras.backend.variable(
                np.random.normal(size=[1, 10, 10, 4]))
            inpt2 = keras.backend.variable(
                np.random.normal(size=[1, 10, 20, 4]))
            outp1_shape = layer(inpt1).shape
            _ = layer(inpt2).shape
            self.assertEqual(outp1_shape, layer(inpt1).shape)


@keras_parameterized.run_all_keras_modes
class OctaveConv3DTest(keras_parameterized.TestCase):

    def test_octave_conv3D_padding_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'padding': 'same'
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 28, 2], [2, 14, 14, 14, 1]))

    def test_octave_conv3D_dilation_rate_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'dilation_rate': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 28, 2], [2, 14, 14, 14, 1]))

    def test_octave_conv3D_strides_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'strides': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 14, 14, 14, 2], [2, 7, 7, 7, 1]))

    def test_octave_conv3D_regularizers(self):
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
            layer.build((None, 10, 10, 10, 4))
            self.assertEqual(len(layer.losses), 4)
            layer(keras.backend.variable(np.ones((1, 10, 10, 10, 4))))
            self.assertEqual(len(layer.losses), 8)

    def test_octave_conv3D_constraints(self):
        k_constraint = lambda x: x
        b_constraint = lambda x: x

        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'low_freq_ratio': 0.5,
            'kernel_constraint': k_constraint,
            'bias_constraint': b_constraint,
            'strides': 1
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
            layer.build((None, 10, 10, 10, 4))
            # list of 2 kernels: one for self.conv_high_to_high and the other
            # for self.conv_high_to_low
            self.assertEqual(len(layer.kernel), 2)
            self.assertEqual(len(layer.bias), 2)
            self.assertEqual(layer.kernel[0].constraint, k_constraint)
            self.assertEqual(layer.bias[0].constraint, b_constraint)

    def test_octave_conv3D_recreate_conv(self):
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3D(filters=1,
                                 kernel_size=3,
                                 low_freq_ratio=0.5,
                                 strides=1,
                                 dilation_rate=2,
                                 padding='same')
            inpt1 = keras.backend.variable(
                np.random.normal(size=[1, 10, 10, 10, 4]))
            inpt2 = keras.backend.variable(
                np.random.normal(size=[1, 10, 10, 20, 4]))
            outp1_shape = layer(inpt1).shape
            _ = layer(inpt2).shape
            self.assertEqual(outp1_shape, layer(inpt1).shape)


@keras_parameterized.run_all_keras_modes
class OctaveConv2DTransposeTest(keras_parameterized.TestCase):

    def test_octave_conv2D_transpose_padding_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'padding': 'same'
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 2], [2, 14, 14, 1]))

    def test_octave_conv2D_transpose_dilation_rate_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'dilation_rate': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 2], [2, 14, 14, 1]))

    def test_octave_conv2D_transpose_strides_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'strides': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 56, 56, 2], [2, 28, 28, 1]))


@keras_parameterized.run_all_keras_modes
class OctaveConv3DTransposeTest(keras_parameterized.TestCase):

    def test_octave_conv3D_transpose_padding_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'padding': 'same'
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 28, 2], [2, 14, 14, 14, 1]))

    def test_octave_conv3D_transpose_dilation_rate_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'dilation_rate': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 28, 28, 28, 2], [2, 14, 14, 14, 1]))

    def test_octave_conv3D_transpose_strides_output_shape(self):
        # layer_test can't be used because we need to put a list of shapes as
        # expected_output_shape
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
            'strides': 2,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv3DTranspose(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 28, 3))))
            self.assertEqual((y[0].shape.as_list(), y[1].shape.as_list()),
                             ([2, 56, 56, 56, 2], [2, 28, 28, 28, 1]))


@keras_parameterized.run_all_keras_modes
class OctaveConvAddTest(keras_parameterized.TestCase):

    def test_octave_conv_add(self):
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'low_freq_ratio': 0.5,
        }
        with self.cached_session(use_gpu=True):
            layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
            y = layer(keras.backend.variable(np.ones((2, 28, 28, 1))))
            y_add = keras.layers.octave_convolutional.OctaveConvAdd()(
                y, builder=keras.layers.MaxPooling2D(strides=2))
            # check that MaxPooling was applied on both tensors
            self.assertEqual((y_add[0].shape.as_list(),
                              y_add[1].shape.as_list()),
                             ([2, 14, 14, 2], [2, 7, 7, 1]))


if __name__ == '__main__':
    test.main()

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pooling layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import test


class GlobalPoolingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_globalpooling_1d(self):
    testing_utils.layer_test(keras.layers.pooling.GlobalMaxPooling1D,
                             input_shape=(3, 4, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D, input_shape=(3, 4, 5))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_globalpooling_2d(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling2D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 5, 6))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling2D,
        kwargs={'data_format': 'channels_last'},
        input_shape=(3, 5, 6, 4))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling2D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 5, 6))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling2D,
        kwargs={'data_format': 'channels_last'},
        input_shape=(3, 5, 6, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_globalpooling_3d(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling3D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 3, 4, 3))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling3D,
        kwargs={'data_format': 'channels_last'},
        input_shape=(3, 4, 3, 4, 3))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling3D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 3, 4, 3))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling3D,
        kwargs={'data_format': 'channels_last'},
        input_shape=(3, 4, 3, 4, 3))


class Pooling2DTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_maxpooling_2d(self):
    pool_size = (3, 3)
    for strides in [(1, 1), (2, 2)]:
      testing_utils.layer_test(
          keras.layers.MaxPooling2D,
          kwargs={
              'strides': strides,
              'padding': 'valid',
              'pool_size': pool_size
          },
          input_shape=(3, 5, 6, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_averagepooling_2d(self):
    testing_utils.layer_test(
        keras.layers.AveragePooling2D,
        kwargs={'strides': (2, 2),
                'padding': 'same',
                'pool_size': (2, 2)},
        input_shape=(3, 5, 6, 4))
    testing_utils.layer_test(
        keras.layers.AveragePooling2D,
        kwargs={'strides': (2, 2),
                'padding': 'valid',
                'pool_size': (3, 3)},
        input_shape=(3, 5, 6, 4))

    # This part of the test can only run on GPU but doesn't appear
    # to be properly assigned to a GPU when running in eager mode.
    if not context.executing_eagerly():
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      if test.is_gpu_available(cuda_only=True):
        testing_utils.layer_test(
            keras.layers.AveragePooling2D,
            kwargs={
                'strides': (1, 1),
                'padding': 'valid',
                'pool_size': (2, 2),
                'data_format': 'channels_first'
            },
            input_shape=(3, 4, 5, 6))


class Pooling3DTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_maxpooling_3d(self):
    pool_size = (3, 3, 3)
    testing_utils.layer_test(
        keras.layers.MaxPooling3D,
        kwargs={'strides': 2,
                'padding': 'valid',
                'pool_size': pool_size},
        input_shape=(3, 11, 12, 10, 4))
    testing_utils.layer_test(
        keras.layers.MaxPooling3D,
        kwargs={
            'strides': 3,
            'padding': 'valid',
            'data_format': 'channels_first',
            'pool_size': pool_size
        },
        input_shape=(3, 4, 11, 12, 10))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_averagepooling_3d(self):
    pool_size = (3, 3, 3)
    testing_utils.layer_test(
        keras.layers.AveragePooling3D,
        kwargs={'strides': 2,
                'padding': 'valid',
                'pool_size': pool_size},
        input_shape=(3, 11, 12, 10, 4))
    testing_utils.layer_test(
        keras.layers.AveragePooling3D,
        kwargs={
            'strides': 3,
            'padding': 'valid',
            'data_format': 'channels_first',
            'pool_size': pool_size
        },
        input_shape=(3, 4, 11, 12, 10))


class Pooling1DTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_maxpooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.MaxPooling1D,
            kwargs={'strides': stride,
                    'padding': padding},
            input_shape=(3, 5, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_averagepooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.AveragePooling1D,
            kwargs={'strides': stride,
                    'padding': padding},
            input_shape=(3, 5, 4))


if __name__ == '__main__':
  test.main()

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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class GlobalPoolingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_globalpooling_1d(self):
    testing_utils.layer_test(keras.layers.pooling.GlobalMaxPooling1D,
                             input_shape=(3, 4, 5))
    testing_utils.layer_test(keras.layers.pooling.GlobalMaxPooling1D,
                             kwargs={'data_format': 'channels_first'},
                             input_shape=(3, 4, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D, input_shape=(3, 4, 5))
    testing_utils.layer_test(keras.layers.pooling.GlobalAveragePooling1D,
                             kwargs={'data_format': 'channels_first'},
                             input_shape=(3, 4, 5))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_globalpooling_1d_masking_support(self):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(None, 4)))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.compile(loss='mae', optimizer='rmsprop')

    model_input = np.random.random((2, 3, 4))
    model_input[0, 1:, :] = 0
    output = model.predict(model_input)
    self.assertAllClose(output[0], model_input[0, 0, :])

  @tf_test_util.run_in_graph_and_eager_modes
  def test_globalpooling_1d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant([
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0]]], ragged_rank=1)
    dense_data = ragged_data.to_tensor()

    inputs = keras.Input(shape=(None, 2), dtype='float32', ragged=True)
    out = keras.layers.GlobalAveragePooling1D()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_ragged = model.predict(ragged_data, steps=1)

    inputs = keras.Input(shape=(None, 2), dtype='float32')
    masking = keras.layers.Masking(mask_value=0., input_shape=(3, 2))(inputs)
    out = keras.layers.GlobalAveragePooling1D()(masking)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_dense = model.predict(dense_data, steps=1)

    self.assertAllEqual(output_ragged, output_dense)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_globalpooling_2d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant([
        [[[1.0], [1.0]], [[2.0], [2.0]], [[3.0], [3.0]]],
        [[[1.0], [1.0]], [[2.0], [2.0]]]], ragged_rank=1)
    dense_data = ragged_data.to_tensor()

    inputs = keras.Input(shape=(None, 2, 1), dtype='float32', ragged=True)
    out = keras.layers.GlobalMaxPooling2D()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_ragged = model.predict(ragged_data, steps=1)

    inputs = keras.Input(shape=(None, 2, 1), dtype='float32')
    out = keras.layers.GlobalMaxPooling2D()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_dense = model.predict(dense_data, steps=1)

    self.assertAllEqual(output_ragged, output_dense)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_globalpooling_3d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant([
        [[[[1.0]], [[1.0]]], [[[2.0]], [[2.0]]], [[[3.0]], [[3.0]]]],
        [[[[1.0]], [[1.0]]], [[[2.0]], [[2.0]]]]], ragged_rank=1)

    inputs = keras.Input(shape=(None, 2, 1, 1), dtype='float32', ragged=True)
    out = keras.layers.GlobalAveragePooling3D()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_ragged = model.predict(ragged_data, steps=1)
    # Because GlobalAveragePooling3D doesn't support masking, the results
    # cannot be compared with its dense equivalent.
    expected_output = constant_op.constant([[2.0], [1.5]])
    self.assertAllEqual(output_ragged, expected_output)

  @tf_test_util.run_in_graph_and_eager_modes
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

  @tf_test_util.run_in_graph_and_eager_modes
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

  @tf_test_util.run_in_graph_and_eager_modes
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

  @tf_test_util.run_in_graph_and_eager_modes
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_maxpooling_3d(self):
    if test.is_built_with_rocm():
      self.skipTest('Pooling with 3D tensors is not supported in ROCm')
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_averagepooling_3d(self):
    if test.is_built_with_rocm():
      self.skipTest('Pooling with 3D tensors is not supported in ROCm')
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_maxpooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.MaxPooling1D,
            kwargs={'strides': stride,
                    'padding': padding},
            input_shape=(3, 5, 4))
    testing_utils.layer_test(
        keras.layers.MaxPooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 2, 6))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_averagepooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.AveragePooling1D,
            kwargs={'strides': stride,
                    'padding': padding},
            input_shape=(3, 5, 4))

    testing_utils.layer_test(
        keras.layers.AveragePooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 2, 6))


if __name__ == '__main__':
  test.main()

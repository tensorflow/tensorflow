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

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import combinations
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class GlobalPoolingTest(test.TestCase, parameterized.TestCase):

  @testing_utils.enable_v2_dtype_behavior
  def test_mixed_float16_policy(self):
    with policy.policy_scope('mixed_float16'):
      inputs1 = keras.Input(shape=(36, 512), dtype='float16')
      inputs2 = keras.Input(shape=(36,), dtype='bool')
      average_layer = keras.layers.pooling.GlobalAveragePooling1D()
      _ = average_layer(inputs1, inputs2)

  def test_globalpooling_1d(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling1D, input_shape=(3, 4, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D, input_shape=(3, 4, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 4, 5))

  def test_globalpooling_1d_masking_support(self):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(None, 4)))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.compile(loss='mae', optimizer='rmsprop')

    model_input = np.random.random((2, 3, 4))
    model_input[0, 1:, :] = 0
    output = model.predict(model_input)
    self.assertAllClose(output[0], model_input[0, 0, :])

  def test_globalpooling_1d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[1.0, 1.0], [2.0, 2.0]]],
        ragged_rank=1)
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

  def test_globalpooling_2d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant(
        [[[[1.0], [1.0]], [[2.0], [2.0]], [[3.0], [3.0]]],
         [[[1.0], [1.0]], [[2.0], [2.0]]]],
        ragged_rank=1)
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

  def test_globalpooling_3d_with_ragged(self):
    ragged_data = ragged_factory_ops.constant(
        [[[[[1.0]], [[1.0]]], [[[2.0]], [[2.0]]], [[[3.0]], [[3.0]]]],
         [[[[1.0]], [[1.0]]], [[[2.0]], [[2.0]]]]],
        ragged_rank=1)

    inputs = keras.Input(shape=(None, 2, 1, 1), dtype='float32', ragged=True)
    out = keras.layers.GlobalAveragePooling3D()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=out)
    output_ragged = model.predict(ragged_data, steps=1)
    # Because GlobalAveragePooling3D doesn't support masking, the results
    # cannot be compared with its dense equivalent.
    expected_output = constant_op.constant([[2.0], [1.5]])
    self.assertAllEqual(output_ragged, expected_output)

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

  def test_globalpooling_1d_keepdims(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling1D,
        kwargs={'keepdims': True},
        input_shape=(3, 4, 5),
        expected_output_shape=(None, 1, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling1D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 5),
        expected_output_shape=(None, 4, 1))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D,
        kwargs={'keepdims': True},
        input_shape=(3, 4, 5),
        expected_output_shape=(None, 1, 5))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling1D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 5),
        expected_output_shape=(None, 4, 1))

  def test_globalpooling_2d_keepdims(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling2D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 5, 6),
        expected_output_shape=(None, 4, 1, 1))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling2D,
        kwargs={'data_format': 'channels_last', 'keepdims': True},
        input_shape=(3, 4, 5, 6),
        expected_output_shape=(None, 1, 1, 6))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling2D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 5, 6),
        expected_output_shape=(None, 4, 1, 1))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling2D,
        kwargs={'data_format': 'channels_last', 'keepdims': True},
        input_shape=(3, 4, 5, 6),
        expected_output_shape=(None, 1, 1, 6))

  def test_globalpooling_3d_keepdims(self):
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling3D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 3, 4, 3),
        expected_output_shape=(None, 4, 1, 1, 1))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalMaxPooling3D,
        kwargs={'data_format': 'channels_last', 'keepdims': True},
        input_shape=(3, 4, 3, 4, 3),
        expected_output_shape=(None, 1, 1, 1, 3))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling3D,
        kwargs={'data_format': 'channels_first', 'keepdims': True},
        input_shape=(3, 4, 3, 4, 3),
        expected_output_shape=(None, 4, 1, 1, 1))
    testing_utils.layer_test(
        keras.layers.pooling.GlobalAveragePooling3D,
        kwargs={'data_format': 'channels_last', 'keepdims': True},
        input_shape=(3, 4, 3, 4, 3),
        expected_output_shape=(None, 1, 1, 1, 3))

  def test_globalpooling_1d_keepdims_masking_support(self):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(None, 4)))
    model.add(keras.layers.GlobalAveragePooling1D(keepdims=True))
    model.compile(loss='mae', optimizer='rmsprop')

    model_input = np.random.random((2, 3, 4))
    model_input[0, 1:, :] = 0
    output = model.predict(model_input)
    self.assertAllEqual((2, 1, 4), output.shape)
    self.assertAllClose(output[0, 0], model_input[0, 0, :])


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class Pooling2DTest(test.TestCase, parameterized.TestCase):

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

  def test_averagepooling_2d(self):
    testing_utils.layer_test(
        keras.layers.AveragePooling2D,
        kwargs={
            'strides': (2, 2),
            'padding': 'same',
            'pool_size': (2, 2)
        },
        input_shape=(3, 5, 6, 4))
    testing_utils.layer_test(
        keras.layers.AveragePooling2D,
        kwargs={
            'strides': (2, 2),
            'padding': 'valid',
            'pool_size': (3, 3)
        },
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


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class Pooling3DTest(test.TestCase, parameterized.TestCase):

  def test_maxpooling_3d(self):
    pool_size = (3, 3, 3)
    testing_utils.layer_test(
        keras.layers.MaxPooling3D,
        kwargs={
            'strides': 2,
            'padding': 'valid',
            'pool_size': pool_size
        },
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

  def test_averagepooling_3d(self):
    pool_size = (3, 3, 3)
    testing_utils.layer_test(
        keras.layers.AveragePooling3D,
        kwargs={
            'strides': 2,
            'padding': 'valid',
            'pool_size': pool_size
        },
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


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class Pooling1DTest(test.TestCase, parameterized.TestCase):

  def test_maxpooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.MaxPooling1D,
            kwargs={
                'strides': stride,
                'padding': padding
            },
            input_shape=(3, 5, 4))
    testing_utils.layer_test(
        keras.layers.MaxPooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 2, 6))

  def test_averagepooling_1d(self):
    for padding in ['valid', 'same']:
      for stride in [1, 2]:
        testing_utils.layer_test(
            keras.layers.AveragePooling1D,
            kwargs={
                'strides': stride,
                'padding': padding
            },
            input_shape=(3, 5, 4))

    testing_utils.layer_test(
        keras.layers.AveragePooling1D,
        kwargs={'data_format': 'channels_first'},
        input_shape=(3, 2, 6))


if __name__ == '__main__':
  test.main()

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for imagenet_utils."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.applications import imagenet_utils as utils
from tensorflow.python.platform import test


class TestImageNetUtils(keras_parameterized.TestCase):

  def test_preprocess_input(self):
    # Test invalid mode check
    x = np.random.uniform(0, 255, (10, 10, 3))
    with self.assertRaises(ValueError):
      utils.preprocess_input(x, mode='some_unknown_mode')

    # Test image batch with float and int image input
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int32')
    self.assertEqual(utils.preprocess_input(x).shape, x.shape)
    self.assertEqual(utils.preprocess_input(xint).shape, xint.shape)

    out1 = utils.preprocess_input(x, 'channels_last')
    out1int = utils.preprocess_input(xint, 'channels_last')
    out2 = utils.preprocess_input(
        np.transpose(x, (0, 3, 1, 2)), 'channels_first')
    out2int = utils.preprocess_input(
        np.transpose(xint, (0, 3, 1, 2)), 'channels_first')
    self.assertAllClose(out1, out2.transpose(0, 2, 3, 1))
    self.assertAllClose(out1int, out2int.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    xint = x.astype('int32')
    self.assertEqual(utils.preprocess_input(x).shape, x.shape)
    self.assertEqual(utils.preprocess_input(xint).shape, xint.shape)

    out1 = utils.preprocess_input(x, 'channels_last')
    out1int = utils.preprocess_input(xint, 'channels_last')
    out2 = utils.preprocess_input(np.transpose(x, (2, 0, 1)), 'channels_first')
    out2int = utils.preprocess_input(
        np.transpose(xint, (2, 0, 1)), 'channels_first')
    self.assertAllClose(out1, out2.transpose(1, 2, 0))
    self.assertAllClose(out1int, out2int.transpose(1, 2, 0))

    # Test that writing over the input data works predictably
    for mode in ['torch', 'tf']:
      x = np.random.uniform(0, 255, (2, 10, 10, 3))
      xint = x.astype('int')
      x2 = utils.preprocess_input(x, mode=mode)
      xint2 = utils.preprocess_input(xint)
      self.assertAllClose(x, x2)
      self.assertNotEqual(xint.astype('float').max(), xint2.max())

    # Caffe mode works differently from the others
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int')
    x2 = utils.preprocess_input(x, data_format='channels_last', mode='caffe')
    xint2 = utils.preprocess_input(xint)
    self.assertAllClose(x, x2[..., ::-1])
    self.assertNotEqual(xint.astype('float').max(), xint2.max())

  @parameterized.named_parameters([
      {
          'testcase_name': 'mode_torch',
          'mode': 'torch'
      },
      {
          'testcase_name': 'mode_tf',
          'mode': 'tf'
      },
      {
          'testcase_name': 'mode_caffe',
          'mode': 'caffe'
      },
  ])
  def test_preprocess_input_symbolic(self, mode):
    # Test image batch
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    inputs = keras.layers.Input(shape=x.shape[1:])
    outputs = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, mode=mode),
        output_shape=x.shape[1:])(
            inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model.predict(x).shape, x.shape)

    outputs1 = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, 'channels_last', mode=mode),
        output_shape=x.shape[1:])(
            inputs)
    model1 = keras.Model(inputs, outputs1)
    out1 = model1.predict(x)
    x2 = np.transpose(x, (0, 3, 1, 2))
    inputs2 = keras.layers.Input(shape=x2.shape[1:])
    outputs2 = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, 'channels_first', mode=mode),
        output_shape=x2.shape[1:])(
            inputs2)
    model2 = keras.Model(inputs2, outputs2)
    out2 = model2.predict(x2)
    self.assertAllClose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    inputs = keras.layers.Input(shape=x.shape)
    outputs = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, mode=mode), output_shape=x.shape)(
            inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model.predict(x[np.newaxis])[0].shape, x.shape)

    outputs1 = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, 'channels_last', mode=mode),
        output_shape=x.shape)(
            inputs)
    model1 = keras.Model(inputs, outputs1)
    out1 = model1.predict(x[np.newaxis])[0]
    x2 = np.transpose(x, (2, 0, 1))
    inputs2 = keras.layers.Input(shape=x2.shape)
    outputs2 = keras.layers.Lambda(
        lambda x: utils.preprocess_input(x, 'channels_first', mode=mode),
        output_shape=x2.shape)(
            inputs2)
    model2 = keras.Model(inputs2, outputs2)
    out2 = model2.predict(x2[np.newaxis])[0]
    self.assertAllClose(out1, out2.transpose(1, 2, 0))

  @parameterized.named_parameters([
      {'testcase_name': 'channels_last_format',
       'data_format': 'channels_last'},
      {'testcase_name': 'channels_first_format',
       'data_format': 'channels_first'},
  ])
  def test_obtain_input_shape(self, data_format):
    # input_shape and default_size are not identical.
    with self.assertRaises(ValueError):
      utils.obtain_input_shape(
          input_shape=(224, 224, 3),
          default_size=299,
          min_size=139,
          data_format='channels_last',
          require_flatten=True,
          weights='imagenet')

    # Test invalid use cases

    shape = (139, 139)
    if data_format == 'channels_last':
      input_shape = shape + (99,)
    else:
      input_shape = (99,) + shape

    # input_shape is smaller than min_size.
    shape = (100, 100)
    if data_format == 'channels_last':
      input_shape = shape + (3,)
    else:
      input_shape = (3,) + shape
    with self.assertRaises(ValueError):
      utils.obtain_input_shape(
          input_shape=input_shape,
          default_size=None,
          min_size=139,
          data_format=data_format,
          require_flatten=False)

    # shape is 1D.
    shape = (100,)
    if data_format == 'channels_last':
      input_shape = shape + (3,)
    else:
      input_shape = (3,) + shape
    with self.assertRaises(ValueError):
      utils.obtain_input_shape(
          input_shape=input_shape,
          default_size=None,
          min_size=139,
          data_format=data_format,
          require_flatten=False)

    # the number of channels is 5 not 3.
    shape = (100, 100)
    if data_format == 'channels_last':
      input_shape = shape + (5,)
    else:
      input_shape = (5,) + shape
    with self.assertRaises(ValueError):
      utils.obtain_input_shape(
          input_shape=input_shape,
          default_size=None,
          min_size=139,
          data_format=data_format,
          require_flatten=False)

    # require_flatten=True with dynamic input shape.
    with self.assertRaises(ValueError):
      utils.obtain_input_shape(
          input_shape=None,
          default_size=None,
          min_size=139,
          data_format='channels_first',
          require_flatten=True)

    # test include top
    self.assertEqual(utils.obtain_input_shape(
        input_shape=(3, 200, 200),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=True), (3, 200, 200))

    self.assertEqual(utils.obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False), (None, None, 3))

    self.assertEqual(utils.obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False), (3, None, None))

    self.assertEqual(utils.obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False), (None, None, 3))

    self.assertEqual(utils.obtain_input_shape(
        input_shape=(150, 150, 3),
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False), (150, 150, 3))

    self.assertEqual(utils.obtain_input_shape(
        input_shape=(3, None, None),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False), (3, None, None))


if __name__ == '__main__':
  test.main()

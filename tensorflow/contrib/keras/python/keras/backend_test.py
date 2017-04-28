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
"""Tests for Keras backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def compare_single_input_op_to_numpy(keras_op,
                                     np_op,
                                     input_shape,
                                     dtype='float32',
                                     negative_values=True,
                                     keras_args=None,
                                     keras_kwargs=None,
                                     np_args=None,
                                     np_kwargs=None):
  keras_args = keras_args or []
  keras_kwargs = keras_kwargs or {}
  np_args = np_args or []
  np_kwargs = np_kwargs or {}
  inputs = 2. * np.random.random(input_shape)
  if negative_values:
    inputs -= 1.
  keras_output = keras_op(keras.backend.variable(inputs, dtype=dtype),
                          *keras_args, **keras_kwargs)
  keras_output = keras.backend.eval(keras_output)
  np_output = np_op(inputs.astype(dtype), *np_args, **np_kwargs)
  try:
    np.testing.assert_allclose(keras_output, np_output, atol=1e-4)
  except AssertionError:
    raise AssertionError('Test for op `' + str(keras_op.__name__) + '` failed; '
                         'Expected ' + str(np_output) + ' but got ' +
                         str(keras_output))


def compare_two_inputs_op_to_numpy(keras_op,
                                   np_op,
                                   input_shape_a,
                                   input_shape_b,
                                   dtype='float32',
                                   keras_args=None,
                                   keras_kwargs=None,
                                   np_args=None,
                                   np_kwargs=None):
  keras_args = keras_args or []
  keras_kwargs = keras_kwargs or {}
  np_args = np_args or []
  np_kwargs = np_kwargs or {}
  input_a = np.random.random(input_shape_a)
  input_b = np.random.random(input_shape_b)
  keras_output = keras_op(keras.backend.variable(input_a, dtype=dtype),
                          keras.backend.variable(input_b, dtype=dtype),
                          *keras_args, **keras_kwargs)
  keras_output = keras.backend.eval(keras_output)
  np_output = np_op(input_a.astype(dtype), input_b.astype(dtype),
                    *np_args, **np_kwargs)
  try:
    np.testing.assert_allclose(keras_output, np_output, atol=1e-4)
  except AssertionError:
    raise AssertionError('Test for op `' + str(keras_op.__name__) + '` failed; '
                         'Expected ' + str(np_output) + ' but got ' +
                         str(keras_output))


class BackendUtilsTest(test.TestCase):

  def test_backend(self):
    self.assertEqual(keras.backend.backend(), 'tensorflow')

  def test_espilon(self):
    epsilon = 1e-2
    keras.backend.set_epsilon(epsilon)
    self.assertEqual(keras.backend.epsilon(), epsilon)
    keras.backend.set_epsilon(1e-7)

  def test_floatx(self):
    floatx = 'float64'
    keras.backend.set_floatx(floatx)
    self.assertEqual(keras.backend.floatx(), floatx)
    keras.backend.set_floatx('float32')

  def test_image_data_format(self):
    image_data_format = 'channels_first'
    keras.backend.set_image_data_format(image_data_format)
    self.assertEqual(keras.backend.image_data_format(), image_data_format)
    keras.backend.set_image_data_format('channels_last')

  def test_get_uid(self):
    self.assertEqual(keras.backend.get_uid('foo'), 1)
    self.assertEqual(keras.backend.get_uid('foo'), 2)


class BackendVariableTest(test.TestCase):

  def test_zeros(self):
    with self.test_session():
      x = keras.backend.zeros((3, 4))
      val = keras.backend.eval(x)
      self.assertAllClose(val, np.zeros((3, 4)))

  def test_ones(self):
    with self.test_session():
      x = keras.backend.ones((3, 4))
      val = keras.backend.eval(x)
      self.assertAllClose(val, np.ones((3, 4)))

  def test_eye(self):
    with self.test_session():
      x = keras.backend.eye(4)
      val = keras.backend.eval(x)
      self.assertAllClose(val, np.eye(4))

  def test_zeros_like(self):
    with self.test_session():
      x = keras.backend.zeros((3, 4))
      y = keras.backend.zeros_like(x)
      val = keras.backend.eval(y)
      self.assertAllClose(val, np.zeros((3, 4)))

  def test_ones_like(self):
    with self.test_session():
      x = keras.backend.zeros((3, 4))
      y = keras.backend.ones_like(x)
      val = keras.backend.eval(y)
      self.assertAllClose(val, np.ones((3, 4)))

  def test_random_uniform_variable(self):
    with self.test_session():
      x = keras.backend.random_uniform_variable((30, 20), low=1, high=2, seed=0)
      val = keras.backend.eval(x)
      self.assertAllClose(val.mean(), 1.5, atol=1e-1)
      self.assertAllClose(val.max(), 2., atol=1e-1)
      self.assertAllClose(val.min(), 1., atol=1e-1)

  def test_random_normal_variable(self):
    with self.test_session():
      x = keras.backend.random_normal_variable((30, 20), 1., 0.5,
                                               seed=0)
      val = keras.backend.eval(x)
      self.assertAllClose(val.mean(), 1., atol=1e-1)
      self.assertAllClose(val.std(), 0.5, atol=1e-1)

  def test_count_params(self):
    with self.test_session():
      x = keras.backend.zeros((4, 5))
      val = keras.backend.count_params(x)
      self.assertAllClose(val, 20)


class BackendLinearAlgebraTest(test.TestCase):

  def test_dot(self):
    x = keras.backend.placeholder(shape=(2, 3))
    y = keras.backend.placeholder(shape=(3, 4))
    xy = keras.backend.dot(x, y)
    self.assertEqual(xy.get_shape().as_list(), [2, 4])

    x = keras.backend.placeholder(shape=(32, 28, 3))
    y = keras.backend.placeholder(shape=(3, 4))
    xy = keras.backend.dot(x, y)
    self.assertEqual(xy.get_shape().as_list(), [32, 28, 4])

  def test_batch_dot(self):
    x = keras.backend.ones(shape=(32, 20, 1))
    y = keras.backend.ones(shape=(32, 30, 20))
    xy = keras.backend.batch_dot(x, y, axes=[1, 2])
    self.assertEqual(xy.get_shape().as_list(), [32, 1, 30])

  def test_reduction_ops(self):
    ops_to_test = [
        (keras.backend.max, np.max),
        (keras.backend.min, np.min),
        (keras.backend.sum, np.sum),
        (keras.backend.prod, np.prod),
        (keras.backend.var, np.var),
        (keras.backend.std, np.std),
        (keras.backend.mean, np.mean),
        (keras.backend.argmin, np.argmin),
        (keras.backend.argmax, np.argmax),
    ]
    for keras_op, np_op in ops_to_test:
      with self.test_session():
        compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7, 5),
                                         keras_kwargs={'axis': 1},
                                         np_kwargs={'axis': 1})
        compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7, 5),
                                         keras_kwargs={'axis': -1},
                                         np_kwargs={'axis': -1})
        if 'keepdims' in tf_inspect.getargspec(keras_op).args:
          compare_single_input_op_to_numpy(keras_op, np_op,
                                           input_shape=(4, 7, 5),
                                           keras_kwargs={'axis': 1,
                                                         'keepdims': True},
                                           np_kwargs={'axis': 1,
                                                      'keepdims': True})

  def test_elementwise_ops(self):
    ops_to_test = [
        (keras.backend.square, np.square),
        (keras.backend.abs, np.abs),
        (keras.backend.round, np.round),
        (keras.backend.sign, np.sign),
        (keras.backend.sin, np.sin),
        (keras.backend.cos, np.cos),
        (keras.backend.exp, np.exp),
    ]
    for keras_op, np_op in ops_to_test:
      with self.test_session():
        compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7))

    ops_to_test = [
        (keras.backend.sqrt, np.sqrt),
        (keras.backend.log, np.log),
    ]
    for keras_op, np_op in ops_to_test:
      with self.test_session():
        compare_single_input_op_to_numpy(keras_op, np_op,
                                         input_shape=(4, 7),
                                         negative_values=False)

    with self.test_session():
      compare_single_input_op_to_numpy(
          keras.backend.clip, np.clip,
          input_shape=(6, 4),
          keras_kwargs={'min_value': 0.1, 'max_value': 2.4},
          np_kwargs={'a_min': 0.1, 'a_max': 1.4})

    with self.test_session():
      compare_single_input_op_to_numpy(
          keras.backend.pow, np.power,
          input_shape=(6, 4),
          keras_args=[3],
          np_args=[3])

  def test_two_tensor_ops(self):
    ops_to_test = [
        (keras.backend.equal, np.equal),
        (keras.backend.not_equal, np.not_equal),
        (keras.backend.greater, np.greater),
        (keras.backend.greater_equal, np.greater_equal),
        (keras.backend.less, np.less),
        (keras.backend.less_equal, np.less_equal),
        (keras.backend.maximum, np.maximum),
        (keras.backend.minimum, np.minimum),
    ]
    for keras_op, np_op in ops_to_test:
      with self.test_session():
        compare_two_inputs_op_to_numpy(keras_op, np_op,
                                       input_shape_a=(4, 7),
                                       input_shape_b=(4, 7))


class BackendShapeOpsTest(test.TestCase):

  def test_reshape(self):
    with self.test_session():
      compare_single_input_op_to_numpy(keras.backend.reshape, np.reshape,
                                       input_shape=(4, 7),
                                       keras_args=[(2, 14)],
                                       np_args=[(2, 14)])

  def test_concatenate(self):
    a = keras.backend.variable(np.ones((1, 2, 3)))
    b = keras.backend.variable(np.ones((1, 2, 2)))
    y = keras.backend.concatenate([a, b], axis=-1)
    self.assertEqual(y.get_shape().as_list(), [1, 2, 5])

  def test_permute_dimensions(self):
    with self.test_session():
      compare_single_input_op_to_numpy(keras.backend.permute_dimensions,
                                       np.transpose,
                                       input_shape=(4, 7),
                                       keras_args=[(1, 0)],
                                       np_args=[(1, 0)])

  def test_resize_images(self):
    height_factor = 2
    width_factor = 2
    data_format = 'channels_last'
    x = keras.backend.variable(np.ones((1, 2, 2, 3)))
    y = keras.backend.resize_images(x,
                                    height_factor,
                                    width_factor,
                                    data_format)
    self.assertEqual(y.get_shape().as_list(), [1, 4, 4, 3])

    data_format = 'channels_first'
    x = keras.backend.variable(np.ones((1, 3, 2, 2)))
    y = keras.backend.resize_images(x,
                                    height_factor,
                                    width_factor,
                                    data_format)
    self.assertEqual(y.get_shape().as_list(), [1, 3, 4, 4])

  def test_resize_volumes(self):
    height_factor = 2
    width_factor = 2
    depth_factor = 2
    data_format = 'channels_last'
    x = keras.backend.variable(np.ones((1, 2, 2, 2, 3)))
    y = keras.backend.resize_volumes(x,
                                     depth_factor,
                                     height_factor,
                                     width_factor,
                                     data_format)
    self.assertEqual(y.get_shape().as_list(), [1, 4, 4, 4, 3])

    data_format = 'channels_first'
    x = keras.backend.variable(np.ones((1, 3, 2, 2, 2)))
    y = keras.backend.resize_volumes(x,
                                     depth_factor,
                                     height_factor,
                                     width_factor,
                                     data_format)
    self.assertEqual(y.get_shape().as_list(), [1, 3, 4, 4, 4])

  def test_repeat_elements(self):
    x = keras.backend.variable(np.ones((1, 3, 2)))
    y = keras.backend.repeat_elements(x, 3, axis=1)
    self.assertEqual(y.get_shape().as_list(), [1, 9, 2])

  def test_repeat(self):
    x = keras.backend.variable(np.ones((1, 3)))
    y = keras.backend.repeat(x, 2)
    self.assertEqual(y.get_shape().as_list(), [1, 2, 3])

  def test_flatten(self):
    with self.test_session():
      compare_single_input_op_to_numpy(keras.backend.flatten,
                                       np.reshape,
                                       input_shape=(4, 7, 6),
                                       np_args=[(4 * 7 * 6,)])

  def test_batch_flatten(self):
    with self.test_session():
      compare_single_input_op_to_numpy(keras.backend.batch_flatten,
                                       np.reshape,
                                       input_shape=(4, 7, 6),
                                       np_args=[(4, 7 * 6)])

  def test_temporal_padding(self):

    def ref_op(x, padding):
      shape = list(x.shape)
      shape[1] += padding[0] + padding[1]
      y = np.zeros(tuple(shape))
      y[:, padding[0]:-padding[1], :] = x
      return y

    with self.test_session():
      compare_single_input_op_to_numpy(keras.backend.temporal_padding,
                                       ref_op,
                                       input_shape=(4, 7, 6),
                                       keras_args=[(2, 3)],
                                       np_args=[(2, 3)])

  def test_spatial_2d_padding(self):

    def ref_op(x, padding, data_format='channels_last'):
      shape = list(x.shape)
      if data_format == 'channels_last':
        shape[1] += padding[0][0] + padding[0][1]
        shape[2] += padding[1][0] + padding[1][1]
        y = np.zeros(tuple(shape))
        y[:, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], :] = x
      else:
        shape[2] += padding[0][0] + padding[0][1]
        shape[3] += padding[1][0] + padding[1][1]
        y = np.zeros(tuple(shape))
        y[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]] = x
      return y

    with self.test_session():
      compare_single_input_op_to_numpy(
          keras.backend.spatial_2d_padding,
          ref_op,
          input_shape=(2, 3, 2, 3),
          keras_args=[((2, 3), (1, 2))],
          keras_kwargs={'data_format': 'channels_last'},
          np_args=[((2, 3), (1, 2))],
          np_kwargs={'data_format': 'channels_last'})
      compare_single_input_op_to_numpy(
          keras.backend.spatial_2d_padding,
          ref_op,
          input_shape=(2, 3, 2, 3),
          keras_args=[((2, 3), (1, 2))],
          keras_kwargs={'data_format': 'channels_first'},
          np_args=[((2, 3), (1, 2))],
          np_kwargs={'data_format': 'channels_first'})

  def test_spatial_3d_padding(self):

    def ref_op(x, padding, data_format='channels_last'):
      shape = list(x.shape)
      if data_format == 'channels_last':
        shape[1] += padding[0][0] + padding[0][1]
        shape[2] += padding[1][0] + padding[1][1]
        shape[3] += padding[2][0] + padding[2][1]
        y = np.zeros(tuple(shape))
        y[:,
          padding[0][0]:-padding[0][1],
          padding[1][0]:-padding[1][1],
          padding[2][0]:-padding[2][1],
          :] = x
      else:
        shape[2] += padding[0][0] + padding[0][1]
        shape[3] += padding[1][0] + padding[1][1]
        shape[4] += padding[2][0] + padding[2][1]
        y = np.zeros(tuple(shape))
        y[:, :,
          padding[0][0]:-padding[0][1],
          padding[1][0]:-padding[1][1],
          padding[2][0]:-padding[2][1]] = x
      return y

    with self.test_session():
      compare_single_input_op_to_numpy(
          keras.backend.spatial_3d_padding,
          ref_op,
          input_shape=(2, 3, 2, 3, 2),
          keras_args=[((2, 3), (1, 2), (2, 3))],
          keras_kwargs={'data_format': 'channels_last'},
          np_args=[((2, 3), (1, 2), (2, 3))],
          np_kwargs={'data_format': 'channels_last'})
      compare_single_input_op_to_numpy(
          keras.backend.spatial_3d_padding,
          ref_op,
          input_shape=(2, 3, 2, 3, 2),
          keras_args=[((2, 3), (1, 2), (2, 3))],
          keras_kwargs={'data_format': 'channels_first'},
          np_args=[((2, 3), (1, 2), (2, 3))],
          np_kwargs={'data_format': 'channels_first'})


if __name__ == '__main__':
  test.main()

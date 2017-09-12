# pylint: disable=g-bad-file-header
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for contrib.resampler.python.ops.resampler_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib import resampler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _bilinearly_interpolate(data, x, y):
  """Performs bilinenar interpolation of grid data at user defined coordinates.

  This interpolation function:
    a) implicitly pads the input data with 0s.
    b) returns 0 when sampling outside the (padded) image.
  The effect is that the sampled signal smoothly goes to 0 outside the original
  input domain, rather than producing a jump discontinuity at the image
  boundaries.

  Args:
    data: numpy array of shape `[data_height, data_width]` containing data
      samples assumed to be defined at the corresponding pixel coordinates.
    x: numpy array of shape `[warp_height, warp_width]` containing x coordinates
      at which interpolation will be performed.
    y: numpy array of shape `[warp_height, warp_width]` containing y coordinates
      at which interpolation will be performed.

  Returns:
    Numpy array of shape `[warp_height, warp_width]` containing interpolated
      values.
  """
  shape = x.shape
  x = np.asarray(x) + 1
  y = np.asarray(y) + 1
  data = np.lib.pad(data, 1, "constant", constant_values=0)

  x_0 = np.floor(x).astype(int)
  x_1 = x_0 + 1
  y_0 = np.floor(y).astype(int)
  y_1 = y_0 + 1

  x_0 = np.clip(x_0, 0, data.shape[1] - 1)
  x_1 = np.clip(x_1, 0, data.shape[1] - 1)
  y_0 = np.clip(y_0, 0, data.shape[0] - 1)
  y_1 = np.clip(y_1, 0, data.shape[0] - 1)

  i_a = data[y_0, x_0]
  i_b = data[y_1, x_0]
  i_c = data[y_0, x_1]
  i_d = data[y_1, x_1]

  w_a = (x_1 - x) * (y_1 - y)
  w_b = (x_1 - x) * (y - y_0)
  w_c = (x - x_0) * (y_1 - y)
  w_d = (x - x_0) * (y - y_0)

  samples = (w_a * i_a + w_b * i_b + w_c * i_c + w_d * i_d)
  samples.reshape(shape)

  return samples


def _make_warp(batch_size, warp_height, warp_width, dtype):
  """Creates batch of warping coordinates."""
  x, y = np.meshgrid(np.linspace(0, warp_width - 1, warp_width),
                     np.linspace(0, warp_height - 1, warp_height))
  warp = np.concatenate((x.reshape([warp_height, warp_width, 1]),
                         y.reshape([warp_height, warp_width, 1])), 2)
  warp = np.tile(warp.reshape([1, warp_height, warp_width, 2]),
                 [batch_size, 1, 1, 1])
  warp += np.random.randn(*warp.shape)
  return warp.astype(dtype)


class ResamplerTest(test.TestCase):

  def test_op_forward_pass_gpu_float32(self):
    self._test_op_forward_pass(True, dtypes.float32, 1e-4)

  def test_op_forward_pass_gpu_float64(self):
    self._test_op_forward_pass(True, dtypes.float64, 1e-5)

  def test_op_forward_pass_cpu_float16(self):
    self._test_op_forward_pass(False, dtypes.float16, 1e-2)

  def test_op_forward_pass_cpu_float32(self):
    self._test_op_forward_pass(False, dtypes.float32, 1e-4)

  def test_op_forward_pass_cpu_float64(self):
    self._test_op_forward_pass(False, dtypes.float64, 1e-5)

  def test_op_backward_pass_gpu_float32(self):
    self._test_op_backward_pass(True, dtypes.float32, 1e-3)

  def test_op_backward_pass_cpu_float16(self):
    self._test_op_backward_pass(False, dtypes.float16, 1e-3)

  def test_op_backward_pass_cpu_float32(self):
    self._test_op_backward_pass(False, dtypes.float32, 1e-4)

  def test_op_backward_pass_cpu_float64(self):
    self._test_op_backward_pass(False, dtypes.float64, 1e-6)

  def _test_op_forward_pass(self, on_gpu, dtype, tol):
    np.random.seed(0)
    data_width = 7
    data_height = 9
    data_channels = 5
    warp_width = 4
    warp_height = 8
    batch_size = 10

    warp = _make_warp(batch_size, warp_height, warp_width, dtype.as_numpy_dtype)
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.random.rand(*data_shape).astype(dtype.as_numpy_dtype)

    with self.test_session(use_gpu=on_gpu, force_gpu=False) as sess:
      data_ph = array_ops.placeholder(dtype, shape=(None,) + data.shape[1:])
      warp_ph = array_ops.placeholder(dtype, shape=(None,) + warp.shape[1:])
      outputs = resampler.resampler(data=data_ph, warp=warp_ph)
      self.assertEqual(outputs.get_shape().as_list(),
                       [None, warp_height, warp_width, data_channels])
      out = sess.run(outputs, feed_dict={data_ph: data, warp_ph: warp})

    # Generate reference output via bilinear interpolation in numpy
    reference_output = np.zeros_like(out)
    for batch in xrange(batch_size):
      for c in xrange(data_channels):
        reference_output[batch, :, :, c] = _bilinearly_interpolate(
            data[batch, :, :, c],
            warp[batch, :, :, 0],
            warp[batch, :, :, 1])

    self.assertAllClose(out, reference_output, rtol=tol, atol=tol)

  def _test_op_backward_pass(self, on_gpu, dtype, tol):
    np.random.seed(13)
    data_width = 5
    data_height = 4
    data_channels = 3
    warp_width = 2
    warp_height = 6
    batch_size = 3

    warp = _make_warp(batch_size, warp_height, warp_width, dtype.as_numpy_dtype)
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.random.rand(*data_shape).astype(dtype.as_numpy_dtype)

    with self.test_session(use_gpu=on_gpu, force_gpu=False):
      data_tensor = constant_op.constant(data)
      warp_tensor = constant_op.constant(warp)
      output_tensor = resampler.resampler(data=data_tensor, warp=warp_tensor)

      grads = test.compute_gradient([data_tensor, warp_tensor], [
          data_tensor.get_shape().as_list(),
          warp_tensor.get_shape().as_list()
      ], output_tensor, output_tensor.get_shape().as_list(), [data, warp])

      if not on_gpu:
        # On CPU we perform numerical differentiation at the best available
        # precision, and compare against that. This is necessary for test to
        # pass for float16.
        data_tensor_64 = constant_op.constant(data, dtype=dtypes.float64)
        warp_tensor_64 = constant_op.constant(warp, dtype=dtypes.float64)
        output_tensor_64 = resampler.resampler(data=data_tensor_64,
                                               warp=warp_tensor_64)
        grads_64 = test.compute_gradient([data_tensor_64, warp_tensor_64], [
            data_tensor.get_shape().as_list(),
            warp_tensor.get_shape().as_list()
        ], output_tensor_64, output_tensor.get_shape().as_list(), [data, warp])

        for g, g_64 in zip(grads, grads_64):
          self.assertLess(np.fabs(g[0] - g_64[1]).max(), tol)

      else:
        for g in grads:
          self.assertLess(np.fabs(g[0] - g[1]).max(), tol)

  def test_op_errors(self):
    data_width = 7
    data_height = 9
    data_depth = 3
    data_channels = 5
    warp_width = 4
    warp_height = 8
    batch_size = 10

    # Input data shape is not defined over a 2D grid, i.e. its shape is not like
    # (batch_size, data_height, data_width, data_channels).
    with self.test_session() as sess:
      data_shape = (batch_size, data_height, data_width, data_depth,
                    data_channels)
      data = np.zeros(data_shape)
      warp_shape = (batch_size, warp_height, warp_width, 2)
      warp = np.zeros(warp_shape)
      outputs = resampler.resampler(constant_op.constant(data),
                                    constant_op.constant(warp))

      with self.assertRaisesRegexp(errors_impl.UnimplementedError,
                                   "Only bilinear interpolation is currently "
                                   "supported."):
        sess.run(outputs)

    # Warp tensor must be at least a matrix, with shape [batch_size, 2].
    with self.test_session() as sess:
      data_shape = (batch_size, data_height, data_width, data_channels)
      data = np.zeros(data_shape)
      warp_shape = (batch_size,)
      warp = np.zeros(warp_shape)
      outputs = resampler.resampler(constant_op.constant(data),
                                    constant_op.constant(warp))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "warp should be at least a matrix"):
        sess.run(outputs)

    # The batch size of the data and warp tensors must be the same.
    with self.test_session() as sess:
      data_shape = (batch_size, data_height, data_width, data_channels)
      data = np.zeros(data_shape)
      warp_shape = (batch_size+1, warp_height, warp_width, 2)
      warp = np.zeros(warp_shape)
      outputs = resampler.resampler(constant_op.constant(data),
                                    constant_op.constant(warp))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Batch size of data and warp tensor"):
        sess.run(outputs)

    # The warp tensor must contain 2D coordinates, i.e. its shape last dimension
    # must be 2.
    with self.test_session() as sess:
      data_shape = (batch_size, data_height, data_width, data_channels)
      data = np.zeros(data_shape)
      warp_shape = (batch_size, warp_height, warp_width, 3)
      warp = np.zeros(warp_shape)
      outputs = resampler.resampler(constant_op.constant(data),
                                    constant_op.constant(warp))

      with self.assertRaisesRegexp(errors_impl.UnimplementedError,
                                   "Only bilinear interpolation is supported, "
                                   "warping"):
        sess.run(outputs)


if __name__ == "__main__":
  test.main()

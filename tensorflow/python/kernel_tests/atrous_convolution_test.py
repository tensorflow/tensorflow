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
"""Tests for atrous convolution functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def upsample_filters(filters, rate):
  """Upsamples the filters by a factor of rate along the spatial dimensions.

  Args:
    filters: spatial_shape + [in_channels, out_channels]
      Original filters.
    rate: A list of len(spatial_shape) positive ints, specifying the
      upsampling rate.

  Returns:
    filters_up: output_spatial_shape + [in_channels, out_channels].
      Upsampled filters with
      output_spatial_shape[i] = (spatial_shape[i] - 1) * rate[i] + 1
      containing (rate[i] - 1) zeros between consecutive filter values along
      spatial dimension i.
  """
  num_spatial_dims = len(rate)
  spatial_shape = np.array(filters.shape[:num_spatial_dims])
  output_spatial_shape = (spatial_shape - 1) * rate + 1
  output = np.zeros(
      tuple(output_spatial_shape) + tuple(filters.shape[-2:]), filters.dtype)
  output[tuple(np.s_[::rate[i]] for i in range(num_spatial_dims))] = filters
  return output


class AtrousConvolutionTest(test.TestCase):

  @contextlib.contextmanager
  def _delay_checks(self):
    """Context manager for combining checks depending on tensor evaluations.

    Each call to Session.run has some overhead, and this overhead can easily
    account for the majority of the time spent in tests that call Session.run
    (or Tensor.eval) many times.

    This context manager provides a mechanism for registering callback functions
    and associated tensors.  When the context is exited, all of the tensors
    associated with all of the registrations are evaluated with a single call to
    Session.run, and then each registered callback function is called with the
    values of its associated tensors.

    Yields:
      A function `add_check(check, *args, **kwargs)` where `check` is the
      callback function to be invoked, and `*args` and `**kwargs` specify the
      associated Tensors. When in EAGER mode, check is executed in add_check,
      otherwise, it's delayed after the context.
    """
    checks = []

    def add_check(check, *args, **kwargs):
      if context.executing_eagerly():
        args_val, kwargs_val = self.evaluate([args, kwargs])
        check(*args_val, **kwargs_val)
      else:
        checks.append((check, args, kwargs))

    yield add_check
    if not context.executing_eagerly():
      all_values = self.evaluate([[args, kwargs] for _, args, kwargs in checks])
      for (check, _, _), (args, kwargs) in zip(checks, all_values):
        check(*args, **kwargs)

  def _test_atrous_convolution(self, add_check, input_shape, filter_shape,
                               dilation_rate, **kwargs):
    filters = np.arange(
        np.prod(filter_shape), dtype=np.float32).reshape(filter_shape)
    filters_upsampled = upsample_filters(filters, dilation_rate)
    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    y1 = nn_ops.convolution(
        input=x, filter=filters, dilation_rate=dilation_rate, **kwargs)
    y2 = nn_ops.convolution(input=x, filter=filters_upsampled, **kwargs)

    def check(y1_eval, y2_eval):
      self.assertAllClose(y1_eval, y2_eval, rtol=1e-2, atol=1e-2)

    add_check(check, y1, y2)

  def test_unknown_spatial_dims_for_channel_last_format(self):
    x = array_ops.placeholder(dtypes.float32, [1, None, None, 10])
    w = array_ops.zeros([3, 3, 10, 20])
    y = nn_ops.convolution(
        x, w, "VALID", dilation_rate=[2, 2], data_format="NHWC")
    self.assertEqual(y.shape.as_list(), [1, None, None, 20])

  def test_unknown_spatial_dims_for_channel_first_format(self):
    x = array_ops.placeholder(dtypes.float32, [1, 10, None, None])
    w = array_ops.zeros([3, 3, 10, 20])
    y = nn_ops.convolution(
        x, w, "VALID", dilation_rate=[2, 2], data_format="NCHW")
    self.assertEqual(y.shape.as_list(), [1, 20, None, None])

  @test_util.run_in_graph_and_eager_modes()
  def testAtrousConvolution2D(self):
    with self._delay_checks() as add_check:
      for padding in ["SAME", "VALID"]:
        for height, width in [[9, 9], [9, 10]]:
          for kernel_height, kernel_width in [[1, 1], [2, 2], [2, 3]]:
            for dilation_rate in [[1, 1], [3, 2], [2, 1]]:
              self._test_atrous_convolution(
                  add_check=add_check,
                  input_shape=[2, height, width, 2],
                  filter_shape=[kernel_height, kernel_width, 2, 2],
                  padding=padding,
                  dilation_rate=dilation_rate,
              )

  @test_util.run_in_graph_and_eager_modes()
  def testAtrousConvolution3D(self):
    with self._delay_checks() as add_check:
      for padding in ["SAME", "VALID"]:
        for depth, height, width in [[9, 9, 10], [9, 10, 9]]:
          for kernel_depth, kernel_height, kernel_width in [[3, 3,
                                                             3], [3, 2, 2],
                                                            [2, 1, 3]]:
            for dilation_rate in [[1, 1, 1], [3, 3, 3], [3, 2, 3], [3, 1, 2]]:
              self._test_atrous_convolution(
                  add_check=add_check,
                  input_shape=[2, depth, height, width, 2],
                  filter_shape=[
                      kernel_depth, kernel_height, kernel_width, 2, 2
                  ],
                  padding=padding,
                  dilation_rate=dilation_rate,
              )

  @test_util.run_in_graph_and_eager_modes()
  def testAtrousConvolution1D(self):
    with self._delay_checks() as add_check:
      for padding in ["SAME", "VALID"]:
        for width in [9, 10]:
          for kernel_width in range(1, 4):
            for rate in range(1, 4):
              self._test_atrous_convolution(
                  add_check=add_check,
                  input_shape=[2, width, 2],
                  filter_shape=[kernel_width, 2, 2],
                  padding=padding,
                  dilation_rate=[rate],
              )

  @test_util.run_in_graph_and_eager_modes()
  def testAtrousConvolutionNC(self):
    if test.is_gpu_available(cuda_only=True):
      # "NCW" and "NCHW" formats are currently supported only on CUDA.
      with test_util.device(use_gpu=True):
        with self._delay_checks() as add_check:
          for padding in ["SAME", "VALID"]:
            self._test_atrous_convolution(
                add_check=add_check,
                input_shape=[2, 2, 9],
                padding=padding,
                filter_shape=[3, 2, 2],
                dilation_rate=[2],
                data_format="NCW",
            )
            self._test_atrous_convolution(
                add_check=add_check,
                input_shape=[2, 2, 9, 5],
                padding=padding,
                filter_shape=[3, 3, 2, 2],
                dilation_rate=[2, 1],
                data_format="NCHW",
            )

  @test_util.run_in_graph_and_eager_modes()
  def testAtrousSequence(self):
    """Tests optimization of sequence of atrous convolutions.

    See the documentation of with_space_to_batch.
    """
    with self._delay_checks() as add_check:
      for padding in ["SAME", "VALID"]:
        for height in range(15, 17):
          for width in range(15, 17):
            x_shape = [3, height, width, 2]
            x = np.random.random_sample(x_shape).astype(np.float32)

            kernel_sizes = [1, 3] if padding == "SAME" else range(1, 3)
            for kernel in kernel_sizes:
              f_shape = [kernel, kernel, 2, 2]
              f1 = 1e-2 * np.random.random_sample(f_shape).astype(np.float32)
              f2 = 1e-2 * np.random.random_sample(f_shape).astype(np.float32)

              def combined_op(converted_input, num_spatial_dims, padding_arg):  # pylint: disable=unused-argument
                # pylint: disable=cell-var-from-loop
                result = nn_ops.convolution(
                    input=converted_input, filter=f1, padding=padding)
                result = nn_ops.convolution(
                    input=result, filter=f2, padding=padding)
                # pylint: enable=cell-var-from-loop
                return result

              for rate_height in range(2, 4):
                for rate_width in range(2, 4):
                  dilation_rate = [rate_height, rate_width]
                  y1 = nn_ops.convolution(
                      input=x,
                      filter=f1,
                      padding=padding,
                      dilation_rate=dilation_rate)
                  y1 = nn_ops.convolution(
                      input=y1,
                      filter=f2,
                      padding=padding,
                      dilation_rate=dilation_rate)
                  y2 = nn_ops.with_space_to_batch(
                      input=x,
                      dilation_rate=dilation_rate,
                      op=combined_op,
                      padding="VALID")

                  def check(y1_eval, y2_eval):
                    self.assertAllClose(y1_eval, y2_eval, rtol=1e-2, atol=1e-2)

                  add_check(check, y1, y2)

  def _test_gradient(self, x_shape, f_shape, dilation_rate, padding):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    f_val = np.random.random_sample(f_shape).astype(np.float32)
    x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
    f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)
    output = nn_ops.convolution(
        input=x, filter=f, dilation_rate=dilation_rate, padding=padding)
    y_shape = output.get_shape().as_list()
    err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape],
                                                  output, y_shape)
    err_tolerance = 1e-3
    self.assertLess(err, err_tolerance)

  def testGradient(self):
    with self.test_session():
      for padding in ["SAME", "VALID"]:
        for rate_width in range(1, 3):
          for rate_height in range(1, 3):
            self._test_gradient(
                x_shape=[2, 5, 6, 2],
                f_shape=[3, 3, 2, 2],
                dilation_rate=[rate_height, rate_width],
                padding=padding)


if __name__ == "__main__":
  test.main()

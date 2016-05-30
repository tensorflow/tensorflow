# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Functional tests for 3d pooling operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class PoolingTest(tf.test.TestCase):

  def _VerifyValues(self, pool_func, input_sizes, window, strides, padding,
                    expected, use_gpu):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called: co.MaxPool, co.AvgPool.
      input_sizes: Input tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
    """
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      t = tf.constant(x, shape=input_sizes)
      t = pool_func(t,
                    ksize=[1, window[0], window[1], window[2], 1],
                    strides=[1, strides[0], strides[1], strides[2], 1],
                    padding=padding)
      vals = sess.run(t)
    # Verifies values.
    actual = vals.flatten()
    self.assertAllClose(expected, actual)

  def _testAvgPool3dValidPadding(self, use_gpu):
    expected_output = [20.5, 21.5, 22.5]
    self._VerifyValues(tf.nn.avg_pool3d,
                       input_sizes=[1, 3, 3, 3, 3],
                       window=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding="VALID",
                       expected=expected_output,
                       use_gpu=use_gpu)

  def _testAvgPool3dSamePadding(self, use_gpu):
    expected_output = [20.5, 21.5, 22.5, 26.5, 27.5, 28.5]
    self._VerifyValues(tf.nn.avg_pool3d,
                       input_sizes=[1, 2, 2, 4, 3],
                       window=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding="SAME",
                       expected=expected_output,
                       use_gpu=use_gpu)

  def _testMaxPool3dValidPadding(self, use_gpu):
    expected_output = [40.0, 41.0, 42.0]
    self._VerifyValues(tf.nn.max_pool3d,
                       input_sizes=[1, 3, 3, 3, 3],
                       window=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding="VALID",
                       expected=expected_output,
                       use_gpu=use_gpu)

  def _testMaxPool3dSamePadding(self, use_gpu):
    expected_output = [31., 32., 33., 34., 35., 36.]
    self._VerifyValues(tf.nn.max_pool3d,
                       input_sizes=[1, 2, 2, 3, 3],
                       window=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding="SAME",
                       expected=expected_output,
                       use_gpu=use_gpu)

    # Test pooling on a larger input, with different stride and kernel
    # size for the 'z' dimension.

    # Simulate max pooling in numpy to get the expected output.
    input_data = np.arange(1, 5 * 27 * 27 * 64 + 1).reshape((5, 27, 27, 64))
    input_data = np.pad(input_data, [[0, 0], [0, 1], [0, 1], [0, 0]],
                        mode="constant")
    expected_output = input_data[:, 1::2, 1::2, :]
    expected_output[:, -1, :, :] = input_data[:, -2, 1::2, :]
    expected_output[:, :, -1, :] = input_data[:, 1::2, -2, :]
    expected_output[:, -1, -1, :] = input_data[:, -2, -2, :]

    self._VerifyValues(tf.nn.max_pool3d,
                       input_sizes=[1, 5, 27, 27, 64],
                       window=(1, 2, 2),
                       strides=(1, 2, 2),
                       padding="SAME",
                       expected=expected_output.flatten(),
                       use_gpu=use_gpu)

  def testAvgPooling3d(self):
    for use_gpu in [False, True]:
      self._testAvgPool3dValidPadding(use_gpu)
      self._testAvgPool3dSamePadding(use_gpu)

  def testMaxPooling3d(self):
    for use_gpu in [False, True]:
      self._testMaxPool3dValidPadding(use_gpu)
      self._testMaxPool3dSamePadding(use_gpu)

  def testKernelSmallerThanStride(self):
    for use_gpu in [True, False]:
      self._VerifyValues(tf.nn.max_pool3d, input_sizes=[1, 3, 3, 3, 1],
                         window=[1, 1, 1], strides=[2, 2, 2],
                         padding="SAME",
                         expected=[1, 3, 7, 9, 19, 21, 25, 27],
                         use_gpu=use_gpu)

      self._VerifyValues(tf.nn.max_pool3d, input_sizes=[1, 7, 7, 7, 1],
                         window=[2, 2, 2], strides=[3, 3, 3],
                         padding="VALID",
                         expected=[58, 61, 79, 82, 205, 208, 226, 229],
                         use_gpu=use_gpu)

      self._VerifyValues(tf.nn.avg_pool3d, input_sizes=[1, 3, 3, 3, 1],
                         window=[1, 1, 1], strides=[2, 2, 2],
                         padding="SAME",
                         expected=[1, 3, 7, 9, 19, 21, 25, 27],
                         use_gpu=use_gpu)

      self._VerifyValues(tf.nn.avg_pool3d, input_sizes=[1, 7, 7, 7, 1],
                         window=[2, 2, 2], strides=[3, 3, 3],
                         padding="VALID",
                         expected=[29.5, 32.5, 50.5, 53.5,
                                   176.5, 179.5, 197.5, 200.5],
                         use_gpu=use_gpu)

  def _ConstructAndTestGradient(self,
                                pool_func,
                                input_sizes,
                                output_sizes,
                                window,
                                strides,
                                padding,
                                x_init_value=None,
                                use_gpu=False):
    """Verifies the gradients of the avg pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
      padding: Padding type.
      x_init_value: Values to be passed to the gradient checker.
      use_gpu: Whether to run pooling on GPU.
    """
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.test_session(use_gpu=use_gpu):
      input_tensor = tf.constant(x, shape=input_sizes, name="input")
      err_margin = 1e-3
      if pool_func == tf.nn.avg_pool3d:
        func_name = "avg_pool3d"
      else:
        if x_init_value is None:
          x_init_value = np.asfarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
          func_name = "max_pool3d"

      t = pool_func(input_tensor,
                    ksize=[1, window[0], window[1], window[2], 1],
                    strides=[1, strides[0], strides[1], strides[2], 1],
                    padding=padding,
                    name=func_name)

      err = tf.test.compute_gradient_error(input_tensor,
                                           input_sizes,
                                           t,
                                           output_sizes,
                                           x_init_value=x_init_value,
                                           delta=1e-2)
    print("%s gradient error = " % func_name, err)
    self.assertLess(err, err_margin)

  def testMaxPoolGradValidPadding1_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[1, 3, 3, 3, 1],
                                     output_sizes=[1, 3, 3, 3, 1],
                                     window=(1, 1, 1),
                                     strides=(1, 1, 1),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testMaxPoolGradValidPadding2_1_6_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 3, 3, 6, 3],
                                     output_sizes=[2, 2, 2, 5, 3],
                                     window=(2, 2, 2),
                                     strides=(1, 1, 1),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testMaxPoolGradValidPadding2_1_7_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 3, 5, 7, 3],
                                     output_sizes=[2, 2, 4, 6, 3],
                                     window=(2, 2, 2),
                                     strides=(1, 1, 1),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testMaxPoolGradValidPadding2_2_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 2, 2, 2, 3],
                                     output_sizes=[2, 1, 1, 1, 3],
                                     window=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testMaxPoolGradSamePadding1_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 3, 2, 4, 1],
                                     output_sizes=[2, 3, 2, 4, 1],
                                     window=(1, 1, 1),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testMaxPoolGradSamePadding2_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 3, 2, 4, 1],
                                     output_sizes=[2, 3, 2, 4, 1],
                                     window=(2, 2, 2),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testMaxPoolGradSamePadding2_2_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[2, 5, 2, 4, 3],
                                     output_sizes=[2, 3, 1, 2, 3],
                                     window=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testMaxPoolGradSamePadding3_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.max_pool3d,
                                     input_sizes=[1, 3, 3, 7, 1],
                                     output_sizes=[1, 3, 3, 7, 1],
                                     window=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testAvgPoolGradValidPadding1_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[2, 3, 3, 3, 3],
                                     output_sizes=[2, 3, 3, 3, 3],
                                     window=(1, 1, 1),
                                     strides=(1, 1, 1),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testAvgPoolGradValidPadding2_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[2, 3, 3, 3, 3],
                                     output_sizes=[2, 2, 2, 2, 3],
                                     window=(2, 2, 2),
                                     strides=(1, 1, 1),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testAvgPoolGradValidPadding2_2_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[2, 2, 2, 2, 3],
                                     output_sizes=[2, 1, 1, 1, 3],
                                     window=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding="VALID",
                                     use_gpu=use_gpu)

  def testAvgPoolGradSamePadding1_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[2, 3, 2, 4, 3],
                                     output_sizes=[2, 3, 2, 4, 3],
                                     window=(1, 1, 1),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testAvgPoolGradSamePadding2_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[1, 2, 2, 2, 1],
                                     output_sizes=[1, 2, 2, 2, 1],
                                     window=(2, 2, 2),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testAvgPoolGradSamePadding2_2_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[2, 5, 2, 4, 3],
                                     output_sizes=[2, 3, 1, 2, 3],
                                     window=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding="SAME",
                                     use_gpu=use_gpu)

  def testAvgPoolGradSamePadding3_1_3d(self):
    for use_gpu in (False, True):
      self._ConstructAndTestGradient(tf.nn.avg_pool3d,
                                     input_sizes=[1, 3, 6, 7, 1],
                                     output_sizes=[1, 3, 6, 7, 1],
                                     window=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     padding="SAME",
                                     use_gpu=use_gpu)


if __name__ == "__main__":
  tf.test.main()

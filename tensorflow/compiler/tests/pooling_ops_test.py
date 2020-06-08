# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for pooling operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


def NHWCToNCHW(input_tensor):
  """Convert the input from NHWC format to NCHW.

  Args:
    input_tensor:  a 4-D tensor, or a 4-element array representing the same.

  Returns:
    the converted tensor or a shape array
  """
  if isinstance(input_tensor, ops.Tensor):
    return array_ops.transpose(input_tensor, [0, 3, 1, 2])
  else:
    return [input_tensor[0], input_tensor[3], input_tensor[1], input_tensor[2]]


def NCHWToNHWC(input_tensor):
  """Convert the input from NCHW format to NHWC.

  Args:
    input_tensor:  a 4-D tensor, or a 4-element array representing the same.

  Returns:
    the converted tensor or a shape array
  """
  if isinstance(input_tensor, ops.Tensor):
    return array_ops.transpose(input_tensor, [0, 2, 3, 1])
  else:
    return [input_tensor[0], input_tensor[2], input_tensor[3], input_tensor[1]]


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs
  """
  test_configs = ["NHWC", "NCHW"]
  return test_configs


class PoolingTest(xla_test.XLATestCase):

  def _VerifyOneTest(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, expected):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, currently only co.MaxPool.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      expected: An array containing the expected operation outputs.
    """
    total_size = np.prod(input_sizes)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = np.array([f * 1.0 for f in range(1, total_size + 1)], dtype=np.float32)
    x = x.reshape(input_sizes)
    with self.session() as sess:
      with self.test_scope():
        inputs = array_ops.placeholder(dtypes.float32)
        t = inputs
        if data_format == "NCHW":
          t = NHWCToNCHW(t)
          ksize = NHWCToNCHW(ksize)
          strides = NHWCToNCHW(strides)
        t = pool_func(t,
                      ksize=ksize,
                      strides=strides,
                      padding=padding,
                      data_format=data_format)
        if data_format == "NCHW":
          t = NCHWToNHWC(t)
      actual = sess.run(t, {inputs: x})
      self.assertAllClose(expected, actual.flatten(), rtol=1e-5, atol=1e-6)

  def _VerifyValues(self, pool_func, input_sizes, ksize, strides, padding,
                    expected):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    for data_format in GetTestConfigs():
      self._VerifyOneTest(pool_func, input_sizes, ksize, strides, padding,
                          data_format, expected)

  def testMaxPoolValidPadding(self):
    expected_output = [13.0, 14.0, 15.0]
    self._VerifyValues(nn_ops.max_pool,
                       input_sizes=[1, 3, 3, 3],
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding="VALID",
                       expected=expected_output)

  def testMaxPoolSamePadding(self):
    expected_output = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    self._VerifyValues(nn_ops.max_pool,
                       input_sizes=[1, 2, 3, 3],
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output)

  def testMaxPoolSamePaddingNonSquareWindow(self):
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #
    #  [max(1.0, 2.0), max(2.0, padded0),
    #   max(3.0, 4.0), max(4.0, padded0)]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 2.0, 4.0, 4.0])

  def testMaxPoolValidPaddingUnevenStride(self):
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0])

  def testMaxPoolSamePaddingFilter4(self):
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0, 54.0, 55.0, 56.0,
        61.0, 62.0, 63.0, 64.0
    ]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output)

  def testMaxPoolSamePaddingFilter8(self):
    expected_output = [
        145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 161.0, 162.0,
        163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 177.0, 178.0, 179.0, 180.0,
        181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0,
        191.0, 192.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0,
        289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 305.0, 306.0,
        307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0,
        317.0, 318.0, 319.0, 320.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0,
        407.0, 408.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0,
        433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0, 441.0, 442.0,
        443.0, 444.0, 445.0, 446.0, 447.0, 448.0, 465.0, 466.0, 467.0, 468.0,
        469.0, 470.0, 471.0, 472.0, 481.0, 482.0, 483.0, 484.0, 485.0, 486.0,
        487.0, 488.0, 497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0,
        505.0, 506.0, 507.0, 508.0, 509.0, 510.0, 511.0, 512.0
    ]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output)

  # Tests for DepthwiseMaxPooling on CPU only.
  def testDepthwiseMaxPool1x1DepthWindow1(self):
    # input is:
    # [1.0, ..., 10.0] along depth,
    #
    # We maxpool by depth in patches of 2.
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 1, 1, 10],
        ksize=[1, 1, 1, 2],
        strides=[1, 1, 1, 2],
        padding="SAME",
        expected=[2.0, 4.0, 6.0, 8.0, 10.0])

  def testDepthwiseMaxPool2x2DepthWindow3(self):
    # input is:
    #
    # a 2x2x6 cube, and we depthwise max across 3 to produce a 2x2x2
    # output.  Each node has contiguous values, so the depthwise max
    # should be multiples of 3.0.
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 2, 2, 6],
        ksize=[1, 1, 1, 3],
        strides=[1, 1, 1, 3],
        padding="SAME",
        expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

  def testKernelSmallerThanStrideValid(self):
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID",
        expected=[9, 12, 30, 33])

  def testKernelSmallerThanStrideSame(self):
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 7, 9])

    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 9, 11])

  # Average pooling
  def testAvgPoolValidPadding(self):
    expected_output = [7, 8, 9]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=expected_output)

  def testAvgPoolSamePadding(self):
    expected_output = [7., 8., 9., 11.5, 12.5, 13.5]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 2, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output)


class PoolGradTest(xla_test.XLATestCase):

  CPU_DEVICE = "/job:localhost/replica:0/task:0/cpu:0"

  def _VerifyOneTest(self,
                     pool_func,
                     pool_grad_func,
                     input_sizes,
                     ksize,
                     strides,
                     padding,
                     data_format,
                     pool_grad_grad_func=None):
    """Verifies the output values of the pooling gradient function.

    Args:
      pool_func: Forward pooling function
      pool_grad_func: Pooling gradient function for pool_grad_func
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      pool_grad_grad_func: Second-order gradient function, if available.
    """
    total_size = np.prod(input_sizes)
    # TODO(b/73062247): MaxPoolGradGrad can confuse gradients when x is equally
    # maximal at 16 bits. Switch to np.random.randn when resolved.
    x = np.arange(1, total_size + 1, dtype=np.float32)
    x *= (np.random.randint(2, size=total_size) * 2 - 1)  # Flip signs randomly
    # Verify some specifically interesting values...
    x[np.random.choice(total_size)] = np.inf
    x[np.random.choice(total_size)] = -np.inf
    # TODO(b/74222344): Fix nan handling for max pool grad.
    # x[np.random.choice(total_size)] = np.nan
    x = x.reshape(input_sizes)
    with self.session() as sess:
      # Use the forward pool function to compute some corresponding outputs
      # (needed for the CPU device, and we need the shape in both cases).
      with ops.device(self.CPU_DEVICE):
        inputs = array_ops.placeholder(dtypes.float32, shape=input_sizes)
        outputs = pool_func(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format="NHWC")

      output_vals = np.array(sess.run(outputs, {inputs: x}))
      output_gradient_vals = np.arange(
          1, output_vals.size + 1, dtype=np.float32)
      output_gradient_vals = output_gradient_vals.reshape(output_vals.shape)
      output_grad_grad_vals = np.arange(1, x.size + 1, dtype=np.float32)
      output_grad_grad_vals = output_grad_grad_vals.reshape(x.shape)

      # Use the Tensorflow CPU pooling gradient to compute the expected input
      # gradients.
      with ops.device(self.CPU_DEVICE):
        output_gradients = array_ops.placeholder(
            dtypes.float32, shape=output_vals.shape)
        expected_input_gradients = pool_grad_func(
            inputs,
            outputs,
            output_gradients,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format="NHWC")
        expected_input_gradient_vals = sess.run(
            expected_input_gradients,
            {inputs: x,
             output_gradients: output_gradient_vals})

        output_grad_gradients = array_ops.placeholder(
            dtypes.float32, shape=expected_input_gradient_vals.shape)
        if pool_grad_grad_func is not None:
          expected_grad_gradients = pool_grad_grad_func(
              inputs,
              outputs,
              output_grad_gradients,
              ksize=ksize,
              strides=strides,
              padding=padding,
              data_format="NHWC")
          expected_grad_gradients_vals = sess.run(expected_grad_gradients, {
              inputs: x,
              output_grad_gradients: output_grad_grad_vals
          })

      # Run the gradient op on the XLA device
      with self.test_scope():
        outputs = array_ops.placeholder(dtypes.float32, shape=output_vals.shape)
        xla_inputs = inputs
        xla_outputs = outputs
        xla_output_gradients = output_gradients
        xla_output_grad_gradients = output_grad_gradients
        xla_ksize = ksize
        xla_strides = strides
        if data_format == "NCHW":
          xla_inputs = NHWCToNCHW(inputs)
          xla_outputs = NHWCToNCHW(outputs)
          xla_output_gradients = NHWCToNCHW(output_gradients)
          xla_output_grad_gradients = NHWCToNCHW(output_grad_gradients)
          xla_ksize = NHWCToNCHW(ksize)
          xla_strides = NHWCToNCHW(strides)
        actual_input_gradients = pool_grad_func(
            xla_inputs,
            xla_outputs,
            xla_output_gradients,
            ksize=xla_ksize,
            strides=xla_strides,
            padding=padding,
            data_format=data_format)
        if data_format == "NCHW":
          actual_input_gradients = NCHWToNHWC(actual_input_gradients)
        if pool_grad_grad_func is not None:
          actual_grad_gradients = pool_grad_grad_func(
              xla_inputs,
              xla_outputs,
              xla_output_grad_gradients,
              ksize=xla_ksize,
              strides=xla_strides,
              padding=padding,
              data_format=data_format)
          if data_format == "NCHW":
            actual_grad_gradients = NCHWToNHWC(actual_grad_gradients)
      actual_input_gradients_vals = sess.run(actual_input_gradients, {
          inputs: x,
          outputs: output_vals,
          output_gradients: output_gradient_vals
      })
      # Compare the Tensorflow and XLA results.
      self.assertAllClose(
          expected_input_gradient_vals,
          actual_input_gradients_vals,
          rtol=1e-4,
          atol=1e-6)
      self.assertShapeEqual(actual_input_gradients_vals, inputs)

      if pool_grad_grad_func is not None:
        actual_grad_gradients_vals = sess.run(
            actual_grad_gradients, {
                inputs: x,
                outputs: output_vals,
                output_grad_gradients: output_grad_grad_vals
            })

        # Compare the Tensorflow and XLA results.
        self.assertAllClose(
            expected_grad_gradients_vals,
            actual_grad_gradients_vals,
            rtol=1e-4,
            atol=1e-6)
        self.assertShapeEqual(actual_grad_gradients_vals, outputs)

  def _VerifyValues(self,
                    pool_func,
                    pool_grad_func,
                    input_sizes,
                    ksize,
                    strides,
                    padding,
                    pool_grad_grad_func=None):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Pooling function to be called, e.g., tf.nn.max_pool2d
      pool_grad_func: Corresponding pooling gradient function.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      pool_grad_grad_func: Second-order gradient function, if available.
    """
    for data_format in GetTestConfigs():
      self._VerifyOneTest(
          pool_func,
          pool_grad_func,
          input_sizes,
          ksize,
          strides,
          padding,
          data_format,
          pool_grad_grad_func=pool_grad_grad_func)

  def _TestPooling(self, forward_op, backward_op, pool_grad_grad_func=None):
    # VALID padding
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        pool_grad_grad_func=pool_grad_grad_func)

    # SAME padding
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 2, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        pool_grad_grad_func=pool_grad_grad_func)

    # SAME padding, non square window
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        pool_grad_grad_func=pool_grad_grad_func)

    # VALID padding, uneven stride
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        pool_grad_grad_func=pool_grad_grad_func)
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        pool_grad_grad_func=pool_grad_grad_func)

    # SAME padding, size 4 input
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        pool_grad_grad_func=pool_grad_grad_func)

    # SAME padding, size 8 input
    self._VerifyValues(
        forward_op,
        backward_op,
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        pool_grad_grad_func=pool_grad_grad_func)

  def testMaxPool(self):
    self._TestPooling(
        nn_ops.max_pool,
        gen_nn_ops.max_pool_grad,
        pool_grad_grad_func=gen_nn_ops.max_pool_grad_grad)

  def testAvgPool(self):
    # Wrapper around AvgPoolGrad that ignores extra arguments needed by
    # MaxPoolGrad.
    def AvgPoolGrad(inputs, outputs, output_gradients, ksize, strides, padding,
                    data_format):
      del outputs  # Unused by average-pooling gradients.
      return gen_nn_ops.avg_pool_grad(
          inputs.get_shape().as_list(),
          output_gradients,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format)

    self._TestPooling(nn_ops.avg_pool, AvgPoolGrad)

  # The CPU implementation of AvgPoolGrad doesn't accept kernels smaller than
  # the stride size, so we only run the following tests on MaxPoolGrad.

  def testMaxPoolKernelSmallerThanStrideValid(self):
    self._VerifyValues(
        nn_ops.max_pool,
        gen_nn_ops.max_pool_grad,
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID")

  def testMaxPoolKernelSmallerThanStrideSame(self):
    self._VerifyValues(
        nn_ops.max_pool,
        gen_nn_ops.max_pool_grad,
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME")

    self._VerifyValues(
        nn_ops.max_pool,
        gen_nn_ops.max_pool_grad,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME")


if __name__ == "__main__":
  googletest.main()

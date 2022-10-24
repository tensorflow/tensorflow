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
"""Tests for Python ops defined in nn_grad.py."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SoftmaxOpTest(test.TestCase):

  # This test is for bfloat16, but the type has a problem with compute_gradient.
  # TODO(penporn): Change the data type back to bfloat16 once b/157773623 is
  # fixed. (compute_gradient internally converts bfloat16 to float32 for
  # calculation anyway.)
  def testSoftmaxGradGradExtendType(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.softmax(x)
        return tape.gradient(y, x)

      x = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]],
                               dtype=dtypes.float32)
      error = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
      self.assertLess(error, 1e-4)


class Relu6OpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testRelu6GradGrad(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    x_init_value = np.array([[-3.5, -1.5, 2, 4], [4.5, 7.5, 8.5, 11]])
    r = nn_ops.relu6(inputs)
    r_g = gradients_impl.gradients(r, inputs)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(),
          r_g,
          r_g.get_shape().as_list(),
          x_init_value=x_init_value)
      self.assertLess(error, 1e-4)


class Conv2dOpTest(test.TestCase):

  def run_test(self, x, y):
    with self.test_session():
      error = gradient_checker.compute_gradient_error(x,
                                                      x.get_shape().as_list(),
                                                      y,
                                                      y.get_shape().as_list())
      self.assertLess(error, 1e-3)

  @test_util.run_deprecated_v1
  def testConv2dGradWRTInput(self):
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
    self.run_test(x, y)

  @test_util.run_deprecated_v1
  def testConv2dGradWRTFilter(self):
    x = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[1, 4, 4, 3],
                             name='input')
    f = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
    y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
    self.run_test(f, y)

  @test_util.run_deprecated_v1
  def testConv2dBackpropFilterGrad(self):
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out = nn_impl.depthwise_conv2d(x, f, strides, padding)

    grad_wrt_input = gradients_impl.gradients(out, x)[0]
    self.run_test(f, grad_wrt_input)

    grad_wrt_filter = gradients_impl.gradients(out, f)[0]
    self.run_test(x, grad_wrt_filter)


class DepthwiseConv2dTest(test.TestCase):

  def run_test(self, x, y):
    with self.test_session():
      error = gradient_checker.compute_gradient_error(x,
                                                      x.get_shape().as_list(),
                                                      y,
                                                      y.get_shape().as_list())
      self.assertLess(error, 1e-3)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dGradWRTInput(self):
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    y = nn_impl.depthwise_conv2d(x, f, strides, padding)
    self.run_test(x, y)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dGradWRTFilter(self):
    x = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[1, 4, 4, 3],
                             name='input')
    f = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    y = nn_impl.depthwise_conv2d(x, f, strides, padding)
    self.run_test(f, y)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dBackpropFilterGrad(self):
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out = nn_impl.depthwise_conv2d(x, f, strides, padding)

    grad_wrt_input = gradients_impl.gradients(out, x)[0]
    self.run_test(f, grad_wrt_input)

    grad_wrt_filter = gradients_impl.gradients(out, f)[0]
    self.run_test(x, grad_wrt_filter)


class EluGradOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testEluGradGradWRTgrad_ys(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    elu = gen_nn_ops.elu(inputs)
    elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          dummy,
          dummy.shape,
          elu_grad,
          elu_grad.shape)
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testEluGradGradWRTinputs(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    elu = gen_nn_ops.elu(inputs)
    elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.shape,
          elu_grad,
          elu_grad.shape)
      self.assertLess(error, 1e-4)


class SeluGradOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSeluGradGradWRTgrad_ys(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    selu = gen_nn_ops.selu(inputs)
    selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          dummy,
          dummy.shape,
          selu_grad,
          selu_grad.shape)
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSeluGradGradWRTinputs(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    selu = gen_nn_ops.selu(inputs)
    selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.shape,
          selu_grad,
          selu_grad.shape)
      self.assertLess(error, 1e-4)


class SwishGradOpTest(test.TestCase):

  def testSwishGrad(self):
    features = constant_op.constant([[-2, -1, 1, 3]],
                                    dtype=dtypes.float32)
    beta = constant_op.constant(0.25, dtype=dtypes.float32)

    with self.cached_session():
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.swish, [features, beta])
      error = gradient_checker_v2.max_error(theoretical, numerical)
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()

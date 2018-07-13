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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class Relu6OpTest(test.TestCase):

  def testRelu6GradGrad(self):
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    x_init_value = np.array([[-3.5, -1.5, 2, 4], [4.5, 7.5, 8.5, 11]])
    r = nn_ops.relu6(inputs)
    r_g = gradients_impl.gradients(r, inputs)[0]
    with self.test_session():
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
            error = gradient_checker.compute_gradient_error(
                x,
                x.get_shape().as_list(),
                y,
                y.get_shape().as_list())
            self.assertLess(error, 1e-3)


    def testConv2dGradWRTInput(self):
        input = array_ops.placeholder(dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = constant_op.constant([0.5], dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        y = nn_ops.conv2d(input, filter, [1,1,1,1], "SAME")
        self.run_test(input, y)

    def testConv2dGradWRTFilter(self):
        input = constant_op.constant([0.5], dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = array_ops.placeholder(dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        y = nn_ops.conv2d(input, filter, [1,1,1,1], "SAME")
        self.run_test(filter, y)

    def testConv2dBackpropFilterGrad(self):
        input = array_ops.placeholder(dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = constant_op.constant([0.5], dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        strides = [1,1,1,1]
        padding = "SAME"
        out = nn_impl.depthwise_conv2d(input, filter, strides, padding)

        grad_wrt_input = gradients_impl.gradients(out, input)[0]
        self.run_test(filter, grad_wrt_input)

        grad_wrt_filter = gradients_impl.gradients(out, filter)[0]
        self.run_test(input, grad_wrt_filter)


class DepthwiseConv2dTest(test.TestCase):

    def run_test(self, x, y):
        with self.test_session():
            error = gradient_checker.compute_gradient_error(
                x,
                x.get_shape().as_list(),
                y,
                y.get_shape().as_list())
            self.assertLess(error, 1e-3)

    def testDepthwiseConv2dGradWRTInput(self):
        input = array_ops.placeholder(dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = constant_op.constant([0.5], dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        strides = [1,1,1,1]
        padding = "SAME"
        y = nn_impl.depthwise_conv2d(input, filter, strides, padding)
        self.run_test(input, y)

    def testDepthwiseConv2dGradWRTFilter(self):
        input = constant_op.constant([0.5], dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = array_ops.placeholder(dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        strides = [1,1,1,1]
        padding = "SAME"
        y = nn_impl.depthwise_conv2d(input, filter, strides, padding)
        self.run_test(filter, y)

    def testDepthwiseConv2dBackpropFilterGrad(self):
        input = array_ops.placeholder(dtype = dtypes.float32, shape=[1,4,4,3], name='input')
        filter = constant_op.constant([0.5], dtype = dtypes.float32, shape=[2,2,3,2], name='filter')
        strides = [1,1,1,1]
        padding = "SAME"
        out = nn_impl.depthwise_conv2d(input, filter, strides, padding)

        grad_wrt_input = gradients_impl.gradients(out, input)[0]
        self.run_test(filter, grad_wrt_input)

        grad_wrt_filter = gradients_impl.gradients(out, filter)[0]
        self.run_test(input, grad_wrt_filter)


if __name__ == "__main__":
  test.main()

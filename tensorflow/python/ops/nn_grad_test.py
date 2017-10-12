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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class Relu6OpTest(test.TestCase):
  def testRelu6GradGrad(self):
    inputs = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]],
                                  dtype=dtypes.float32)
    r = nn_ops.relu6(inputs)
    r_g = gradients_impl.gradients(r, inputs)[0]
    with self.test_session():
      error = gradient_checker.compute_gradient_error(
        inputs, inputs.get_shape().as_list(),
        r_g, r_g.get_shape().as_list())
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()

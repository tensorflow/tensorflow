# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for new version of accumulate_n op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class AccumulateNV2Test(test_util.TensorFlowTestCase):
  """Tests of the new, differentiable version of accumulate_n."""

  def testFloat(self):
    np.random.seed(12345)
    x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(5)]
    tf_x = ops.convert_n_to_tensor(x)
    with self.test_session(use_gpu=True):
      self.assertAllClose(sum(x), math_ops.accumulate_n(tf_x).eval())
      self.assertAllClose(x[0] * 5,
                          math_ops.accumulate_n([tf_x[0]] * 5).eval())

  def testInt(self):
    np.random.seed(54321)
    x = [np.random.randint(-128, 128, (5, 4, 3, 2, 1)) for _ in range(6)]
    tf_x = ops.convert_n_to_tensor(x)
    with self.test_session(use_gpu=True):
      self.assertAllEqual(sum(x), math_ops.accumulate_n(tf_x).eval())
      self.assertAllEqual(x[0] * 6,
                          math_ops.accumulate_n([tf_x[0]] * 6).eval())

  def testGrad(self):
    np.random.seed(42)
    for num_inputs in range(1, 10):
      with self.test_session(use_gpu=True) as sess:
        input_vars = [
            variables.Variable(10.0 * np.random.random())
            for _ in range(0, num_inputs)
        ]
        accum_n = math_ops.accumulate_n(input_vars)
        sess.run(variables.global_variables_initializer())
        accum_n_grad = gradients.gradients(accum_n, input_vars)
        self.assertAllEqual(
            np.repeat(1.0, num_inputs),  # d/dx (x + y + ...) = 1
            [g.eval() for g in accum_n_grad])

  # The tests below used to be in a separate class under cwise_ops_test.py,
  # which did not run in the default test target.
  # Putting them here so that everything that exercises AccumulateNV2 is in
  # one place and the default build runs all unit tests.
  def testSimple(self):
    with self.test_session():
      random_arrays = [
          np.random.rand(16, 16, 16, 16).astype(np.float32) for _ in range(20)
      ]
      random_tensors = [
          ops.convert_to_tensor(x, dtype=dtypes_lib.float32)
          for x in random_arrays
      ]
      tf_val = math_ops.accumulate_n(random_tensors)
      np_val = random_arrays[0]
      for random_array in random_arrays[1:]:
        np_val += random_array
      self.assertAllClose(np_val, tf_val.eval())

  def testZeroArgs(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf_val = math_ops.accumulate_n([])
        tf_val.eval()

  def testWrongShape(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        a = variables.Variable(0.2)
        b = variables.Variable(0.1)
        math_ops.accumulate_n([a, b], shape=[2, 2])  # Should be shape=[]

  def testIncompatibleShapes(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        a = variables.Variable(np.array([0.1, 0.2]))
        b = variables.Variable(np.array([[0.3], [0.4]]))
        math_ops.accumulate_n([a, b])

  def testWrongType(self):
    with self.test_session():
      with self.assertRaises(TypeError):
        a = variables.Variable(0.2, dtype=np.float32)
        b = variables.Variable(0.1, dtype=np.float32)
        math_ops.accumulate_n([a, b], tensor_dtype=np.int32)

  def testWrongTypeOneInput(self):
    # Scenario that used to trigger a bug, even when testWrongType() worked
    with self.test_session():
      with self.assertRaises(TypeError):
        a = variables.Variable(0.2, dtype=np.float32)
        math_ops.accumulate_n([a], tensor_dtype=np.int32)


if __name__ == "__main__":
  googletest.main()

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
"""Tests for XLA JIT compiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class VariableOpsTest(XLATestCase):
  """Test cases for resource variable operators."""

  def testReadWrite(self):
    """Tests initialization, reading, and writing a resource variable."""
    with self.test_session() as session:
      with self.test_scope():
        with variable_scope.variable_scope("ascope", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(2))
          a = x.read_value()
          with ops.control_dependencies([a]):
            b = state_ops.assign(x, 47)
          with ops.control_dependencies([b]):
            c = x.read_value()
          with ops.control_dependencies([c]):
            d = state_ops.assign_add(x, 3)
          with ops.control_dependencies([d]):
            e = x.read_value()

      session.run(variables.initialize_all_variables())
      v1, v2, v3 = session.run([a, c, e])
      self.assertAllClose(2.0, v1)
      self.assertAllClose(47.0, v2)
      self.assertAllClose(50.0, v3)

  def testTraining(self):
    """Tests a gradient descent step for a simple model."""
    with self.test_session() as session:
      with self.test_scope():
        with variable_scope.variable_scope("ascope", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

          x = array_ops.placeholder(dtypes.float32, shape=[1, 4])
          y = math_ops.matmul(x, w) + b
          loss = math_ops.reduce_sum(y)
          optimizer = GradientDescentOptimizer(0.1)
          train = optimizer.minimize(loss)

      session.run(variables.initialize_all_variables())
      session.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      vw, vb = session.run([w, b])
      self.assertAllClose(
          np.array(
              [[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]],
              dtype=np.float32),
          vw,
          rtol=1e-4)
      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32), vb, rtol=1e-4)


if __name__ == "__main__":
  googletest.main()

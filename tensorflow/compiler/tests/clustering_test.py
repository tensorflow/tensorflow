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
"""Tests for the behavior of the auto-compilation pass."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

CPU_DEVICE = "/job:localhost/replica:0/task:0/cpu:0"


class ClusteringTest(xla_test.XLATestCase):

  def testAdd(self):
    val1 = np.array([4, 3, 2, 1], dtype=np.float32)
    val2 = np.array([5, 6, 7, 8], dtype=np.float32)
    expected = val1 + val2
    with self.session():
      with self.test_scope():
        input1 = constant_op.constant(val1, name="const1")
        input2 = constant_op.constant(val2, name="const2")
        output = math_ops.add(input1, input2)
      result = self.evaluate(output)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testAddFromCpuMultiple(self):
    val1 = np.array([4, 3, 2, 1]).astype(np.float32)
    val2 = np.array([5, 6, 7, 8]).astype(np.float32)
    expected = val1 + val2
    with self.session():
      with ops.device(CPU_DEVICE):
        input1 = constant_op.constant(val1, name="const1")
        input2 = constant_op.constant(val2, name="const2")
      with self.test_scope():
        output = math_ops.add(input1, input2)
      for _ in range(10):
        result = self.evaluate(output)
        self.assertAllClose(result, expected, rtol=1e-3)

  def testDeadlock(self):
    # Builds a graph of the form:
    #  x -> y
    #       | \
    #       z -> w
    # where x and z are placed on the CPU and y and w are placed on the XLA
    # device. If y and w are clustered for compilation, then the graph will
    # deadlock since the clustered graph will contain a self-loop.
    with self.session() as sess:
      with ops.device(CPU_DEVICE):
        x = array_ops.placeholder(dtypes.float32, [2])
      with self.test_scope():
        y = x * 2
      with ops.device(CPU_DEVICE):
        z = y * y
      with self.test_scope():
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)

  def testHostMemory(self):
    with self.session() as sess:
      x = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        y = x + 1
      with ops.device(CPU_DEVICE):
        # Place a computation on the CPU, so y and w cannot be merged into the
        # same JIT compilation.
        z = y * 2
      with self.test_scope():
        # Argument 'y' is a non-constant output of a previous cluster. Make sure
        # it is properly copied to host memory so it can be used as a
        # compile-time constant input for this cluster.
        w = array_ops.reshape(z, y)
      result = sess.run(w, {x: [1, 0]})
      expected = np.array([[4], [2]], dtype=np.int32)
      self.assertAllClose(expected, result, rtol=1e-3)


if __name__ == "__main__":
  googletest.main()

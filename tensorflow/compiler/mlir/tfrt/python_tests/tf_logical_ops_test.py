# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for logical operations JIT compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

specializations = [
    tf_jitrt.Specialization.ENABLED,
    tf_jitrt.Specialization.DISABLED,
    tf_jitrt.Specialization.ALWAYS,
]


def logical_op_1d(op_name):
  return f"""
  func.func @test(%arg0: tensor<?xi1>, %arg1: tensor<?xi1>) -> tensor<?xi1> {{
    %0 = "tf.{op_name}"(%arg0, %arg1)
        : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
    func.return %0 : tensor<?xi1>
  }}"""


jitrt = tf_jitrt.TfJitRtExecutor()


def test_logical_op(mlir_blob, reference_fn, rank):
  for specialize in specializations:
    compiled = jitrt.compile(mlir_blob, "test", specialize)

    for _ in range(100):
      shape = np.random.randint(0, 100, size=(rank))
      arg0 = np.random.choice([True, False], size=shape)
      arg1 = np.random.choice([True, False], size=shape)

      [res] = jitrt.execute(compiled, [arg0, arg1])
      np.testing.assert_equal(res, reference_fn(arg0, arg1))


class TfLogicalOpsTest(test.TestCase):

  def test_logical_and_1d(self):
    test_logical_op(logical_op_1d("LogicalAnd"), np.logical_and, 1)

  def test_logical_or_1d(self):
    test_logical_op(logical_op_1d("LogicalOr"), np.logical_or, 1)


if __name__ == "__main__":
  np.random.seed(0)
  test.main()

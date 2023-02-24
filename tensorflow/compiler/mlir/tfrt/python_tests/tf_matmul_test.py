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
"""Tests for tf.MatMul JIT compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test


def matmul():
  return """
  func.func @matmul(%arg0: tensor<?x?xf32>,
               %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = "tf.MatMul"(%arg0, %arg1) {
           transpose_a = false,
           transpose_b = false
         } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %0 : tensor<?x?xf32>
  }"""


jitrt = tf_jitrt.TfJitRtExecutor()


def verify_matmul(compiled, m, k, n):
  lhs = np.random.uniform(0.0, 1.0, size=(m, k)).astype(np.float32)
  rhs = np.random.uniform(0.0, 1.0, size=(k, n)).astype(np.float32)

  [res] = jitrt.execute(compiled, [lhs, rhs])
  np.testing.assert_allclose(res, np.matmul(lhs, rhs), rtol=1e-05)


class TfMatMulTest(test.TestCase):

  # Matmul: [1, k] x [k, 1]
  def test_dot_product(self):
    compiled = jitrt.compile(matmul(), "matmul", vectorize=True)
    for _ in range(100):
      k = np.random.randint(1, 10)
      verify_matmul(compiled, 1, k, 1)

  # Matmul: [1, k] x [k, n]
  def test_vec_mat(self):
    compiled = jitrt.compile(matmul(), "matmul", vectorize=True)
    for _ in range(100):
      k = np.random.randint(1, 10)
      n = np.random.randint(1, 10)
      verify_matmul(compiled, 1, k, n)

  # Matmul: [n, k] x [k, 1]
  def test_mat_vec(self):
    compiled = jitrt.compile(matmul(), "matmul", vectorize=True)
    for _ in range(100):
      m = np.random.randint(1, 10)
      k = np.random.randint(1, 10)
      verify_matmul(compiled, m, k, 1)

  # Matmul: [m, k] x [k, n]
  def test_matmul(self):
    compiled = jitrt.compile(matmul(), "matmul", vectorize=True)
    for _ in range(100):
      m = np.random.randint(1, 10)
      k = np.random.randint(1, 10)
      n = np.random.randint(1, 10)
      verify_matmul(compiled, m, k, n)


if __name__ == "__main__":
  np.random.seed(0)
  test.main()

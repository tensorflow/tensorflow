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
"""Tests for Tensorflow -> CPURT compilation."""

import numpy as np

import unittest
from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt

cpurt = tf_cpurt.TfCpurtExecutor()


class TfReductionTest(googletest.TestCase):

  def test_2d_column_reduction(self):
    mlir_function = """
        func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          return %0 : tensor<?xf32>
      }"""

    compiled = cpurt.compile(mlir_function, 'test', codegen_reductions=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=1), atol=0.01)

  def test_2d_column_reduction_static(self):
    mlir_function = """
        func @test(%input: tensor<8x8xf32>) -> tensor<8xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<8x8xf32>, tensor<1xi32>) -> tensor<8xf32>
          return %0 : tensor<8xf32>
      }"""

    compiled = cpurt.compile(mlir_function, 'test', codegen_reductions=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 8)).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=1), atol=1)


if __name__ == '__main__':
  googletest.main()

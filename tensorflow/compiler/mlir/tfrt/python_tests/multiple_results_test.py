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
"""Tests for JIT compilation of functions with multiple results."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

specializations = [
    tf_cpurt.Specialization.ENABLED,
    tf_cpurt.Specialization.DISABLED,
    tf_cpurt.Specialization.ALWAYS,
]

cpurt = tf_cpurt.TfCpurtExecutor()


class MultipleResultsTest(test.TestCase):

  def test_two_results(self):
    for specialize in specializations:
      mlir_function = """
        func @test(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
          %0 = "tf.Const"() { value = dense<1.0> : tensor<f32> }
               : () -> tensor<f32>
          %1 = "tf.AddV2"(%arg0, %0)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          %2 = "tf.AddV2"(%1, %0)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          return %1, %2 : tensor<?xf32>, tensor<?xf32>
        }"""

      compiled = cpurt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 10)
      arg0 = np.zeros(d0, np.float32)

      [res0, res1] = cpurt.execute(compiled, [arg0])
      np.testing.assert_allclose(res0, arg0 + 1.0, atol=0.0)
      np.testing.assert_allclose(res1, arg0 + 2.0, atol=0.0)

  def test_three_results(self):
    for specialize in specializations:
      mlir_function = """
        func @test(%arg0: tensor<?xf32>) ->
            (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
          %0 = "tf.Const"() { value = dense<1.0> : tensor<f32> }
               : () -> tensor<f32>
          %1 = "tf.AddV2"(%arg0, %0)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          %2 = "tf.AddV2"(%1, %0)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          %3 = "tf.AddV2"(%2, %0)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          return %1, %2, %3 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
        }"""

      compiled = cpurt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 10)
      arg0 = np.zeros(d0, np.float32)

      [res0, res1, res2] = cpurt.execute(compiled, [arg0])
      np.testing.assert_allclose(res0, arg0 + 1.0, atol=0.0)
      np.testing.assert_allclose(res1, arg0 + 2.0, atol=0.0)
      np.testing.assert_allclose(res2, arg0 + 3.0, atol=0.0)


if __name__ == '__main__':
  np.random.seed(0)
  test.main()

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Tensorflow -> jitrt compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

specializations = [
    # TODO(ezhulenev): Fix memrefCopy msan warnings to enable these tests.
    # tf_jitrt.Specialization.ENABLED,
    # tf_jitrt.Specialization.DISABLED,
    tf_jitrt.Specialization.ALWAYS,
]

vectorization = [False, True]

jitrt = tf_jitrt.TfJitRtExecutor()


class TfFunction(test.TestCase):

  def test_func_0(self):
    for specialize in specializations:
      for vectorize in vectorization:
        mlir_function = """
        func.func @test(%arg0: tensor<1x?xf32>,
                       %arg1: tensor<1x?xf32>,
                       %arg2: tensor<1x?xf32>) -> tensor<1x?xf32> {
          %c = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
               : () -> tensor<f32>
          %0 = "tf.Tanh"(%arg0)
               : (tensor<1x?xf32>) -> tensor<1x?xf32>
          %1 = "tf.Mul"(%arg1, %arg2)
               : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
          %2 = "tf.Sub"(%c, %arg2)
               : (tensor<f32>, tensor<1x?xf32>) -> tensor<1x?xf32>
          %3 = "tf.Mul"(%0, %2)
               : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
          %4 = "tf.AddV2"(%1, %3)
               : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
          return %4 : tensor<1x?xf32>
        }"""

        compiled = jitrt.compile(mlir_function, 'test', specialize, vectorize)

        d0 = np.random.randint(128, 256)
        arg0 = np.random.uniform(1.0, 10.0, size=(1, d0)).astype(np.float32)
        arg1 = np.random.uniform(1.0, 10.0, size=(1, d0)).astype(np.float32)
        arg2 = np.random.uniform(1.0, 10.0, size=(1, d0)).astype(np.float32)

        [res] = jitrt.execute(compiled, [arg0, arg1, arg2])

        # Function under test spelled in NumPy
        v0 = np.tanh(arg0)
        v1 = arg1 * arg2
        v2 = 1.0 - arg2
        v3 = v0 * v2
        v4 = v1 + v3

        np.testing.assert_allclose(res, v4, atol=1e-06)

  def test_func_1(self):
    for vectorize in vectorization:
      mlir_function = """
        func @test(%arg0: tensor<*xf32> {jitrt.constraint = "rank"})
            -> (tensor<*xf32>, tensor<*xf32>) {
          %c = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
               : () -> tensor<f32>
          %0 = "tf.Sub"(%c, %arg0)
               : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
          %1 = "tf.Sub"(%c, %0)
               : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
          return %0, %1 : tensor<*xf32>, tensor<*xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test',
                               tf_jitrt.Specialization.ALWAYS, vectorize)

      d0 = np.random.randint(128, 256)
      arg0 = np.random.uniform(1.0, 10.0, size=(1, d0)).astype(np.float32)

      [res0, res1] = jitrt.execute(compiled, [arg0])

      # Function under test spelled in NumPy
      v0 = 1.0 - arg0
      v1 = 1.0 - v0

      np.testing.assert_allclose(res0, v0, atol=0.0)
      np.testing.assert_allclose(res1, v1, atol=0.0)

  def test_func_2(self):
    for vectorize in vectorization:
      mlir_function = """
        func @test(%arg0: tensor<*xf32> {jitrt.constraint = "rank"},
                   %arg1: tensor<?x?xf32>,
                   %arg2: tensor<?x?xf32>,
                   %arg3: tensor<?x?xf32>) -> tensor<*xf32> {
          %0 = "tf.Mul"(%arg0, %arg1)
               : (tensor<*xf32>, tensor<?x?xf32>) -> tensor<*xf32>
          %1 = "tf.Mul"(%arg2, %arg3)
               : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
          %2 = "tf.AddV2"(%0, %1)
               : (tensor<*xf32>, tensor<?x?xf32>) -> tensor<*xf32>
          return %2 : tensor<*xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test',
                               tf_jitrt.Specialization.ALWAYS, vectorize)

      d0 = np.random.randint(4, 8)
      d1 = np.random.randint(4, 8)

      arg1 = np.random.uniform(1.0, 10.0, size=(d0, d1)).astype(np.float32)
      arg2 = np.random.uniform(1.0, 10.0, size=(d0, d1)).astype(np.float32)
      arg3 = np.random.uniform(1.0, 10.0, size=(d0, d1)).astype(np.float32)

      for shape in [(), (d1), (d0, d1)]:
        arg0 = np.random.uniform(1.0, 10.0, size=shape).astype(np.float32)
        [res] = jitrt.execute(compiled, [arg0, arg1, arg2, arg3])

        # Function under test spelled in NumPy
        v0 = arg0 * arg1
        v1 = arg2 * arg3
        v3 = v0 + v1

        np.testing.assert_allclose(res, v3, atol=0.0)

  def test_func_3(self):
    for vectorize in vectorization:
      mlir_function = """
        func @test(%arg0: tensor<i32>, %arg1: tensor<i32>)
            -> (tensor<i32>, tensor<i32>) {
          %c = "tf.Const"() {value = dense<1> : tensor<i32>}
               : () -> tensor<i32>
          %0 = "tf.Maximum"(%c, %arg0)
               : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %1 = "tf.Minimum"(%arg1, %0)
               : (tensor<i32>, tensor<i32>) -> tensor<i32>
        return %0, %1 : tensor<i32>, tensor<i32>
      }"""

      compiled = jitrt.compile(mlir_function, 'test',
                               tf_jitrt.Specialization.ALWAYS, vectorize)

      arg0 = np.random.uniform(-100, 100, size=()).astype(np.int32)
      arg1 = np.random.uniform(-100, 100, size=()).astype(np.int32)

      [res0, res1] = jitrt.execute(compiled, [arg0, arg1])

      # Function under test spelled in NumPy
      v0 = np.maximum(1, arg0)
      v1 = np.minimum(arg1, v0)

      np.testing.assert_allclose(res0, v0, atol=0.0)
      np.testing.assert_allclose(res1, v1, atol=0.0)


if __name__ == '__main__':
  test.main()

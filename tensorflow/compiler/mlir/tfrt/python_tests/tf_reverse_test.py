# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

jitrt = tf_jitrt.TfJitRtExecutor()


class TfReverseTest(test.TestCase):

  def test_1d_static(self):
    mlir_function = """
        func.func @test(%input: tensor<10xf32>) -> tensor<10xf32> {
          %reverse_dims =  "tf.Const"() {value = dense<[0]> : tensor<1xi64>}
             : () -> tensor<1xi64>
          %0 = "tf.ReverseV2"(%input, %reverse_dims)
              : (tensor<10xf32>, tensor<1xi64>) -> tensor<10xf32>
          func.return %0 : tensor<10xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 10.0, size=(10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.flip(arg0, axis=0))

  def test_1d_dynamic(self):
    mlir_function = """
        func.func @test(%input: tensor<?xf32>) -> tensor<?xf32> {
          %reverse_dims =  "tf.Const"() {value = dense<[0]> : tensor<1xi64>}
             : () -> tensor<1xi64>
          %0 = "tf.ReverseV2"(%input, %reverse_dims)
              : (tensor<?xf32>, tensor<1xi64>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 15.0, size=(15)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.flip(arg0, axis=0))

  def test_2d_dynamic(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %reverse_dims =  "tf.Const"() {value = dense<[1]> : tensor<1xi64>}
             : () -> tensor<1xi64>
          %0 = "tf.ReverseV2"(%input, %reverse_dims)
              : (tensor<?x?xf32>, tensor<1xi64>) -> tensor<?x?xf32>
          func.return %0 : tensor<?x?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 10.0, size=(2, 2)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.flip(arg0, axis=1))

  def test_3d_dynamic(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %reverse_dims =  "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>}
             : () -> tensor<2xi64>
          %0 = "tf.ReverseV2"(%input, %reverse_dims)
              : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?x?xf32>
          func.return %0 : tensor<?x?x?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 30.0, size=(2, 3, 4)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.flip(arg0, axis=(0, 1)))

  def test_3d_dynamic_reverse_last(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %reverse_dims =  "tf.Const"() {value = dense<[0, 2]> : tensor<2xi64>}
             : () -> tensor<2xi64>
          %0 = "tf.ReverseV2"(%input, %reverse_dims)
              : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?x?xf32>
          func.return %0 : tensor<?x?x?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 30.0, size=(2, 3, 4)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.flip(arg0, axis=(0, 2)))


if __name__ == '__main__':
  test.main()

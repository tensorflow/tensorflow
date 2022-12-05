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
"""Tests for Tensorflow -> jitrt compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

specializations = [
    tf_jitrt.Specialization.ENABLED,
    tf_jitrt.Specialization.DISABLED,
    tf_jitrt.Specialization.ALWAYS,
]

jitrt = tf_jitrt.TfJitRtExecutor()


class TfTransposeTest(test.TestCase):

  def test_transpose_2d(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %0 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32> }
               : () -> tensor<2xi32>
          %1 = "tf.Transpose"(%arg0, %0)
               : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
          func.return %1 : tensor<?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      d0 = np.random.randint(1, 10)
      d1 = np.random.randint(1, 10)

      arg0 = np.random.uniform(0, 10.0, size=(d0, d1)).astype(np.float32)

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_allclose(res, np.transpose(arg0), atol=0.0)

  def test_transpose_3d_0_2_1(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi64> }
            : () -> tensor<3xi64>
          %1 = "tf.Transpose"(%arg0, %0)
            : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
          func.return %1 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      dim_size = 32
      arg0 = np.arange(0, dim_size * dim_size * dim_size, 1,
                       np.float32).reshape((dim_size, dim_size, dim_size))

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_array_equal(res, np.transpose(arg0, (0, 2, 1)))

  def test_transpose_3d_2_0_1(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi64> }
            : () -> tensor<3xi64>
          %1 = "tf.Transpose"(%arg0, %0)
            : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
          func.return %1 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      dim_size = 32
      arg0 = np.arange(0, dim_size * dim_size * dim_size, 1,
                       np.float32).reshape((dim_size, dim_size, dim_size))

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_array_equal(res, np.transpose(arg0, (2, 0, 1)))

  def test_transpose_3d_2_1_0(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi64> }
            : () -> tensor<3xi64>
          %1 = "tf.Transpose"(%arg0, %0)
            : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
          func.return %1 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      dim_size = 32
      arg0 = np.arange(0, dim_size * dim_size * dim_size, 1,
                       np.float32).reshape((dim_size, dim_size, dim_size))

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_array_equal(res, np.transpose(arg0, (2, 1, 0)))

  def test_transpose_3d_1_2_0(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi64> }
            : () -> tensor<3xi64>
          %1 = "tf.Transpose"(%arg0, %0)
            : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
          func.return %1 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      dim_size = 32
      arg0 = np.arange(0, dim_size * dim_size * dim_size, 1,
                       np.float32).reshape((dim_size, dim_size, dim_size))

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_array_equal(res, np.transpose(arg0, (1, 2, 0)))

  def test_transpose_3d_1_0_2(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi64> }
            : () -> tensor<3xi64>
          %1 = "tf.Transpose"(%arg0, %0)
            : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
          func.return %1 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      dim_size = 32
      arg0 = np.arange(0, dim_size * dim_size * dim_size, 1,
                       np.float32).reshape((dim_size, dim_size, dim_size))

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_array_equal(res, np.transpose(arg0, (1, 0, 2)))

  def test_double_transpose_3d(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32> }
               : () -> tensor<3xi32>
          %1 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32> }
               : () -> tensor<3xi32>
          %2 = "tf.Transpose"(%arg0, %0)
               : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
          %3 = "tf.Transpose"(%2, %1)
               : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
          func.return %3 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(
          mlir_function,
          'test',
          specialize,
          vectorize=True,
          codegen_transpose=True)

      d0 = np.random.randint(1, 10)
      d1 = np.random.randint(1, 10)
      d2 = np.random.randint(1, 10)

      arg0 = np.random.uniform(0, 10.0, size=(d0, d1, d2)).astype(np.float32)

      [res] = jitrt.execute(compiled, [arg0])
      ref = np.transpose(np.transpose(arg0, (0, 2, 1)), (2, 1, 0))
      np.testing.assert_allclose(res, ref, atol=0.0)

  # Without value specialization, the below tf.Transpose won't compile because
  # the permutation vector must be statically shaped.
  def test_transpose_value_specialization_i32(self):
    mlir_function = """
      func.func @compute(%arg0: tensor<*xf32>,
                    %arg1: tensor<?xi32> {rt.constraint = "value"})
          -> tensor<*xf32> {
        %0 = "tf.Transpose"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
        func.return %0 : tensor<*xf32>
      }"""
    compiled = jitrt.compile(mlir_function, 'compute')
    tensor = np.random.uniform(0, 10.0, size=(3, 3)).astype(np.float32)
    perm0 = np.array([1, 0]).astype(np.int32)
    perm1 = np.array([0, 1]).astype(np.int32)

    # Test that the same compiled module with two different value-specialized
    # arguments is handled correctly, i.e. it is specialized twice.
    [res0] = jitrt.execute(compiled, [tensor, perm0])
    [res1] = jitrt.execute(compiled, [tensor, perm1])
    np.testing.assert_allclose(res0, np.transpose(tensor, perm0), atol=0.0)
    np.testing.assert_allclose(res1, np.transpose(tensor, perm1), atol=0.0)

  # Test value specialization of two i64 operands.
  def test_transpose_value_specialization_i64(self):
    mlir_function = """
      func.func @compute(%arg0: tensor<*xf32>,
                    %arg1: tensor<?xi64> {rt.constraint = "value"},
                    %arg2: tensor<?xi64> {rt.constraint = "value"})
          -> tensor<*xf32> {
        %0 = "tf.Transpose"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
        %1 = "tf.Transpose"(%0, %arg2)
             : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
        func.return %1 : tensor<*xf32>
      }"""
    compiled = jitrt.compile(mlir_function, 'compute')
    tensor = np.random.uniform(0, 10.0, size=(3, 3)).astype(np.float32)
    perm0 = np.array([1, 0]).astype(np.int64)
    perm1 = np.array([0, 1]).astype(np.int64)

    [res] = jitrt.execute(compiled, [tensor, perm0, perm1])
    np.testing.assert_allclose(
        res, np.transpose(np.transpose(tensor, perm0), perm1), atol=0.0)

  # Test that without the value constraint the function cannot compile
  # because the permutation vector is not statically shaped.
  def test_transpose_die_without_value_specialization(self):
    mlir_function = """
      func.func @compute(%arg0: tensor<*xf32>,
                    %arg1: tensor<?xi64>) -> tensor<*xf32> {
        %0 = "tf.Transpose"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
        func.return %0 : tensor<*xf32>
      }"""
    try:
      jitrt.compile(mlir_function, 'compute')
    except Exception:  # pylint: disable=broad-except
      return
    raise RuntimeError('Compilation should have failed')


if __name__ == '__main__':
  test.main()

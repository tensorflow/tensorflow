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

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

cpurt = tf_cpurt.TfCpurtExecutor()

specializations = [
    tf_cpurt.Specialization.ENABLED,
    tf_cpurt.Specialization.DISABLED,
    tf_cpurt.Specialization.ALWAYS,
]


class TfBinaryBcastTest(test.TestCase):

  def test_bcast_2d_1d(self):
    mlir_function = """
      func @test(%arg0: tensor<?x4xf32>,
                 %arg1: tensor<4xf32>,
                 %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
        %0 = "tf.Log1p"(%arg0)
             : (tensor<?x4xf32>) -> tensor<?x4xf32>
        %1 = "tf.Sub"(%0, %arg1)
             : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
        %2 = "tf.Mul"(%1, %arg2)
             : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
        %3 = "tf.Atan2"(%2, %arg2)
             : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
        return %3 : tensor<?x4xf32>
      }"""

    n = np.random.randint(1, 10)

    arg0 = np.random.uniform(0, 10.0, size=(n, 4)).astype(np.float32)
    arg1 = np.random.uniform(0, 10.0, size=(4)).astype(np.float32)
    arg2 = np.random.uniform(0, 10.0, size=(4)).astype(np.float32)

    for specialize in specializations:
      compiled = cpurt.compile(mlir_function, 'test', specialize)

      [res] = cpurt.execute(compiled, [arg0, arg1, arg2])
      ref = np.arctan2((np.log1p(arg0) - arg1) * arg2, arg2)
      np.testing.assert_allclose(res, ref, atol=1e-05)

  def test_bcast_2d_2d(self):
    mlir_function = """
      func @test(%arg0: tensor<?x?xf32>,
                 %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
        %0 = "tf.Mul"(%arg0, %arg1)
             : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
        return %0 : tensor<?x?xf32>
      }"""

    m = np.random.randint(1, 10)
    n = np.random.randint(1, 10)

    lhs0 = np.random.uniform(0, 10.0, size=(1, 1)).astype(np.float32)
    lhs1 = np.random.uniform(0, 10.0, size=(1, n)).astype(np.float32)
    lhs2 = np.random.uniform(0, 10.0, size=(m, 1)).astype(np.float32)
    lhs3 = np.random.uniform(0, 10.0, size=(m, n)).astype(np.float32)

    rhs0 = np.random.uniform(0, 10.0, size=(1, 1)).astype(np.float32)
    rhs1 = np.random.uniform(0, 10.0, size=(1, n)).astype(np.float32)
    rhs2 = np.random.uniform(0, 10.0, size=(m, 1)).astype(np.float32)
    rhs3 = np.random.uniform(0, 10.0, size=(m, n)).astype(np.float32)

    for specialize in specializations:
      compiled = cpurt.compile(mlir_function, 'test', specialize)

      for lhs in [lhs0, lhs1, lhs2, lhs3]:
        for rhs in [rhs0, rhs1, rhs2, rhs3]:
          [res] = cpurt.execute(compiled, [lhs, rhs])
          np.testing.assert_allclose(res, lhs * rhs, atol=1e-07)

  def test_bcast_2d_1d_0d(self):
    mlir_function = """
      func @compute(%arg0: tensor<?x4xf32>,
                    %arg1: tensor<4xf32>,
                    %arg2: tensor<f32>) -> tensor<?x4xf32> {
        %0 = "tf.AddV2"(%arg1, %arg2)
             : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
        %1 = "tf.AddV2"(%arg0, %0)
             : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
        %2 = "tf.AddV2"(%1, %0)
             : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
        return %2 : tensor<?x4xf32>
      }"""

    for specialize in specializations:
      compiled = cpurt.compile(mlir_function, 'compute', specialize)

      arg0 = np.random.uniform(0, 10.0, size=(1, 4)).astype(np.float32)
      arg1 = np.random.uniform(0, 10.0, size=(4)).astype(np.float32)
      arg2 = np.random.uniform(0, 10.0, size=()).astype(np.float32)

      [res] = cpurt.execute(compiled, [arg0, arg1, arg2])

      # Reference implementation with numpy
      t_0 = np.add(arg1, arg2)
      t_1 = np.add(arg0, t_0)
      t_2 = np.add(t_1, t_0)

      np.testing.assert_allclose(res, t_2, atol=0.0)

  def test_bcast_unranked_0d(self):
    mlir_function = """
      func @compute(%arg0: tensor<*xf32> {cpurt.constraint = "rank"},
                    %arg1: tensor<f32>) -> tensor<*xf32> {
        %0 = "tf.AddV2"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }"""

    compiled = cpurt.compile(mlir_function, 'compute')

    arg0 = np.random.uniform(0, 10.0, size=(4, 4)).astype(np.float32)
    arg1 = np.random.uniform(0, 10.0, size=()).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg0, arg1])

    np.testing.assert_allclose(res, np.add(arg0, arg1), atol=0.0)

  def test_bcast_unranked_unranked(self):
    mlir_function = """
      func @compute(%arg0: tensor<*xf32> {cpurt.constraint = "rank"},
                    %arg1: tensor<*xf32> {cpurt.constraint = "rank"})
          -> tensor<*xf32> {
        %0 = "tf.AddV2"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }"""

    compiled = cpurt.compile(mlir_function, 'compute')

    arg0 = np.random.uniform(0, 10.0, size=(1, 4)).astype(np.float32)
    arg1 = np.random.uniform(0, 10.0, size=(4, 1)).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg0, arg1])

    np.testing.assert_allclose(res, np.add(arg0, arg1), atol=0.0)

  # Test that the non-broadcastable shapes error is handled at run time.
  def test_bcast_1d_1d_error(self):
    mlir_function = """
      func @compute(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
          -> tensor<?xf32> {
        %0 = "tf.AddV2"(%arg0, %arg1)
             : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
        return %0 : tensor<?xf32>
      }"""

    arg0 = np.random.uniform(0, 10.0, size=(2)).astype(np.float32)
    arg1 = np.random.uniform(0, 10.0, size=(3)).astype(np.float32)

    for specialize in specializations:
      compiled = cpurt.compile(mlir_function, 'compute', specialize)

      with self.assertRaisesRegex(Exception, 'required broadcastable shapes'):
        cpurt.execute(compiled, [arg0, arg1])

  # Test that 0-ranked operands are correctly specialized.
  def test_bcast_value_rank0(self):
    mlir_function = """
      func @compute(%arg0: tensor<*xi32>,
                    %arg1: tensor<i32> {cpurt.constraint = "value"})
          -> tensor<*xi32> {
        %0 = "tf.AddV2"(%arg0, %arg1)
             : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
        return %0 : tensor<*xi32>
      }"""
    compiled = cpurt.compile(mlir_function, 'compute')
    # Test that the same compiled module with two different value-specialized
    # arguments is handled correctly.
    tensor = np.random.uniform(0, 10.0, size=(3)).astype(np.int32)
    rhs0 = np.random.uniform(0, 10.0, size=()).astype(np.int32)
    rhs1 = np.random.uniform(0, 10.0, size=()).astype(np.int32)
    [res0] = cpurt.execute(compiled, [tensor, rhs0])
    [res1] = cpurt.execute(compiled, [tensor, rhs1])
    np.testing.assert_allclose(res0, np.add(tensor, rhs0), atol=0.0)
    np.testing.assert_allclose(res1, np.add(tensor, rhs1), atol=0.0)

  # Test that the function does not compile when value-specializing an f32.
  def test_bcast_value_die_if_unsinkable(self):
    mlir_function = """
      func @compute(%arg0: tensor<*xf32>,
                    %arg1: tensor<f32> {cpurt.constraint = "value"})
          -> tensor<*xf32> {
        %0 = "tf.AddV2"(%arg0, %arg1)
             : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }"""

    with self.assertRaisesRegex(Exception,
                                'Cannot sink operand type: tensor<f32>'):
      cpurt.compile(mlir_function, 'compute')


if __name__ == '__main__':
  test.main()

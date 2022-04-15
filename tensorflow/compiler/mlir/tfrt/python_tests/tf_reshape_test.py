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


class TfReshapeTest(test.TestCase):

  def test_reshape_unknown_1d(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>)
            -> tensor<?x?xf32> {
          %0 = "tf.Reshape"(%arg0, %arg1)
              : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
          func.return %0 : tensor<?x?xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 10) * 2

      arg0 = np.random.uniform(0, 10.0, size=(d0)).astype(np.float32)

      shape = np.array([2, d0 / 2]).astype(np.int32)
      [res] = jitrt.execute(compiled, [arg0, shape])
      np.testing.assert_allclose(res, np.reshape(arg0, shape), atol=0.0)

      shape = np.array([2, -1]).astype(np.int32)
      [res] = jitrt.execute(compiled, [arg0, shape])
      np.testing.assert_allclose(res, np.reshape(arg0, shape), atol=0.0)

      with self.assertRaises(RuntimeError):
        shape = np.array([30, -1]).astype(np.int32)
        [res] = jitrt.execute(compiled, [arg0, shape])

  def test_reshape_unknown_2d(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?x?xf32>, %arg1: tensor<1xi32>)
            -> tensor<?xf32> {
          %0 = "tf.Reshape"(%arg0, %arg1)
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 10) * 2
      d1 = np.random.randint(1, 10) * 2

      arg0 = np.random.uniform(0, 10.0, size=(d0, d1)).astype(np.float32)

      shape = np.array([d0 * d1]).astype(np.int32)
      [res] = jitrt.execute(compiled, [arg0, shape])
      np.testing.assert_allclose(res, np.reshape(arg0, shape), atol=0.0)

      shape = np.array([-1]).astype(np.int32)
      [res] = jitrt.execute(compiled, [arg0, shape])
      np.testing.assert_allclose(res, np.reshape(arg0, shape), atol=0.0)

      with self.assertRaises(RuntimeError):
        shape = np.array([d0]).astype(np.int32)
        [res] = jitrt.execute(compiled, [arg0, shape])

  def test_reshape_zero_dim(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<1xi32>)
            -> tensor<?xf32> {
          %0 = "tf.Reshape"(%arg0, %arg1)
              : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test', specialize)

      empty = np.array([]).astype(np.float32)

      zero = np.array([0]).astype(np.int32)
      [res] = jitrt.execute(compiled, [empty, zero])
      np.testing.assert_equal(res.shape, [0])

      neg1 = np.array([-1]).astype(np.int32)
      [res] = jitrt.execute(compiled, [empty, neg1])
      np.testing.assert_equal(res.shape, [0])

      with self.assertRaises(RuntimeError):
        neg2 = np.array([-2]).astype(np.int32)
        [res] = jitrt.execute(compiled, [empty, neg2])

      with self.assertRaises(RuntimeError):
        one = np.array([1]).astype(np.int32)
        [res] = jitrt.execute(compiled, [empty, one])

  def test_reshape_zero_dim_3d(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<3xi32>)
            -> tensor<?x?x?xf32> {
          %0 = "tf.Const"() { value = dense<[3, 0, 5]> : tensor<3xi32> }
              : () -> tensor<3xi32>
          %1 = "tf.Reshape"(%arg0, %0)
              : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
          %2 = "tf.Reshape"(%1, %arg1)
              : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
          func.return %2 : tensor<?x?x?xf32>
        }"""

      compiled = jitrt.compile(mlir_function, 'test', specialize)

      empty = np.array([]).astype(np.float32)

      shape = np.array([3, 0, -1]).astype(np.int32)
      [res] = jitrt.execute(compiled, [empty, shape])
      # TODO(kramerb): This should be [3, 0, 5]
      np.testing.assert_equal(res.shape, [3, 0, 0])

      with self.assertRaises(RuntimeError):
        shape = np.array([3, -1, -1]).astype(np.int32)
        [res] = jitrt.execute(compiled, [empty, shape])

if __name__ == '__main__':
  test.main()

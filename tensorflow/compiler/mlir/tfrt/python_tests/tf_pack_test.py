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


class TfPackTest(test.TestCase):

  def pack_and_check(self, src, shape, dtype):
    compiled = cpurt.compile(src, 'test')

    arg0 = np.random.uniform(0, 10.0, size=shape).astype(dtype)
    arg1 = np.random.uniform(0, 10.0, size=shape).astype(dtype)

    [res] = cpurt.execute(compiled, [arg0, arg1])
    np.testing.assert_allclose(res, np.array([arg0, arg1]), atol=0.0)

  def test_pack_0d_f32(self):
    self.pack_and_check(
        """
      func @test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<2xf32> {
        %1 = "tf.Pack"(%arg0, %arg1) {axis = 0 : i64}
             : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
        return %1 : tensor<2xf32>
      }""", (), np.float32)

  def test_pack_0d_i32(self):
    self.pack_and_check(
        """
      func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<2xi32> {
        %1 = "tf.Pack"(%arg0, %arg1) {axis = 0 : i64}
             : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
        return %1 : tensor<2xi32>
      }""", (), np.int32)

  def test_pack_0d_i64(self):
    self.pack_and_check(
        """
      func @test(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<2xi64> {
        %1 = "tf.Pack"(%arg0, %arg1) {axis = 0 : i64}
             : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
        return %1 : tensor<2xi64>
      }""", (), np.int64)

  def test_pack_0d_i1(self):
    self.pack_and_check(
        """
      func @test(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<2xi1> {
        %1 = "tf.Pack"(%arg0, %arg1) {axis = 0 : i64}
             : (tensor<i1>, tensor<i1>) -> tensor<2xi1>
        return %1 : tensor<2xi1>
      }""", (), bool)

  def test_pack_1d_i32(self):
    self.pack_and_check(
        """
      func @test(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>)
         -> tensor<2x4xi32> {
        %1 = "tf.Pack"(%arg0, %arg1) {axis = 0 : i64}
             : (tensor<4xi32>, tensor<4xi32>) -> tensor<2x4xi32>
        return %1 : tensor<2x4xi32>
      }""", (4), np.int32)


if __name__ == '__main__':
  test.main()

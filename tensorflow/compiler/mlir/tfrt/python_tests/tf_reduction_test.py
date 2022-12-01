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

jitrt = tf_jitrt.TfJitRtExecutor()


class TfReductionTest(test.TestCase):

  def test_1d_sum_dynamic(self):
    mlir_function = """
        func.func @test(%input: tensor<?xf32>) -> tensor<f32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
          func.return %0 : tensor<f32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 5.0, size=(10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=0), atol=0.01)

  def test_1d_max_static(self):
    mlir_function = """
        func.func @test(%input: tensor<10xf32>) -> tensor<f32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Max"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<10xf32>, tensor<1xi32>) -> tensor<f32>
          func.return %0 : tensor<f32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 1.0, size=(10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.max(arg0, axis=0), atol=0.01)

  def test_1d_max_static_no_dims_to_reduce(self):
    mlir_function = """
        func.func @test(%input: tensor<10xf32>) -> tensor<10xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[]> : tensor<0xi32>}
             : () -> tensor<0xi32>
          %0 = "tf.Max"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<10xf32>, tensor<0xi32>) -> tensor<10xf32>
          func.return %0 : tensor<10xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1.0, 1.0, size=(10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, arg0, atol=0.01)

  def test_2d_row_max(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Max"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.max(arg0, axis=1), atol=0.01)

  def test_2d_row_min(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Min"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.min(arg0, axis=1), atol=0.01)

  def test_2d_row_sum(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=1), atol=0.01)

  def test_2d_row_prod(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Prod"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(
        res, np.prod(arg0, axis=1), rtol=3e-07, atol=0.01)

  def test_2d_column_mean(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Mean"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(
        res, np.mean(arg0, axis=1), rtol=3e-07, atol=0.01)

  def test_2d_row_any(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xi1>) -> tensor<?xi1> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Any"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
          func.return %0 : tensor<?xi1>
      }"""

    compiled = jitrt.compile(
        mlir_function, 'test', vectorize=True, legalize_i1_tensors=True)

    arg0 = np.random.choice(a=[False, True], size=(8, 10)).astype(np.bool)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_equal(res, np.any(arg0, axis=1))

  def test_2d_row_all(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xi1>) -> tensor<?xi1> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.All"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
          func.return %0 : tensor<?xi1>
      }"""

    compiled = jitrt.compile(
        mlir_function, 'test', vectorize=True, legalize_i1_tensors=True)

    arg0 = np.random.choice(a=[False, True], size=(40, 2)).astype(np.bool)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_equal(res, np.all(arg0, axis=1))

  def test_2d_row_sum_static(self):
    mlir_function = """
        func.func @test(%input: tensor<8x8xf32>) -> tensor<8xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<8x8xf32>, tensor<1xi32>) -> tensor<8xf32>
          func.return %0 : tensor<8xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 8)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=1), atol=1)

  def test_2d_column_sum(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=0), atol=0.01)

  def test_2d_column_sum_static(self):
    mlir_function = """
        func.func @test(%input: tensor<8x8xf32>) -> tensor<8xf32> {
          %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
             : () -> tensor<1xi32>
          %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
              : (tensor<8x8xf32>, tensor<1xi32>) -> tensor<8xf32>
          func.return %0 : tensor<8xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 8)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.sum(arg0, axis=0), atol=1)

  def test_2d_row_argmax(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?xi64> {
          %dim_to_reduce = "tf.Const"() {value = dense<1> : tensor<i32>}
             : () -> tensor<i32>
          %0 = "tf.ArgMax"(%input, %dim_to_reduce)
              : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xi64>
          func.return %0 : tensor<?xi64>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(0.0, 10.0, size=(8, 10)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_equal(res, np.argmax(arg0, axis=1))

if __name__ == '__main__':
  test.main()

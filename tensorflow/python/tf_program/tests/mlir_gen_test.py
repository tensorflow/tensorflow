# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `mlir_gen` module"""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.types import core
from tensorflow.python.tf_program.mlir_gen import mlir_gen

import tensorflow.compiler.mlir.python.mlir_wrapper.filecheck_wrapper as fw


class MLIRGenTestBase(test.TestCase):

  def _check_code(self, mlir_code, exp_mlir_code):
    return self.assertTrue(fw.check(str(mlir_code), exp_mlir_code))


class MLIRGenTest(MLIRGenTestBase):
  """MLIR Generation Tests for Tensorflow Program"""

  def test_simple(self):

    def test_fn():
      pass

    mlir_code = mlir_gen(test_fn)
    mlir_code_exp = r"""
      CHECK-LABEL: @test_fn
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_argument(self):

    def test_fn(x: core.Tensor) -> core.Tensor:
      return x

    mlir_code = mlir_gen(test_fn)
    mlir_code_exp = r"""
      CHECK-LABEL: @test_fn(%arg0: tensor<*xi32>) -> tensor<*xi32> {
        CHECK-NEXT: return %arg0 : tensor<*xi32>
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_constant(self):

    def test_fn() -> int:
      return 23

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn() -> i32
      CHECK: %[[r0:[0-9]+]] = "tf.Const"() {value = dense<23> : tensor<i32>} : () -> tensor<i32>
      CHECK: return %[[r0]] : tensor<i32>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_BoolOp(self):

    def test_fn(x: bool, y: bool) -> bool:
      return x or y or x and x and y

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn(%arg0: i1, %arg1: i1) -> i1
      CHECK: %[[r0:[0-9]+]] = "tfp.And"(%arg0, %arg0, %arg1) : (i1, i1, i1) -> tensor<*xi1>
      CHECK: %[[r1:[0-9]+]] = "tfp.Or"(%arg0, %arg1, %[[r0]]) : (i1, i1, tensor<*xi1>) -> tensor<*xi1>
      CHECK: return %[[r1]] : tensor<*xi1>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_Call(self):

    def test_fn():

      def f1():
        return 23

      def f2():
        return f1()

      f2()

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn()
        CHECK: "tf.LegacyCall"() {_disable_call_shape_inference = false, f = @f2} : () -> ()
      CHECK: }
      CHECK-LABEL: func @f1() {
        CHECK: %[[r0:[0-9]+]] = "tf.Const"() {value = dense<23> : tensor<i32>} : () -> tensor<i32>
        CHECK: return %[[r0]] : tensor<i32>
      CHECK: }
      CHECK-LABEL: func @f2() {
        CHECK: "tf.LegacyCall"() {_disable_call_shape_inference = false, f = @f1} : () -> ()
      }
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_Compare(self):

    def test_fn(x: core.Tensor, y: core.Tensor, z: core.Tensor):
      return x > y < z

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<*xi32>)
      CHECK: %[[r0:[0-9]+]] = "tf.Greater"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
      CHECK: %[[r1:[0-9]+]] = "tf.Less"(%[[r0]], %arg2) : (tensor<*xi1>, tensor<*xi32>) -> tensor<*xi1>
      CHECK: return %[[r1]] : tensor<*xi1>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_Assign_BinOp(self):

    def test_fn() -> int:
      y = 12 + 23 - 24
      return y

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn() -> i32
      CHECK: %[[r0:[0-9]+]] = "tf.AddV2"(%{{[0-9]+}}, %{{[0-9]+}}) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      CHECK: %[[r1:[0-9]+]] = "tf.Sub"(%{{[0-9]+}}, %{{[0-9]+}}) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      CHECK: return %[[r1]] : tensor<i32>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_if(self):

    def test_fn(x: core.Tensor) -> int:
      res = 0
      if x > 0:
        res = 1
      elif x < 0:
        res = -1
      else:
        res = 0
      return res

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn(%arg0: tensor<*xi32>) -> i32

      CHECK: %[[r1:[0-9]+]] = "tf.Greater"(%arg0, %{{[0-9]+}}) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      CHECK-NEXT: %[[r2:[0-9]+]] = "tfp.If"(%[[r1]]) ( {
        CHECK: return %{{[0-9]+}} : tensor<i32>
      CHECK-NEXT: },  {
        CHECK: %[[r3:[0-9]+]] = "tf.Less"(%arg0, %{{[0-9]+}}) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
        CHECK: %[[r4:[0-9]+]] = "tfp.If"(%[[r3]]) ( {
          CHECK: %[[r5:[0-9]+]] = "tf.Neg"(%{{[0-9]+}}) : (tensor<i32>) -> tensor<i32>
          CHECK: return %[[r5]] : tensor<i32>
        CHECK-NEXT: },  {
          CHECK: return %{{[0-9]+}} : tensor<i32>
        CHECK-NEXT: }) : (tensor<*xi1>) -> tensor<i32>
        CHECK: return %[[r4]] : tensor<i32>
      CHECK-NEXT: }) : (tensor<*xi1>) -> tensor<i32>
      CHECK-NEXT: return %[[r2]] : tensor<i32>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_while(self):

    def test_fn(x: core.Tensor) -> core.Tensor:
      s = 0
      while x > 0:
        s = s + x
      return s

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: func @test_fn(%arg0: tensor<*xi32>) -> tensor<*xi32>

      CHECK: %[[r1:[0-9]+]] = "tfp.While"(%0) ( {
      CHECK-NEXT: ^{{[^ ]+}}(%arg1: tensor<i32>):
        CHECK: %[[r2:[0-9]+]] = "tf.Greater"(%arg0, %{{[0-9]+}}) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
        CHECK-NEXT: return %[[r2]] : tensor<*xi1>
      CHECK-NEXT: },  {
      CHECK-NEXT: ^{{[^ ]+}}(%arg1: tensor<i32>):
        CHECK: %[[r3:[0-9]+]] = "tf.AddV2"(%arg1, %arg0) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi32>
        CHECK-NEXT: return %[[r3]] : tensor<*xi32>
      CHECK-NEXT: }) : (tensor<i32>) -> tensor<i32>
      CHECK-NEXT: return %[[r1]] : tensor<i32>
    """
    self._check_code(mlir_code, exp_mlir_code)

  def test_fibonacci(self):

    def test_fn(x: core.Tensor) -> core.Tensor:
      res, idx = 0, 2
      a, b = 0, 1
      if x == 0 or x == 1:
        res = x
      else:
        while idx <= x:
          res = a + b
          a = b
          b = res
          idx = idx + 1
      return res

    mlir_code = mlir_gen(test_fn)
    exp_mlir_code = r"""
      CHECK-LABEL: @test_fn(%arg0: tensor<*xi32>) -> tensor<*xi32>
      CHECK: %[[r5:[0-9]+]] = "tf.Equal"(%arg0, %{{[0-9]+}}) {incompatible_shape_error = true} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      CHECK: %[[r7:[0-9]+]] = "tf.Equal"(%arg0, %{{[0-9]+}}) {incompatible_shape_error = true} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      CHECK: %[[r8:[0-9]+]] = "tfp.Or"(%[[r5]], %[[r7]]) : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>

      CHECK: %[[r9:[0-9]+]]:4 = "tfp.If"(%[[r8]]) ( {
        CHECK-NEXT: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<{{(\*x)?}}i32>, tensor<{{(\*x)?}}i32>, tensor<{{(\*x)?}}i32>, tensor<{{(\*x)?}}i32>
        CHECK-NEXT: },  {
        CHECK-NEXT: %[[r10:[0-9]+]]:4 = "tfp.While"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
          CHECK-NEXT: ^{{[^ ]*}}(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
          CHECK-NEXT: %[[r11:[0-9]+]] = "tf.LessEqual"(%arg{{[0-9]+}}, %arg{{[0-9]+}}) : (tensor<{{(\*x)?}}i32>, tensor<{{(\*x)?}}i32>) -> tensor<*xi1>
          CHECK-NEXT: return %[[r11]] : tensor<*xi1>
        CHECK-NEXT: },  {
          CHECK-NEXT: ^{{[^ ]*}}(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
          CHECK-NEXT: %[[r12:[0-9]+]] = "tf.AddV2"(%arg{{[0-9]+}}, %arg{{[0-9]+}}) : (tensor<i32>, tensor<i32>) -> tensor<i32>
          CHECK: %[[r13:[0-9]+]] = "tf.AddV2"(%arg{{[0-9]+}}, %{{[0-9]+}}) : (tensor<i32>, tensor<i32>) -> tensor<i32>
          CHECK-NEXT: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
        CHECK-NEXT: }) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)
        CHECK-NEXT: return %[[r10]]#{{[0-9]+}}, %[[r10]]#{{[0-9]+}}, %[[r10]]#{{[0-9]+}}, %[[r10]]#{{[0-9]+}} : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
      CHECK-NEXT: }) : (tensor<*xi1>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)
      CHECK-NEXT: return %[[r9]]#{{[0-9]+}} : tensor<i32>
    """
    self._check_code(mlir_code, exp_mlir_code)


if __name__ == '__main__':
  test.main()

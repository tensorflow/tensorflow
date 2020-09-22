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
"""Tests for functional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FunctionalOpsTest(test.TestCase):

  def testIfWithDefun(self):
    # Defun should only be used in graph mode
    with ops.Graph().as_default():
      @function.Defun(dtypes.float32)
      def Then(x):
        return x + 1

      @function.Defun(dtypes.float32)
      def Else(x):
        return x - 1

      inputs = [10.]
      result = self.evaluate(functional_ops.If(False, inputs, Then, Else))
      self.assertEqual([9.0], result)

  def testIfWithFunction(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Then(x):
      return x + 1

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Else(x):
      return x - 1

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = self.evaluate(functional_ops.If(False, inputs, then_cf, else_cf))
    self.assertEqual([9.0], result)

  def testIfWithFunctionComposite(self):

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    @def_function.function(input_signature=signature)
    def Then(x):
      return sparse_tensor.SparseTensor([[0]], [x + 1], [1])

    @def_function.function(input_signature=signature)
    def Else(x):
      return sparse_tensor.SparseTensor([[0]], [x - 1], [1])

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = functional_ops.If(False, inputs, then_cf, else_cf)
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual([9.0], result.values)

  def testWhileWithDefun(self):
    # Defun should only be used in graph mode
    with ops.Graph().as_default():
      @function.Defun(dtypes.int32)
      def Body(n):
        return n - 1

      @function.Defun(dtypes.int32)
      def Cond(n):
        return math_ops.reduce_min(n) > 0

      n = constant_op.constant([2])
      result = self.evaluate(functional_ops.While([n], Cond, Body))
      self.assertAllEqual([[0]], result)

  def testWhileWithFunction(self):

    @def_function.function
    def Body(n):
      return n - 1

    @def_function.function
    def Cond(n):
      return math_ops.reduce_min(n) > 0

    n = constant_op.constant([2])
    cond_cf = Cond.get_concrete_function(n)
    body_cf = Body.get_concrete_function(n)
    result = functional_ops.While([n], cond_cf, body_cf)
    self.assertAllEqual([0], result)

  def testWhileWithFunctionComposite(self):

    @def_function.function
    def Body(n):
      return sparse_tensor.SparseTensor([[0]], n.values - 1, [1])

    @def_function.function
    def Cond(n):
      return math_ops.reduce_min(n.values) > 0

    n = constant_op.constant([2])
    n = sparse_tensor.SparseTensor([[0]], [2], [1])
    cond_cf = Cond.get_concrete_function(n)
    body_cf = Body.get_concrete_function(n)
    result = functional_ops.While([n], cond_cf, body_cf)
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual([0], result.values)


if __name__ == '__main__':
  test.main()

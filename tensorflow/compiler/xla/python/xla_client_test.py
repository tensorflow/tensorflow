# Lint as: python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the Python extension-based XLA client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import threading

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import custom_call_for_test
from tensorflow.compiler.xla.python import xla_client

bfloat16 = xla_client.bfloat16


class ComputationTest(absltest.TestCase):
  """Base class for running an XLA Computation through the local client."""

  def _NewComputation(self, name=None):
    if name is None:
      name = self.id()
    return xla_client.ComputationBuilder(name)

  def _Execute(self, c, arguments):
    compiled_c = c.Build().Compile()
    return xla_client.execute_with_python_values(compiled_c, arguments)

  def _ExecuteAndAssertWith(self, assert_func, c, arguments, expected):
    assert expected is not None
    result = self._Execute(c, arguments)
    # Numpy's comparison methods are a bit too lenient by treating inputs as
    # "array-like", meaning that scalar 4 will be happily compared equal to
    # [[4]]. We'd like to be more strict so assert shapes as well.
    self.assertEqual(np.asanyarray(result).shape, np.asanyarray(expected).shape)
    assert_func(result, expected)

  def _ExecuteAndCompareExact(self, c, arguments=(), expected=None):
    self._ExecuteAndAssertWith(np.testing.assert_equal, c, arguments, expected)

  def _ExecuteAndCompareClose(self,
                              c,
                              arguments=(),
                              expected=None,
                              rtol=1e-7,
                              atol=0):
    self._ExecuteAndAssertWith(
        functools.partial(np.testing.assert_allclose, rtol=rtol, atol=atol), c,
        arguments, expected)


def NumpyArrayF32(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.float32 dtype."""
  return np.array(*args, dtype=np.float32, **kwargs)


def NumpyArrayF64(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.float64 dtype."""
  return np.array(*args, dtype=np.float64, **kwargs)


def NumpyArrayS32(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.int32 dtype."""
  return np.array(*args, dtype=np.int32, **kwargs)


def NumpyArrayS64(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.int64 dtype."""
  return np.array(*args, dtype=np.int64, **kwargs)


def NumpyArrayBool(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.bool dtype."""
  return np.array(*args, dtype=np.bool, **kwargs)


class ComputationPrinting(absltest.TestCase):

  def ExampleComputation(self):
    builder = xla_client.ComputationBuilder("acomputation")
    p0 = builder.ParameterFromNumpy(np.float32(0))
    p1 = builder.ParameterFromNumpy(np.zeros((4,), np.float32))
    builder.Mul(p0, p1)
    return builder.Build()

  def testComputationToHloText(self):
    computation = self.ExampleComputation()
    hlo_text = computation.GetHloText()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))

  def testComputationToHloGraph(self):
    computation = self.ExampleComputation()
    hlo_dot_graph = computation.GetHloDotGraph()
    self.assertTrue(hlo_dot_graph.startswith("digraph "))


class ComputationHashTest(absltest.TestCase):

  def testHash(self):
    builder0 = xla_client.ComputationBuilder("computation0")
    p0 = builder0.ParameterFromNumpy(np.float32(0))
    p1 = builder0.ParameterFromNumpy(np.zeros((4,), np.float32))
    builder0.Mul(p0, p1)
    computation0 = builder0.Build()

    builder1 = xla_client.ComputationBuilder("computation1")
    p0 = builder1.ParameterFromNumpy(np.float32(0))
    p1 = builder1.ParameterFromNumpy(np.zeros((4,), np.float32))
    builder1.Mul(p0, p1)
    computation1 = builder1.Build()

    self.assertEqual(computation0.Hash(), computation1.Hash())


class ComputationsWithConstantsTest(ComputationTest):
  """Tests focusing on Constant ops."""

  def testConstantScalarSumS8(self):
    c = self._NewComputation()
    c.Add(c.Constant(np.int8(1)), c.Constant(np.int8(2)))
    self._ExecuteAndCompareExact(c, expected=np.int8(3))

  def testConstantScalarSumBF16(self):
    c = self._NewComputation()
    c.Add(c.Constant(bfloat16(1.11)), c.Constant(bfloat16(3.14)))
    self._ExecuteAndCompareClose(c, expected=bfloat16(4.25))

  def testConstantScalarSumF32(self):
    c = self._NewComputation()
    c.Add(c.ConstantF32Scalar(1.11), c.ConstantF32Scalar(3.14))
    self._ExecuteAndCompareClose(c, expected=4.25)

  def testConstantScalarSumF64(self):
    c = self._NewComputation()
    c.Add(c.ConstantF64Scalar(1.11), c.ConstantF64Scalar(3.14))
    self._ExecuteAndCompareClose(c, expected=4.25)

  def testConstantScalarSumS32(self):
    c = self._NewComputation()
    c.Add(c.ConstantS32Scalar(1), c.ConstantS32Scalar(2))
    self._ExecuteAndCompareClose(c, expected=3)

  def testConstantScalarSumS64(self):
    c = self._NewComputation()
    c.Add(c.ConstantS64Scalar(1), c.ConstantS64Scalar(2))
    self._ExecuteAndCompareClose(c, expected=3)

  def testConstantVectorMulF16(self):
    c = self._NewComputation()
    c.Mul(
        c.Constant(np.array([2.5, 3.3, -1.2, 0.7], np.float16)),
        c.Constant(np.array([-1.2, 2, -2, -3], np.float16)))
    self._ExecuteAndCompareClose(
        c, expected=np.array([-3, 6.6, 2.4, -2.1], np.float16), rtol=2e-3)

  def testConstantVectorMulF32(self):
    c = self._NewComputation()
    c.Mul(
        c.Constant(NumpyArrayF32([2.5, 3.3, -1.2, 0.7])),
        c.Constant(NumpyArrayF32([-1.2, 2, -2, -3])))
    self._ExecuteAndCompareClose(c, expected=[-3, 6.6, 2.4, -2.1])

  def testConstantVectorMulF64(self):
    c = self._NewComputation()
    c.Mul(
        c.Constant(NumpyArrayF64([2.5, 3.3, -1.2, 0.7])),
        c.Constant(NumpyArrayF64([-1.2, 2, -2, -3])))
    self._ExecuteAndCompareClose(c, expected=[-3, 6.6, 2.4, -2.1])

  def testConstantVectorScalarDivF32(self):
    c = self._NewComputation()
    c.Div(
        c.Constant(NumpyArrayF32([1.5, 2.5, 3.0, -10.8])),
        c.ConstantF32Scalar(2.0))
    self._ExecuteAndCompareClose(c, expected=[0.75, 1.25, 1.5, -5.4])

  def testConstantVectorScalarDivF64(self):
    c = self._NewComputation()
    c.Div(
        c.Constant(NumpyArrayF64([1.5, 2.5, 3.0, -10.8])),
        c.ConstantF64Scalar(2.0))
    self._ExecuteAndCompareClose(c, expected=[0.75, 1.25, 1.5, -5.4])

  def testConstantVectorScalarPowF32(self):
    c = self._NewComputation()
    c.Pow(c.Constant(NumpyArrayF32([1.5, 2.5, 3.0])), c.ConstantF32Scalar(2.))
    self._ExecuteAndCompareClose(c, expected=[2.25, 6.25, 9.])

  def testConstantVectorScalarPowF64(self):
    c = self._NewComputation()
    c.Pow(c.Constant(NumpyArrayF64([1.5, 2.5, 3.0])), c.ConstantF64Scalar(2.))
    self._ExecuteAndCompareClose(c, expected=[2.25, 6.25, 9.])

  def testIota(self):
    c = self._NewComputation()
    c.Iota(np.float32, 10)
    self._ExecuteAndCompareExact(c, expected=np.arange(10, dtype=np.float32))

  def testBroadcastedIota(self):
    c = self._NewComputation()
    c.BroadcastedIota(np.int64, (2, 3), 1)
    expected = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    self._ExecuteAndCompareExact(c, expected=expected)

  def testBooleanAnd(self):
    c = self._NewComputation()
    c.And(
        c.Constant(NumpyArrayBool([True, False, True, False])),
        c.Constant(NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[True, False, False, False])

  def testBooleanOr(self):
    c = self._NewComputation()
    c.Or(
        c.Constant(NumpyArrayBool([True, False, True, False])),
        c.Constant(NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[True, True, True, False])

  def testBooleanXor(self):
    c = self._NewComputation()
    c.Xor(
        c.Constant(NumpyArrayBool([True, False, True, False])),
        c.Constant(NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[False, True, True, False])

  def testSum2DF32(self):
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF32([[1, 2, 3], [4, 5, 6]])),
        c.Constant(NumpyArrayF32([[1, -1, 1], [-1, 1, -1]])))
    self._ExecuteAndCompareClose(c, expected=[[2, 1, 4], [3, 6, 5]])

  def testShiftLeft(self):
    c = self._NewComputation()
    c.ShiftLeft(c.Constant(NumpyArrayS32([3])), c.Constant(NumpyArrayS32([2])))
    self._ExecuteAndCompareClose(c, expected=[12])

  def testShiftRightArithmetic(self):
    c = self._NewComputation()
    c.ShiftRightArithmetic(
        c.Constant(NumpyArrayS32([-2])), c.Constant(NumpyArrayS32([1])))
    self._ExecuteAndCompareClose(c, expected=[-1])

  def testShiftRightLogical(self):
    c = self._NewComputation()
    c.ShiftRightLogical(
        c.Constant(NumpyArrayS32([-1])), c.Constant(NumpyArrayS32([1])))
    self._ExecuteAndCompareClose(c, expected=[2**31 - 1])

  def testSum2DF64(self):
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF64([[1, 2, 3], [4, 5, 6]])),
        c.Constant(NumpyArrayF64([[1, -1, 1], [-1, 1, -1]])))
    self._ExecuteAndCompareClose(c, expected=[[2, 1, 4], [3, 6, 5]])

  def testSum2DWith1DBroadcastDim0F32(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 0 to match the former's shape.
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF32([10, 20, 30])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareClose(
        c, expected=[[11, 12, 13], [24, 25, 26], [37, 38, 39]])

  def testSum2DWith1DBroadcastDim0F64(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 0 to match the former's shape.
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF64([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF64([10, 20, 30])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareClose(
        c, expected=[[11, 12, 13], [24, 25, 26], [37, 38, 39]])

  def testSum2DWith1DBroadcastDim1F32(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 1 to match the former's shape.
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF32([10, 20, 30])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareClose(
        c, expected=[[11, 22, 33], [14, 25, 36], [17, 28, 39]])

  def testSum2DWith1DBroadcastDim1F64(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 1 to match the former's shape.
    c = self._NewComputation()
    c.Add(
        c.Constant(NumpyArrayF64([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF64([10, 20, 30])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareClose(
        c, expected=[[11, 22, 33], [14, 25, 36], [17, 28, 39]])

  def testConstantAxpyF32(self):
    c = self._NewComputation()
    c.Add(
        c.Mul(
            c.ConstantF32Scalar(2),
            c.Constant(NumpyArrayF32([2.2, 3.3, 4.4, 5.5]))),
        c.Constant(NumpyArrayF32([100, -100, 200, -200])))
    self._ExecuteAndCompareClose(c, expected=[104.4, -93.4, 208.8, -189])

  def testConstantAxpyF64(self):
    c = self._NewComputation()
    c.Add(
        c.Mul(
            c.ConstantF64Scalar(2),
            c.Constant(NumpyArrayF64([2.2, 3.3, 4.4, 5.5]))),
        c.Constant(NumpyArrayF64([100, -100, 200, -200])))
    self._ExecuteAndCompareClose(c, expected=[104.4, -93.4, 208.8, -189])

  def testCustomCall(self):
    c = self._NewComputation()
    for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
      xla_client.register_custom_call_target(name, fn, platform="cpu")
    c.CustomCall(
        b"test_subtract_f32",
        operands=(c.ConstantF32Scalar(1.25), c.ConstantF32Scalar(0.5)),
        shape_with_layout=xla_client.Shape.array_shape(
            np.dtype(np.float32), (), ()),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(np.dtype(np.float32), (), ()),
            xla_client.Shape.array_shape(np.dtype(np.float32), (), ()),
        ))
    self._ExecuteAndCompareClose(c, expected=0.75)


class ParametersTest(ComputationTest):
  """Tests focusing on Parameter ops and argument-passing."""

  def setUp(self):
    self.f32_scalar_2 = NumpyArrayF32(2.0)
    self.f32_4vector = NumpyArrayF32([-2.3, 3.3, -4.3, 5.3])
    self.f64_scalar_2 = NumpyArrayF64(2.0)
    self.f64_4vector = NumpyArrayF64([-2.3, 3.3, -4.3, 5.3])
    self.s32_scalar_3 = NumpyArrayS32(3)
    self.s32_4vector = NumpyArrayS32([10, 15, -2, 7])
    self.s64_scalar_3 = NumpyArrayS64(3)
    self.s64_4vector = NumpyArrayS64([10, 15, -2, 7])

  def testScalarTimesVectorAutonumberF32(self):
    c = self._NewComputation()
    p0 = c.ParameterFromNumpy(self.f32_scalar_2)
    p1 = c.ParameterFromNumpy(self.f32_4vector)
    c.Mul(p0, p1)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f32_scalar_2, self.f32_4vector],
        expected=[-4.6, 6.6, -8.6, 10.6])

  def testScalarTimesVectorAutonumberF64(self):
    c = self._NewComputation()
    p0 = c.ParameterFromNumpy(self.f64_scalar_2)
    p1 = c.ParameterFromNumpy(self.f64_4vector)
    c.Mul(p0, p1)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f64_scalar_2, self.f64_4vector],
        expected=[-4.6, 6.6, -8.6, 10.6])

  def testScalarTimesVectorS32(self):
    c = self._NewComputation()
    p0 = c.ParameterFromNumpy(self.s32_scalar_3)
    p1 = c.ParameterFromNumpy(self.s32_4vector)
    c.Mul(p0, p1)
    self._ExecuteAndCompareExact(
        c,
        arguments=[self.s32_scalar_3, self.s32_4vector],
        expected=[30, 45, -6, 21])

  def testScalarTimesVectorS64(self):
    c = self._NewComputation()
    p0 = c.ParameterFromNumpy(self.s64_scalar_3)
    p1 = c.ParameterFromNumpy(self.s64_4vector)
    c.Mul(p0, p1)
    self._ExecuteAndCompareExact(
        c,
        arguments=[self.s64_scalar_3, self.s64_4vector],
        expected=[30, 45, -6, 21])

  def testScalarMinusVectorExplicitNumberingF32(self):
    # Use explicit numbering and pass parameter_num first. Sub is used since
    # it's not commutative and can help catch parameter reversal within the
    # computation.
    c = self._NewComputation()
    p1 = c.ParameterFromNumpy(self.f32_4vector, parameter_num=1)
    p0 = c.ParameterFromNumpy(self.f32_scalar_2, parameter_num=0)
    c.Sub(p1, p0)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f32_scalar_2, self.f32_4vector],
        expected=[-4.3, 1.3, -6.3, 3.3])

  def testScalarMinusVectorExplicitNumberingF64(self):
    # Use explicit numbering and pass parameter_num first. Sub is used since
    # it's not commutative and can help catch parameter reversal within the
    # computation.
    c = self._NewComputation()
    p1 = c.ParameterFromNumpy(self.f64_4vector, parameter_num=1)
    p0 = c.ParameterFromNumpy(self.f64_scalar_2, parameter_num=0)
    c.Sub(p1, p0)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f64_scalar_2, self.f64_4vector],
        expected=[-4.3, 1.3, -6.3, 3.3])


class BufferTest(ComputationTest):
  """Tests focusing on execution with Buffers."""

  def _Execute(self, c, arguments):
    compiled_c = c.Build().Compile()
    arg_buffers = [xla_client.Buffer.from_pyval(arg) for arg in arguments]
    result_buffer = compiled_c.Execute(arg_buffers)
    return result_buffer.to_py()

  def testConstantSum(self):
    c = self._NewComputation()
    c.Add(c.ConstantF32Scalar(1.11), c.ConstantF32Scalar(3.14))
    self._ExecuteAndCompareClose(c, expected=4.25)

  def testOneParameterSum(self):
    c = self._NewComputation()
    c.Add(c.ParameterFromNumpy(NumpyArrayF32(0.)), c.ConstantF32Scalar(3.14))
    self._ExecuteAndCompareClose(
        c, arguments=[NumpyArrayF32(1.11)], expected=4.25)

  def testTwoParameterSum(self):
    c = self._NewComputation()
    c.Add(
        c.ParameterFromNumpy(NumpyArrayF32(0.)),
        c.ParameterFromNumpy(NumpyArrayF32(0.)))
    self._ExecuteAndCompareClose(
        c, arguments=[NumpyArrayF32(1.11),
                      NumpyArrayF32(3.14)], expected=4.25)

  def testCannotCallWithDeletedBuffers(self):
    c = self._NewComputation()
    c.Add(c.ParameterFromNumpy(NumpyArrayF32(0.)), c.ConstantF32Scalar(3.14))
    arg = NumpyArrayF32(1.11)
    compiled_c = c.Build().Compile()
    arg_buffer = xla_client.Buffer.from_pyval(arg)
    arg_buffer.delete()
    with self.assertRaises(RuntimeError):
      compiled_c.Execute([arg_buffer])

  def testDestructureTupleEmpty(self):
    t = ()
    local_buffer = xla_client.Buffer.from_pyval(t)
    pieces = local_buffer.destructure()
    self.assertFalse(local_buffer.is_deleted())
    self.assertEmpty(pieces)

  def testDestructureTupleOneArrayElement(self):
    t = (np.array([1, 2, 3, 4], dtype=np.int32),)
    local_buffer = xla_client.Buffer.from_pyval(t)
    pieces = local_buffer.destructure()
    self.assertFalse(local_buffer.is_deleted())
    self.assertLen(pieces, 1)
    array = pieces[0]
    got = array.to_py()
    want = NumpyArrayS32([1, 2, 3, 4])
    np.testing.assert_equal(want, got)

  def testDestructureTupleTwoArrayElementDifferentType(self):
    t = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([2, 3, 4, 5], dtype=np.int32),
    )
    local_buffer = xla_client.Buffer.from_pyval(t)
    # Run the test twice to verify that the original tuple buffer remains valid
    # even after destructuring.
    for _ in range(2):
      pieces = local_buffer.destructure()
      self.assertFalse(local_buffer.is_deleted())
      self.assertLen(pieces, 2)
      array0, array1 = pieces
      got = array0.to_py()
      want = NumpyArrayF32([1.0, 2.0, 3.0, 4.0])
      np.testing.assert_equal(want, got)
      got = array1.to_py()
      want = NumpyArrayS32([2, 3, 4, 5])
      np.testing.assert_equal(want, got)

  def testDestructureTupleNested(self):
    t = ((NumpyArrayF32([1.0, 2.0]), NumpyArrayS32([3, 4])), NumpyArrayS32([5]))
    local_buffer = xla_client.Buffer.from_pyval(t)
    pieces = local_buffer.destructure()
    self.assertFalse(local_buffer.is_deleted())
    self.assertLen(pieces, 2)
    tuple0, array1 = pieces
    got = array1.to_py()
    want = NumpyArrayS32([5])
    np.testing.assert_equal(want, got)
    got = tuple0.to_py()
    self.assertEqual(type(got), tuple)
    self.assertLen(got, 2)
    np.testing.assert_equal(NumpyArrayF32([1.0, 2.0]), got[0])
    np.testing.assert_equal(NumpyArrayS32([3, 4]), got[1])

  def testMakeTuple(self):
    t = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([2, 3, 4, 5], dtype=np.int32),
    )
    b0 = xla_client.Buffer.from_pyval(t[0])
    b1 = xla_client.Buffer.from_pyval(t[1])
    device = xla_client.get_local_backend().local_devices()[0]
    btup = xla_client.Buffer.make_tuple([b0, b1], device=device)
    pieces = btup.destructure()
    self.assertLen(pieces, 2)
    array0, array1 = pieces
    np.testing.assert_equal(
        np.array([1, 2, 3, 4], dtype=np.float32), array0.to_py())
    np.testing.assert_equal(
        np.array([2, 3, 4, 5], dtype=np.int32), array1.to_py())

  def testShape(self):
    pyval = np.array([[1., 2.]], np.float32)
    local_buffer = xla_client.Buffer.from_pyval(pyval)
    xla_shape = local_buffer.shape()
    self.assertEqual(xla_shape.dimensions(), (1, 2))
    self.assertEqual(np.dtype(xla_shape.element_type()), np.dtype(np.float32))

  def testTupleShape(self):
    t = (
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        np.array([2, 3, 4, 5], dtype=np.int32),
    )
    b0 = xla_client.Buffer.from_pyval(t[0])
    b1 = xla_client.Buffer.from_pyval(t[1])
    device = xla_client.get_local_backend().local_devices()[0]
    tuple_buffer = xla_client.Buffer.make_tuple([b0, b1], device=device)
    tuple_shape = tuple_buffer.shape()
    self.assertEqual(tuple_shape.leaf_count(), 2)
    shapes = tuple_shape.tuple_shapes()
    self.assertLen(shapes, 2)
    shape1, shape2 = shapes
    self.assertEqual(shape1.dimensions(), (1, 4))
    self.assertEqual(shape2.dimensions(), (4,))

  def testBlockHostUntilReadyWorks(self):
    arg = np.array([[1., 2.]], np.float32)
    arg_buffer = xla_client.Buffer.from_pyval(arg)
    arg_buffer.block_host_until_ready()
    # This test merely checks that nothing goes awry when we call
    # block_host_until_ready(); it's difficult to test anything else.

  def testCopyToHost(self):
    arg0 = np.array([[1., 2.]], np.float32)
    arg1 = np.array([[3., 4.]], np.float32)
    arg0_buffer = xla_client.Buffer.from_pyval(arg0)
    arg1_buffer = xla_client.Buffer.from_pyval(arg1)
    # Prefetch two buffers using copy_to_host_async, and then retrieve their
    # values using to_py.
    arg0_buffer.copy_to_host_async()
    arg0_buffer.copy_to_host_async()  # Duplicate calls don't do anything.
    arg1_buffer.copy_to_host_async()
    np.testing.assert_equal(arg0, arg0_buffer.to_py())
    np.testing.assert_equal(arg1, arg1_buffer.to_py())
    # copy_to_host_async does nothing after to_py is called.
    arg0_buffer.copy_to_host_async()
    np.testing.assert_equal(arg0, arg0_buffer.to_py())

  def testDevice(self):
    x = np.arange(8)
    for device in xla_client.get_local_backend().local_devices():
      buf = xla_client.Buffer.from_pyval(x, device=device)
      self.assertEqual(buf.device(), device)
      np.testing.assert_equal(x, buf.to_py())


class SingleOpTest(ComputationTest):
  """Tests for single ops.

  The goal here is smoke testing - to exercise the most basic functionality of
  single XLA ops. As minimal as possible number of additional ops are added
  around the op being tested.
  """

  def testConcatenateF32(self):
    c = self._NewComputation()
    args = (
        c.Constant(NumpyArrayF32([1.0, 2.0, 3.0])),
        c.Constant(NumpyArrayF32([4.0, 5.0, 6.0])),
    )
    c.Concatenate(args, dimension=0)
    self._ExecuteAndCompareClose(c, expected=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

  def testConcatenateF64(self):
    c = self._NewComputation()
    args = (
        c.Constant(NumpyArrayF64([1.0, 2.0, 3.0])),
        c.Constant(NumpyArrayF64([4.0, 5.0, 6.0])),
    )
    c.Concatenate(args, dimension=0)
    self._ExecuteAndCompareClose(c, expected=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

  def testConvertElementType(self):
    xla_types = {
        np.bool: xla_client.PrimitiveType.PRED,
        np.int32: xla_client.PrimitiveType.S32,
        np.int64: xla_client.PrimitiveType.S64,
        np.float32: xla_client.PrimitiveType.F32,
        np.float64: xla_client.PrimitiveType.F64,
    }

    def _ConvertAndTest(template, src_dtype, dst_dtype):
      c = self._NewComputation()
      x = c.Constant(np.array(template, dtype=src_dtype))
      c.ConvertElementType(x, xla_types[dst_dtype])

      result = xla_client.execute_with_python_values(c.Build().Compile())
      expected = np.array(template, dtype=dst_dtype)

      self.assertEqual(result.shape, expected.shape)
      self.assertEqual(result.dtype, expected.dtype)
      np.testing.assert_equal(result, expected)

    x = [0, 1, 0, 0, 1]
    for src_dtype, dst_dtype in itertools.product(xla_types, xla_types):
      _ConvertAndTest(x, src_dtype, dst_dtype)

  def testBitcastConvertType(self):
    xla_x32_types = {
        np.int32: xla_client.PrimitiveType.S32,
        np.float32: xla_client.PrimitiveType.F32,
    }

    xla_x64_types = {
        np.int64: xla_client.PrimitiveType.S64,
        np.float64: xla_client.PrimitiveType.F64,
    }

    def _ConvertAndTest(template, src_dtype, dst_dtype, dst_etype):
      c = self._NewComputation()
      x = c.Constant(np.array(template, dtype=src_dtype))
      c.BitcastConvertType(x, dst_etype)

      result = xla_client.execute_with_python_values(c.Build().Compile())
      expected = np.array(template, src_dtype).view(dst_dtype)

      self.assertEqual(result.shape, expected.shape)
      self.assertEqual(result.dtype, expected.dtype)
      np.testing.assert_equal(result, expected)

    x = [0, 1, 0, 0, 1]
    for xla_types in [xla_x32_types, xla_x64_types]:
      for src_dtype, dst_dtype in itertools.product(xla_types, xla_types):
        _ConvertAndTest(x, src_dtype, dst_dtype, xla_types[dst_dtype])

  # TODO(b/123523486) implement AllToAll on CPU
  def DISABLED_testAllToAllOneReplica(self):
    samples = [
        NumpyArrayF32([97.0]),
        NumpyArrayF32([64.0, 117.0]),
        NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]]),
    ]
    for lhs in samples[:1]:
      c = self._NewComputation()
      c.AllToAll(c.Constant(lhs), 0, 0)
      self._ExecuteAndCompareExact(c, expected=lhs)

  def testCrossReplicaSumOneReplica(self):
    samples = [
        NumpyArrayF32(42.0),
        NumpyArrayF32([97.0]),
        NumpyArrayF32([64.0, 117.0]),
        NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]]),
    ]
    for lhs in samples:
      c = self._NewComputation()
      c.CrossReplicaSum(c.Constant(lhs))
      self._ExecuteAndCompareExact(c, expected=lhs)

  def testReplicaId(self):
    c = self._NewComputation()
    _ = c.ReplicaId()
    self._ExecuteAndCompareExact(c, expected=0)

  def testCrossReplicaSumOneReplicaWithSingletonGroup(self):
    samples = [
        NumpyArrayF32(42.0),
        NumpyArrayF32([97.0]),
        NumpyArrayF32([64.0, 117.0]),
        NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]]),
    ]
    for lhs in samples:
      c = self._NewComputation()
      c.CrossReplicaSum(c.Constant(lhs), [[0]])
      self._ExecuteAndCompareExact(c, expected=lhs)

  def testDotMatrixVectorF32(self):
    c = self._NewComputation()
    lhs = NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF32([[10.0], [20.0]])
    c.Dot(c.Constant(lhs), c.Constant(rhs))
    self._ExecuteAndCompareClose(c, expected=np.dot(lhs, rhs))

  def testDotMatrixVectorF64(self):
    c = self._NewComputation()
    lhs = NumpyArrayF64([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF64([[10.0], [20.0]])
    c.Dot(c.Constant(lhs), c.Constant(rhs))
    self._ExecuteAndCompareClose(c, expected=np.dot(lhs, rhs))

  def testDotMatrixMatrixF32(self):
    c = self._NewComputation()
    lhs = NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF32([[10.0, 20.0], [100.0, 200.0]])
    c.Dot(c.Constant(lhs), c.Constant(rhs))
    self._ExecuteAndCompareClose(c, expected=np.dot(lhs, rhs))

  def testDotMatrixMatrixF64(self):
    c = self._NewComputation()
    lhs = NumpyArrayF64([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF64([[10.0, 20.0], [100.0, 200.0]])
    c.Dot(c.Constant(lhs), c.Constant(rhs))
    self._ExecuteAndCompareClose(c, expected=np.dot(lhs, rhs))

  def testDotGeneral(self):
    c = self._NewComputation()
    rng = np.random.RandomState(0)
    lhs = NumpyArrayF32(rng.randn(10, 3, 4))
    rhs = NumpyArrayF32(rng.randn(10, 4, 5))
    dimension_numbers = (([2], [1]), ([0], [0]))
    c.DotGeneral(c.Constant(lhs), c.Constant(rhs), dimension_numbers)
    self._ExecuteAndCompareClose(c, expected=np.matmul(lhs, rhs), rtol=1e-6)

  def testDotGeneralWithDotDimensionNumbersProto(self):
    c = self._NewComputation()
    rng = np.random.RandomState(0)
    lhs = NumpyArrayF32(rng.randn(10, 3, 4))
    rhs = NumpyArrayF32(rng.randn(10, 4, 5))

    dimension_numbers = xla_client.DotDimensionNumbers()
    dimension_numbers.lhs_contracting_dimensions.append(2)
    dimension_numbers.rhs_contracting_dimensions.append(1)
    dimension_numbers.lhs_batch_dimensions.append(0)
    dimension_numbers.rhs_batch_dimensions.append(0)

    c.DotGeneral(c.Constant(lhs), c.Constant(rhs), dimension_numbers)
    self._ExecuteAndCompareClose(c, expected=np.matmul(lhs, rhs), rtol=1e-6)

  def testDotGeneralWithPrecisionConfig(self):
    c = self._NewComputation()
    rng = np.random.RandomState(0)
    lhs = NumpyArrayF32(rng.randn(10, 3, 4))
    rhs = NumpyArrayF32(rng.randn(10, 4, 5))
    dimension_numbers = (([2], [1]), ([0], [0]))
    config = xla_client.PrecisionConfig()
    config.operand_precision.append(config.Precision.HIGH)
    config.operand_precision.append(config.Precision.HIGHEST)
    c.DotGeneral(
        c.Constant(lhs),
        c.Constant(rhs),
        dimension_numbers,
        precision_config=config)
    self._ExecuteAndCompareClose(c, expected=np.matmul(lhs, rhs), rtol=1e-6)

  def testConvF32Same(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 2, 3, 4)
    rhs = a(1, 2, 1, 2) * 10
    c.Conv(
        c.Constant(lhs), c.Constant(rhs), [1, 1], xla_client.PaddingType.SAME)
    result = np.array([[[
        [640., 700., 760., 300.],
        [880., 940., 1000., 380.],
        [1120., 1180., 1240., 460.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testConvF32Valid(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 2, 3, 4)
    rhs = a(1, 2, 1, 2) * 10
    c.Conv(
        c.Constant(lhs), c.Constant(rhs), [2, 1], xla_client.PaddingType.VALID)
    result = np.array([[[
        [640., 700., 760.],
        [1120., 1180., 1240.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testConvWithGeneralPaddingF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    c.ConvWithGeneralPadding(
        c.Constant(lhs), c.Constant(rhs), strides, pads, lhs_dilation,
        rhs_dilation)
    result = np.array([[[
        [0., 0., 0.],
        [10., 20., 0.],
        [0., 0., 0.],
        [40., 50., 0.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testConvGeneralDilatedF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = ("NCHW", "OIHW", "NCHW")
    c.ConvGeneralDilated(
        c.Constant(lhs), c.Constant(rhs), strides, pads, lhs_dilation,
        rhs_dilation, dimension_numbers)
    result = np.array([[[
        [0., 0., 0.],
        [10., 20., 0.],
        [0., 0., 0.],
        [40., 50., 0.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testConvGeneralDilatedF32WithPrecisionConfig(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = ("NCHW", "OIHW", "NCHW")
    config = xla_client.PrecisionConfig()
    config.operand_precision.append(config.Precision.HIGHEST)
    config.operand_precision.append(config.Precision.DEFAULT)
    c.ConvGeneralDilated(
        c.Constant(lhs),
        c.Constant(rhs),
        strides,
        pads,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        precision_config=config)
    result = np.array([[[
        [0., 0., 0.],
        [10., 20., 0.],
        [0., 0., 0.],
        [40., 50., 0.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testConvGeneralDilatedPermutedF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)

    dimension_numbers = ("NHWC", "OIHW", "CWNH")
    c.ConvGeneralDilated(
        c.Constant(np.transpose(lhs, (0, 2, 3, 1))), c.Constant(rhs), strides,
        pads, lhs_dilation, rhs_dilation, dimension_numbers)
    result = np.array([[[[0., 0., 0.], [10., 20., 0.], [0., 0., 0.],
                         [40., 50., 0.]]]])
    self._ExecuteAndCompareClose(c, expected=np.transpose(result, (1, 3, 0, 2)))

  def testConvGeneralDilatedGroupedConvolutionF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 2, 2, 3)
    rhs = a(2, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = ("NCHW", "OIHW", "NCHW")
    feature_group_count = 2
    c.ConvGeneralDilated(
        c.Constant(lhs), c.Constant(rhs), strides, pads, lhs_dilation,
        rhs_dilation, dimension_numbers, feature_group_count)
    result = np.array([[[
        [0., 0., 0.],
        [10., 20., 0.],
        [0., 0., 0.],
        [40., 50., 0.],
    ], [
        [0., 0., 0.],
        [330., 380., 160.],
        [0., 0., 0.],
        [480., 530., 220.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=result)

  def testBooleanNot(self):
    c = self._NewComputation()
    arr = NumpyArrayBool([True, False, True])
    c.Not(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=~arr)

  def testCountLeadingZeros(self):
    c = self._NewComputation()
    arr = NumpyArrayS32([0x7FFF, 0x12345678])
    c.Clz(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=[17, 3])

  def testExp(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Exp(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.exp(arr))

  def testExpm1(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Expm1(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.expm1(arr))

  def testRound(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Round(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.round(arr))

  def testLog(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Log(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.log(arr))

  def testLog1p(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Log1p(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.log1p(arr))

  def testNeg(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Neg(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=-arr)

  def testFloor(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Floor(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.floor(arr))

  def testCeil(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Ceil(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.ceil(arr))

  def testAbs(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, -12.1, 2.4, -1.])
    c.Abs(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.abs(arr))

  def testTanh(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    c.Tanh(c.Constant(arr))
    self._ExecuteAndCompareClose(c, expected=np.tanh(arr))

  def testTrans(self):

    def _TransposeAndTest(array):
      c = self._NewComputation()
      c.Trans(c.Constant(array))
      self._ExecuteAndCompareClose(c, expected=array.T)

    # Test square and non-square matrices in both default (C) and F orders.
    for array_fun in [NumpyArrayF32, NumpyArrayF64]:
      _TransposeAndTest(array_fun([[1, 2, 3], [4, 5, 6]]))
      _TransposeAndTest(array_fun([[1, 2, 3], [4, 5, 6]], order="F"))
      _TransposeAndTest(array_fun([[1, 2], [4, 5]]))
      _TransposeAndTest(array_fun([[1, 2], [4, 5]], order="F"))

  def testTranspose(self):

    def _TransposeAndTest(array, permutation):
      c = self._NewComputation()
      c.Transpose(c.Constant(array), permutation)
      expected = np.transpose(array, permutation)
      self._ExecuteAndCompareClose(c, expected=expected)

    _TransposeAndTest(NumpyArrayF32([[1, 2, 3], [4, 5, 6]]), [0, 1])
    _TransposeAndTest(NumpyArrayF32([[1, 2, 3], [4, 5, 6]]), [1, 0])
    _TransposeAndTest(NumpyArrayF32([[1, 2], [4, 5]]), [0, 1])
    _TransposeAndTest(NumpyArrayF32([[1, 2], [4, 5]]), [1, 0])

    arr = np.random.RandomState(0).randn(2, 3, 4).astype(np.float32)
    for permutation in itertools.permutations(range(arr.ndim)):
      _TransposeAndTest(arr, permutation)
      _TransposeAndTest(np.asfortranarray(arr), permutation)

  def testEq(self):
    c = self._NewComputation()
    c.Eq(
        c.Constant(NumpyArrayS32([1, 2, 3, 4])),
        c.Constant(NumpyArrayS32([4, 2, 3, 1])))
    self._ExecuteAndCompareExact(c, expected=[False, True, True, False])

  def testNe(self):
    c = self._NewComputation()
    c.Ne(
        c.Constant(NumpyArrayS32([1, 2, 3, 4])),
        c.Constant(NumpyArrayS32([4, 2, 3, 1])))
    self._ExecuteAndCompareExact(c, expected=[True, False, False, True])

    c.Ne(
        c.Constant(NumpyArrayF32([-2.0, 0.0,
                                  float("nan"),
                                  float("nan")])),
        c.Constant(NumpyArrayF32([2.0, -0.0, 1.0, float("nan")])))
    self._ExecuteAndAssertWith(
        np.testing.assert_allclose, c, (), expected=[True, False, True, True])

  def testGt(self):
    c = self._NewComputation()
    c.Gt(
        c.Constant(NumpyArrayS32([1, 2, 3, 4, 9])),
        c.Constant(NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[False, True, True, False, False])

  def testGe(self):
    c = self._NewComputation()
    c.Ge(
        c.Constant(NumpyArrayS32([1, 2, 3, 4, 9])),
        c.Constant(NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[True, True, True, False, False])

  def testLt(self):
    c = self._NewComputation()
    c.Lt(
        c.Constant(NumpyArrayS32([1, 2, 3, 4, 9])),
        c.Constant(NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[False, False, False, True, True])

  def testLe(self):
    c = self._NewComputation()
    c.Le(
        c.Constant(NumpyArrayS32([1, 2, 3, 4, 9])),
        c.Constant(NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[True, False, False, True, True])

  def testMax(self):
    c = self._NewComputation()
    c.Max(
        c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0, 9.0])),
        c.Constant(NumpyArrayF32([1.0, 0.0, 2.0, 7.0, 12.0])))
    self._ExecuteAndCompareExact(c, expected=[1.0, 2.0, 3.0, 7.0, 12.0])

  def testMaxExplicitBroadcastDim0(self):
    c = self._NewComputation()
    c.Max(
        c.Constant(NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF32([3, 4, 5])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareExact(c, expected=[[3, 3, 3], [4, 5, 6], [7, 8, 9]])

  def testMaxExplicitBroadcastDim1(self):
    c = self._NewComputation()
    c.Max(
        c.Constant(NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayF32([3, 4, 5])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareExact(c, expected=[[3, 4, 5], [4, 5, 6], [7, 8, 9]])

  def testMin(self):
    c = self._NewComputation()
    c.Min(
        c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0, 9.0])),
        c.Constant(NumpyArrayF32([1.0, 0.0, 2.0, 7.0, 12.0])))
    self._ExecuteAndCompareExact(c, expected=[1.0, 0.0, 2.0, 4.0, 9.0])

  def testPad(self):
    c = self._NewComputation()
    c.Pad(
        c.Constant(NumpyArrayF32([[1.0, 2.0], [3.0, 4.0]])),
        c.Constant(NumpyArrayF32(0.0)), [(1, 2, 1), (0, 1, 0)])
    self._ExecuteAndCompareClose(
        c,
        expected=[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0],
                  [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  def testPadWithPaddingConfig(self):
    c = self._NewComputation()
    padding_config = xla_client.PaddingConfig()
    for lo, hi, interior in [(1, 2, 1), (0, 1, 0)]:
      dimension = xla_client.PaddingConfigDimension()
      dimension.edge_padding_low = lo
      dimension.edge_padding_high = hi
      dimension.interior_padding = interior
      padding_config.dimensions.append(dimension)
    c.Pad(
        c.Constant(NumpyArrayF32([[1.0, 2.0], [3.0, 4.0]])),
        c.Constant(NumpyArrayF32(0.0)), padding_config)
    self._ExecuteAndCompareClose(
        c,
        expected=[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0],
                  [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  def testReshape(self):
    c = self._NewComputation()
    c.Reshape(
        c.Constant(NumpyArrayS32([[1, 2], [3, 4], [5, 6]])),
        dimensions=[0, 1],
        new_sizes=[2, 3])
    self._ExecuteAndCompareExact(c, expected=[[1, 2, 3], [4, 5, 6]])

  def testCollapse(self):
    c = self._NewComputation()
    c.Collapse(
        c.Constant(NumpyArrayS32([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
        dimensions=[1, 2])
    self._ExecuteAndCompareExact(c, expected=[[1, 2, 3, 4], [5, 6, 7, 8]])

  def testRev(self):
    c = self._NewComputation()
    c.Rev(
        c.Constant(NumpyArrayS32([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
        dimensions=[0, 2])
    self._ExecuteAndCompareExact(
        c, expected=[[[6, 5], [8, 7]], [[2, 1], [4, 3]]])

  def testReducePrecision(self):
    c = self._NewComputation()
    c.ReducePrecision(
        c.Constant(NumpyArrayF32([float.fromhex("0x1.32fffep-3")])),
        exponent_bits=8,
        mantissa_bits=7)
    self._ExecuteAndCompareClose(c, expected=[float.fromhex("0x1.32p-3")])

  def testClampF32(self):
    c = self._NewComputation()
    c.Clamp(
        c.Constant(NumpyArrayF32(-1)),
        c.Constant(NumpyArrayF32([-2, -1, 0, 1, 2, 3])),
        c.Constant(NumpyArrayF32(2)))
    self._ExecuteAndCompareExact(c, expected=[-1, -1, 0, 1, 2, 2])

  def testClampS32(self):
    c = self._NewComputation()
    c.Clamp(
        c.Constant(NumpyArrayS32(-1)),
        c.Constant(NumpyArrayS32([-2, -1, 0, 1, 2, 3])),
        c.Constant(NumpyArrayS32(2)))
    self._ExecuteAndCompareExact(c, expected=[-1, -1, 0, 1, 2, 2])

  def testSelect(self):
    c = self._NewComputation()
    c.Select(
        c.Constant(NumpyArrayBool([True, False, False, True, False])),
        c.Constant(NumpyArrayS32([1, 2, 3, 4, 5])),
        c.Constant(NumpyArrayS32([-1, -2, -3, -4, -5])))
    self._ExecuteAndCompareExact(c, expected=[1, -2, -3, 4, -5])

  def testSlice(self):
    c = self._NewComputation()
    c.Slice(
        c.Constant(NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), [1, 0],
        [3, 2])
    self._ExecuteAndCompareExact(c, expected=[[4, 5], [7, 8]])

  def testSliceInDim(self):
    c = self._NewComputation()
    c.SliceInDim(
        c.Constant(NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        start_index=1,
        limit_index=2,
        stride=1,
        dimno=1)
    self._ExecuteAndCompareExact(c, expected=[[2], [5], [8]])
    c.SliceInDim(
        c.Constant(NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        start_index=0,
        limit_index=3,
        stride=2,
        dimno=0)
    self._ExecuteAndCompareExact(c, expected=[[1, 2, 3], [7, 8, 9]])

  def testDynamicSlice(self):
    c = self._NewComputation()
    c.DynamicSlice(
        c.Constant(NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayS32([1, 0])), [2, 2])
    self._ExecuteAndCompareExact(c, expected=[[4, 5], [7, 8]])

  def testDynamicUpdateSlice(self):
    c = self._NewComputation()
    c.DynamicUpdateSlice(
        c.Constant(NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        c.Constant(NumpyArrayS32([[1, 2], [3, 4]])),
        c.Constant(NumpyArrayS32([1, 1])))
    self._ExecuteAndCompareExact(c, expected=[[1, 2, 3], [4, 1, 2], [7, 3, 4]])

  def testTuple(self):
    c = self._NewComputation()
    c.Tuple(
        c.ConstantS32Scalar(42), c.Constant(NumpyArrayF32([1.0, 2.0])),
        c.Constant(NumpyArrayBool([True, False, False, True])))
    result = xla_client.execute_with_python_values(c.Build().Compile())
    self.assertIsInstance(result, tuple)
    np.testing.assert_equal(result[0], 42)
    np.testing.assert_allclose(result[1], [1.0, 2.0])
    np.testing.assert_equal(result[2], [True, False, False, True])

  def testGetTupleElement(self):
    c = self._NewComputation()
    c.GetTupleElement(
        c.Tuple(
            c.ConstantS32Scalar(42), c.Constant(NumpyArrayF32([1.0, 2.0])),
            c.Constant(NumpyArrayBool([True, False, False, True]))), 1)
    self._ExecuteAndCompareClose(c, expected=[1.0, 2.0])

  def testBroadcast(self):
    c = self._NewComputation()
    c.Broadcast(c.Constant(NumpyArrayS32([10, 20, 30, 40])), sizes=(3,))
    self._ExecuteAndCompareExact(
        c, expected=[[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]])

  def testBroadcastInDim(self):
    c = self._NewComputation()
    c.BroadcastInDim(c.Constant(NumpyArrayS32([1, 2])), [2, 2], [0])
    self._ExecuteAndCompareExact(c, expected=[[1, 1], [2, 2]])
    c.BroadcastInDim(c.Constant(NumpyArrayS32([1, 2])), [2, 2], [1])
    self._ExecuteAndCompareExact(c, expected=[[1, 2], [1, 2]])

  def testRngNormal(self):
    shape = (2, 3)
    c = self._NewComputation()
    c.RngNormal(
        c.Constant(NumpyArrayF32(0.)),
        c.Constant(NumpyArrayF32(1.)),
        dims=shape)
    result = xla_client.execute_with_python_values(c.Build().Compile())
    # since the result is random, we just check shape and uniqueness
    self.assertEqual(result.shape, shape)
    self.assertLen(np.unique(result), np.prod(shape))

  def testRngUniformF32(self):
    lo, hi = 2., 4.
    shape = (2, 3)
    c = self._NewComputation()
    c.RngUniform(
        c.Constant(NumpyArrayF32(lo)),
        c.Constant(NumpyArrayF32(hi)),
        dims=shape)
    result = xla_client.execute_with_python_values(c.Build().Compile())
    # since the result is random, we just check shape, uniqueness, and range
    self.assertEqual(result.shape, shape)
    self.assertLen(np.unique(result), np.prod(shape))
    self.assertTrue(np.all(lo <= result))
    self.assertTrue(np.all(result < hi))

  def testRngUniformS32(self):
    lo, hi = 2, 4
    shape = (2, 3)
    c = self._NewComputation()
    c.RngUniform(
        c.Constant(NumpyArrayS32(lo)),
        c.Constant(NumpyArrayS32(hi)),
        dims=shape)
    result = xla_client.execute_with_python_values(c.Build().Compile())
    # since the result is random, we just check shape, integrality, and range
    self.assertEqual(result.shape, shape)
    self.assertEqual(result.dtype, np.int32)
    self.assertTrue(np.all(lo <= result))
    self.assertTrue(np.all(result < hi))

  def testCholesky(self):
    l = np.array([[4, 0, 0, 0], [6, 5, 0, 0], [2, 14, 16, 0], [3, 6, 1, 4]],
                 dtype=np.float32)
    c = self._NewComputation()
    c.Cholesky(c.Constant(np.dot(l, l.T)))
    self._ExecuteAndCompareClose(c, expected=l, rtol=1e-4)

  def testSort(self):
    keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
    c = self._NewComputation()
    c.Sort(c.Constant(keys))
    self._ExecuteAndCompareClose(
        c, expected=np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32))

  def testSortKeyVal(self):
    keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
    values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    c = self._NewComputation()
    c.Sort((c.Constant(keys), c.Constant(values)), dimension=0)
    result = xla_client.execute_with_python_values(c.Build().Compile())
    self.assertIsInstance(result, tuple)
    np.testing.assert_allclose(result[0], [[2, 1, 1, 2], [3, 4, 4, 3]])
    np.testing.assert_equal(result[1], [[0, 5, 2, 7], [4, 1, 6, 3]])

  def testSortCustomComparator(self):
    b = self._NewComputation("comparator")
    p0 = b.ParameterFromNumpy(NumpyArrayF32(0))
    q0 = b.ParameterFromNumpy(NumpyArrayF32(0))
    p1 = b.ParameterFromNumpy(NumpyArrayS32(0))
    q1 = b.ParameterFromNumpy(NumpyArrayS32(0))
    b.Or(b.Lt(p0, q0), b.And(b.Eq(p0, q0), b.Gt(p1, q1)))
    comparator = b.Build()

    keys = np.array([[2, 3, 1, 3], [3, 1, 2, 2]], dtype=np.float32)
    values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    c = self._NewComputation()
    c.Sort((c.Constant(keys), c.Constant(values)),
           dimension=1,
           comparator=comparator)
    result = xla_client.execute_with_python_values(c.Build().Compile())
    self.assertIsInstance(result, tuple)
    np.testing.assert_allclose(result[0], [[1, 2, 3, 3], [1, 2, 2, 3]])
    np.testing.assert_equal(result[1], [[2, 0, 3, 1], [5, 7, 6, 4]])

  def testQR(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    c = self._NewComputation()
    c.QR(c.Constant(a), full_matrices=True)
    q, r = self._Execute(c, ())
    np.testing.assert_allclose(np.dot(q, r), a, rtol=1e-4)

  def testEigh(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    a = (a + a.T) / 2

    c = self._NewComputation()
    c.Eigh(c.Constant(a), full_matrices=True)
    # TODO(b/129396575): Turn this test back on when it passes without fastmath.
    # v, w = self._Execute(c, ())
    # self.assertLess(np.linalg.norm(np.dot(a, v) - w * v), 1e-3)

  def testSVD(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    c = self._NewComputation()
    c.SVD(c.Constant(a))
    u, d, v = self._Execute(c, ())
    self.assertLess(np.linalg.norm(a - np.matmul(u * d, v.T)), 1e-3)

  def testTriangularSolve(self):
    a_vals = np.array(
        [[2, 0, 0, 0], [3, 6, 0, 0], [4, 7, 9, 0], [5, 8, 10, 11]],
        dtype=np.float32)
    b_vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      dtype=np.float32)

    c = self._NewComputation()
    c.TriangularSolve(
        c.Constant(a_vals),
        c.Constant(b_vals),
        left_side=False,
        lower=True,
        transpose_a=True)
    self._ExecuteAndCompareClose(
        c,
        expected=np.array([
            [0.5, 0.08333334, 0.04629629, 0.03367003],
            [2.5, -0.25, -0.1388889, -0.1010101],
            [4.5, -0.58333331, -0.32407406, -0.23569024],
        ],
                          dtype=np.float32),
        rtol=1e-4)

  def testIsConstant(self):
    c = self._NewComputation()
    a = c.ConstantS32Scalar(3)
    b = c.ConstantS32Scalar(1)
    x = c.ParameterFromNumpy(NumpyArrayS32(0))
    const_expr = c.Sub(b, a)
    non_const_expr = c.Mul(const_expr, x)
    self.assertTrue(c.IsConstant(const_expr))
    self.assertFalse(c.IsConstant(non_const_expr))
    # self.assertTrue(c.IsConstant(c.Sub(c.Add(x, a), x)))  # TODO(b/77245564)

  def testGather(self):
    a = np.arange(9).astype(np.int32).reshape((3, 3))
    indices = np.array([[[0, 2], [2, 1]], [[1, 2], [2, 0]]], dtype=np.int32)
    dnums = xla_client.GatherDimensionNumbers()
    dnums.offset_dims.append(1)
    dnums.offset_dims.append(2)
    dnums.start_index_map.append(0)
    dnums.start_index_map.append(1)
    dnums.index_vector_dim = 2
    c = self._NewComputation()
    c.Gather(c.Constant(a), c.Constant(indices), dnums, slice_sizes=[1, 1])
    g = self._Execute(c, ())
    expected = np.array([[[[2, 7]]], [[[5, 6]]]], dtype=np.int32)
    np.testing.assert_allclose(g, expected, rtol=1e-4)

  def testFft(self):
    shape = [2, 3, 4, 5]
    rng = np.random.RandomState(0)
    a = rng.randn(*shape) + 1.0j * rng.randn(*shape)
    a = a.astype(np.complex64)
    # FFT
    c = self._NewComputation()
    c.Fft(c.Constant(a), xla_client.FftType.FFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=np.fft.fftn(a, axes=(1, 2, 3)), rtol=1e-4)
    # IFFT
    c = self._NewComputation()
    c.Fft(c.Constant(a), xla_client.FftType.IFFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=np.fft.ifftn(a, axes=(1, 2, 3)), rtol=1e-4)
    # RFFT
    b = rng.randn(*shape).astype(np.float32)
    c = self._NewComputation()
    c.Fft(c.Constant(b), xla_client.FftType.RFFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=np.fft.rfftn(b, axes=(1, 2, 3)), rtol=1e-4)
    # IRFFT
    c = self._NewComputation()
    c.Fft(c.Constant(a), xla_client.FftType.IRFFT, [3, 4, 8])
    self._ExecuteAndCompareClose(
        c, expected=np.fft.irfftn(a, axes=(1, 2, 3)), rtol=1e-4)

  def testNextAfter(self):
    c = self._NewComputation()
    c.NextAfter(
        c.Constant(np.array([1, 2], dtype=np.float32)),
        c.Constant(np.array([2, 1], dtype=np.float32)))
    out = self._Execute(c, ())
    eps = np.finfo(np.float32).eps
    np.testing.assert_equal(np.array([eps + 1, 2 - eps], dtype=np.float32), out)

  def testRegularizedIncompleteBeta(self):
    x = np.array([0.53787335, 0.24015466, 0.47494545, 0.13567594, 0.95114538])
    a = np.array([0.00753073, 0.34813385, 0.30485708, 1.29298632, 0.51472606])
    b = np.array([0.55688389, 0.59794214, 0.42661022, 1.59748339, 0.95047677])
    c = self._NewComputation()
    c.RegularizedIncompleteBeta(c.Constant(a), c.Constant(b), c.Constant(x))
    expected = np.array(
        [0.98923271, 0.48575411, 0.57952568, 0.12579775, 0.96989155])
    self._ExecuteAndCompareClose(c, expected=expected, rtol=1e-4)


class EmbeddedComputationsTest(ComputationTest):
  """Tests for XLA graphs with embedded computations (such as maps)."""

  def _CreateConstantS32Computation(self):
    """Computation (f32) -> s32 that returns a constant 1 for any input."""
    c = self._NewComputation("constant_s32_one")
    # TODO(eliben): consider adding a nicer way to create new parameters without
    # having to create dummy Numpy arrays or populating Shape messages. Perhaps
    # we need our own (Python-client-own) way to represent Shapes conveniently.
    c.ParameterFromNumpy(NumpyArrayF32(0))
    c.ConstantS32Scalar(1)
    return c.Build()

  def _CreateConstantS64Computation(self):
    """Computation (f64) -> s64 that returns a constant 1 for any input."""
    c = self._NewComputation("constant_s64_one")
    # TODO(eliben): consider adding a nicer way to create new parameters without
    # having to create dummy Numpy arrays or populating Shape messages. Perhaps
    # we need our own (Python-client-own) way to represent Shapes conveniently.
    c.ParameterFromNumpy(NumpyArrayF64(0))
    c.ConstantS64Scalar(1)
    return c.Build()

  def _CreateConstantF32Computation(self):
    """Computation (f32) -> f32 that returns a constant 1.0 for any input."""
    c = self._NewComputation("constant_f32_one")
    c.ParameterFromNumpy(NumpyArrayF32(0))
    c.ConstantF32Scalar(1.0)
    return c.Build()

  def _CreateConstantF64Computation(self):
    """Computation (f64) -> f64 that returns a constant 1.0 for any input."""
    c = self._NewComputation("constant_f64_one")
    c.ParameterFromNumpy(NumpyArrayF64(0))
    c.ConstantF64Scalar(1.0)
    return c.Build()

  def _CreateMulF32By2Computation(self):
    """Computation (f32) -> f32 that multiplies its parameter by 2."""
    c = self._NewComputation("mul_f32_by2")
    c.Mul(c.ParameterFromNumpy(NumpyArrayF32(0)), c.ConstantF32Scalar(2.0))
    return c.Build()

  def _CreateMulF32ByParamComputation(self):
    """Computation (f32) -> f32 that multiplies one parameter by the other."""
    c = self._NewComputation("mul_f32_by_param")
    c.Mul(
        c.ParameterFromNumpy(NumpyArrayF32(0)),
        c.ParameterFromNumpy(NumpyArrayF32(0)))
    return c.Build()

  def _CreateMulF64By2Computation(self):
    """Computation (f64) -> f64 that multiplies its parameter by 2."""
    c = self._NewComputation("mul_f64_by2")
    c.Mul(c.ParameterFromNumpy(NumpyArrayF64(0)), c.ConstantF64Scalar(2.0))
    return c.Build()

  def _CreateBinaryAddS32Computation(self):
    """Computation (s32, s32) -> s32 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    c.Add(
        c.ParameterFromNumpy(NumpyArrayS32(0)),
        c.ParameterFromNumpy(NumpyArrayS32(0)))
    return c.Build()

  def _CreateBinaryAddF32Computation(self):
    """Computation (f32, f32) -> f32 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    c.Add(
        c.ParameterFromNumpy(NumpyArrayF32(0)),
        c.ParameterFromNumpy(NumpyArrayF32(0)))
    return c.Build()

  def _CreateBinaryAddF64Computation(self):
    """Computation (f64, f64) -> f64 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    c.Add(
        c.ParameterFromNumpy(NumpyArrayF64(0)),
        c.ParameterFromNumpy(NumpyArrayF64(0)))
    return c.Build()

  def _CreateBinaryDivF32Computation(self):
    """Computation (f32, f32) -> f32 that divides its two parameters."""
    c = self._NewComputation("div_param0_by_param1")
    c.Div(
        c.ParameterFromNumpy(NumpyArrayF32(0)),
        c.ParameterFromNumpy(NumpyArrayF32(0)))
    return c.Build()

  def _CreateBinaryDivF64Computation(self):
    """Computation (f64, f64) -> f64 that divides its two parameters."""
    c = self._NewComputation("div_param0_by_param1")
    c.Div(
        c.ParameterFromNumpy(NumpyArrayF64(0)),
        c.ParameterFromNumpy(NumpyArrayF64(0)))
    return c.Build()

  def _CreateTestF32Lt10Computation(self):
    """Computation (f32) -> bool that tests if its parameter is less than 10."""
    c = self._NewComputation("test_f32_lt_10")
    c.Lt(c.ParameterFromNumpy(NumpyArrayF32(0)), c.ConstantF32Scalar(10.))
    return c.Build()

  def _CreateTestF64Lt10Computation(self):
    """Computation (f64) -> bool that tests if its parameter is less than 10."""
    c = self._NewComputation("test_f64_lt_10")
    c.Lt(c.ParameterFromNumpy(NumpyArrayF64(0)), c.ConstantF64Scalar(10.))
    return c.Build()

  def _CreateBinaryGeF32Computation(self):
    """Computation (f32, f32) -> bool that tests first_param >= second_param."""
    c = self._NewComputation("param0_lt_param1")
    c.Ge(
        c.ParameterFromNumpy(NumpyArrayF32(0)),
        c.ParameterFromNumpy(NumpyArrayF32(0)))
    return c.Build()

  def _CreateBinaryGeF64Computation(self):
    """Computation (f64, f64) -> bool that tests first_param >= second_param."""
    c = self._NewComputation("param0_lt_param1")
    c.Ge(
        c.ParameterFromNumpy(NumpyArrayF64(0)),
        c.ParameterFromNumpy(NumpyArrayF64(0)))
    return c.Build()

  def _MakeSample3DArrayF32(self):
    return NumpyArrayF32([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                          [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

  def _MakeSample3DArrayF64(self):
    return NumpyArrayF64([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                          [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

  def testCallF32(self):
    c = self._NewComputation()
    c.Call(
        self._CreateMulF32By2Computation(),
        operands=(c.ConstantF32Scalar(5.0),))
    self._ExecuteAndCompareClose(c, expected=10.0)

  def testCallF64(self):
    c = self._NewComputation()
    c.Call(
        self._CreateMulF64By2Computation(),
        operands=(c.ConstantF64Scalar(5.0),))
    self._ExecuteAndCompareClose(c, expected=10.0)

  def testMapEachElementToS32Constant(self):
    c = self._NewComputation()
    c.Map([c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
          self._CreateConstantS32Computation(), [0])
    self._ExecuteAndCompareExact(c, expected=[1, 1, 1, 1])

  def testMapEachElementToS64Constant(self):
    c = self._NewComputation()
    c.Map([c.Constant(NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
          self._CreateConstantS64Computation(), [0])
    self._ExecuteAndCompareExact(c, expected=[1, 1, 1, 1])

  def testMapMulBy2F32(self):
    c = self._NewComputation()
    c.Map([c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
          self._CreateMulF32By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[2.0, 4.0, 6.0, 8.0])

  def testMapMulBy2F64(self):
    c = self._NewComputation()
    c.Map([c.Constant(NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
          self._CreateMulF64By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[2.0, 4.0, 6.0, 8.0])

  def testSimpleMapChainF32(self):
    # Chains a map of constant-f32 with a map of mul-by-2
    c = self._NewComputation()
    const_f32 = c.Map([c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
                      self._CreateConstantF32Computation(), [0])
    c.Map([const_f32], self._CreateMulF32By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[2.0, 2.0, 2.0, 2.0])

  def testSimpleMapChainF64(self):
    # Chains a map of constant-f64 with a map of mul-by-2
    c = self._NewComputation()
    const_f64 = c.Map([c.Constant(NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
                      self._CreateConstantF64Computation(), [0])
    c.Map([const_f64], self._CreateMulF64By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[2.0, 2.0, 2.0, 2.0])

  def testDivVectorsWithMapF32(self):
    c = self._NewComputation()
    c.Map((c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0])),
           c.Constant(NumpyArrayF32([5.0, 5.0, 4.0, 4.0]))),
          self._CreateBinaryDivF32Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[0.2, 0.4, 0.75, 1.0])

  def testDivVectorsWithMapF64(self):
    c = self._NewComputation()
    c.Map((c.Constant(NumpyArrayF64([1.0, 2.0, 3.0, 4.0])),
           c.Constant(NumpyArrayF64([5.0, 5.0, 4.0, 4.0]))),
          self._CreateBinaryDivF64Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[0.2, 0.4, 0.75, 1.0])

  def testSelectAndScatterF32(self):
    c = self._NewComputation()
    c.SelectAndScatter(
        c.Constant(NumpyArrayF32([[1., 2., 6.], [4., 5., 3.]])),
        select=self._CreateBinaryGeF32Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 2),
        padding=xla_client.PaddingType.VALID,
        source=c.Constant(NumpyArrayF32([[0.1, 0.2]])),
        init_value=c.Constant(NumpyArrayF32(1)),
        scatter=self._CreateBinaryAddF32Computation())
    self._ExecuteAndCompareClose(c, expected=[[1., 1., 1.2], [1.1, 1., 1.]])

  def testSelectAndScatterF64(self):
    c = self._NewComputation()
    c.SelectAndScatter(
        c.Constant(NumpyArrayF64([[1., 2., 6.], [4., 5., 3.]])),
        select=self._CreateBinaryGeF64Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 2),
        padding=xla_client.PaddingType.VALID,
        source=c.Constant(NumpyArrayF64([[0.1, 0.2]])),
        init_value=c.Constant(NumpyArrayF64(1)),
        scatter=self._CreateBinaryAddF64Computation())
    self._ExecuteAndCompareClose(c, expected=[[1., 1., 1.2], [1.1, 1., 1.]])

  def testReduce1DtoScalarF32(self):
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(NumpyArrayF32([1.0, 2.0, 3.0, 4.0])),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        dimensions=[0])
    self._ExecuteAndCompareClose(c, expected=10)

  def testReduce1DtoScalarF64(self):
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(NumpyArrayF64([1.0, 2.0, 3.0, 4.0])),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        dimensions=[0])
    self._ExecuteAndCompareClose(c, expected=10)

  def testReduce2DTo1DDim0F32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(input_array),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        dimensions=[0])
    self._ExecuteAndCompareClose(c, expected=[5, 7, 9])

  def testReduce2DTo1DDim0F64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(input_array),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        dimensions=[0])
    self._ExecuteAndCompareClose(c, expected=[5, 7, 9])

  def testReduce2DTo1DDim1F32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(input_array),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        dimensions=[1])
    self._ExecuteAndCompareClose(c, expected=[6, 15])

  def testReduce2DTo1DDim1F64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.Reduce(
        operand=c.Constant(input_array),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        dimensions=[1])
    self._ExecuteAndCompareClose(c, expected=[6, 15])

  def testReduce3DAllPossibleWaysF32(self):
    input_array = self._MakeSample3DArrayF32()

    def _ReduceAndTest(*dims):
      c = self._NewComputation()
      c.Reduce(
          operand=c.Constant(input_array),
          init_value=c.ConstantF32Scalar(0),
          computation_to_apply=self._CreateBinaryAddF32Computation(),
          dimensions=dims)
      self._ExecuteAndCompareClose(
          c, expected=np.sum(input_array, axis=tuple(dims)))

    _ReduceAndTest(0)
    _ReduceAndTest(0, 1)
    _ReduceAndTest(0, 2)
    _ReduceAndTest(1, 2)
    _ReduceAndTest(0, 1, 2)

  def testReduce3DAllPossibleWaysF64(self):
    input_array = self._MakeSample3DArrayF64()

    def _ReduceAndTest(*dims):
      c = self._NewComputation()
      c.Reduce(
          operand=c.Constant(input_array),
          init_value=c.ConstantF64Scalar(0),
          computation_to_apply=self._CreateBinaryAddF64Computation(),
          dimensions=dims)
      self._ExecuteAndCompareClose(
          c, expected=np.sum(input_array, axis=tuple(dims)))

    _ReduceAndTest(0)
    _ReduceAndTest(0)
    _ReduceAndTest(0, 1)
    _ReduceAndTest(0, 2)
    _ReduceAndTest(1, 2)
    _ReduceAndTest(0, 1, 2)

  def testReduceWindowValidUnitStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 1),
        padding=xla_client.PaddingType.VALID)
    self._ExecuteAndCompareClose(c, expected=[[5., 7., 9.]])

  def testReduceWindowSameUnitStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 1),
        padding=xla_client.PaddingType.SAME)
    self._ExecuteAndCompareClose(c, expected=[[5., 7., 9.], [4., 5., 6.]])

  def testReduceWindowValidGeneralStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF32Scalar(0),
        computation_to_apply=self._CreateBinaryAddF32Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 2),
        padding=xla_client.PaddingType.VALID)
    self._ExecuteAndCompareClose(c, expected=[[5., 9.]])

  def testReduceWindowValidUnitStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 1),
        padding=xla_client.PaddingType.VALID)
    self._ExecuteAndCompareClose(c, expected=[[5., 7., 9.]])

  def testReduceWindowSameUnitStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 1),
        padding=xla_client.PaddingType.SAME)
    self._ExecuteAndCompareClose(c, expected=[[5., 7., 9.], [4., 5., 6.]])

  def testReduceWindowValidGeneralStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    c.ReduceWindow(
        operand=c.Constant(input_array),
        init_value=c.ConstantF64Scalar(0),
        computation_to_apply=self._CreateBinaryAddF64Computation(),
        window_dimensions=(2, 1),
        window_strides=(1, 2),
        padding=xla_client.PaddingType.VALID)
    self._ExecuteAndCompareClose(c, expected=[[5., 9.]])

  def testWhileF32(self):
    cond = self._CreateTestF32Lt10Computation()
    body = self._CreateMulF32By2Computation()
    c = self._NewComputation()
    init = c.ConstantF32Scalar(1.)
    c.While(cond, body, init)
    self._ExecuteAndCompareClose(c, expected=16.)

  def testWhileF64(self):
    cond = self._CreateTestF64Lt10Computation()
    body = self._CreateMulF64By2Computation()
    c = self._NewComputation()
    init = c.ConstantF64Scalar(1.)
    c.While(cond, body, init)
    self._ExecuteAndCompareClose(c, expected=16.)

  def testConditionalTrue(self):
    c = self._NewComputation()
    pred = c.ConstantPredScalar(True)
    true_operand = c.ConstantF32Scalar(3.)
    true_computation = self._CreateMulF32By2Computation()
    false_operand = c.ConstantF32Scalar(2.)
    false_computation = self._CreateConstantF32Computation()
    c.Conditional(pred, true_operand, true_computation, false_operand,
                  false_computation)
    self._ExecuteAndCompareClose(c, expected=6.)

  def testConditionalFalse(self):
    c = self._NewComputation()
    pred = c.ConstantPredScalar(False)
    true_operand = c.ConstantF32Scalar(3.)
    true_computation = self._CreateMulF32By2Computation()
    false_operand = c.ConstantF32Scalar(2.)
    false_computation = self._CreateConstantF32Computation()
    c.Conditional(pred, true_operand, true_computation, false_operand,
                  false_computation)
    self._ExecuteAndCompareClose(c, expected=1.)

  def testInfeedS32Values(self):
    to_infeed = NumpyArrayS32([1, 2, 3, 4])
    c = self._NewComputation()
    c.GetTupleElement(c.Infeed(xla_client.shape_from_pyval(to_infeed[0])), 0)
    compiled_c = c.Build().Compile()
    for item in to_infeed:
      xla_client.transfer_to_infeed(item)

    for item in to_infeed:
      result = xla_client.execute_with_python_values(compiled_c)
      self.assertEqual(result, item)

  def testInfeedTuple(self):
    to_infeed = (NumpyArrayS32([1, 2, 3, 4]), NumpyArrayS32([[7], [8]]))
    c = self._NewComputation()
    c.GetTupleElement(c.Infeed(xla_client.shape_from_pyval(to_infeed)), 0)
    compiled_c = c.Build().Compile()
    xla_client.transfer_to_infeed(to_infeed)

    result = xla_client.execute_with_python_values(compiled_c)
    np.testing.assert_equal(result[0], to_infeed[0])
    np.testing.assert_equal(result[1], to_infeed[1])

  def testInfeedThenOutfeedS32(self):
    to_round_trip = NumpyArrayS32([1, 2, 3, 4])
    c = self._NewComputation()
    x_and_token = c.Infeed(xla_client.shape_from_pyval(to_round_trip[0]))
    x = c.GetTupleElement(x_and_token, 0)
    token = c.GetTupleElement(x_and_token, 1)
    c.Outfeed(x, token)

    compiled_c = c.Build().Compile()

    for want in to_round_trip:
      execution = threading.Thread(target=lambda: compiled_c.Execute([]))
      execution.start()
      xla_client.transfer_to_infeed(want)
      got = xla_client.transfer_from_outfeed(
          xla_client.shape_from_pyval(to_round_trip[0]))
      execution.join()
      self.assertEqual(want, got)

  def testScatter(self):
    a = np.arange(9).astype(np.int32).reshape((3, 3))
    scatter_indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[10, 20, 30], [70, 80, 90]], dtype=np.int32)

    dnums = xla_client.ScatterDimensionNumbers()
    dnums.update_window_dims.append(1)
    dnums.inserted_window_dims.append(0)
    dnums.scatter_dims_to_operand_dims.append(0)
    dnums.index_vector_dim = 1

    c = self._NewComputation()
    c.Scatter(
        c.Constant(a), c.Constant(scatter_indices), c.Constant(updates),
        self._CreateBinaryAddS32Computation(), dnums)
    expected = np.array([[10, 21, 32], [3, 4, 5], [76, 87, 98]], dtype=np.int32)
    self._ExecuteAndCompareClose(c, expected=expected)


class ErrorTest(ComputationTest):

  def setUp(self):
    self.f32_scalar_2 = NumpyArrayF32(2.0)
    self.s32_scalar_2 = NumpyArrayS32(2)

  def testCompileWithWrongElementTypeInLayout(self):
    c = self._NewComputation()
    c.SetOpMetadata(xla_client.CurrentSourceInfoMetadata())
    c.ParameterFromNumpy(self.s32_scalar_2)
    c.ClearOpMetadata()

    options = xla_client.CompileOptions()
    options.argument_layouts = [
        xla_client.Shape.array_shape(np.dtype(np.float32), [])
    ]

    def TestFun():
      return c.Build().Compile(compile_options=options)

    self.assertRaisesRegex(
        RuntimeError, r".*Invalid argument shape.*"
        r"expected s32\[\], got f32\[\].*", TestFun)

  def testInvokeWithWrongElementType(self):
    c = self._NewComputation()
    c.SetOpMetadata(xla_client.CurrentSourceInfoMetadata())
    c.ParameterFromNumpy(self.s32_scalar_2)
    c.ClearOpMetadata()

    def TestFun():
      return xla_client.execute_with_python_values(c.Build().Compile(),
                                                   [self.f32_scalar_2])

    self.assertRaisesRegex(
        RuntimeError, r"Invalid argument: Argument does not match.*"
        r"want s32\[\], got f32\[\].*", TestFun)


class ComputationRootTest(ComputationTest):
  """Tests related to setting the root of the computation."""

  def testComputationRootDifferentFromLastOp(self):
    c = self._NewComputation()
    x = c.ParameterFromNumpy(NumpyArrayF32(2.0))
    result = c.Add(x, c.ConstantF32Scalar(3.14))
    extra = c.Add(result, c.ConstantF32Scalar(1.618))  # pylint: disable=unused-variable

    arg = NumpyArrayF32(1.0)
    compiled_c = c.Build(result).Compile()
    ans = xla_client.execute_with_python_values(compiled_c, [arg])
    np.testing.assert_allclose(ans, 4.14)


class SetShardingTest(ComputationTest):
  """Tests related to set OpSharding."""

  def testSetSharding(self):
    c = self._NewComputation()
    sharding = xla_client.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])
    # Set Sharding.
    c.SetSharding(sharding)
    x = c.ParameterFromNumpy(NumpyArrayF32(2.0))
    # Clear Sharding.
    c.ClearSharding()

    result = c.Add(x, c.ConstantF32Scalar(3.14))
    extra = c.Add(result, c.ConstantF32Scalar(1.618))  # pylint: disable=unused-variable
    arg = NumpyArrayF32(1.0)
    compiled_c = c.Build(result).Compile()
    ans = xla_client.execute_with_python_values(compiled_c, [arg])
    np.testing.assert_allclose(ans, 4.14)


dlpack_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64, np.float16, np.float32, np.float64, bfloat16
]


class DLPackTest(parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters({
      "testcase_name":
          "_{}[{}]".format(dtype.__name__, ",".join(map(str, shape))),
      "dtype":
          dtype,
      "shape":
          shape
  } for dtype in dlpack_dtypes for shape in [
      (),
      (1,),
      (2, 3),
      (2, 0),
      (0, 7),
      (4, 1, 2),
      (2, 1, 3),
      (2, 4, 1),
      (3, 1),
      (1, 3),
  ])
  def testRoundTrip(self, dtype, shape):
    x = np.array(np.random.rand(*shape) * 100, dtype=dtype)
    backend = xla_client.get_local_backend()
    buffer = xla_client.Buffer.from_pyval(x, backend=backend)
    dlt = xla_client._xla.BufferToDLPackManagedTensor(buffer)
    del buffer  # Free "buffer" to make sure dlt retains ownership.
    self.assertEqual(type(dlt).__name__, "PyCapsule")
    y = xla_client._xla.DLPackManagedTensorToBuffer(dlt, backend.client)
    np.testing.assert_array_equal(x, y.to_py())

  def testTensorsCanBeConsumedOnceOnly(self):
    x = np.array(np.random.rand(3, 4, 5, 6), dtype=np.float32)
    backend = xla_client.get_local_backend()
    buffer = xla_client.Buffer.from_pyval(x, backend=backend)
    dlt = xla_client._xla.BufferToDLPackManagedTensor(buffer)

    def ConsumeDLPackTensor():
      _ = xla_client._xla.DLPackManagedTensorToBuffer(dlt, backend.client)

    ConsumeDLPackTensor()
    self.assertRaisesRegex(RuntimeError,
                           ".*a DLPack tensor may be consumed at most once.*",
                           ConsumeDLPackTensor)


if __name__ == "__main__":
  absltest.main()

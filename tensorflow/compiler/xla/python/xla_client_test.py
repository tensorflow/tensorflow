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
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import custom_call_for_test
from tensorflow.compiler.xla.python import xla_client

# pylint: disable=g-import-not-at-top
try:
  import portpicker
except ImportError:
  portpicker = None
# pylint: enable=g-import-not-at-top

bfloat16 = xla_client.bfloat16
ops = xla_client.ops


class ComputationTest(absltest.TestCase):
  """Base class for running an XLA Computation through the local client."""

  def _NewComputation(self, name=None):
    if name is None:
      name = self.id()
    return xla_client.XlaBuilder(name)

  def _Execute(self, c, arguments):
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build())
    return xla_client.execute_with_python_values(compiled_c, arguments)

  def _ExecuteAndAssertWith(self, assert_func, c, arguments, expected):
    assert expected is not None
    results = self._Execute(c, arguments)
    self.assertLen(results, len(expected))
    for result, e in zip(results, expected):
      # Numpy's comparison methods are a bit too lenient by treating inputs as
      # "array-like", meaning that scalar 4 will be happily compared equal to
      # [[4]]. We'd like to be more strict so assert shapes as well.
      self.assertEqual(np.asanyarray(result).shape, np.asanyarray(e).shape)
      assert_func(result, e)

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
    builder = xla_client.XlaBuilder("acomputation")
    p0 = ops.Parameter(builder, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    x = ops.Mul(p0, p1)
    ops.Add(x, x)
    return builder.Build()

  def testComputationToHloText(self):
    computation = self.ExampleComputation()
    hlo_text = computation.GetHloText()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))

  def testComputationToHloGraph(self):
    computation = self.ExampleComputation()
    hlo_dot_graph = computation.GetHloDotGraph()
    self.assertTrue(hlo_dot_graph.startswith("digraph "))

  def testHloModuleToHloText(self):
    computation = self.ExampleComputation()
    hlo_text = computation.get_hlo_module().to_string()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))

  def testHloModuleToHloGraph(self):
    computation = self.ExampleComputation()
    hlo_dot_graph = xla_client._xla.hlo_module_to_dot_graph(
        computation.get_hlo_module())
    self.assertTrue(hlo_dot_graph.startswith("digraph "))

  def testCompiledHloModuleToHloText(self):
    computation = self.ExampleComputation()
    backend = xla_client.get_local_backend()
    executable = backend.compile(computation)
    hlo_modules = executable.get_hlo_modules()
    self.assertLen(hlo_modules, 1)
    hlo_text = hlo_modules[0].to_string()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))
    self.assertIn("fusion", hlo_text)


class ComputationHashTest(absltest.TestCase):

  def testHash(self):
    builder0 = xla_client.XlaBuilder("computation0")
    p0 = ops.Parameter(builder0, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder0, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    ops.Mul(p0, p1)
    computation0 = builder0.Build()

    builder1 = xla_client.XlaBuilder("computation1")
    p0 = ops.Parameter(builder1, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder1, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    ops.Mul(p0, p1)
    computation1 = builder1.Build()

    self.assertEqual(computation0.Hash(), computation1.Hash())


class ComputationsWithConstantsTest(ComputationTest):
  """Tests focusing on Constant ops."""

  def testConstantScalarSumS8(self):
    c = self._NewComputation()
    ops.Add(ops.Constant(c, np.int8(1)), ops.Constant(c, np.int8(2)))
    self._ExecuteAndCompareExact(c, expected=[np.int8(3)])

  def testConstantScalarSumBF16(self):
    c = self._NewComputation()
    ops.Add(ops.Constant(c, bfloat16(1.11)), ops.Constant(c, bfloat16(3.14)))
    self._ExecuteAndCompareClose(c, expected=[bfloat16(4.25)])

  def testConstantScalarSumF32(self):
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, np.float32(1.11)), ops.Constant(c, np.float32(3.14)))
    self._ExecuteAndCompareClose(c, expected=[4.25])

  def testConstantScalarSumF64(self):
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, np.float64(1.11)), ops.Constant(c, np.float64(3.14)))
    self._ExecuteAndCompareClose(c, expected=[4.25])

  def testConstantScalarSumS32(self):
    c = self._NewComputation()
    ops.Add(ops.Constant(c, np.int32(1)), ops.Constant(c, np.int32(2)))
    self._ExecuteAndCompareClose(c, expected=[3])

  def testConstantScalarSumS64(self):
    c = self._NewComputation()
    ops.Add(ops.Constant(c, np.int64(1)), ops.Constant(c, np.int64(2)))
    self._ExecuteAndCompareClose(c, expected=[3])

  def testConstantVectorMulF16(self):
    c = self._NewComputation()
    ops.Mul(
        ops.Constant(c, np.array([2.5, 3.3, -1.2, 0.7], np.float16)),
        ops.Constant(c, np.array([-1.2, 2, -2, -3], np.float16)))
    self._ExecuteAndCompareClose(
        c, expected=[np.array([-3, 6.6, 2.4, -2.1], np.float16)], rtol=2e-3)

  def testConstantVectorMulF32(self):
    c = self._NewComputation()
    ops.Mul(
        ops.Constant(c, NumpyArrayF32([2.5, 3.3, -1.2, 0.7])),
        ops.Constant(c, NumpyArrayF32([-1.2, 2, -2, -3])))
    self._ExecuteAndCompareClose(c, expected=[[-3, 6.6, 2.4, -2.1]])

  def testConstantVectorMulF64(self):
    c = self._NewComputation()
    ops.Mul(
        ops.Constant(c, NumpyArrayF64([2.5, 3.3, -1.2, 0.7])),
        ops.Constant(c, NumpyArrayF64([-1.2, 2, -2, -3])))
    self._ExecuteAndCompareClose(c, expected=[[-3, 6.6, 2.4, -2.1]])

  def testConstantVectorScalarDivF32(self):
    c = self._NewComputation()
    ops.Div(
        ops.Constant(c, NumpyArrayF32([1.5, 2.5, 3.0, -10.8])),
        ops.Constant(c, np.float32(2.0)))
    self._ExecuteAndCompareClose(c, expected=[[0.75, 1.25, 1.5, -5.4]])

  def testConstantVectorScalarDivF64(self):
    c = self._NewComputation()
    ops.Div(
        ops.Constant(c, NumpyArrayF64([1.5, 2.5, 3.0, -10.8])),
        ops.Constant(c, np.float64(2.0)))
    self._ExecuteAndCompareClose(c, expected=[[0.75, 1.25, 1.5, -5.4]])

  def testConstantVectorScalarPowF32(self):
    c = self._NewComputation()
    ops.Pow(
        ops.Constant(c, NumpyArrayF32([1.5, 2.5, 3.0])),
        ops.Constant(c, np.float32(2.)))
    self._ExecuteAndCompareClose(c, expected=[[2.25, 6.25, 9.]])

  def testConstantVectorScalarPowF64(self):
    c = self._NewComputation()
    ops.Pow(
        ops.Constant(c, NumpyArrayF64([1.5, 2.5, 3.0])),
        ops.Constant(c, np.float64(2.)))
    self._ExecuteAndCompareClose(c, expected=[[2.25, 6.25, 9.]])

  def testIota(self):
    c = self._NewComputation()
    ops.Iota(c, xla_client.PrimitiveType.F32, 10)
    self._ExecuteAndCompareExact(c, expected=[np.arange(10, dtype=np.float32)])

  def testBroadcastedIota(self):
    c = self._NewComputation()
    shape = xla_client.Shape.array_shape(xla_client.PrimitiveType.S64, (2, 3))
    ops.Iota(c, shape, 1)
    expected = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    self._ExecuteAndCompareExact(c, expected=[expected])

  def testBooleanAnd(self):
    c = self._NewComputation()
    ops.And(
        ops.Constant(c, NumpyArrayBool([True, False, True, False])),
        ops.Constant(c, NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[[True, False, False, False]])

  def testBooleanOr(self):
    c = self._NewComputation()
    ops.Or(
        ops.Constant(c, NumpyArrayBool([True, False, True, False])),
        ops.Constant(c, NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[[True, True, True, False]])

  def testBooleanXor(self):
    c = self._NewComputation()
    ops.Xor(
        ops.Constant(c, NumpyArrayBool([True, False, True, False])),
        ops.Constant(c, NumpyArrayBool([True, True, False, False])))
    self._ExecuteAndCompareExact(c, expected=[[False, True, True, False]])

  def testSum2DF32(self):
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF32([[1, 2, 3], [4, 5, 6]])),
        ops.Constant(c, NumpyArrayF32([[1, -1, 1], [-1, 1, -1]])))
    self._ExecuteAndCompareClose(c, expected=[[[2, 1, 4], [3, 6, 5]]])

  def testShiftLeft(self):
    c = self._NewComputation()
    ops.ShiftLeft(
        ops.Constant(c, NumpyArrayS32([3])),
        ops.Constant(c, NumpyArrayS32([2])))
    self._ExecuteAndCompareClose(c, expected=[[12]])

  def testShiftRightArithmetic(self):
    c = self._NewComputation()
    ops.ShiftRightArithmetic(
        ops.Constant(c, NumpyArrayS32([-2])),
        ops.Constant(c, NumpyArrayS32([1])))
    self._ExecuteAndCompareClose(c, expected=[[-1]])

  def testShiftRightLogical(self):
    c = self._NewComputation()
    ops.ShiftRightLogical(
        ops.Constant(c, NumpyArrayS32([-1])),
        ops.Constant(c, NumpyArrayS32([1])))
    self._ExecuteAndCompareClose(c, expected=[[2**31 - 1]])

  def testSum2DF64(self):
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF64([[1, 2, 3], [4, 5, 6]])),
        ops.Constant(c, NumpyArrayF64([[1, -1, 1], [-1, 1, -1]])))
    self._ExecuteAndCompareClose(c, expected=[[[2, 1, 4], [3, 6, 5]]])

  def testSum2DWith1DBroadcastDim0F32(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 0 to match the former's shape.
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF32([10, 20, 30])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareClose(
        c, expected=[[[11, 12, 13], [24, 25, 26], [37, 38, 39]]])

  def testSum2DWith1DBroadcastDim0F64(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 0 to match the former's shape.
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF64([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF64([10, 20, 30])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareClose(
        c, expected=[[[11, 12, 13], [24, 25, 26], [37, 38, 39]]])

  def testSum2DWith1DBroadcastDim1F32(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 1 to match the former's shape.
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF32([10, 20, 30])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareClose(
        c, expected=[[[11, 22, 33], [14, 25, 36], [17, 28, 39]]])

  def testSum2DWith1DBroadcastDim1F64(self):
    # sum of a 2D array with a 1D array where the latter is replicated across
    # dimension 1 to match the former's shape.
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, NumpyArrayF64([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF64([10, 20, 30])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareClose(
        c, expected=[[[11, 22, 33], [14, 25, 36], [17, 28, 39]]])

  def testConstantAxpyF32(self):
    c = self._NewComputation()
    ops.Add(
        ops.Mul(
            ops.Constant(c, np.float32(2)),
            ops.Constant(c, NumpyArrayF32([2.2, 3.3, 4.4, 5.5]))),
        ops.Constant(c, NumpyArrayF32([100, -100, 200, -200])))
    self._ExecuteAndCompareClose(c, expected=[[104.4, -93.4, 208.8, -189]])

  def testConstantAxpyF64(self):
    c = self._NewComputation()
    ops.Add(
        ops.Mul(
            ops.Constant(c, np.float64(2)),
            ops.Constant(c, NumpyArrayF64([2.2, 3.3, 4.4, 5.5]))),
        ops.Constant(c, NumpyArrayF64([100, -100, 200, -200])))
    self._ExecuteAndCompareClose(c, expected=[[104.4, -93.4, 208.8, -189]])

  def testCustomCall(self):
    c = self._NewComputation()
    for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
      xla_client.register_custom_call_target(name, fn, platform="cpu")
    ops.CustomCallWithLayout(
        c,
        b"test_subtract_f32",
        operands=[
            ops.Constant(c, np.float32(1.25)),
            ops.Constant(c, np.float32(0.5))
        ],
        shape_with_layout=xla_client.Shape.array_shape(
            np.dtype(np.float32), (), ()),
        operand_shapes_with_layout=[
            xla_client.Shape.array_shape(np.dtype(np.float32), (), ()),
            xla_client.Shape.array_shape(np.dtype(np.float32), (), ()),
        ])
    self._ExecuteAndCompareClose(c, expected=[0.75])


class ComputationFromProtoTest(absltest.TestCase):
  """Test computation execution from HLO proto."""

  def testExecuteFromProto(self):
    # Build the HLO proto
    b = xla_client.XlaBuilder("computation")
    ops.Add(ops.Constant(b, np.int8(1)), ops.Constant(b, np.int8(2)))
    serialized_proto = b.Build().GetSerializedProto()

    # Load and execute the proto
    c = xla_client.Computation(xla_client._xla.XlaComputation(serialized_proto))
    ans, = xla_client.execute_with_python_values(c.Compile())
    np.testing.assert_equal(ans, np.int8(3))


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

  def testScalarTimesVectorS32(self):
    c = self._NewComputation()
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_3))
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(self.s32_4vector))
    ops.Mul(p0, p1)
    self._ExecuteAndCompareExact(
        c,
        arguments=[self.s32_scalar_3, self.s32_4vector],
        expected=[[30, 45, -6, 21]])

  def testScalarTimesVectorS64(self):
    c = self._NewComputation()
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s64_scalar_3))
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(self.s64_4vector))
    ops.Mul(p0, p1)
    self._ExecuteAndCompareExact(
        c,
        arguments=[self.s64_scalar_3, self.s64_4vector],
        expected=[[30, 45, -6, 21]])

  def testScalarMinusVectorExplicitNumberingF32(self):
    # Use explicit numbering and pass parameter_num first. Sub is used since
    # it's not commutative and can help catch parameter reversal within the
    # computation.
    c = self._NewComputation()
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(self.f32_4vector))
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(self.f32_scalar_2))
    ops.Sub(p1, p0)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f32_scalar_2, self.f32_4vector],
        expected=[[-4.3, 1.3, -6.3, 3.3]])

  def testScalarMinusVectorExplicitNumberingF64(self):
    # Use explicit numbering and pass parameter_num first. Sub is used since
    # it's not commutative and can help catch parameter reversal within the
    # computation.
    c = self._NewComputation()
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(self.f64_4vector))
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(self.f64_scalar_2))
    ops.Sub(p1, p0)
    self._ExecuteAndCompareClose(
        c,
        arguments=[self.f64_scalar_2, self.f64_4vector],
        expected=[[-4.3, 1.3, -6.3, 3.3]])


class BufferTest(ComputationTest):
  """Tests focusing on execution with Buffers."""

  def testConstantSum(self):
    c = self._NewComputation()
    ops.Add(
        ops.Constant(c, np.float32(1.11)), ops.Constant(c, np.float32(3.14)))
    self._ExecuteAndCompareClose(c, expected=[4.25])

  def testOneParameterSum(self):
    c = self._NewComputation()
    ops.Add(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
        ops.Constant(c, np.float32(3.14)))
    self._ExecuteAndCompareClose(
        c, arguments=[NumpyArrayF32(1.11)], expected=[4.25])

  def testTwoParameterSum(self):
    c = self._NewComputation()
    ops.Add(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
        ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0.))))
    self._ExecuteAndCompareClose(
        c,
        arguments=[NumpyArrayF32(1.11),
                   NumpyArrayF32(3.14)],
        expected=[4.25])

  def testCannotCallWithDeletedBuffers(self):
    c = self._NewComputation()
    ops.Add(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
        ops.Constant(c, np.float32(3.14)))
    arg = NumpyArrayF32(1.11)
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build())
    arg_buffer = xla_client.Buffer.from_pyval(arg)
    arg_buffer.delete()
    with self.assertRaises(RuntimeError):
      compiled_c.Execute([arg_buffer])

  def testShape(self):
    pyval = np.array([[1., 2.]], np.float32)
    local_buffer = xla_client.Buffer.from_pyval(pyval)
    xla_shape = local_buffer.shape()
    self.assertEqual(xla_shape.dimensions(), (1, 2))
    self.assertEqual(np.dtype(xla_shape.element_type()), np.dtype(np.float32))

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
        ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0])),
        ops.Constant(c, NumpyArrayF32([4.0, 5.0, 6.0])),
    )
    ops.ConcatInDim(c, args, dimension=0)
    self._ExecuteAndCompareClose(c, expected=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

  def testConcatenateF64(self):
    c = self._NewComputation()
    args = (
        ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0])),
        ops.Constant(c, NumpyArrayF64([4.0, 5.0, 6.0])),
    )
    ops.ConcatInDim(c, args, dimension=0)
    self._ExecuteAndCompareClose(c, expected=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

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
      x = ops.Constant(c, np.array(template, dtype=src_dtype))
      ops.ConvertElementType(x, xla_types[dst_dtype])

      backend = xla_client.get_local_backend()
      result = xla_client.execute_with_python_values(backend.compile(c.Build()))
      self.assertLen(result, 1)
      expected = np.array(template, dtype=dst_dtype)

      self.assertEqual(result[0].shape, expected.shape)
      self.assertEqual(result[0].dtype, expected.dtype)
      np.testing.assert_equal(result[0], expected)

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
      x = ops.Constant(c, np.array(template, dtype=src_dtype))
      ops.BitcastConvertType(x, dst_etype)

      backend = xla_client.get_local_backend()
      result = xla_client.execute_with_python_values(backend.compile(c.Build()))
      self.assertLen(result, 1)
      expected = np.array(template, src_dtype).view(dst_dtype)

      self.assertEqual(result[0].shape, expected.shape)
      self.assertEqual(result[0].dtype, expected.dtype)
      np.testing.assert_equal(result[0], expected)

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
      ops.AllToAll(ops.Constant(c, lhs), 0, 0)
      self._ExecuteAndCompareExact(c, expected=[lhs])

  def testCrossReplicaSumOneReplica(self):
    samples = [
        NumpyArrayF32(42.0),
        NumpyArrayF32([97.0]),
        NumpyArrayF32([64.0, 117.0]),
        NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]]),
    ]
    for lhs in samples:
      c = self._NewComputation()
      ops.CrossReplicaSum(ops.Constant(c, lhs))
      self._ExecuteAndCompareExact(c, expected=[lhs])

  def testReplicaId(self):
    c = self._NewComputation()
    _ = ops.ReplicaId(c)
    self._ExecuteAndCompareExact(c, expected=[0])

  def testCrossReplicaSumOneReplicaWithSingletonGroup(self):
    samples = [
        NumpyArrayF32(42.0),
        NumpyArrayF32([97.0]),
        NumpyArrayF32([64.0, 117.0]),
        NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]]),
    ]
    for lhs in samples:
      c = self._NewComputation()
      ops.CrossReplicaSum(
          ops.Constant(c, lhs), xla_client.make_replica_groups([[0]]))
      self._ExecuteAndCompareExact(c, expected=[lhs])

  def testDotMatrixVectorF32(self):
    c = self._NewComputation()
    lhs = NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF32([[10.0], [20.0]])
    ops.Dot(ops.Constant(c, lhs), ops.Constant(c, rhs))
    self._ExecuteAndCompareClose(c, expected=[np.dot(lhs, rhs)])

  def testDotMatrixVectorF64(self):
    c = self._NewComputation()
    lhs = NumpyArrayF64([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF64([[10.0], [20.0]])
    ops.Dot(ops.Constant(c, lhs), ops.Constant(c, rhs))
    self._ExecuteAndCompareClose(c, expected=[np.dot(lhs, rhs)])

  def testDotMatrixMatrixF32(self):
    c = self._NewComputation()
    lhs = NumpyArrayF32([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF32([[10.0, 20.0], [100.0, 200.0]])
    ops.Dot(ops.Constant(c, lhs), ops.Constant(c, rhs))
    self._ExecuteAndCompareClose(c, expected=[np.dot(lhs, rhs)])

  def testDotMatrixMatrixF64(self):
    c = self._NewComputation()
    lhs = NumpyArrayF64([[2.0, 3.0], [4.0, 5.0]])
    rhs = NumpyArrayF64([[10.0, 20.0], [100.0, 200.0]])
    ops.Dot(ops.Constant(c, lhs), ops.Constant(c, rhs))
    self._ExecuteAndCompareClose(c, expected=[np.dot(lhs, rhs)])

  def testDotGeneral(self):
    c = self._NewComputation()
    rng = np.random.RandomState(0)
    lhs = NumpyArrayF32(rng.randn(10, 3, 4))
    rhs = NumpyArrayF32(rng.randn(10, 4, 5))
    dimension_numbers = xla_client.make_dot_dimension_numbers(
        (([2], [1]), ([0], [0])))
    ops.DotGeneral(
        ops.Constant(c, lhs), ops.Constant(c, rhs), dimension_numbers)
    self._ExecuteAndCompareClose(c, expected=[np.matmul(lhs, rhs)], rtol=1e-6)

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

    ops.DotGeneral(
        ops.Constant(c, lhs), ops.Constant(c, rhs), dimension_numbers)
    self._ExecuteAndCompareClose(c, expected=[np.matmul(lhs, rhs)], rtol=1e-6)

  def testDotGeneralWithPrecisionConfig(self):
    c = self._NewComputation()
    rng = np.random.RandomState(0)
    lhs = NumpyArrayF32(rng.randn(10, 3, 4))
    rhs = NumpyArrayF32(rng.randn(10, 4, 5))
    dimension_numbers = xla_client.make_dot_dimension_numbers(
        (([2], [1]), ([0], [0])))
    config = xla_client.PrecisionConfig()
    config.operand_precision.append(config.Precision.HIGH)
    config.operand_precision.append(config.Precision.HIGHEST)
    ops.DotGeneral(
        ops.Constant(c, lhs),
        ops.Constant(c, rhs),
        dimension_numbers,
        precision_config=config)
    self._ExecuteAndCompareClose(c, expected=[np.matmul(lhs, rhs)], rtol=1e-6)

  def testConvGeneralDilatedF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = xla_client.make_convolution_dimension_numbers(
        ("NCHW", "OIHW", "NCHW"), 2)
    ops.ConvGeneralDilated(
        ops.Constant(c, lhs), ops.Constant(c, rhs), strides, pads, lhs_dilation,
        rhs_dilation, dimension_numbers)
    result = np.array([[[
        [0., 0., 0.],
        [10., 20., 0.],
        [0., 0., 0.],
        [40., 50., 0.],
    ]]])
    self._ExecuteAndCompareClose(c, expected=[result])

  def testConvGeneralDilatedF32WithPrecisionConfig(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = xla_client.make_convolution_dimension_numbers(
        ("NCHW", "OIHW", "NCHW"), 2)
    config = xla_client.PrecisionConfig()
    config.operand_precision.append(config.Precision.HIGHEST)
    config.operand_precision.append(config.Precision.DEFAULT)
    ops.ConvGeneralDilated(
        ops.Constant(c, lhs),
        ops.Constant(c, rhs),
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
    self._ExecuteAndCompareClose(c, expected=[result])

  def testConvGeneralDilatedPermutedF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 1, 2, 3)
    rhs = a(1, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)

    dimension_numbers = xla_client.make_convolution_dimension_numbers(
        ("NHWC", "OIHW", "CWNH"), 2)
    ops.ConvGeneralDilated(
        ops.Constant(c, np.transpose(lhs, (0, 2, 3, 1))), ops.Constant(c, rhs),
        strides, pads, lhs_dilation, rhs_dilation, dimension_numbers)
    result = np.array([[[[0., 0., 0.], [10., 20., 0.], [0., 0., 0.],
                         [40., 50., 0.]]]])
    self._ExecuteAndCompareClose(
        c, expected=[np.transpose(result, (1, 3, 0, 2))])

  def testConvGeneralDilatedGroupedConvolutionF32(self):
    c = self._NewComputation()
    a = lambda *dims: np.arange(np.prod(dims)).reshape(dims).astype("float32")
    lhs = a(1, 2, 2, 3)
    rhs = a(2, 1, 1, 2) * 10
    strides = [1, 1]
    pads = [(1, 0), (0, 1)]
    lhs_dilation = (2, 1)
    rhs_dilation = (1, 1)
    dimension_numbers = xla_client.make_convolution_dimension_numbers(
        ("NCHW", "OIHW", "NCHW"), 2)
    feature_group_count = 2
    ops.ConvGeneralDilated(
        ops.Constant(c, lhs), ops.Constant(c, rhs), strides, pads, lhs_dilation,
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
    self._ExecuteAndCompareClose(c, expected=[result])

  def testBooleanNot(self):
    c = self._NewComputation()
    arr = NumpyArrayBool([True, False, True])
    ops.Not(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[~arr])

  def testPopulationCount(self):
    c = self._NewComputation()
    arr = NumpyArrayS32([3, 0, 1])
    ops.PopulationCount(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.array([2, 0, 1])])

  def testCountLeadingZeros(self):
    c = self._NewComputation()
    arr = NumpyArrayS32([0x7FFF, 0x12345678])
    ops.Clz(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[[17, 3]])

  def testExp(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Exp(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.exp(arr)])

  def testExpm1(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Expm1(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.expm1(arr)])

  def testRound(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Round(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.round(arr)])

  def testLog(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Log(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.log(arr)])

  def testLog1p(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Log1p(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.log1p(arr)])

  def testNeg(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Neg(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[-arr])

  def testFloor(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Floor(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.floor(arr)])

  def testCeil(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Ceil(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.ceil(arr)])

  def testAbs(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, -12.1, 2.4, -1.])
    ops.Abs(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.abs(arr)])

  def testTanh(self):
    c = self._NewComputation()
    arr = NumpyArrayF32([3.3, 12.1])
    ops.Tanh(ops.Constant(c, arr))
    self._ExecuteAndCompareClose(c, expected=[np.tanh(arr)])

  def testTranspose(self):

    def _TransposeAndTest(array, permutation):
      c = self._NewComputation()
      ops.Transpose(ops.Constant(c, array), permutation)
      expected = np.transpose(array, permutation)
      self._ExecuteAndCompareClose(c, expected=[expected])

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
    ops.Eq(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4])),
        ops.Constant(c, NumpyArrayS32([4, 2, 3, 1])))
    self._ExecuteAndCompareExact(c, expected=[[False, True, True, False]])

  def testNe(self):
    c = self._NewComputation()
    ops.Ne(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4])),
        ops.Constant(c, NumpyArrayS32([4, 2, 3, 1])))
    self._ExecuteAndCompareExact(c, expected=[[True, False, False, True]])

    ops.Ne(
        ops.Constant(c, NumpyArrayF32([-2.0, 0.0,
                                       float("nan"),
                                       float("nan")])),
        ops.Constant(c, NumpyArrayF32([2.0, -0.0, 1.0,
                                       float("nan")])))
    self._ExecuteAndAssertWith(
        np.testing.assert_allclose, c, (), expected=[[True, False, True, True]])

  def testGt(self):
    c = self._NewComputation()
    ops.Gt(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4, 9])),
        ops.Constant(c, NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(
        c, expected=[[False, True, True, False, False]])

  def testGe(self):
    c = self._NewComputation()
    ops.Ge(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4, 9])),
        ops.Constant(c, NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[[True, True, True, False, False]])

  def testLt(self):
    c = self._NewComputation()
    ops.Lt(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4, 9])),
        ops.Constant(c, NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(
        c, expected=[[False, False, False, True, True]])

  def testLe(self):
    c = self._NewComputation()
    ops.Le(
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4, 9])),
        ops.Constant(c, NumpyArrayS32([1, 0, 2, 7, 12])))
    self._ExecuteAndCompareExact(c, expected=[[True, False, False, True, True]])

  def testMax(self):
    c = self._NewComputation()
    ops.Max(
        ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0, 9.0])),
        ops.Constant(c, NumpyArrayF32([1.0, 0.0, 2.0, 7.0, 12.0])))
    self._ExecuteAndCompareExact(c, expected=[[1.0, 2.0, 3.0, 7.0, 12.0]])

  def testMaxExplicitBroadcastDim0(self):
    c = self._NewComputation()
    ops.Max(
        ops.Constant(c, NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF32([3, 4, 5])),
        broadcast_dimensions=(0,))
    self._ExecuteAndCompareExact(
        c, expected=[[[3, 3, 3], [4, 5, 6], [7, 8, 9]]])

  def testMaxExplicitBroadcastDim1(self):
    c = self._NewComputation()
    ops.Max(
        ops.Constant(c, NumpyArrayF32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayF32([3, 4, 5])),
        broadcast_dimensions=(1,))
    self._ExecuteAndCompareExact(
        c, expected=[[[3, 4, 5], [4, 5, 6], [7, 8, 9]]])

  def testMin(self):
    c = self._NewComputation()
    ops.Min(
        ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0, 9.0])),
        ops.Constant(c, NumpyArrayF32([1.0, 0.0, 2.0, 7.0, 12.0])))
    self._ExecuteAndCompareExact(c, expected=[[1.0, 0.0, 2.0, 4.0, 9.0]])

  def testPad(self):
    c = self._NewComputation()
    ops.Pad(
        ops.Constant(c, NumpyArrayF32([[1.0, 2.0], [3.0, 4.0]])),
        ops.Constant(c, NumpyArrayF32(0.0)),
        xla_client.make_padding_config([(1, 2, 1), (0, 1, 0)]))
    self._ExecuteAndCompareClose(
        c,
        expected=[[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0],
                   [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

  def testPadWithPaddingConfig(self):
    c = self._NewComputation()
    padding_config = xla_client.PaddingConfig()
    for lo, hi, interior in [(1, 2, 1), (0, 1, 0)]:
      dimension = xla_client.PaddingConfigDimension()
      dimension.edge_padding_low = lo
      dimension.edge_padding_high = hi
      dimension.interior_padding = interior
      padding_config.dimensions.append(dimension)
    ops.Pad(
        ops.Constant(c, NumpyArrayF32([[1.0, 2.0], [3.0, 4.0]])),
        ops.Constant(c, NumpyArrayF32(0.0)), padding_config)
    self._ExecuteAndCompareClose(
        c,
        expected=[[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0],
                   [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

  def testReshape(self):
    c = self._NewComputation()
    ops.Reshape(
        ops.Constant(c, NumpyArrayS32([[1, 2], [3, 4], [5, 6]])),
        dimensions=[0, 1],
        new_sizes=[2, 3])
    self._ExecuteAndCompareExact(c, expected=[[[1, 2, 3], [4, 5, 6]]])

  def testCollapse(self):
    c = self._NewComputation()
    ops.Collapse(
        ops.Constant(c, NumpyArrayS32([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
        dimensions=[1, 2])
    self._ExecuteAndCompareExact(c, expected=[[[1, 2, 3, 4], [5, 6, 7, 8]]])

  def testRev(self):
    c = self._NewComputation()
    ops.Rev(
        ops.Constant(c, NumpyArrayS32([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
        dimensions=[0, 2])
    self._ExecuteAndCompareExact(
        c, expected=[[[[6, 5], [8, 7]], [[2, 1], [4, 3]]]])

  def testReducePrecision(self):
    c = self._NewComputation()
    ops.ReducePrecision(
        ops.Constant(c, NumpyArrayF32([float.fromhex("0x1.32fffep-3")])),
        exponent_bits=8,
        mantissa_bits=7)
    self._ExecuteAndCompareClose(c, expected=[[float.fromhex("0x1.32p-3")]])

  def testClampF32(self):
    c = self._NewComputation()
    ops.Clamp(
        ops.Constant(c, NumpyArrayF32(-1)),
        ops.Constant(c, NumpyArrayF32([-2, -1, 0, 1, 2, 3])),
        ops.Constant(c, NumpyArrayF32(2)))
    self._ExecuteAndCompareExact(c, expected=[[-1, -1, 0, 1, 2, 2]])

  def testClampS32(self):
    c = self._NewComputation()
    ops.Clamp(
        ops.Constant(c, NumpyArrayS32(-1)),
        ops.Constant(c, NumpyArrayS32([-2, -1, 0, 1, 2, 3])),
        ops.Constant(c, NumpyArrayS32(2)))
    self._ExecuteAndCompareExact(c, expected=[[-1, -1, 0, 1, 2, 2]])

  def testSelect(self):
    c = self._NewComputation()
    ops.Select(
        ops.Constant(c, NumpyArrayBool([True, False, False, True, False])),
        ops.Constant(c, NumpyArrayS32([1, 2, 3, 4, 5])),
        ops.Constant(c, NumpyArrayS32([-1, -2, -3, -4, -5])))
    self._ExecuteAndCompareExact(c, expected=[[1, -2, -3, 4, -5]])

  def testSlice(self):
    c = self._NewComputation()
    ops.Slice(
        ops.Constant(c, NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        [1, 0], [3, 2], [1, 1])
    self._ExecuteAndCompareExact(c, expected=[[[4, 5], [7, 8]]])

  def testSliceInDim(self):
    c = self._NewComputation()
    ops.SliceInDim(
        ops.Constant(c, NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        start_index=1,
        limit_index=2,
        stride=1,
        dimno=1)
    self._ExecuteAndCompareExact(c, expected=[[[2], [5], [8]]])
    ops.SliceInDim(
        ops.Constant(c, NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        start_index=0,
        limit_index=3,
        stride=2,
        dimno=0)
    self._ExecuteAndCompareExact(c, expected=[[[1, 2, 3], [7, 8, 9]]])

  def testDynamicSlice(self):
    c = self._NewComputation()
    ops.DynamicSlice(
        ops.Constant(c, NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        [ops.Constant(c, NumpyArrayS32([1, 0]))], [2, 2])
    self._ExecuteAndCompareExact(c, expected=[[[4, 5], [7, 8]]])

  def testDynamicUpdateSlice(self):
    c = self._NewComputation()
    ops.DynamicUpdateSlice(
        ops.Constant(c, NumpyArrayS32([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ops.Constant(c, NumpyArrayS32([[1, 2], [3, 4]])),
        [ops.Constant(c, NumpyArrayS32([1, 1]))])
    self._ExecuteAndCompareExact(
        c, expected=[[[1, 2, 3], [4, 1, 2], [7, 3, 4]]])

  def testTuple(self):
    c = self._NewComputation()
    ops.Tuple(c, [
        ops.Constant(c, np.int32(42)),
        ops.Constant(c, NumpyArrayF32([1.0, 2.0])),
        ops.Constant(c, NumpyArrayBool([True, False, False, True]))
    ])
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    self.assertLen(result, 3)
    np.testing.assert_equal(result[0], 42)
    np.testing.assert_allclose(result[1], [1.0, 2.0])
    np.testing.assert_equal(result[2], [True, False, False, True])

  def testGetTupleElement(self):
    c = self._NewComputation()
    ops.GetTupleElement(
        ops.Tuple(c, [
            ops.Constant(c, np.int32(42)),
            ops.Constant(c, NumpyArrayF32([1.0, 2.0])),
            ops.Constant(c, NumpyArrayBool([True, False, False, True]))
        ]), 1)
    self._ExecuteAndCompareClose(c, expected=[[1.0, 2.0]])

  def testBroadcast(self):
    c = self._NewComputation()
    ops.Broadcast(ops.Constant(c, NumpyArrayS32([10, 20, 30, 40])), sizes=(3,))
    self._ExecuteAndCompareExact(
        c, expected=[[[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]]])

  def testBroadcastInDim(self):
    c = self._NewComputation()
    ops.BroadcastInDim(ops.Constant(c, NumpyArrayS32([1, 2])), [2, 2], [0])
    self._ExecuteAndCompareExact(c, expected=[[[1, 1], [2, 2]]])
    ops.BroadcastInDim(ops.Constant(c, NumpyArrayS32([1, 2])), [2, 2], [1])
    self._ExecuteAndCompareExact(c, expected=[[[1, 2], [1, 2]]])

  def testRngNormal(self):
    shape = (2, 3)
    c = self._NewComputation()
    ops.RngNormal(
        ops.Constant(c, NumpyArrayF32(0.)),
        ops.Constant(c, NumpyArrayF32(1.)),
        shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, shape))
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    # since the result is random, we just check shape and uniqueness
    self.assertLen(result, 1)
    self.assertEqual(result[0].shape, shape)
    self.assertLen(np.unique(result[0]), np.prod(shape))

  def testRngUniformF32(self):
    lo, hi = 2., 4.
    shape = (2, 3)
    c = self._NewComputation()
    ops.RngUniform(
        ops.Constant(c, NumpyArrayF32(lo)),
        ops.Constant(c, NumpyArrayF32(hi)),
        shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, shape))
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    # since the result is random, we just check shape, uniqueness, and range
    self.assertLen(result, 1)
    self.assertEqual(result[0].shape, shape)
    self.assertLen(np.unique(result[0]), np.prod(shape))
    self.assertTrue(np.all(lo <= result[0]))
    self.assertTrue(np.all(result[0] < hi))

  def testRngUniformS32(self):
    lo, hi = 2, 4
    shape = (2, 3)
    c = self._NewComputation()
    ops.RngUniform(
        ops.Constant(c, NumpyArrayS32(lo)),
        ops.Constant(c, NumpyArrayS32(hi)),
        shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.S32, shape))
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    # since the result is random, we just check shape, integrality, and range
    self.assertLen(result, 1)
    self.assertEqual(result[0].shape, shape)
    self.assertEqual(result[0].dtype, np.int32)
    self.assertTrue(np.all(lo <= result[0]))
    self.assertTrue(np.all(result[0] < hi))

  def testCholesky(self):
    l = np.array([[4, 0, 0, 0], [6, 5, 0, 0], [2, 14, 16, 0], [3, 6, 1, 4]],
                 dtype=np.float32)
    c = self._NewComputation()
    ops.Cholesky(ops.Constant(c, np.dot(l, l.T)))
    self._ExecuteAndCompareClose(c, expected=[l], rtol=1e-4)

  def testSort(self):
    keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
    c = self._NewComputation()
    ops.Sort(c, [ops.Constant(c, keys)])
    self._ExecuteAndCompareClose(
        c, expected=[np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32)])

  def testSortKeyVal(self):
    keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
    values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    c = self._NewComputation()
    ops.Sort(c, (ops.Constant(c, keys), ops.Constant(c, values)), dimension=0)
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    self.assertLen(result, 2)
    np.testing.assert_allclose(result[0], [[2, 1, 1, 2], [3, 4, 4, 3]])
    np.testing.assert_equal(result[1], [[0, 5, 2, 7], [4, 1, 6, 3]])

  def testSortCustomComparator(self):
    b = self._NewComputation("comparator")
    p0 = ops.Parameter(b, 0, xla_client.shape_from_pyval(NumpyArrayF32(0)))
    q0 = ops.Parameter(b, 1, xla_client.shape_from_pyval(NumpyArrayF32(0)))
    p1 = ops.Parameter(b, 2, xla_client.shape_from_pyval(NumpyArrayS32(0)))
    q1 = ops.Parameter(b, 3, xla_client.shape_from_pyval(NumpyArrayS32(0)))
    ops.Or(ops.Lt(p0, q0), ops.And(ops.Eq(p0, q0), ops.Gt(p1, q1)))
    comparator = b.Build()

    keys = np.array([[2, 3, 1, 3], [3, 1, 2, 2]], dtype=np.float32)
    values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    c = self._NewComputation()
    ops.Sort(
        c, (ops.Constant(c, keys), ops.Constant(c, values)),
        dimension=1,
        comparator=comparator)
    backend = xla_client.get_local_backend()
    result = xla_client.execute_with_python_values(backend.compile(c.Build()))
    self.assertLen(result, 2)
    np.testing.assert_allclose(result[0], [[1, 2, 3, 3], [1, 2, 2, 3]])
    np.testing.assert_equal(result[1], [[2, 0, 3, 1], [5, 7, 6, 4]])

  def testQR(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    c = self._NewComputation()
    ops.Tuple(c, ops.QR(ops.Constant(c, a), full_matrices=True))
    q, r = self._Execute(c, ())
    np.testing.assert_allclose(np.dot(q, r), a, rtol=1e-4)

  def testEigh(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    a = (a + a.T) / 2

    c = self._NewComputation()
    ops.Tuple(c, ops.Eigh(ops.Constant(c, a), lower=True))
    # TODO(b/129396575): Turn this test back on when it passes without fastmath.
    # v, w = self._Execute(c, ())
    # self.assertLess(np.linalg.norm(np.dot(a, v) - w * v), 1e-3)

  def testSVD(self):
    a = np.array(
        [[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166], [10, 63, 166, 310]],
        dtype=np.float32)
    c = self._NewComputation()
    ops.Tuple(c, ops.SVD(ops.Constant(c, a)))
    u, d, v = self._Execute(c, ())
    self.assertLess(np.linalg.norm(a - np.matmul(u * d, v.T)), 1e-3)

  def testTriangularSolve(self):
    a_vals = np.array(
        [[2, 0, 0, 0], [3, 6, 0, 0], [4, 7, 9, 0], [5, 8, 10, 11]],
        dtype=np.float32)
    b_vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      dtype=np.float32)

    c = self._NewComputation()
    ops.TriangularSolve(
        ops.Constant(c, a_vals),
        ops.Constant(c, b_vals),
        left_side=False,
        lower=True,
        transpose_a=ops.TriangularSolveOptions_Transpose.TRANSPOSE,
        unit_diagonal=False)
    self._ExecuteAndCompareClose(
        c,
        expected=[
            np.array([
                [0.5, 0.08333334, 0.04629629, 0.03367003],
                [2.5, -0.25, -0.1388889, -0.1010101],
                [4.5, -0.58333331, -0.32407406, -0.23569024],
            ],
                     dtype=np.float32)
        ],
        rtol=1e-4)

  def testIsConstant(self):
    c = self._NewComputation()
    a = ops.Constant(c, np.int32(3))
    b = ops.Constant(c, np.int32(1))
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayS32(0)))
    const_expr = ops.Sub(b, a)
    non_const_expr = ops.Mul(const_expr, x)
    self.assertTrue(c.IsConstant(const_expr))
    self.assertFalse(c.IsConstant(non_const_expr))

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
    ops.Gather(
        ops.Constant(c, a), ops.Constant(c, indices), dnums, slice_sizes=[1, 1])
    g, = self._Execute(c, ())
    expected = np.array([[[[2, 7]]], [[[5, 6]]]], dtype=np.int32)
    np.testing.assert_allclose(g, expected, rtol=1e-4)

  def testFft(self):
    shape = [2, 3, 4, 5]
    rng = np.random.RandomState(0)
    a = rng.randn(*shape) + 1.0j * rng.randn(*shape)
    a = a.astype(np.complex64)
    # FFT
    c = self._NewComputation()
    ops.Fft(ops.Constant(c, a), xla_client.FftType.FFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=[np.fft.fftn(a, axes=(1, 2, 3))], rtol=1e-4)
    # IFFT
    c = self._NewComputation()
    ops.Fft(ops.Constant(c, a), xla_client.FftType.IFFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=[np.fft.ifftn(a, axes=(1, 2, 3))], rtol=1e-4)
    # RFFT
    b = rng.randn(*shape).astype(np.float32)
    c = self._NewComputation()
    ops.Fft(ops.Constant(c, b), xla_client.FftType.RFFT, shape[-3:])
    self._ExecuteAndCompareClose(
        c, expected=[np.fft.rfftn(b, axes=(1, 2, 3))], rtol=1e-4)
    # IRFFT
    c = self._NewComputation()
    ops.Fft(ops.Constant(c, a), xla_client.FftType.IRFFT, [3, 4, 8])
    self._ExecuteAndCompareClose(
        c, expected=[np.fft.irfftn(a, axes=(1, 2, 3))], rtol=1e-4)

  def testNextAfter(self):
    c = self._NewComputation()
    ops.NextAfter(
        ops.Constant(c, np.array([1, 2], dtype=np.float32)),
        ops.Constant(c, np.array([2, 1], dtype=np.float32)))
    out, = self._Execute(c, ())
    eps = np.finfo(np.float32).eps
    np.testing.assert_equal(np.array([eps + 1, 2 - eps], dtype=np.float32), out)

  def testRegularizedIncompleteBeta(self):
    x = np.array([0.53787335, 0.24015466, 0.47494545, 0.13567594, 0.95114538])
    a = np.array([0.00753073, 0.34813385, 0.30485708, 1.29298632, 0.51472606])
    b = np.array([0.55688389, 0.59794214, 0.42661022, 1.59748339, 0.95047677])
    c = self._NewComputation()
    ops.RegularizedIncompleteBeta(
        ops.Constant(c, a), ops.Constant(c, b), ops.Constant(c, x))
    expected = np.array(
        [0.98923271, 0.48575411, 0.57952568, 0.12579775, 0.96989155])
    self._ExecuteAndCompareClose(c, expected=[expected], rtol=1e-4)


class EmbeddedComputationsTest(ComputationTest):
  """Tests for XLA graphs with embedded computations (such as maps)."""

  def _CreateConstantS32Computation(self):
    """Computation (f32) -> s32 that returns a constant 1 for any input."""
    c = self._NewComputation("constant_s32_one")
    # TODO(eliben): consider adding a nicer way to create new parameters without
    # having to create dummy Numpy arrays or populating Shape messages. Perhaps
    # we need our own (Python-client-own) way to represent Shapes conveniently.
    ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0)))
    ops.Constant(c, np.int32(1))
    return c.Build()

  def _CreateConstantS64Computation(self):
    """Computation (f64) -> s64 that returns a constant 1 for any input."""
    c = self._NewComputation("constant_s64_one")
    # TODO(eliben): consider adding a nicer way to create new parameters without
    # having to create dummy Numpy arrays or populating Shape messages. Perhaps
    # we need our own (Python-client-own) way to represent Shapes conveniently.
    ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF64(0)))
    ops.Constant(c, np.int64(1))
    return c.Build()

  def _CreateConstantF32Computation(self):
    """Computation (f32) -> f32 that returns a constant 1.0 for any input."""
    c = self._NewComputation("constant_f32_one")
    ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0)))
    ops.Constant(c, np.float32(1.0))
    return c.Build()

  def _CreateConstantF64Computation(self):
    """Computation (f64) -> f64 that returns a constant 1.0 for any input."""
    c = self._NewComputation("constant_f64_one")
    ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF64(0)))
    ops.Constant(c, np.float64(1.0))
    return c.Build()

  def _CreateMulF32By2Computation(self):
    """Computation (f32) -> f32 that multiplies its parameter by 2."""
    c = self._NewComputation("mul_f32_by2")
    ops.Mul(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF32(0)).with_major_to_minor_layout_if_absent()),
        ops.Constant(c, np.float32(2.0)))
    return c.Build()

  def _CreateMulF32ByParamComputation(self):
    """Computation (f32) -> f32 that multiplies one parameter by the other."""
    c = self._NewComputation("mul_f32_by_param")
    ops.Mul(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0))),
        ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0))))
    return c.Build()

  def _CreateMulF64By2Computation(self):
    """Computation (f64) -> f64 that multiplies its parameter by 2."""
    c = self._NewComputation("mul_f64_by2")
    ops.Mul(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF64(0)).with_major_to_minor_layout_if_absent()),
        ops.Constant(c, np.float64(2.0)))
    return c.Build()

  def _CreateBinaryAddS32Computation(self):
    """Computation (s32, s32) -> s32 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    ops.Add(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayS32(0))),
        ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayS32(0))))
    return c.Build()

  def _CreateBinaryAddF32Computation(self):
    """Computation (f32, f32) -> f32 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    ops.Add(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF32(0)).with_major_to_minor_layout_if_absent()),
        ops.Parameter(
            c, 1,
            xla_client.shape_from_pyval(
                NumpyArrayF32(0)).with_major_to_minor_layout_if_absent()))
    return c.Build()

  def _CreateBinaryAddF64Computation(self):
    """Computation (f64, f64) -> f64 that adds its two parameters."""
    c = self._NewComputation("add_param0_by_param1")
    ops.Add(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF64(0)).with_major_to_minor_layout_if_absent()),
        ops.Parameter(
            c, 1,
            xla_client.shape_from_pyval(
                NumpyArrayF64(0)).with_major_to_minor_layout_if_absent()))
    return c.Build()

  def _CreateBinaryDivF32Computation(self):
    """Computation (f32, f32) -> f32 that divides its two parameters."""
    c = self._NewComputation("div_param0_by_param1")
    ops.Div(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0))),
        ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0))))
    return c.Build()

  def _CreateBinaryDivF64Computation(self):
    """Computation (f64, f64) -> f64 that divides its two parameters."""
    c = self._NewComputation("div_param0_by_param1")
    ops.Div(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF64(0))),
        ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF64(0))))
    return c.Build()

  def _CreateTestF32Lt10Computation(self):
    """Computation (f32) -> bool that tests if its parameter is less than 10."""
    c = self._NewComputation("test_f32_lt_10")
    ops.Lt(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0))),
        ops.Constant(c, np.float32(10.)))
    return c.Build()

  def _CreateTestF64Lt10Computation(self):
    """Computation (f64) -> bool that tests if its parameter is less than 10."""
    c = self._NewComputation("test_f64_lt_10")
    ops.Lt(
        ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF64(0))),
        ops.Constant(c, np.float64(10.)))
    return c.Build()

  def _CreateBinaryGeF32Computation(self):
    """Computation (f32, f32) -> bool that tests first_param >= second_param."""
    c = self._NewComputation("param0_lt_param1")
    ops.Ge(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF32(0)).with_major_to_minor_layout_if_absent()),
        ops.Parameter(
            c, 1,
            xla_client.shape_from_pyval(
                NumpyArrayF32(0)).with_major_to_minor_layout_if_absent()))
    return c.Build()

  def _CreateBinaryGeF64Computation(self):
    """Computation (f64, f64) -> bool that tests first_param >= second_param."""
    c = self._NewComputation("param0_lt_param1")
    ops.Ge(
        ops.Parameter(
            c, 0,
            xla_client.shape_from_pyval(
                NumpyArrayF64(0)).with_major_to_minor_layout_if_absent()),
        ops.Parameter(
            c, 1,
            xla_client.shape_from_pyval(
                NumpyArrayF64(0)).with_major_to_minor_layout_if_absent()))
    return c.Build()

  def _MakeSample3DArrayF32(self):
    return NumpyArrayF32([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                          [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

  def _MakeSample3DArrayF64(self):
    return NumpyArrayF64([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                          [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

  def testCallF32(self):
    c = self._NewComputation()
    ops.Call(
        c,
        self._CreateMulF32By2Computation(),
        operands=(ops.Constant(c, np.float32(5.0)),))
    self._ExecuteAndCompareClose(c, expected=[10.0])

  def testCallF64(self):
    c = self._NewComputation()
    ops.Call(
        c,
        self._CreateMulF64By2Computation(),
        operands=(ops.Constant(c, np.float64(5.0)),))
    self._ExecuteAndCompareClose(c, expected=[10.0])

  def testMapEachElementToS32Constant(self):
    c = self._NewComputation()
    ops.Map(c, [ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
            self._CreateConstantS32Computation(), [0])
    self._ExecuteAndCompareExact(c, expected=[[1, 1, 1, 1]])

  def testMapEachElementToS64Constant(self):
    c = self._NewComputation()
    ops.Map(c, [ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
            self._CreateConstantS64Computation(), [0])
    self._ExecuteAndCompareExact(c, expected=[[1, 1, 1, 1]])

  def testMapMulBy2F32(self):
    c = self._NewComputation()
    ops.Map(c, [ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
            self._CreateMulF32By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[2.0, 4.0, 6.0, 8.0]])

  def testMapMulBy2F64(self):
    c = self._NewComputation()
    ops.Map(c, [ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
            self._CreateMulF64By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[2.0, 4.0, 6.0, 8.0]])

  def testSimpleMapChainF32(self):
    # Chains a map of constant-f32 with a map of mul-by-2
    c = self._NewComputation()
    const_f32 = ops.Map(c,
                        [ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
                        self._CreateConstantF32Computation(), [0])
    ops.Map(c, [const_f32], self._CreateMulF32By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[2.0, 2.0, 2.0, 2.0]])

  def testSimpleMapChainF64(self):
    # Chains a map of constant-f64 with a map of mul-by-2
    c = self._NewComputation()
    const_f64 = ops.Map(c,
                        [ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
                        self._CreateConstantF64Computation(), [0])
    ops.Map(c, [const_f64], self._CreateMulF64By2Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[2.0, 2.0, 2.0, 2.0]])

  def testDivVectorsWithMapF32(self):
    c = self._NewComputation()
    ops.Map(c, (ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0])),
                ops.Constant(c, NumpyArrayF32([5.0, 5.0, 4.0, 4.0]))),
            self._CreateBinaryDivF32Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[0.2, 0.4, 0.75, 1.0]])

  def testDivVectorsWithMapF64(self):
    c = self._NewComputation()
    ops.Map(c, (ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0, 4.0])),
                ops.Constant(c, NumpyArrayF64([5.0, 5.0, 4.0, 4.0]))),
            self._CreateBinaryDivF64Computation(), [0])
    self._ExecuteAndCompareClose(c, expected=[[0.2, 0.4, 0.75, 1.0]])

  def testSelectAndScatterF32(self):
    c = self._NewComputation()
    operand = ops.Constant(c, NumpyArrayF32([[1., 2., 6.], [4., 5., 3.]]))
    window_dimensions = (2, 1)
    window_strides = (1, 2)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID,
        c.GetShape(operand).dimensions(), window_dimensions, window_strides)
    ops.SelectAndScatterWithGeneralPadding(
        operand,
        select=self._CreateBinaryGeF32Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        source=ops.Constant(c, NumpyArrayF32([[0.1, 0.2]])),
        init_value=ops.Constant(c, NumpyArrayF32(1)),
        scatter=self._CreateBinaryAddF32Computation())
    self._ExecuteAndCompareClose(c, expected=[[[1., 1., 1.2], [1.1, 1., 1.]]])

  def testSelectAndScatterF64(self):
    c = self._NewComputation()
    operand = ops.Constant(c, NumpyArrayF64([[1., 2., 6.], [4., 5., 3.]]))
    window_dimensions = (2, 1)
    window_strides = (1, 2)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID,
        c.GetShape(operand).dimensions(), window_dimensions, window_strides)
    ops.SelectAndScatterWithGeneralPadding(
        operand,
        select=self._CreateBinaryGeF64Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        source=ops.Constant(c, NumpyArrayF64([[0.1, 0.2]])),
        init_value=ops.Constant(c, NumpyArrayF64(1)),
        scatter=self._CreateBinaryAddF64Computation())
    self._ExecuteAndCompareClose(c, expected=[[[1., 1., 1.2], [1.1, 1., 1.]]])

  def testReduce1DtoScalarF32(self):
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, NumpyArrayF32([1.0, 2.0, 3.0, 4.0]))],
        init_values=[ops.Constant(c, np.float32(0))],
        computation=self._CreateBinaryAddF32Computation(),
        dimensions_to_reduce=[0])
    self._ExecuteAndCompareClose(c, expected=[10])

  def testReduce1DtoScalarF64(self):
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, NumpyArrayF64([1.0, 2.0, 3.0, 4.0]))],
        init_values=[ops.Constant(c, np.float64(0))],
        computation=self._CreateBinaryAddF64Computation(),
        dimensions_to_reduce=[0])
    self._ExecuteAndCompareClose(c, expected=[10])

  def testReduce2DTo1DDim0F32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, input_array)],
        init_values=[ops.Constant(c, np.float32(0))],
        computation=self._CreateBinaryAddF32Computation(),
        dimensions_to_reduce=[0])
    self._ExecuteAndCompareClose(c, expected=[[5, 7, 9]])

  def testReduce2DTo1DDim0F64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, input_array)],
        init_values=[ops.Constant(c, np.float64(0))],
        computation=self._CreateBinaryAddF64Computation(),
        dimensions_to_reduce=[0])
    self._ExecuteAndCompareClose(c, expected=[[5, 7, 9]])

  def testReduce2DTo1DDim1F32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, input_array)],
        init_values=[ops.Constant(c, np.float32(0))],
        computation=self._CreateBinaryAddF32Computation(),
        dimensions_to_reduce=[1])
    self._ExecuteAndCompareClose(c, expected=[[6, 15]])

  def testReduce2DTo1DDim1F64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    ops.Reduce(
        c,
        operands=[ops.Constant(c, input_array)],
        init_values=[ops.Constant(c, np.float64(0))],
        computation=self._CreateBinaryAddF64Computation(),
        dimensions_to_reduce=[1])
    self._ExecuteAndCompareClose(c, expected=[[6, 15]])

  def testReduce3DAllPossibleWaysF32(self):
    input_array = self._MakeSample3DArrayF32()

    def _ReduceAndTest(*dims):
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[ops.Constant(c, input_array)],
          init_values=[ops.Constant(c, np.float32(0))],
          computation=self._CreateBinaryAddF32Computation(),
          dimensions_to_reduce=dims)
      self._ExecuteAndCompareClose(
          c, expected=[np.sum(input_array, axis=tuple(dims))])

    _ReduceAndTest(0)
    _ReduceAndTest(0, 1)
    _ReduceAndTest(0, 2)
    _ReduceAndTest(1, 2)
    _ReduceAndTest(0, 1, 2)

  def testReduce3DAllPossibleWaysF64(self):
    input_array = self._MakeSample3DArrayF64()

    def _ReduceAndTest(*dims):
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[ops.Constant(c, input_array)],
          init_values=[ops.Constant(c, np.float64(0))],
          computation=self._CreateBinaryAddF64Computation(),
          dimensions_to_reduce=dims)
      self._ExecuteAndCompareClose(
          c, expected=[np.sum(input_array, axis=tuple(dims))])

    _ReduceAndTest(0)
    _ReduceAndTest(0)
    _ReduceAndTest(0, 1)
    _ReduceAndTest(0, 2)
    _ReduceAndTest(1, 2)
    _ReduceAndTest(0, 1, 2)

  def testReduceWindowValidUnitStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 1)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float32(0)),
        computation=self._CreateBinaryAddF32Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.]]])

  def testReduceWindowSameUnitStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 1)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.SAME, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float32(0)),
        computation=self._CreateBinaryAddF32Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.], [4., 5., 6.]]])

  def testReduceWindowValidGeneralStridesF32(self):
    input_array = NumpyArrayF32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 2)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float32(0)),
        computation=self._CreateBinaryAddF32Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 9.]]])

  def testReduceWindowValidUnitStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 1)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float64(0)),
        computation=self._CreateBinaryAddF64Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.]]])

  def testReduceWindowSameUnitStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 1)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.SAME, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float64(0)),
        computation=self._CreateBinaryAddF64Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.], [4., 5., 6.]]])

  def testReduceWindowValidGeneralStridesF64(self):
    input_array = NumpyArrayF64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = self._NewComputation()
    window_dimensions = (2, 1)
    window_strides = (1, 2)
    padding = xla_client.window_padding_type_to_pad_values(
        xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
        window_strides)
    ops.ReduceWindowWithGeneralPadding(
        operand=ops.Constant(c, input_array),
        init_value=ops.Constant(c, np.float64(0)),
        computation=self._CreateBinaryAddF64Computation(),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        base_dilations=[],
        window_dilations=[],
        padding=padding)
    self._ExecuteAndCompareClose(c, expected=[[[5., 9.]]])

  def testWhileF32(self):
    cond = self._CreateTestF32Lt10Computation()
    body = self._CreateMulF32By2Computation()
    c = self._NewComputation()
    init = ops.Constant(c, np.float32(1.))
    ops.While(cond, body, init)
    self._ExecuteAndCompareClose(c, expected=[16.])

  def testWhileF64(self):
    cond = self._CreateTestF64Lt10Computation()
    body = self._CreateMulF64By2Computation()
    c = self._NewComputation()
    init = ops.Constant(c, np.float64(1.))
    ops.While(cond, body, init)
    self._ExecuteAndCompareClose(c, expected=[16.])

  def testConditionalTrue(self):
    c = self._NewComputation()
    pred = ops.Constant(c, np.bool_(True))
    true_operand = ops.Constant(c, np.float32(3.))
    true_computation = self._CreateMulF32By2Computation()
    false_operand = ops.Constant(c, np.float32(2.))
    false_computation = self._CreateConstantF32Computation()
    ops.Conditional(pred, true_operand, true_computation, false_operand,
                    false_computation)
    self._ExecuteAndCompareClose(c, expected=[6.])

  def testConditionalFalse(self):
    c = self._NewComputation()
    pred = ops.Constant(c, np.bool_(False))
    true_operand = ops.Constant(c, np.float32(3.))
    true_computation = self._CreateMulF32By2Computation()
    false_operand = ops.Constant(c, np.float32(2.))
    false_computation = self._CreateConstantF32Computation()
    ops.Conditional(pred, true_operand, true_computation, false_operand,
                    false_computation)
    self._ExecuteAndCompareClose(c, expected=[1.])

  def testInfeedS32Values(self):
    to_infeed = NumpyArrayS32([1, 2, 3, 4])
    c = self._NewComputation()
    ops.GetTupleElement(
        ops.InfeedWithToken(
            ops.CreateToken(c),
            xla_client.shape_from_pyval(
                to_infeed[0]).with_major_to_minor_layout_if_absent()), 0)
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build())
    for item in to_infeed:
      xla_client.transfer_to_infeed(item)

    for item in to_infeed:
      result, = xla_client.execute_with_python_values(compiled_c)
      self.assertEqual(result, item)

  def testInfeedTuple(self):
    to_infeed = (NumpyArrayS32([1, 2, 3, 4]), NumpyArrayS32([[7], [8]]))
    c = self._NewComputation()
    ops.GetTupleElement(
        ops.InfeedWithToken(
            ops.CreateToken(c),
            xla_client.shape_from_pyval(
                to_infeed).with_major_to_minor_layout_if_absent()), 0)
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build())
    xla_client.transfer_to_infeed(to_infeed)

    result = xla_client.execute_with_python_values(compiled_c)
    self.assertLen(result, 2)
    np.testing.assert_equal(result[0], to_infeed[0])
    np.testing.assert_equal(result[1], to_infeed[1])

  def testInfeedThenOutfeedS32(self):
    to_round_trip = NumpyArrayS32([1, 2, 3, 4])
    c = self._NewComputation()
    x_and_token = ops.InfeedWithToken(
        ops.CreateToken(c),
        xla_client.shape_from_pyval(
            to_round_trip[0]).with_major_to_minor_layout_if_absent())
    x = ops.GetTupleElement(x_and_token, 0)
    token = ops.GetTupleElement(x_and_token, 1)
    outfeed_shape = xla_client.shape_from_pyval(
        to_round_trip[0]).with_major_to_minor_layout_if_absent()
    ops.OutfeedWithToken(x, token, outfeed_shape)

    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build())

    for want in to_round_trip:
      execution = threading.Thread(target=lambda: compiled_c.Execute([]))
      execution.start()
      xla_client.transfer_to_infeed(want)
      got = xla_client.transfer_from_outfeed(outfeed_shape)
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
    ops.Scatter(
        ops.Constant(c, a), ops.Constant(c, scatter_indices),
        ops.Constant(c, updates), self._CreateBinaryAddS32Computation(), dnums)
    expected = np.array([[10, 21, 32], [3, 4, 5], [76, 87, 98]], dtype=np.int32)
    self._ExecuteAndCompareClose(c, expected=[expected])


class ErrorTest(ComputationTest):

  def setUp(self):
    self.f32_scalar_2 = NumpyArrayF32(2.0)
    self.s32_scalar_2 = NumpyArrayS32(2)

  def testCompileWithWrongElementTypeInLayout(self):
    c = self._NewComputation()
    c.SetOpMetadata(xla_client.CurrentSourceInfoMetadata())
    ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
    c.ClearOpMetadata()

    options = xla_client.CompileOptions()
    options.argument_layouts = [
        xla_client.Shape.array_shape(np.dtype(np.float32), [])
    ]

    backend = xla_client.get_local_backend()

    def TestFun():
      return backend.compile(c.Build(), compile_options=options)

    self.assertRaisesRegex(
        RuntimeError, r".*Invalid argument shape.*"
        r"expected s32\[\], got f32\[\].*", TestFun)

  def testInvokeWithWrongElementType(self):
    c = self._NewComputation()
    c.SetOpMetadata(xla_client.CurrentSourceInfoMetadata())
    ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
    c.ClearOpMetadata()

    backend = xla_client.get_local_backend()

    def TestFun():
      return xla_client.execute_with_python_values(
          backend.compile(c.Build()), [self.f32_scalar_2])

    self.assertRaisesRegex(
        RuntimeError, r"Invalid argument: Argument does not match.*"
        r"want s32\[\], got f32\[\].*", TestFun)


class ComputationRootTest(ComputationTest):
  """Tests related to setting the root of the computation."""

  def testComputationRootDifferentFromLastOp(self):
    c = self._NewComputation()
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
    result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
    ops.Add(result, ops.Constant(c, np.float32(1.618)))

    arg = NumpyArrayF32(1.0)
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build(result))
    ans, = xla_client.execute_with_python_values(compiled_c, [arg])
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
    x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
    # Clear Sharding.
    c.ClearSharding()

    result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
    ops.Add(result, ops.Constant(c, np.float32(1.618)))
    arg = NumpyArrayF32(1.0)
    backend = xla_client.get_local_backend()
    compiled_c = backend.compile(c.Build(result))
    ans, = xla_client.execute_with_python_values(compiled_c, [arg])
    np.testing.assert_allclose(ans, 4.14)


class AliasTest(ComputationTest):

  def testSetUpAlias(self):
    c = self._NewComputation()
    p1 = ops.Parameter(
        c, 0,
        xla_client.shape_from_pyval(
            NumpyArrayF32(1.0)).with_major_to_minor_layout_if_absent())
    p2 = ops.Parameter(
        c, 1,
        xla_client.shape_from_pyval(
            NumpyArrayF32(1.0)).with_major_to_minor_layout_if_absent())
    out = ops.Add(p1, p2)
    c.SetUpAlias([], 0, [])
    c = c.Build(out)
    backend = xla_client.get_local_backend()
    with self.assertRaisesRegex(
        RuntimeError, "Buffer aliasing is not supported "
        "by XLA for non-TPU backends"):
      backend.compile(c)


int_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
]
float_dtypes = [np.float16, np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]
dlpack_dtypes = int_dtypes + float_dtypes + [bfloat16]
standard_dtypes = int_dtypes + float_dtypes + complex_dtypes + [np.bool_]

testcase_shapes = [
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
]


def FormatShapeAndDtype(shape, dtype):
  return "_{}[{}]".format(np.dtype(dtype).name, ",".join(map(str, shape)))


class DLPackTest(parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters({
      "testcase_name": FormatShapeAndDtype(shape, dtype),
      "dtype": dtype,
      "shape": shape
  } for dtype in dlpack_dtypes for shape in testcase_shapes)
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


class BufferProtocolTest(parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters({
      "testcase_name": FormatShapeAndDtype(shape, dtype),
      "dtype": dtype,
      "shape": shape
  } for dtype in standard_dtypes for shape in testcase_shapes)
  def testRoundTrip(self, dtype, shape):
    x = np.array(np.random.rand(*shape) * 100, dtype=dtype)
    x_ptr = x.__array_interface__["data"][0]
    backend = xla_client.get_local_backend("cpu")
    buffer = xla_client.Buffer.from_pyval(x, backend=backend)
    y = np.array(buffer, copy=False)
    y_ptr = y.__array_interface__["data"][0]
    np.testing.assert_array_equal(x, y)
    # If the input was sufficiently aligned, the input and output should alias.
    self.assertTrue((x_ptr & 63) != 0 or x_ptr == y_ptr)
    self.assertEqual(y_ptr, buffer.unsafe_buffer_pointer())

    buffer2 = xla_client.Buffer.from_pyval(x, backend=backend, force_copy=True)
    z = np.array(buffer2, copy=False)
    self.assertNotEqual(x.__array_interface__["data"][0],
                        z.__array_interface__["data"][0])

  def testDeleteWithActiveView(self):
    x = np.random.randn(20, 10)
    backend = xla_client.get_local_backend("cpu")
    buffer = xla_client.Buffer.from_pyval(x, backend=backend)
    buffer_ptr = buffer.unsafe_buffer_pointer()
    y = np.array(buffer, copy=False)
    buffer.delete()
    # It is still legal to access `y`; the array view must keep it alive.
    np.testing.assert_array_equal(x, y)
    self.assertEqual(y.__array_interface__["data"][0], buffer_ptr)


class ProfilerTest(absltest.TestCase):

  def testTraceMe(self):
    # TODO(phawkins): These tests just check that the TraceMe context manager
    # acts like a context manager and doesn't explode. Ideally we'd check that
    # the profiler saw the traceme too.
    with xla_client.profiler.TraceMe("test1"):
      pass
    with xla_client.profiler.TraceMe("test2", foo=123):
      pass
    with self.assertRaises(ValueError):
      with xla_client.profiler.TraceMe("test3"):
        raise ValueError("test")

  @unittest.skipIf(portpicker is None, "Test requires portpicker")
  def testStartServer(self):
    port = portpicker.pick_unused_port()
    server = xla_client.profiler.start_server(port)
    del server


if __name__ == "__main__":
  absltest.main()

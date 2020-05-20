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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import xla_client

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.compiler.xla.python import custom_call_for_test
except ImportError:
  custom_call_for_test = None

try:
  import portpicker
except ImportError:
  portpicker = None
# pylint: enable=g-import-not-at-top

bfloat16 = xla_client.bfloat16
ops = xla_client.ops

FLAGS = flags.FLAGS

# We choose to ignore pylint's complaints about complex comprehensions, which we
# use widely for parameterizing tests.
# pylint: disable=g-complex-comprehension


def TestFactory(xla_backend, cloud_tpu=False):
  tests = []

  if not cloud_tpu:
    int_dtypes = [np.int32, np.int64, np.uint32, np.uint64]
    # TODO(phawkins): test np.float16, where supported.
    float_dtypes = [bfloat16, np.float32, np.float64]
    complex_dtypes = [np.complex64, np.complex128]
    standard_dtypes = int_dtypes + float_dtypes + complex_dtypes + [np.bool_]
  else:
    int_dtypes = [np.int32, np.uint32]
    float_dtypes = [np.float32]
    complex_dtypes = [np.complex64]
    standard_dtypes = int_dtypes + float_dtypes + complex_dtypes + [np.bool_]
  dlpack_dtypes = int_dtypes + float_dtypes

  class ComputationTest(parameterized.TestCase):
    """Base class for running an XLA Computation through the local client."""

    def setUp(self):
      super(ComputationTest, self).setUp()
      self.backend = xla_backend()

    def _NewComputation(self, name=None):
      if name is None:
        name = self.id()
      return xla_client.XlaBuilder(name)

    def _Execute(self, c, arguments):
      compiled_c = self.backend.compile(c.build())
      return xla_client.execute_with_python_values(
          compiled_c, arguments, backend=self.backend)

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
      self._ExecuteAndAssertWith(np.testing.assert_equal, c, arguments,
                                 expected)

    def _ExecuteAndCompareClose(self,
                                c,
                                arguments=(),
                                expected=None,
                                rtol=1e-7,
                                atol=0):
      self._ExecuteAndAssertWith(
          functools.partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
          c, arguments, expected)

  def NumpyArrayF32(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.float32 dtype."""
    return np.array(*args, dtype=np.float32, **kwargs)

  def NumpyArrayS32(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.int32 dtype."""
    return np.array(*args, dtype=np.int32, **kwargs)

  def NumpyArrayBool(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.bool dtype."""
    return np.array(*args, dtype=np.bool, **kwargs)

  class ComputationPrinting(absltest.TestCase):

    def setUp(self):
      super(ComputationPrinting, self).setUp()
      self.backend = xla_backend()

    def ExampleComputation(self):
      builder = xla_client.XlaBuilder("acomputation")
      p0 = ops.Parameter(builder, 0, xla_client.shape_from_pyval(np.float32(0)))
      p1 = ops.Parameter(
          builder, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
      x = ops.Mul(p0, p1)
      ops.Add(x, x)
      return builder.build()

    def testComputationToHloText(self):
      computation = self.ExampleComputation()
      hlo_text = computation.as_hlo_text()
      self.assertTrue(hlo_text.startswith("HloModule acomputation"))

    def testComputationToHloGraph(self):
      computation = self.ExampleComputation()
      hlo_dot_graph = computation.as_hlo_dot_graph()
      self.assertTrue(hlo_dot_graph.startswith("digraph "))

    def testHloModuleToHloText(self):
      computation = self.ExampleComputation()
      hlo_text = computation.as_hlo_module().to_string()
      self.assertTrue(hlo_text.startswith("HloModule acomputation"))

    def testHloModuleToHloGraph(self):
      computation = self.ExampleComputation()
      hlo_dot_graph = xla_client._xla.hlo_module_to_dot_graph(
          computation.as_hlo_module())
      self.assertTrue(hlo_dot_graph.startswith("digraph "))

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testCompiledHloModuleToHloText(self):
      computation = self.ExampleComputation()
      executable = self.backend.compile(computation)
      hlo_modules = executable.hlo_modules()
      self.assertLen(hlo_modules, 1)
      hlo_text = hlo_modules[0].to_string()
      self.assertTrue(hlo_text.startswith("HloModule acomputation"))
      self.assertIn("fusion", hlo_text)

  tests.append(ComputationPrinting)

  class ComputationHashTest(absltest.TestCase):

    def testHash(self):
      builder0 = xla_client.XlaBuilder("computation0")
      p0 = ops.Parameter(builder0, 0,
                         xla_client.shape_from_pyval(np.float32(0)))
      p1 = ops.Parameter(
          builder0, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
      ops.Mul(p0, p1)
      computation0 = builder0.build()

      builder1 = xla_client.XlaBuilder("computation1")
      p0 = ops.Parameter(builder1, 0,
                         xla_client.shape_from_pyval(np.float32(0)))
      p1 = ops.Parameter(
          builder1, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
      ops.Mul(p0, p1)
      computation1 = builder1.build()

      self.assertEqual(computation0.hash(), computation1.hash())

  tests.append(ComputationHashTest)

  class ComputationsWithConstantsTest(ComputationTest):
    """Tests focusing on Constant ops."""

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in int_dtypes + float_dtypes)
    def testConstantScalarSum(self, dtype):
      if dtype == np.int8 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support int8")
      c = self._NewComputation()
      ops.Add(ops.Constant(c, dtype(1.11)), ops.Constant(c, dtype(3.14)))
      self._ExecuteAndCompareClose(c, expected=[dtype(1.11) + dtype(3.14)])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConstantVectorMul(self, dtype):
      c = self._NewComputation()
      ops.Mul(
          ops.Constant(c, np.array([2.5, 3.3, -1.2, 0.7], dtype)),
          ops.Constant(c, np.array([-1.2, 2, -2, -3], dtype)))
      self._ExecuteAndCompareClose(
          c, expected=[[-3, 6.6, 2.4, -2.1]], rtol=3e-3)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConstantVectorScalarDiv(self, dtype):
      c = self._NewComputation()
      ops.Div(
          ops.Constant(c, np.array([1.5, 2.5, 3.0, -10.8], dtype=dtype)),
          ops.Constant(c, dtype(2.0)))
      self._ExecuteAndCompareClose(
          c, expected=[[0.75, 1.25, 1.5, -5.4]], rtol=2e-3)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConstantVectorScalarPow(self, dtype):
      c = self._NewComputation()
      ops.Pow(
          ops.Constant(c, np.array([1.5, 2.5, 3.0], dtype=dtype)),
          ops.Constant(c, dtype(2.)))
      self._ExecuteAndCompareClose(c, expected=[[2.25, 6.25, 9.]])

    def testIota(self):
      c = self._NewComputation()
      ops.Iota(c, xla_client.PrimitiveType.F32, 10)
      self._ExecuteAndCompareExact(
          c, expected=[np.arange(10, dtype=np.float32)])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in int_dtypes)
    def testBroadcastedIota(self, dtype):
      c = self._NewComputation()
      shape = xla_client.Shape.array_shape(
          xla_client.dtype_to_etype(dtype), (2, 3))
      ops.Iota(c, shape, 1)
      expected = np.array([[0, 1, 2], [0, 1, 2]], dtype=dtype)
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

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSum2D(self, dtype):
      c = self._NewComputation()
      ops.Add(
          ops.Constant(c, np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)),
          ops.Constant(c, np.array([[1, -1, 1], [-1, 1, -1]], dtype=dtype)))
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

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSum2DWith1DBroadcastDim0(self, dtype):
      # sum of a 2D array with a 1D array where the latter is replicated across
      # dimension 0 to match the former's shape.
      c = self._NewComputation()
      ops.Add(
          ops.Constant(c,
                       np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                dtype=dtype)),
          ops.Constant(c, np.array([10, 20, 30], dtype=dtype)),
          broadcast_dimensions=(0,))
      self._ExecuteAndCompareClose(
          c, expected=[[[11, 12, 13], [24, 25, 26], [37, 38, 39]]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSum2DWith1DBroadcastDim1(self, dtype):
      # sum of a 2D array with a 1D array where the latter is replicated across
      # dimension 1 to match the former's shape.
      c = self._NewComputation()
      ops.Add(
          ops.Constant(c,
                       np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                dtype=dtype)),
          ops.Constant(c, np.array([10, 20, 30], dtype=dtype)),
          broadcast_dimensions=(1,))
      self._ExecuteAndCompareClose(
          c, expected=[[[11, 22, 33], [14, 25, 36], [17, 28, 39]]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConstantAxpy(self, dtype):
      c = self._NewComputation()
      ops.Add(
          ops.Mul(
              ops.Constant(c, dtype(2)),
              ops.Constant(c, np.array([2.2, 3.3, 4.4, 5.5], dtype=dtype))),
          ops.Constant(c, np.array([100, -100, 200, -200], dtype)))
      self._ExecuteAndCompareClose(
          c, expected=[[104.4, -93.4, 208.8, -189]], rtol=2e-3)

    def testCustomCall(self):
      if self.backend.platform != "cpu":
        self.skipTest("Test requires cpu platform")
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

  tests.append(ComputationsWithConstantsTest)

  class ComputationFromProtoTest(absltest.TestCase):
    """Test computation execution from HLO proto."""

    def setUp(self):
      super(ComputationFromProtoTest, self).setUp()
      self.backend = xla_backend()

    def testExecuteFromProto(self):
      # Build the HLO proto
      b = xla_client.XlaBuilder("computation")
      ops.Add(ops.Constant(b, np.int32(1)), ops.Constant(b, np.int32(2)))
      serialized_proto = b.build().as_serialized_hlo_module_proto()

      # Load and execute the proto
      c = xla_client.XlaComputation(serialized_proto)
      ans, = xla_client.execute_with_python_values(
          self.backend.compile(c), (), backend=self.backend)
      np.testing.assert_equal(ans, np.int32(3))

  tests.append(ComputationFromProtoTest)

  class ParametersTest(ComputationTest):
    """Tests focusing on Parameter ops and argument-passing."""

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in int_dtypes)
    def testScalarTimesVector(self, dtype):
      c = self._NewComputation()
      arg0 = np.array(3, dtype=dtype)
      arg1 = np.array([10, 15, -2, 7], dtype=dtype)
      p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
      p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
      ops.Mul(p0, p1)
      self._ExecuteAndCompareExact(
          c, arguments=[arg0, arg1], expected=[arg0 * arg1])

    # TODO(phawkins): test comparison harness doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testScalarMinusVectorExplicitNumbering(self, dtype):
      # Use explicit numbering and pass parameter_num first. Sub is used since
      # it's not commutative and can help catch parameter reversal within the
      # computation.
      c = self._NewComputation()
      arg0 = np.array(2.0, dtype=dtype)
      arg1 = np.array([-2.3, 3.3, -4.3, 5.3], dtype=dtype)
      p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
      p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
      ops.Sub(p1, p0)
      self._ExecuteAndCompareClose(
          c, arguments=[arg0, arg1], expected=[arg1 - arg0])

  tests.append(ParametersTest)

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

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testCannotCallWithDeletedBuffers(self):
      c = self._NewComputation()
      ops.Add(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
          ops.Constant(c, np.float32(3.14)))
      arg = NumpyArrayF32(1.11)
      compiled_c = self.backend.compile(c.build())
      arg_buffer = self.backend.buffer_from_pyval(arg)
      arg_buffer.delete()
      with self.assertRaises(RuntimeError):
        compiled_c.execute([arg_buffer])

    def testShape(self):
      pyval = np.array([[1., 2.]], np.float32)
      local_buffer = self.backend.buffer_from_pyval(pyval)
      xla_shape = local_buffer.shape()
      self.assertEqual(xla_shape.dimensions(), (1, 2))
      self.assertEqual(np.dtype(xla_shape.element_type()), np.dtype(np.float32))

    def testBlockHostUntilReadyWorks(self):
      arg = np.array([[1., 2.]], np.float32)
      arg_buffer = self.backend.buffer_from_pyval(arg)
      arg_buffer.block_host_until_ready()
      # This test merely checks that nothing goes awry when we call
      # block_host_until_ready(); it's difficult to test anything else.

    def testCopyToHost(self):
      arg0 = np.array([[1., 2.]], np.float32)
      arg1 = np.array([[3., 4.]], np.float32)
      arg0_buffer = self.backend.buffer_from_pyval(arg0)
      arg1_buffer = self.backend.buffer_from_pyval(arg1)
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
      x = np.arange(8, dtype=np.int32)
      for device in self.backend.local_devices():
        buf = self.backend.buffer_from_pyval(x, device=device)
        self.assertEqual(buf.device(), device)
        np.testing.assert_equal(x, buf.to_py())

  tests.append(BufferTest)

  class SingleOpTest(ComputationTest):
    """Tests for single ops.

    The goal here is smoke testing - to exercise the most basic functionality of
    single XLA ops. As minimal as possible number of additional ops are added
    around the op being tested.
    """

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConcatenate(self, dtype):
      c = self._NewComputation()
      args = (
          ops.Constant(c, np.array([1.0, 2.0, 3.0], dtype=dtype)),
          ops.Constant(c, np.array([4.0, 5.0, 6.0], dtype=dtype)),
      )
      ops.ConcatInDim(c, args, dimension=0)
      self._ExecuteAndCompareExact(
          c, expected=[np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)])

    @parameterized.named_parameters({
        "testcase_name": "_{}_{}".format(src_dtype.__name__,
                                         dst_dtype.__name__),
        "src_dtype": src_dtype,
        "dst_dtype": dst_dtype,
    } for src_dtype, dst_dtype in itertools.permutations(
        [np.bool, np.int32, np.int64, np.float32, np.float64], 2))
    def testConvertElementType(self, src_dtype, dst_dtype):
      if ((src_dtype in [np.int64, np.float64] or
           dst_dtype in [np.int64, np.float64]) and
          self.backend.platform == "tpu"):
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      x = np.array([0, 1, 0, 0, 1], dtype=src_dtype)
      ops.ConvertElementType(
          ops.Constant(c, x), xla_client.dtype_to_etype(dst_dtype))

      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
      self.assertLen(result, 1)
      expected = np.array(x, dtype=dst_dtype)

      self.assertEqual(result[0].shape, expected.shape)
      self.assertEqual(result[0].dtype, expected.dtype)
      np.testing.assert_equal(result[0], expected)

    @parameterized.named_parameters(
        {
            "testcase_name": "_{}_{}".format(src_dtype.__name__,
                                             dst_dtype.__name__),
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
        }
        for dtypes in [[np.int32, np.float32], [np.int64, np.float64]]
        for src_dtype, dst_dtype in itertools.permutations(dtypes, 2))
    def testBitcastConvertType(self, src_dtype, dst_dtype):
      if (np.float64 in (src_dtype, dst_dtype) and
          self.backend.platform == "tpu"):
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      x = np.array([0, 1, 0, 0, 1], dtype=src_dtype)
      ops.BitcastConvertType(
          ops.Constant(c, x), xla_client.dtype_to_etype(dst_dtype))

      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
      self.assertLen(result, 1)
      expected = x.view(dst_dtype)

      self.assertEqual(result[0].shape, expected.shape)
      self.assertEqual(result[0].dtype, expected.dtype)
      np.testing.assert_equal(result[0], expected)

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

    # TODO(phawkins): np.dot implementation doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testDotMatrixVector(self, dtype):
      c = self._NewComputation()
      lhs = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=dtype)
      rhs = np.array([[10.0], [20.0]], dtype=dtype)
      ops.Dot(ops.Constant(c, lhs), ops.Constant(c, rhs))
      self._ExecuteAndCompareClose(c, expected=[np.dot(lhs, rhs)])

    # TODO(phawkins): np.dot implementation doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testDotMatrixMatrix(self, dtype):
      c = self._NewComputation()
      lhs = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=dtype)
      rhs = np.array([[10.0, 20.0], [100.0, 200.0]], dtype=dtype)
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
          ops.Constant(c, lhs), ops.Constant(c, rhs), strides, pads,
          lhs_dilation, rhs_dilation, dimension_numbers)
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
          ops.Constant(c, np.transpose(lhs,
                                       (0, 2, 3, 1))), ops.Constant(c, rhs),
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
          ops.Constant(c, lhs), ops.Constant(c, rhs), strides, pads,
          lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count)
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
          np.testing.assert_allclose,
          c, (),
          expected=[[True, False, True, True]])

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
      self._ExecuteAndCompareExact(
          c, expected=[[True, True, True, False, False]])

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
      self._ExecuteAndCompareExact(
          c, expected=[[True, False, False, True, True]])

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
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
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
      ops.Broadcast(
          ops.Constant(c, NumpyArrayS32([10, 20, 30, 40])), sizes=(3,))
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
          shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.F32,
                                             shape))
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
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
          shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.F32,
                                             shape))
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
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
          shape=xla_client.Shape.array_shape(xla_client.PrimitiveType.S32,
                                             shape))
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
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
      ops.Cholesky(ops.Constant(c, np.tril(np.dot(l, l.T))))
      self._ExecuteAndCompareClose(c, expected=[l], rtol=1e-4)

    def testSort(self):
      keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
      c = self._NewComputation()
      ops.Sort(c, [ops.Constant(c, keys)], is_stable=True)
      self._ExecuteAndCompareClose(
          c,
          expected=[np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32)])

    def testSortKeyVal(self):
      keys = np.array([[2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.float32)
      values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
      c = self._NewComputation()
      ops.Sort(c, (ops.Constant(c, keys), ops.Constant(c, values)), dimension=0)
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
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
      comparator = b.build()

      keys = np.array([[2, 3, 1, 3], [3, 1, 2, 2]], dtype=np.float32)
      values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
      c = self._NewComputation()
      ops.Sort(
          c, (ops.Constant(c, keys), ops.Constant(c, values)),
          dimension=1,
          comparator=comparator)
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (), backend=self.backend)
      self.assertLen(result, 2)
      np.testing.assert_allclose(result[0], [[1, 2, 3, 3], [1, 2, 2, 3]])
      np.testing.assert_equal(result[1], [[2, 0, 3, 1], [5, 7, 6, 4]])

    def testQR(self):
      a = np.array([[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166],
                    [10, 63, 166, 310]],
                   dtype=np.float32)
      c = self._NewComputation()
      ops.Tuple(c, ops.QR(ops.Constant(c, a), full_matrices=True))
      q, r = self._Execute(c, ())
      np.testing.assert_allclose(np.dot(q, r), a, rtol=1e-4)

    def testEigh(self):
      a = np.array([[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166],
                    [10, 63, 166, 310]],
                   dtype=np.float32)
      a = (a + a.T) / 2

      c = self._NewComputation()
      ops.Tuple(c, ops.Eigh(ops.Constant(c, a), lower=True))
      # TODO(b/129396575): Turn this test back on when it passes without
      # fastmath.
      # v, w = self._Execute(c, ())
      # self.assertLess(np.linalg.norm(np.dot(a, v) - w * v), 1e-3)

    def testSVD(self):
      a = np.array([[4, 6, 8, 10], [6, 45, 54, 63], [8, 54, 146, 166],
                    [10, 63, 166, 310]],
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
      self.assertTrue(c.is_constant(const_expr))
      self.assertFalse(c.is_constant(non_const_expr))

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
          ops.Constant(c, a),
          ops.Constant(c, indices),
          dnums,
          slice_sizes=[1, 1])
      g, = self._Execute(c, ())
      expected = np.array([[[[2, 7]]], [[[5, 6]]]], dtype=np.int32)
      np.testing.assert_allclose(g, expected, rtol=1e-4)

    def testFft(self):
      if self.backend.platform == "tpu":
        self.skipTest("TPU only supports 1D FFT")
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
      np.testing.assert_equal(
          np.array([eps + 1, 2 - eps], dtype=np.float32), out)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testRegularizedIncompleteBeta(self, dtype):
      x = np.array([0.53787335, 0.24015466, 0.47494545, 0.13567594, 0.95114538],
                   dtype=dtype)
      a = np.array([0.00753073, 0.34813385, 0.30485708, 1.29298632, 0.51472606],
                   dtype=dtype)
      b = np.array([0.55688389, 0.59794214, 0.42661022, 1.59748339, 0.95047677],
                   dtype=dtype)
      c = self._NewComputation()
      ops.RegularizedIncompleteBeta(
          ops.Constant(c, a), ops.Constant(c, b), ops.Constant(c, x))
      expected = np.array(
          [0.98923271, 0.48575411, 0.57952568, 0.12579775, 0.96989155])
      self._ExecuteAndCompareClose(c, expected=[expected], rtol=2e-2)

  tests.append(SingleOpTest)

  class EmbeddedComputationsTest(ComputationTest):
    """Tests for XLA graphs with embedded computations (such as maps)."""

    def _CreateConstantComputation(self, in_dtype, out_dtype):
      """Computation (A) -> B that returns a constant 1 for any input."""
      c = self._NewComputation("constant_{}_{}_one".format(
          in_dtype.__name__, out_dtype.__name__))
      ops.Parameter(c, 0,
                    xla_client.shape_from_pyval(np.array(0, dtype=in_dtype)))
      ops.Constant(c, out_dtype(1))
      return c.build()

    def _CreateMulBy2Computation(self, dtype):
      """Computation (dtype) -> dtype that multiplies its parameter by 2."""
      c = self._NewComputation("mul_f32_by2")
      ops.Mul(
          ops.Parameter(
              c, 0,
              xla_client.shape_from_pyval(np.array(
                  0, dtype=dtype)).with_major_to_minor_layout_if_absent()),
          ops.Constant(c, dtype(2.0)))
      return c.build()

    def _CreateMulF32ByParamComputation(self):
      """Computation (f32) -> f32 that multiplies one parameter by the other."""
      c = self._NewComputation("mul_f32_by_param")
      ops.Mul(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0))),
          ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0))))
      return c.build()

    def _CreateBinaryAddComputation(self, dtype):
      """Computation (dtype, dtype) -> dtype that adds its two parameters."""
      c = self._NewComputation("add_param0_by_param1")
      shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
      shape = shape.with_major_to_minor_layout_if_absent()
      ops.Add(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
      return c.build()

    def _CreateBinaryGeComputation(self, dtype):
      """Computation (dtype, dtype) -> bool that tests param0 >= param1."""
      c = self._NewComputation("param0_lt_param1")
      shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
      shape = shape.with_major_to_minor_layout_if_absent()
      ops.Ge(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
      return c.build()

    def _MakeSample3DArray(self, dtype):
      return np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                       [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]],
                      dtype=dtype)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testCall(self, dtype):
      c = self._NewComputation()
      ops.Call(
          c,
          self._CreateMulBy2Computation(dtype),
          operands=(ops.Constant(c, dtype(5.0)),))
      self._ExecuteAndCompareClose(c, expected=[10.0])

    @parameterized.named_parameters({
        "testcase_name": "_{}_{}".format(in_dtype.__name__, out_dtype.__name__),
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
    } for in_dtype, out_dtype in [[np.float32, np.int32]])
    def testMapEachElementToConstant(self, in_dtype, out_dtype):
      c = self._NewComputation()
      ops.Map(c,
              [ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=in_dtype))],
              self._CreateConstantComputation(in_dtype, out_dtype), [0])
      self._ExecuteAndCompareExact(c, expected=[[1, 1, 1, 1]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testMapMulBy2(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      ops.Map(c, [ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype))],
              self._CreateMulBy2Computation(dtype), [0])
      self._ExecuteAndCompareClose(c, expected=[[2.0, 4.0, 6.0, 8.0]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSimpleMapChain(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      # Chains a map of constant-out with a map of mul-by-2
      c = self._NewComputation()
      const = ops.Map(
          c, [ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype))],
          self._CreateConstantComputation(dtype, dtype), [0])
      ops.Map(c, [const], self._CreateMulBy2Computation(dtype), [0])
      self._ExecuteAndCompareClose(c, expected=[[2.0, 2.0, 2.0, 2.0]])

    # TODO(b/154752816): bfloat16 crashes in evaluator.
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testDivVectorsWithMap(self, dtype):

      def DivComputation():
        c = self._NewComputation("div_param0_by_param1")
        shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
        ops.Div(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
        return c.build()

      c = self._NewComputation()
      ops.Map(c, (ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)),
                  ops.Constant(c, np.array([5.0, 5.0, 4.0, 4.0], dtype=dtype))),
              DivComputation(), [0])
      self._ExecuteAndCompareClose(
          c, expected=[[0.2, 0.4, 0.75, 1.0]], rtol=1e-3)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSelectAndScatter(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      operand = ops.Constant(
          c, np.array([[1., 2., 6.], [4., 5., 3.]], dtype=dtype))
      window_dimensions = (2, 1)
      window_strides = (1, 2)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID,
          c.get_shape(operand).dimensions(), window_dimensions, window_strides)
      ops.SelectAndScatterWithGeneralPadding(
          operand,
          select=self._CreateBinaryGeComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          source=ops.Constant(c, np.array([[0.1, 0.2]], dtype=dtype)),
          init_value=ops.Constant(c, np.array(1, dtype=dtype)),
          scatter=self._CreateBinaryAddComputation(dtype))
      self._ExecuteAndCompareClose(
          c, expected=[[[1., 1., 1.2], [1.1, 1., 1.]]], rtol=5e-3)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduce1DtoScalar(self, dtype):
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[
              ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype))
          ],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=[0])
      self._ExecuteAndCompareClose(c, expected=[10])

    # TODO(phawkins): test comparison harness doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}_dim{}".format(dtype.__name__, dim),
        "dtype": dtype,
        "dim": dim,
    } for dtype in float_dtypes if dtype != bfloat16 for dim in range(2))
    def testReduce2DTo1D(self, dtype, dim):
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[ops.Constant(c, input_array)],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=[dim])
      self._ExecuteAndCompareClose(c, expected=[np.sum(input_array, axis=dim)])

    @parameterized.named_parameters({
        "testcase_name": "_{}_dims[{}]".format(dtype.__name__, dims),
        "dtype": dtype,
        "dims": tuple(dims)
    } for dtype in float_dtypes for dims in itertools.permutations(range(3)))
    def testReduce3DAllPossibleWaysF32(self, dtype, dims):
      input_array = self._MakeSample3DArray(dtype)
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[ops.Constant(c, input_array)],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=dims)
      self._ExecuteAndCompareClose(c, expected=[np.sum(input_array, axis=dims)])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowValidUnitStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 1)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
          window_strides)
      ops.ReduceWindowWithGeneralPadding(
          operand=ops.Constant(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding)
      self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.]]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowSameUnitStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 1)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.SAME, input_array.shape, window_dimensions,
          window_strides)
      ops.ReduceWindowWithGeneralPadding(
          operand=ops.Constant(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding)
      self._ExecuteAndCompareClose(c, expected=[[[5., 7., 9.], [4., 5., 6.]]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowValidGeneralStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 2)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
          window_strides)
      ops.ReduceWindowWithGeneralPadding(
          operand=ops.Constant(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding)
      self._ExecuteAndCompareClose(c, expected=[[[5., 9.]]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testWhile(self, dtype):

      def LessThan10Cond():
        c = self._NewComputation("test_lt_10")
        shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
        ops.Lt(ops.Parameter(c, 0, shape), ops.Constant(c, dtype(10.)))
        return c.build()

      cond = LessThan10Cond()
      body = self._CreateMulBy2Computation(dtype)
      c = self._NewComputation()
      init = ops.Constant(c, dtype(1.))
      ops.While(cond, body, init)
      self._ExecuteAndCompareClose(c, expected=[16.])

    def testConditionalTrue(self):
      c = self._NewComputation()
      pred = ops.Constant(c, np.bool_(True))
      true_operand = ops.Constant(c, np.float32(3.))
      true_computation = self._CreateMulBy2Computation(np.float32)
      false_operand = ops.Constant(c, np.float32(2.))
      false_computation = self._CreateConstantComputation(
          np.float32, np.float32)
      ops.Conditional(pred, true_operand, true_computation, false_operand,
                      false_computation)
      self._ExecuteAndCompareClose(c, expected=[6.])

    def testConditionalFalse(self):
      c = self._NewComputation()
      pred = ops.Constant(c, np.bool_(False))
      true_operand = ops.Constant(c, np.float32(3.))
      true_computation = self._CreateMulBy2Computation(np.float32)
      false_operand = ops.Constant(c, np.float32(2.))
      false_computation = self._CreateConstantComputation(
          np.float32, np.float32)
      ops.Conditional(pred, true_operand, true_computation, false_operand,
                      false_computation)
      self._ExecuteAndCompareClose(c, expected=[1.])

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testInfeedS32Values(self):
      to_infeed = NumpyArrayS32([1, 2, 3, 4])
      c = self._NewComputation()
      ops.GetTupleElement(
          ops.InfeedWithToken(
              ops.CreateToken(c),
              xla_client.shape_from_pyval(
                  to_infeed[0]).with_major_to_minor_layout_if_absent()), 0)
      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]
      for item in to_infeed:
        device.transfer_to_infeed(item)

      for item in to_infeed:
        result, = xla_client.execute_with_python_values(
            compiled_c, (), backend=self.backend)
        self.assertEqual(result, item)

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testInfeedTuple(self):
      to_infeed = (NumpyArrayS32([1, 2, 3, 4]), NumpyArrayS32([[7], [8]]))
      c = self._NewComputation()
      ops.GetTupleElement(
          ops.InfeedWithToken(
              ops.CreateToken(c),
              xla_client.shape_from_pyval(
                  to_infeed).with_major_to_minor_layout_if_absent()), 0)
      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]
      device.transfer_to_infeed(to_infeed)

      result = xla_client.execute_with_python_values(
          compiled_c, (), backend=self.backend)
      self.assertLen(result, 2)
      np.testing.assert_equal(result[0], to_infeed[0])
      np.testing.assert_equal(result[1], to_infeed[1])

    @unittest.skipIf(cloud_tpu, "not implemented")
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

      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]

      for want in to_round_trip:
        execution = threading.Thread(target=lambda: compiled_c.execute([]))
        execution.start()
        device.transfer_to_infeed(want)
        got = device.transfer_from_outfeed(outfeed_shape)
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
          ops.Constant(c, updates), self._CreateBinaryAddComputation(np.int32),
          dnums)
      expected = np.array([[10, 21, 32], [3, 4, 5], [76, 87, 98]],
                          dtype=np.int32)
      self._ExecuteAndCompareClose(c, expected=[expected])

  class ErrorTest(ComputationTest):

    def setUp(self):
      super(ErrorTest, self).setUp()
      self.f32_scalar_2 = NumpyArrayF32(2.0)
      self.s32_scalar_2 = NumpyArrayS32(2)

    def testCompileWithWrongElementTypeInLayout(self):
      c = self._NewComputation()
      c.set_op_metadata(xla_client.CurrentSourceInfoMetadata())
      ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
      c.clear_op_metadata()

      options = xla_client.CompileOptions()
      options.argument_layouts = [
          xla_client.Shape.array_shape(np.dtype(np.float32), [])
      ]

      def TestFun():
        return self.backend.compile(c.build(), compile_options=options)

      self.assertRaisesRegex(
          RuntimeError, r".*Invalid argument shape.*"
          r"expected s32\[\], got f32\[\].*", TestFun)

    def testInvokeWithWrongElementType(self):
      c = self._NewComputation()
      c.set_op_metadata(xla_client.CurrentSourceInfoMetadata())
      ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
      c.clear_op_metadata()

      def TestFun():
        return xla_client.execute_with_python_values(
            self.backend.compile(c.build()), [self.f32_scalar_2], self.backend)

      self.assertRaisesRegex(
          RuntimeError, r"Invalid argument: Argument does not match.*"
          r"want s32\[\], got f32\[\].*", TestFun)

  tests.append(EmbeddedComputationsTest)

  class ComputationRootTest(ComputationTest):
    """Tests related to setting the root of the computation."""

    def testComputationRootDifferentFromLastOp(self):
      c = self._NewComputation()
      x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
      result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
      ops.Add(result, ops.Constant(c, np.float32(1.618)))

      arg = NumpyArrayF32(1.0)
      compiled_c = self.backend.compile(c.build(result))
      ans, = xla_client.execute_with_python_values(
          compiled_c, [arg], backend=self.backend)
      np.testing.assert_allclose(ans, 4.14)

  tests.append(ComputationRootTest)

  class SetShardingTest(ComputationTest):
    """Tests related to set OpSharding."""

    def testSetSharding(self):
      c = self._NewComputation()
      sharding = xla_client.OpSharding()
      sharding.type = sharding.type.REPLICATED
      sharding.tile_assignment_dimensions.extend([1])
      sharding.tile_assignment_devices.extend([0])
      c.set_sharding(sharding)
      x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
      c.clear_sharding()

      result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
      ops.Add(result, ops.Constant(c, np.float32(1.618)))
      arg = NumpyArrayF32(1.0)
      compiled_c = self.backend.compile(c.build(result))
      ans, = xla_client.execute_with_python_values(
          compiled_c, [arg], backend=self.backend)
      np.testing.assert_allclose(ans, 4.14)

  tests.append(SetShardingTest)

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
      c.setup_alias([], 0, [])
      c = c.build(out)
      if self.backend.platform != "tpu":
        with self.assertRaisesRegex(
            RuntimeError, "Buffer aliasing is not supported "
            "by XLA for non-TPU backends"):
          self.backend.compile(c)

  tests.append(AliasTest)

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

    def setUp(self):
      super(DLPackTest, self).setUp()
      self.backend = xla_backend()
      if self.backend.platform not in ("cpu", "gpu"):
        self.skipTest("DLPack requires CPU or GPU")

    # pylint: disable=g-complex-comprehension
    @parameterized.named_parameters({
        "testcase_name": FormatShapeAndDtype(shape, dtype),
        "dtype": dtype,
        "shape": shape
    } for dtype in dlpack_dtypes for shape in testcase_shapes)
    def testRoundTrip(self, dtype, shape):
      x = np.array(np.random.rand(*shape) * 100, dtype=dtype)
      buffer = self.backend.buffer_from_pyval(x)
      dlt = xla_client._xla.buffer_to_dlpack_managed_tensor(buffer)
      del buffer  # Free "buffer" to make sure dlt retains ownership.
      self.assertEqual(type(dlt).__name__, "PyCapsule")
      y = xla_client._xla.dlpack_managed_tensor_to_buffer(
          dlt, self.backend)
      np.testing.assert_array_equal(x, y.to_py())

    def testTensorsCanBeConsumedOnceOnly(self):
      x = np.array(np.random.rand(3, 4, 5, 6), dtype=np.float32)
      buffer = self.backend.buffer_from_pyval(x)
      dlt = xla_client._xla.buffer_to_dlpack_managed_tensor(buffer)

      def ConsumeDLPackTensor():
        _ = xla_client._xla.dlpack_managed_tensor_to_buffer(
            dlt, self.backend)

      ConsumeDLPackTensor()
      self.assertRaisesRegex(
          RuntimeError, ".*a DLPack tensor may be consumed at most once.*",
          ConsumeDLPackTensor)

  tests.append(DLPackTest)

  class BufferProtocolTest(parameterized.TestCase):

    def setUp(self):
      super(BufferProtocolTest, self).setUp()
      self.backend = xla_backend()
      if self.backend.platform != "cpu":
        self.skipTest("Test requires CPU")

    # pylint: disable=g-complex-comprehension
    @parameterized.named_parameters({
        "testcase_name": FormatShapeAndDtype(shape, dtype),
        "dtype": dtype,
        "shape": shape
    } for dtype in standard_dtypes if dtype != bfloat16
                                    for shape in testcase_shapes)
    def testRoundTrip(self, dtype, shape):
      x = np.array(np.random.rand(*shape) * 100, dtype=dtype)
      x_ptr = x.__array_interface__["data"][0]
      buffer = self.backend.buffer_from_pyval(x)
      y = np.array(buffer, copy=False)
      y_ptr = y.__array_interface__["data"][0]
      np.testing.assert_array_equal(x, y)
      # If the input was sufficiently aligned, the input and output should
      # alias.
      self.assertTrue((x_ptr & 15) != 0 or x_ptr == y_ptr)
      self.assertEqual(y_ptr, buffer.unsafe_buffer_pointer())

      buffer2 = self.backend.buffer_from_pyval(x, force_copy=True)
      z = np.array(buffer2, copy=False)
      self.assertNotEqual(x.__array_interface__["data"][0],
                          z.__array_interface__["data"][0])

    def testDeleteWithActiveView(self):
      x = np.random.randn(20, 10)
      buffer = self.backend.buffer_from_pyval(x)
      buffer_ptr = buffer.unsafe_buffer_pointer()
      y = np.array(buffer, copy=False)
      buffer.delete()
      # It is still legal to access `y`; the array view must keep it alive.
      np.testing.assert_array_equal(x, y)
      self.assertEqual(y.__array_interface__["data"][0], buffer_ptr)

  tests.append(BufferProtocolTest)

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

  tests.append(ProfilerTest)
  return tests


def InstantiateTests(globals_dict, backend_fn, test_prefix="", **kw):
  # Avoid creating a new backend per test (this causes GPU OOM, and is probably
  # inefficient).
  backend_fn = functools.lru_cache(maxsize=None)(backend_fn)
  for klass in TestFactory(backend_fn, **kw):
    test = type(test_prefix + klass.__name__, (klass,), {})
    # Clean up the qualified names of the tests to not include the test factory.
    test.__qualname__ = test.__name__
    globals_dict[test.__name__] = test


if __name__ == "__main__":
  flags.DEFINE_string("backend", "cpu", "Target backend.")
  InstantiateTests(globals(),
                   lambda: xla_client.get_local_backend(FLAGS.backend))
  absltest.main()

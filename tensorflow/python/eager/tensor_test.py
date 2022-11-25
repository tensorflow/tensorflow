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
"""Unit tests for TensorFlow "Eager" Mode's Tensor class."""

import copy
import re
import sys

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables


def _create_tensor(value, device=None, dtype=None):
  context.ensure_initialized()
  ctx = context.context()
  if device is None:
    device = ctx.device_name
  if dtype is not None:
    dtype = dtype.as_datatype_enum
  try:
    return ops.EagerTensor(value, device=device, dtype=dtype)
  except core._NotOkStatusException as e:  # pylint: disable=protected-access
    raise core._status_to_exception(e)


class TFETensorTest(test_util.TensorFlowTestCase):

  def testScalarTensor(self):
    t = _create_tensor(3, dtype=dtypes.int32)
    self.assertAllEqual(t, _create_tensor(np.array(3)))
    self.assertEqual(dtypes.int32, t.dtype)
    self.assertEqual(0, t.shape.ndims)
    self.assertAllEqual([], t.shape.as_list())
    self.assertIn("tf.Tensor", str(t))
    self.assertIn("tf.Tensor", repr(t))

  def testBadConstructorArgs(self):
    context.ensure_initialized()
    ctx = context.context()
    device = ctx.device_name
    # Missing device.
    with self.assertRaisesRegex(TypeError, r".*argument 'device' \(pos 2\).*"):
      ops.EagerTensor(1)
    # Bad dtype type.
    with self.assertRaisesRegex(TypeError,
                                "Expecting a DataType value for dtype. Got"):
      ops.EagerTensor(1, device=device, dtype="1")

    # Following errors happen when trying to copy to GPU.
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs found")

    with ops.device("/device:GPU:0"):
      # Bad device.
      with self.assertRaisesRegex(TypeError, "Error parsing device argument"):
        ops.EagerTensor(1.0, device=1)

  def testNumpyValue(self):
    values = np.array([3.0])
    t = _create_tensor(values)
    self.assertAllEqual(values, t)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testNumpyDtypeSurvivesThroughTensorConversion(self):
    scalar_creators = [np.int32, np.int64, np.float32, np.float64]
    conversion_functions = [ops.convert_to_tensor, constant_op.constant]

    for scalar_creator in scalar_creators:
      for conversion_function in conversion_functions:
        np_val = scalar_creator(3)
        tensor_val = conversion_function(np_val)
        self.assertEqual(tensor_val.numpy().dtype, np_val.dtype)
        self.assertEqual(tensor_val.numpy(), np_val)

  def testNumpyValueWithCast(self):
    values = np.array([3.0], dtype=np.float32)
    t = _create_tensor(values, dtype=dtypes.float64)
    self.assertAllEqual(values, t)
    ctx = context.context()
    # Bad dtype value.
    with self.assertRaisesRegex(TypeError, "Invalid dtype argument value"):
      ops.EagerTensor(values, device=ctx.device_name, dtype=12345)

  def testNumpyOrderHandling(self):
    n = np.array([[1, 2], [3, 4]], order="F")
    t = _create_tensor(n)
    self.assertAllEqual([[1, 2], [3, 4]], t)

  def testNumpyArrayDtype(self):
    tensor = constant_op.constant([1.0, 2.0, 3.0])
    numpy_tensor = np.asarray(tensor, dtype=np.int32)
    self.assertAllEqual(numpy_tensor, [1, 2, 3])

  def testNdimsAgreesWithNumpy(self):
    numpy_tensor = np.asarray(1.0)
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

    numpy_tensor = np.asarray([1.0, 2.0, 3.0])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

    numpy_tensor = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

  def testLenAgreesWithNumpy(self):
    numpy_tensor = np.asarray(1.0)
    tensor = constant_op.constant(numpy_tensor)
    with self.assertRaises(TypeError):
      len(numpy_tensor)
    with self.assertRaisesRegex(TypeError, r"Scalar tensor has no `len[(][)]`"):
      len(tensor)

    numpy_tensor = np.asarray([1.0, 2.0, 3.0])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(len(numpy_tensor), len(tensor))

    numpy_tensor = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(len(numpy_tensor), len(tensor))

  def testCopy(self):
    t = constant_op.constant(1.0)
    tt = copy.copy(t)
    self.assertAllEqual(tt, 1.0)
    del tt
    tt = copy.deepcopy(t)
    self.assertAllEqual(tt, 1.0)
    del tt
    self.assertAllEqual(t, 1.0)

  def testConstantDtype(self):
    self.assertEqual(
        constant_op.constant(1, dtype=np.int64).dtype, dtypes.int64)

  def testTensorAndNumpyMatrix(self):
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    actual = _create_tensor([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(expected, actual)
    self.assertEqual(np.float32, actual.dtype)
    self.assertEqual(dtypes.float32, actual.dtype)
    self.assertAllEqual([2, 2], actual.shape.as_list())

  def testNumpyArrayInterface(self):

    class ArrayAsArrayInterface:
      """Simple class that wraps an np.array as an __array_interface__."""

      def __init__(self, array):
        self.array = array

      @property
      def __array_interface__(self):
        return self.array.__array_interface__

    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    array_interface = ArrayAsArrayInterface(expected)
    actual = _create_tensor(array_interface)
    self.assertAllEqual(expected, actual)

  def testFloatDowncast(self):
    # Unless explicitly specified, float64->float32
    t = _create_tensor(3.0)
    self.assertEqual(dtypes.float32, t.dtype)
    t = _create_tensor(3.0, dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t.dtype)

  def testBool(self):
    self.assertFalse(bool(_create_tensor(False)))
    self.assertFalse(bool(_create_tensor([False])))
    self.assertFalse(bool(_create_tensor([[False]])))
    self.assertFalse(bool(_create_tensor([0])))
    self.assertFalse(bool(_create_tensor([0.])))
    self.assertTrue(bool(_create_tensor([1])))
    self.assertTrue(bool(_create_tensor([1.])))

  def testIndex(self):
    self.assertEqual([42][_create_tensor(0)], 42)

    with self.assertRaises(TypeError):
      _ = [42][_create_tensor([0])]

  def testIntDowncast(self):
    t = _create_tensor(3)
    self.assertEqual(dtypes.int32, t.dtype)
    t = _create_tensor(3, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, t.dtype)
    t = _create_tensor(2**33)
    self.assertEqual(dtypes.int64, t.dtype)

  def testTensorCreationFailure(self):
    with self.assertRaises(ValueError):
      # Should fail because the each row of the Python object has a different
      # number of columns.
      self.assertEqual(None, _create_tensor([[1], [1, 2]]))

  def testMultiLineTensorStr(self):
    t = _create_tensor(np.eye(3))
    tensor_str = str(t)
    self.assertIn("shape=%s, dtype=%s" % (t.shape, t.dtype.name), tensor_str)
    self.assertIn(str(t), tensor_str)

  def testMultiLineTensorRepr(self):
    t = _create_tensor(np.eye(3))
    tensor_repr = repr(t)
    self.assertTrue(tensor_repr.startswith("<"))
    self.assertTrue(tensor_repr.endswith(">"))
    self.assertIn(
        "shape=%s, dtype=%s, numpy=\n%r" % (t.shape, t.dtype.name, t.numpy()),
        tensor_repr)

  def testTensorStrReprObeyNumpyPrintOptions(self):
    orig_threshold = np.get_printoptions()["threshold"]
    orig_edgeitems = np.get_printoptions()["edgeitems"]
    np.set_printoptions(threshold=2, edgeitems=1)

    t = _create_tensor(np.arange(10, dtype=np.int32))
    self.assertTrue(re.match(r".*\[.*0.*\.\.\..*9.*\]", str(t)))
    self.assertTrue(re.match(r".*\[.*0.*\.\.\..*9.*\]", repr(t)))

    # Clean up: reset to previous printoptions.
    np.set_printoptions(threshold=orig_threshold, edgeitems=orig_edgeitems)

  def testZeroDimTensorStr(self):
    t = _create_tensor(42)
    self.assertIn("42, shape=(), dtype=int32", str(t))

  def testZeroDimTensorRepr(self):
    t = _create_tensor(42)
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("shape=(), dtype=int32, numpy=42", repr(t))

  def testZeroSizeTensorStr(self):
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertIn("[], shape=(0,), dtype=float32", str(t))

  def testZeroSizeTensorRepr(self):
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("shape=(0,), dtype=float32, numpy=%r" % t.numpy(), repr(t))

  def testStringTensor(self):
    t_np_orig = np.array([[b"a", b"ab"], [b"abc", b"abcd"]])
    t = _create_tensor(t_np_orig)
    t_np = t.numpy()
    self.assertTrue(np.all(t_np == t_np_orig), "%s vs %s" % (t_np, t_np_orig))

  def testIterateOverTensor(self):
    l = [[1, 2], [3, 4]]
    t = _create_tensor(l)
    for list_element, tensor_element in zip(l, t):
      self.assertAllEqual(list_element, tensor_element.numpy())

  def testIterateOverScalarTensorRaises(self):
    t = _create_tensor(1)
    with self.assertRaisesRegex(TypeError,
                                "Cannot iterate over a scalar tensor"):
      iter(t)

  @test_util.run_gpu_only
  def testStringTensorOnGPU(self):
    with ops.device("/device:GPU:0"):
      t = _create_tensor("test string")
      self.assertIn("GPU", t.device)

  def testInvalidUTF8ProducesReasonableError(self):
    if sys.version_info[0] < 3:
      self.skipTest("Test is only valid in python3.")
    with self.assertRaises(UnicodeDecodeError):
      io_ops.read_file(b"\xff")

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorPreferredDtypeIsRespected(self):
    self.assertEqual(
        ops.convert_to_tensor(0.5, preferred_dtype=dtypes.int32).dtype,
        dtypes.float32)
    self.assertEqual(
        ops.convert_to_tensor(0.5, preferred_dtype=dtypes.float64).dtype,
        dtypes.float64)

  @test_util.run_in_graph_and_eager_modes
  def testCompatibility(self):
    integer_types = [
        dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8,
        dtypes.uint16, dtypes.uint32, dtypes.uint64
    ]

    # Floats are not compatible with ints
    for t in integer_types:
      with self.assertRaises(TypeError):
        constant_op.constant(0.5, dtype=t)

    # Ints compatible with floats
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float16)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float32)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float64)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.bfloat16)), 5.0)

    # Ints and floats are compatible with complex types
    self.assertEqual(
        constant_op.constant([[1.0]], dtype=dtypes.complex128).dtype,
        dtypes.complex128)
    self.assertEqual(
        constant_op.constant([[1]], dtype=dtypes.complex128).dtype,
        dtypes.complex128)

    # Quantized types are not compatible with floats
    quantized_types = [
        dtypes.qint16, dtypes.qint32, dtypes.qint8, dtypes.quint16,
        dtypes.quint8
    ]

    for t in quantized_types:
      with self.assertRaises(TypeError):
        constant_op.constant(0.5, dtype=t)

    # TODO(b/118402529): quantized types are broken in eager.

  @test_util.run_in_graph_and_eager_modes
  def testCConvertToTensor(self):
    with self.assertRaises(TypeError):
      _ = constant_op.constant(0) < 0.5

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorAllowsOverflow(self):
    _ = ops.convert_to_tensor(123456789, dtype=dtypes.uint8)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNumpyZeroDim(self):
    for np_type, dtype in [(np.int32, dtypes.int32), (np.half, dtypes.half),
                           (np.float32, dtypes.float32)]:
      x = ops.convert_to_tensor(
          [np.array(65, dtype=np_type),
           np.array(16, dtype=np_type)])
      self.assertEqual(x.dtype, dtype)
      self.assertAllEqual(x, [65, 16])

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNumpyScalar(self):
    x = ops.convert_to_tensor([
        np.array(321, dtype=np.int64).item(),
        np.array(16, dtype=np.int64).item()
    ])
    self.assertAllEqual(x, [321, 16])

  def testEagerTensorError(self):
    with self.assertRaisesRegex(TypeError,
                                "Cannot convert .* to EagerTensor of dtype .*"):
      _ = ops.convert_to_tensor(1., dtype=dtypes.int32)

  def testEagerLargeConstant(self):
    for t in [dtypes.uint64, dtypes.uint32, dtypes.int32, dtypes.int64]:
      self.assertEqual(constant_op.constant(t.max, dtype=t).numpy(), t.max)
      self.assertEqual(constant_op.constant(t.min, dtype=t).numpy(), t.min)

  def test_numpyIsView(self):
    with ops.device("CPU"):
      t = constant_op.constant([0.0])
      t._numpy()[0] = 42.0
      self.assertAllClose(t, constant_op.constant([42.0]))

  def test_numpyFailsForResource(self):
    v = variables.Variable(42)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Cannot convert .+ resource"):
      v._handle._numpy()

  def test_numpyFailsForVariant(self):
    variant_t = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Cannot convert .+ variant"):
      variant_t._numpy()

  def testMemoryviewFailsForResource(self):
    v = variables.Variable(42)
    with self.assertRaisesRegex(BufferError, "Cannot convert .+ resource"):
      np.asarray(memoryview(v._handle))

  def testMemoryviewFailsForVariant(self):
    variant_t = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    with self.assertRaisesRegex(BufferError, "Cannot convert .+ variant"):
      np.asarray(memoryview(variant_t))

  def testMemoryviewIsReadonly(self):
    t = constant_op.constant([0.0])
    self.assertTrue(memoryview(t).readonly)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMemoryviewScalar(self):
    t = constant_op.constant(42.0)
    self.assertAllEqual(
        np.array(memoryview(t)), np.array(42.0, dtype=np.float32))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMemoryviewEmpty(self):
    t = constant_op.constant([], dtype=np.float32)
    self.assertAllEqual(np.array(memoryview(t)), np.array([]))

  @test_util.run_gpu_only
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMemoryviewCopyToCPU(self):
    with ops.device("/device:GPU:0"):
      t = constant_op.constant([0.0])
    self.assertAllEqual(
        np.array(memoryview(t)), np.array([0.0], dtype=np.float32))

  @test_util.disable_tfrt("b/169877776: ResourceVariable is not initialized "
                          "properly in TFRT")
  def testResourceTensorCopy(self):
    if not test_util.is_gpu_available():
      self.skipTest("GPU only")

    with ops.device("GPU:0"):
      v = resource_variable_ops.ResourceVariable(1.)

    read_handle_on_gpu = resource_variable_ops.read_variable_op(
        v.handle, dtypes.float32)
    handle_on_cpu = v.handle.cpu()
    read_handle_on_cpu = resource_variable_ops.read_variable_op(
        handle_on_cpu, dtypes.float32)

    self.assertAllEqual(read_handle_on_cpu, read_handle_on_gpu)

  def testEagerTensorFormat(self):
    t = array_ops.constant(1)
    self.assertEqual(f"{t}", "1")
    self.assertEqual(str(t), "tf.Tensor(1, shape=(), dtype=int32)")
    self.assertEqual(f"{t!s}", "tf.Tensor(1, shape=(), dtype=int32)")
    self.assertEqual(repr(t), "<tf.Tensor: shape=(), dtype=int32, numpy=1>")
    self.assertEqual(f"{t!r}", "<tf.Tensor: shape=(), dtype=int32, numpy=1>")

  def testEagerTensorFormatForResource(self):
    t = resource_variable_ops.VarHandleOp(shape=[], dtype=dtypes.float32)

    # type is compiler-depdendent, as it comes from demangling.
    handle_str = (f"<ResourceHandle("
                  f"name=\"\", "
                  f"device=\"{t.device}\", "
                  f"container=\"localhost\", "
                  f"type=\"@@tensorflow@@Var@@\")>")

    def make_regex(s):
      return re.escape(s).replace("@@", ".*")

    self.assertRegex(f"{t}", make_regex(handle_str))
    self.assertRegex(
        str(t),
        make_regex(f"tf.Tensor({handle_str}, shape=(), dtype=resource)"))
    self.assertRegex(
        f"{t!s}",
        make_regex(f"tf.Tensor({handle_str}, shape=(), dtype=resource)"))
    self.assertRegex(
        repr(t),
        make_regex(
            f"<tf.Tensor: shape=(), dtype=resource, value={handle_str}>"))
    self.assertRegex(
        f"{t!r}",
        make_regex(
            f"<tf.Tensor: shape=(), dtype=resource, value={handle_str}>"))

  def testEagerTensorFormatForVariant(self):
    t = list_ops.tensor_list_reserve(
        element_shape=[1], num_elements=1, element_dtype=dtypes.float32)
    self.assertEqual(f"{t}", "<TensorList>")
    self.assertEqual(str(t), "tf.Tensor(<TensorList>, shape=(), dtype=variant)")
    self.assertEqual(f"{t!s}",
                     "tf.Tensor(<TensorList>, shape=(), dtype=variant)")
    self.assertEqual(
        repr(t), "<tf.Tensor: shape=(), dtype=variant, value=<TensorList>>")
    self.assertEqual(
        f"{t!r}", "<tf.Tensor: shape=(), dtype=variant, value=<TensorList>>")

  def testNumpyTooManyDimensions(self):
    t = constant_op.constant(1., shape=[1] * 33)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cannot convert tensor with 33 dimensions to NumPy array. NumPy arrays "
        "can have at most 32 dimensions"):
      t.numpy()

  def testNumpyDimsTooBig(self):
    # Creating a Numpy array fails in some cases if the product of non-zero
    # dimensions is very big, even if the shape also has a zero in it.
    t = array_ops.ones((0, 2**31, 2**31))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Failed to create numpy array from tensor of shape "
        r"\[0, 2147483648, 2147483648\]. Numpy error.*array is too big"):
      t.numpy()


class TFETensorUtilTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(TFETensorUtilTest, self).setUp()
    context.ensure_initialized()

  def testListOfThree(self):
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([[1, 2, 5], [3, 4, 5]], dtype=dtypes.int32)
    t3 = _create_tensor([[1], [3], [5], [6]], dtype=dtypes.int32)

    r = pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2, t3], 0)
    self.assertAllEqual(np.array([3, 2, 4]), r.numpy())

    r = pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2, t3], 1)
    self.assertAllEqual(np.array([2, 3, 1]), r.numpy())

  def testEmptyTensorList(self):
    a = pywrap_tfe.TFE_Py_TensorShapeSlice([], 0)
    self.assertTrue(isinstance(a, ops.EagerTensor))
    self.assertEqual(0, a.numpy().size)

  def testTensorListContainsNonTensors(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        TypeError,
        r"Expected a list of EagerTensors but element 1 has type \"str\""):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1, "abc"], 0)

    with self.assertRaisesRegex(
        TypeError,
        r"Expected a list of EagerTensors but element 0 has type \"int\""):
      pywrap_tfe.TFE_Py_TensorShapeSlice([2, t1], 0)

  def testTensorListNotList(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        TypeError,
        r"tensors argument must be a list or a tuple. Got.*EagerTensor"):
      pywrap_tfe.TFE_Py_TensorShapeSlice(t1, -2)

  def testNegativeSliceDim(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        ValueError, r"Slice dimension must be non-negative. Got -2"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1], -2)

  def testUnicode(self):
    self.assertEqual(constant_op.constant(u"asdf").numpy(), b"asdf")

  def testFloatTensor(self):
    self.assertEqual(dtypes.float64, _create_tensor(np.float64()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float32, _create_tensor(np.float32()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float16, _create_tensor(np.float16()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float32, _create_tensor(0.0).dtype)

  def testSliceDimOutOfRange(self):
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([1, 2], dtype=dtypes.int32)
    t3 = _create_tensor(2, dtype=dtypes.int32)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(2\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 2"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1], 2)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 1"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t2], 1)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 1 has rank 1"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2], 1)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 0"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t3], 0)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 2 has rank 0"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t2, t1, t3], 0)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTensorDir(self):
    t = array_ops.ones(1)
    t.test_attr = "Test"

    instance_dir = dir(t)
    type_dir = dir(ops.EagerTensor)

    # Monkey patched attributes should show up in dir(t)
    self.assertIn("test_attr", instance_dir)
    instance_dir.remove("test_attr")
    self.assertEqual(instance_dir, type_dir)

  def testNonRectangularPackAsConstant(self):
    l = [array_ops.zeros((10, 1)).numpy(), array_ops.zeros(1).numpy()]

    with self.assertRaisesRegex(ValueError, "non-rectangular Python sequence"):
      constant_op.constant(l)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFloatAndIntAreConvertibleToComplex(self):
    a = [[1., 1], [1j, 2j]]
    np_value = np.array(a, dtype=np.complex128)
    tf_value = ops.convert_to_tensor(a, dtype=dtypes.complex128)
    self.assertAllEqual(tf_value.numpy(), np_value)


if __name__ == "__main__":
  test.main()

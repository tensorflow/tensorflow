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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util


def _create_tensor(value, device=None, dtype=None):
  ctx = context.context()
  if device is None:
    device = ctx.device_name
  if dtype is not None:
    dtype = dtype.as_datatype_enum
  try:
    return ops.EagerTensor(
        value, context=ctx._handle, device=device, dtype=dtype)
  except core._NotOkStatusException as e:  # pylint: disable=protected-access
    raise core._status_to_exception(e.code, e.message)


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
    ctx = context.context()
    handle = ctx._handle
    device = ctx.device_name
    # Missing context.
    with self.assertRaisesRegexp(
        TypeError, r"Required argument 'context' \(pos 2\) not found"):
      ops.EagerTensor(1, device=device)
    # Missing device.
    with self.assertRaisesRegexp(
        TypeError, r"Required argument 'device' \(pos 3\) not found"):
      ops.EagerTensor(1, context=handle)
    # Bad dtype type.
    with self.assertRaisesRegexp(TypeError,
                                 "Expecting a DataType value for dtype. Got"):
      ops.EagerTensor(1, context=handle, device=device, dtype="1")
    # Following errors happen when trying to copy to GPU.
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with ops.device("/device:GPU:0"):
      device = ctx.device_name
      # Bad context.
      with self.assertRaisesRegexp(
          TypeError, "Expecting a PyCapsule encoded context handle. Got"):
        ops.EagerTensor(1.0, context=1, device=device)
      # Bad device.
      with self.assertRaisesRegexp(
          TypeError, "Error parsing device argument to CopyToDevice"):
        ops.EagerTensor(1.0, context=handle, device=1)

  def testNumpyValue(self):
    values = np.array([3.0])
    t = _create_tensor(values)
    self.assertAllEqual(values, t)

  def testNumpyValueWithCast(self):
    values = np.array([3.0], dtype=np.float32)
    t = _create_tensor(values, dtype=dtypes.float64)
    self.assertAllEqual(values, t)
    ctx = context.context()
    # Bad dtype value.
    with self.assertRaisesRegexp(TypeError, "Invalid dtype argument value"):
      ops.EagerTensor(
          values, context=ctx._handle, device=ctx.device_name, dtype=12345)

  def testNumpyOrderHandling(self):
    n = np.array([[1, 2], [3, 4]], order="F")
    t = _create_tensor(n)
    self.assertAllEqual([[1, 2], [3, 4]], t)

  def testNumpyArrayDtype(self):
    tensor = constant_op.constant([1.0, 2.0, 3.0])
    numpy_tensor = np.asarray(tensor, dtype=np.int32)
    self.assertAllEqual(numpy_tensor, [1, 2, 3])

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
    self.assertEqual(constant_op.constant(1.0, dtype=np.int64).dtype,
                     dtypes.int64)

  def testTensorAndNumpyMatrix(self):
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    actual = _create_tensor([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(expected, actual)
    self.assertEqual(np.float32, actual.dtype)
    self.assertEqual(dtypes.float32, actual.dtype)
    self.assertAllEqual([2, 2], actual.shape.as_list())

  def testFloatDowncast(self):
    # Unless explicitly specified, float64->float32
    t = _create_tensor(3.0)
    self.assertEqual(dtypes.float32, t.dtype)
    t = _create_tensor(3.0, dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t.dtype)

  def testBool(self):
    t = _create_tensor(False)
    if t:
      self.assertFalse(True)

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
    self.assertIn("id=%d, shape=%s, dtype=%s, numpy=\n%r" %
                  (t._id, t.shape, t.dtype.name, t.numpy()), tensor_repr)

  def testTensorStrReprObeyNumpyPrintOptions(self):
    orig_threshold = np.get_printoptions()["threshold"]
    orig_edgeitems = np.get_printoptions()["edgeitems"]
    np.set_printoptions(threshold=2, edgeitems=1)

    t = _create_tensor(np.arange(10, dtype=np.int32))
    self.assertIn("[0 ..., 9]", str(t))
    self.assertIn("[0, ..., 9]", repr(t))

    # Clean up: reset to previous printoptions.
    np.set_printoptions(threshold=orig_threshold, edgeitems=orig_edgeitems)

  def testZeroDimTensorStr(self):
    t = _create_tensor(42)
    self.assertIn("42, shape=(), dtype=int32", str(t))

  def testZeroDimTensorRepr(self):
    t = _create_tensor(42)
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("id=%d, shape=(), dtype=int32, numpy=42" % t._id, repr(t))

  def testZeroSizeTensorStr(self):
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertIn("[], shape=(0,), dtype=float32", str(t))

  def testZeroSizeTensorRepr(self):
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("id=%d, shape=(0,), dtype=float32, numpy=%r" % (t._id,
                                                                  t.numpy()),
                  repr(t))

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

  def testStringTensorOnGPU(self):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with ops.device("/device:GPU:0"):
      with self.assertRaisesRegexp(
          RuntimeError, "Can't copy Tensor with type string to device"):
        _create_tensor("test string")


class TFETensorUtilTest(test_util.TensorFlowTestCase):

  def testListOfThree(self):
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([[1, 2, 5], [3, 4, 5]], dtype=dtypes.int32)
    t3 = _create_tensor([[1], [3], [5], [6]], dtype=dtypes.int32)

    r = pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1, t2, t3], 0)
    self.assertAllEqual(np.array([3, 2, 4]), r.numpy())

    r = pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1, t2, t3], 1)
    self.assertAllEqual(np.array([2, 3, 1]), r.numpy())

  def testEmptyTensorList(self):
    a = pywrap_tensorflow.TFE_Py_TensorShapeSlice([], 0)
    self.assertTrue(isinstance(a, ops.EagerTensor))
    self.assertEqual(0, a.numpy().size)

  def testTensorListContainsNonTensors(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegexp(
        TypeError,
        r"Expected a list of EagerTensors but element 1 has type \"str\""):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1, "abc"], 0)

    with self.assertRaisesRegexp(
        TypeError,
        r"Expected a list of EagerTensors but element 0 has type \"int\""):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([2, t1], 0)

  def testTensorListNotList(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegexp(
        TypeError,
        r"tensor_list argument must be a list. Got \"EagerTensor\""):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice(t1, -2)

    with self.assertRaisesRegexp(
        TypeError,
        r"tensor_list argument must be a list. Got \"tuple\""):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice((t1,), -2)

  def testNegativeSliceDim(self):
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegexp(
        ValueError,
        r"Slice dimension must be non-negative. Got -2"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1], -2)

  def testUnicode(self):
    self.assertEqual(constant_op.constant(u"asdf").numpy(), b"asdf")

  def testSliceDimOutOfRange(self):
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([1, 2], dtype=dtypes.int32)
    t3 = _create_tensor(2, dtype=dtypes.int32)

    with self.assertRaisesRegexp(
        IndexError,
        r"Slice dimension \(2\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 2"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1], 2)

    with self.assertRaisesRegexp(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 1"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t2], 1)

    with self.assertRaisesRegexp(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 1 has rank 1"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t1, t2], 1)

    with self.assertRaisesRegexp(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 0"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t3], 0)

    with self.assertRaisesRegexp(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 2 has rank 0"):
      pywrap_tensorflow.TFE_Py_TensorShapeSlice([t2, t1, t3], 0)


if __name__ == "__main__":
  test.main()

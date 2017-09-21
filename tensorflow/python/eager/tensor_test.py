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

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util


class TFETensorTest(test_util.TensorFlowTestCase):

  def testScalarTensor(self):
    t = constant_op.constant(3)
    self.assertEqual(t.numpy(), constant_op.constant(np.array(3)).numpy())
    self.assertEqual(dtypes.int32, t.dtype)
    self.assertEqual(0, t.shape.ndims)
    self.assertAllEqual([], t.shape.as_list())

  def testTensorAndNumpyMatrix(self):
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    actual = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(expected, actual.numpy())
    self.assertEqual(np.float32, actual.numpy().dtype)
    self.assertEqual(dtypes.float32, actual.dtype)
    self.assertAllEqual([2, 2], actual.shape.as_list())

  def testFloatDowncast(self):
    # Unless explicitly specified, float64->float32
    t = constant_op.constant(3.0)
    self.assertEqual(dtypes.float32, t.dtype)
    t = constant_op.constant(3.0, dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t.dtype)

  def testBool(self):
    t = constant_op.constant(False)
    if t:
      self.assertFalse(True)

  def testIntDowncast(self):
    t = constant_op.constant(3)
    self.assertEqual(dtypes.int32, t.dtype)
    t = constant_op.constant(3, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, t.dtype)
    t = constant_op.constant(2**33)
    self.assertEqual(dtypes.int64, t.dtype)

  def testTensorCreationFailure(self):
    with self.assertRaises(Exception):
      # Should fail because the each row of the Python object has a different
      # number of columns.
      self.assertEqual(None, constant_op.constant([[1], [1, 2]]))

  def testNumpyOrderHandling(self):
    n = np.array([[1, 2], [3, 4]], order="F")
    t = constant_op.constant(n)
    self.assertAllEqual([[1, 2], [3, 4]], t.numpy())

  def testMultiLineTensorStr(self):
    t = constant_op.constant(np.eye(3))
    tensor_str = str(t)
    self.assertIn("shape=%s, dtype=%s" % (t.shape, t.dtype.name), tensor_str)
    self.assertIn(str(t.numpy()), tensor_str)

  def testMultiLineTensorRepr(self):
    t = constant_op.constant(np.eye(3))
    tensor_repr = repr(t)
    self.assertTrue(tensor_repr.startswith("<"))
    self.assertTrue(tensor_repr.endswith(">"))
    self.assertIn(
        "id=%d, shape=%s, dtype=%s, numpy=\n%r" % (
            t._id, t.shape, t.dtype.name, t.numpy()), tensor_repr)

  def testTensorStrReprObeyNumpyPrintOptions(self):
    orig_threshold = np.get_printoptions()["threshold"]
    orig_edgeitems = np.get_printoptions()["edgeitems"]
    np.set_printoptions(threshold=2, edgeitems=1)

    t = constant_op.constant(np.arange(10, dtype=np.int32))
    self.assertIn("[0 ..., 9]", str(t))
    self.assertIn("[0, ..., 9]", repr(t))

    # Clean up: reset to previous printoptions.
    np.set_printoptions(threshold=orig_threshold, edgeitems=orig_edgeitems)

  def testZeroDimTensorStr(self):
    t = constant_op.constant(42)
    self.assertIn("42, shape=(), dtype=int32", str(t))

  def testZeroDimTensorRepr(self):
    t = constant_op.constant(42)
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("id=%d, shape=(), dtype=int32, numpy=42" % t._id, repr(t))

  def testZeroSizeTensorStr(self):
    t = constant_op.constant(np.zeros(0, dtype=np.float32))
    self.assertIn("[], shape=(0,), dtype=float32", str(t))

  def testZeroSizeTensorRepr(self):
    t = constant_op.constant(np.zeros(0, dtype=np.float32))
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn(
        "id=%d, shape=(0,), dtype=float32, numpy=%r" % (t._id, t.numpy()),
        repr(t))

  def testStringTensor(self):
    t_np_orig = np.array([[b"a", b"ab"], [b"abc", b"abcd"]])
    t = constant_op.constant(t_np_orig)
    t_np = t.numpy()
    self.assertTrue(np.all(t_np == t_np_orig), "%s vs %s" % (t_np, t_np_orig))

  def testStringTensorOnGPU(self):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with ops.device("/device:GPU:0"):
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Can't copy Tensor with type string to device"):
        constant_op.constant("test string")


if __name__ == "__main__":
  test.main()

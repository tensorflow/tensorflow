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
"""Tests for tensorflow.python.framework.python_tensor_converter."""

from absl.testing import parameterized

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_tensor_converter
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class PythonTensorConverterTest(test_util.TensorFlowTestCase,
                                parameterized.TestCase):

  def setUp(self):
    context.ensure_initialized()
    super(PythonTensorConverterTest, self).setUp()

  def makePythonTensorConverter(self):
    return _pywrap_python_tensor_converter.PythonTensorConverter(
        context.context())

  #=============================================================================
  # Convert int to tensor.

  def testConvertIntWithInferredDType(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(12, types_pb2.DT_INVALID)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, 12)
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertIntWithExplicitDtype(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(12, types_pb2.DT_INT64)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, 12)
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertIntWithIncompatibleDtype(self):
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        TypeError, "Expected string, but got 3 of type 'int'"
        "|Cannot convert 3 to EagerTensor of dtype string"):
      converter.Convert(3, types_pb2.DT_STRING)

  #=============================================================================
  # Convert tensor to tensor.

  def testConvertTensorWithInferredDType(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(
        constant_op.constant([1, 2, 3]), types_pb2.DT_INVALID)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [1, 2, 3])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertFalse(used_fallback)

  def testConvertTensorWithExplicitDtype(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(
        constant_op.constant([1, 2, 3], dtypes.int64), types_pb2.DT_INT64)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [1, 2, 3])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertFalse(used_fallback)

  def testConvertTensorWithIncorrectDtype(self):
    converter = self.makePythonTensorConverter()
    with self.assertRaises((TypeError, ValueError)):
      converter.Convert(
          constant_op.constant([1, 2, 3], dtypes.int32), types_pb2.DT_INT64)

  #=============================================================================
  # Convert list to tensor.

  def testConvertListWithInferredDType(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert([[1, 2, 3], [4, 5, 6]],
                                                     types_pb2.DT_INVALID)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertListWithExplicitDtype(self):
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert([[1, 2, 3], [4, 5, 6]],
                                                     types_pb2.DT_INT64)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertListWithIncompatibleDtype(self):
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        TypeError, "Expected string, but got .* of type 'int'"
        "|Cannot convert .* to EagerTensor of dtype string"):
      converter.Convert([[1, 2, 3], [4, 5, 6]], types_pb2.DT_STRING)

  def testConvertListWithInconsistentDtype(self):
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "Can't convert Python sequence with mixed types to Tensor."
        "|Failed to convert"):
      converter.Convert([[1, 2], ["a", "b"]], types_pb2.DT_INVALID)

  #=============================================================================
  # Convert np.array to tensor.

  def testConvertNumpyArrayWithInferredDType(self):
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INVALID)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertNumpyArrayWithExplicitDtype(self):
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INT64)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertNumpyArrayWithIncompatibleDtype(self):
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_STRING)

  def testConvertNumpyArrayWithUnsupportedDtype(self):
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2], ["a", "b"]], np.object_)
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_INVALID)

  #=============================================================================
  # Convert IndexedSlices to tensor.

  def testConvertIndexedSlicesWithInferredDType(self):
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INVALID)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertTrue(used_fallback)

  def testConvertIndexedSlicesWithExplicitDtype(self):
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INT32)
    self.assertIsInstance(result, ops.Tensor)
    self.assertAllEqual(result, [[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertTrue(used_fallback)

  def testConvertIndexedSlicesWithIncorrectDtype(self):
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_FLOAT)


if __name__ == "__main__":
  googletest.main()

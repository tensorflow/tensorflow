# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for UnicodeEncode op from ragged_string_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


class UnicodeEncodeOpTest(test.TestCase, parameterized.TestCase):

  def testScalar(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        ragged_string_ops.unicode_encode(72, "UTF-8")
    with self.cached_session():
      with self.assertRaises(ValueError):
        ragged_string_ops.unicode_encode(constant_op.constant(72), "UTF-8")

  def testRequireParams(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode()
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(72)
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(encoding="UTF-8")

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrors(self, encoding):
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    with self.cached_session():
      with self.assertRaises(errors.InvalidArgumentError):
        ragged_string_ops.unicode_encode(test_value, encoding, "strict").eval()

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testIgnoreErrors(self, encoding):
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    expected_value = u"Heo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "ignore")
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testReplaceErrors(self, encoding):
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    expected_value = u"He\U0000fffd\U0000fffdo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace")
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    # Test custom replacement character
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    expected_value = u"Heooo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 111)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    # Verify "replace" is default
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    expected_value = u"He\U0000fffd\U0000fffdo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    # Replacement_char must be within range
    test_value = np.array([72, 101, 2147483647, -1, 111], np.int32)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 1114112)
    with self.cached_session():
      with self.assertRaises(errors.InvalidArgumentError):
        unicode_encode_op.eval()

  # -- regular Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testVector(self, encoding):
    test_value = np.array([72, 101, 108, 108, 111], np.int32)
    expected_value = u"Hello".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    test_value = np.array([72, 101, 195, 195, 128516], np.int32)
    expected_value = u"He\xc3\xc3\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    # Single character string
    test_value = np.array([72], np.int32)
    expected_value = u"H".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

    test_value = np.array([128516], np.int32)
    expected_value = u"\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(result, bytes)
      self.assertAllEqual(result, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testMatrix(self, encoding):
    test_value = np.array(
        [[72, 128516, 108, 108, 111], [87, 128516, 114, 108, 100]], np.int32)
    expected_value = [
        u"H\U0001f604llo".encode(encoding), u"W\U0001f604rld".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(unicode_encode_op, ops.Tensor)
      self.assertAllEqual(result, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test3DimMatrix(self, encoding):
    test_value = constant_op.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]],
         [[102, 105, 120, 101, 100], [119, 111, 114, 100, 115]],
         [[72, 121, 112, 101, 114], [99, 117, 98, 101, 46]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World".encode(encoding)],
                      [u"fixed".encode(encoding), u"words".encode(encoding)],
                      [u"Hyper".encode(encoding), u"cube.".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(unicode_encode_op, ops.Tensor)
      self.assertAllEqual(result, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test4DimMatrix(self, encoding):
    test_value = constant_op.constant(
        [[[[72, 101, 108, 108, 111]], [[87, 111, 114, 108, 100]]],
         [[[102, 105, 120, 101, 100]], [[119, 111, 114, 100, 115]]],
         [[[72, 121, 112, 101, 114]], [[99, 117, 98, 101, 46]]]], np.int32)
    expected_value = [[[u"Hello".encode(encoding)],
                       [u"World".encode(encoding)]],
                      [[u"fixed".encode(encoding)],
                       [u"words".encode(encoding)]],
                      [[u"Hyper".encode(encoding)],
                       [u"cube.".encode(encoding)]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(unicode_encode_op, ops.Tensor)
      self.assertAllEqual(result, expected_value)

  # -- Ragged Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testRaggedMatrix(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[72, 195, 108, 108, 111], [87, 128516, 114, 108, 100, 46]], np.int32)
    expected_value = [
        u"H\xc3llo".encode(encoding), u"W\U0001f604rld.".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertIsInstance(unicode_encode_op, ops.Tensor)
      self.assertAllEqual(result, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test3DimMatrixWithRagged2ndDim(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]],
         [[102, 105, 120, 101, 100]],
         [[72, 121, 112, 101, 114], [119, 111, 114, 100, 115],
          [99, 117, 98, 101, 46]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World".encode(encoding)],
                      [u"fixed".encode(encoding)],
                      [
                          u"Hyper".encode(encoding), u"words".encode(encoding),
                          u"cube.".encode(encoding)
                      ]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertEqual(unicode_encode_op.ragged_rank, 1)
      self.assertAllEqual(result.tolist(), expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test3DimMatrixWithRagged3rdDim(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]],
         [[68, 111, 110, 39, 116], [119, 195, 114, 114, 121, 44, 32, 98, 101]],
         [[128516], []]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [
                          u"Don't".encode(encoding),
                          u"w\xc3rry, be".encode(encoding)
                      ], [u"\U0001f604".encode(encoding), u"".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertEqual(unicode_encode_op.ragged_rank, 1)
      self.assertAllEqual(result.tolist(), expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test3DimMatrixWithRagged2ndAnd3rdDim(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]], [],
         [[128516]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [], [u"\U0001f604".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertEqual(unicode_encode_op.ragged_rank, 1)
      self.assertAllEqual(result.tolist(), expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def test4DimRaggedMatrix(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]]],
         [[[]], [[72, 121, 112, 101]]]], np.int32)
    expected_value = [[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                      [[u"".encode(encoding)], [u"Hype".encode(encoding)]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertEqual(unicode_encode_op.ragged_rank, 2)
      self.assertAllEqual(result.tolist(), expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testRaggedMatrixWithMultiDimensionInnerValues(self, encoding):
    test_inner_values = constant_op.constant([[[72, 101, 108, 108, 111],
                                               [87, 111, 114, 108, 100]],
                                              [[102, 105, 120, 101, 100],
                                               [119, 111, 114, 100, 115]],
                                              [[72, 121, 112, 101, 114],
                                               [99, 117, 98, 101, 46]]])
    test_row_splits = [
        constant_op.constant([0, 2, 3], dtype=np.int64),
        constant_op.constant([0, 1, 1, 3], dtype=np.int64)
    ]
    test_value = ragged_factory_ops.from_nested_row_splits(test_inner_values,
                                                           test_row_splits)
    expected_value = [[[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                       []],
                      [[[u"fixed".encode(encoding), u"words".encode(encoding)],
                        [u"Hyper".encode(encoding),
                         u"cube.".encode(encoding)]]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    with self.cached_session():
      result = unicode_encode_op.eval()
      self.assertEqual(unicode_encode_op.ragged_rank, 2)
      self.assertAllEqual(result.tolist(), expected_value)
      # These next two assertions don't necessarily need to be here as they test
      # internal representations and we already verified the value is correct.
      self.assertAllEqual(len(result.nested_row_splits), len(test_row_splits))
      self.assertEqual(unicode_encode_op.inner_values.shape.ndims,
                       test_inner_values.shape.ndims - 1)


if __name__ == "__main__":
  test.main()

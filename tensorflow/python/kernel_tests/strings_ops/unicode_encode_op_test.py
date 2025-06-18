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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test


class UnicodeEncodeOpTest(test.TestCase, parameterized.TestCase):

  def assertAllEqual(self, rt, expected):
    with self.cached_session() as sess:
      value = sess.run(rt)
      if isinstance(value, np.ndarray):
        value = value.tolist()
      elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
        value = value.to_list()
      self.assertEqual(value, expected)

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
        ragged_string_ops.unicode_encode()  # pylint: disable=no-value-for-parameter
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(72)  # pylint: disable=no-value-for-parameter
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(encoding="UTF-8")  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsValid(self, encoding):
    test_value = np.array([ord('H'), ord('e'), ord('o'), 0, ord('\b'), 0x1F600],
                          np.int32)
    expected_value = u"Heo\0\b\U0001f600".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "strict")
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsNegative(self, encoding):
    test_value = np.array([-1], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsTooLarge(self, encoding):
    test_value = np.array([0x110000], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsSurrogatePair(self, encoding):
    test_value = np.array([0xD83D, 0xDE00], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsNonCharacter(self, encoding):
    # This should not be an error.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([0xFDD0], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsLastTwoInBMP(self, encoding):
    # This should not be an error.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([0xFFFE, 0xFFFF], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsLastInPlane16(self, encoding):
    # This should not be an error.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([0x10FFFF], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testIgnoreErrors(self, encoding):
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o'),
                           0, ord('\b'), 0x1F600,  # valid
                           0xD83D, 0xDE00,  # invalid, surrogate code points
                           0xFDD0,  # noncharacter
                           0xFFFE, 0xFFFF,  # last two in BMP
                           0x10FFFF,  # last in plane 16 = last in Unicode
                           0x110000],  # invalid, beyond Unicode
                          np.int32)
    # There are two inconsistencies with the output compared to the
    # strict/replace behavior:
    # 1. Surrogate pairs are not rejected, but are combined, then encoded.
    # 2. noncharacters are not rejected
    expected_value = (
        u"Heo\0\b\U0001F600\U0001F600\ufdd0\ufffe\uffff\U0010ffff"
    ).encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "ignore")
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testReplaceErrors(self, encoding):
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o'),
                           0, ord('\b'), 0x1F600,  # valid
                           0xD83D, 0xDE00,  # invalid, surrogate code points
                           0xFDD0,  # noncharacter
                           0xFFFE, 0xFFFF,  # last two in BMP
                           0x10FFFF,  # last in plane 16 = last in Unicode
                           0x110000],  # invalid, beyond Unicode
                          np.int32)
    expected_value = (
        u"He\ufffd\ufffdo\0\b\U0001F600" + u"\ufffd" * 7).encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace")
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Test custom replacement character
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    expected_value = u"Heooo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", ord('o'))
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Verify "replace" is default
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    expected_value = u"He\U0000fffd\U0000fffdo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Verify non-default replacement with an unpaired surrogate.
    test_value = np.array([0xD801], np.int32)
    expected_value = u"A".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x41)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Test with a noncharacter code point.
    test_value = np.array([0x1FFFF], np.int32)
    expected_value = u"A".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x41)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Replacement_char must be within range
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x110000)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(unicode_encode_op)

  # -- regular Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testVector(self, encoding):
    test_value = np.array([ord('H'), ord('e'), ord('l'), ord('l'), ord('o')],
                          np.int32)
    expected_value = u"Hello".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    test_value = np.array([ord('H'), ord('e'), 0xC3, 0xC3, 0x1F604], np.int32)
    expected_value = u"He\xc3\xc3\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Single character string
    test_value = np.array([ord('H')], np.int32)
    expected_value = u"H".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    test_value = np.array([0x1F604], np.int32)
    expected_value = u"\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testMatrix(self, encoding):
    test_value = np.array(
        [[72, 0x1F604, 108, 108, 111], [87, 0x1F604, 114, 108, 100]], np.int32)
    expected_value = [
        u"H\U0001f604llo".encode(encoding), u"W\U0001f604rld".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrix(self, encoding):
    test_value = constant_op.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]],
         [[102, 105, 120, 101, 100], [119, 111, 114, 100, 115]],
         [[72, 121, 112, 101, 114], [99, 117, 98, 101, 46]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World".encode(encoding)],
                      [u"fixed".encode(encoding), u"words".encode(encoding)],
                      [u"Hyper".encode(encoding), u"cube.".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
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
    self.assertAllEqual(unicode_encode_op, expected_value)

  # -- Ragged Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testRaggedMatrix(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[ord('H'), 0xC3, ord('l'), ord('l'), ord('o')],
         [ord('W'), 0x1F604, ord('r'), ord('l'), ord('d'), ord('.')]], np.int32)
    expected_value = [
        u"H\xc3llo".encode(encoding), u"W\U0001f604rld.".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
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
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrixWithRagged3rdDim(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]],
         [[68, 111, 110, 39, 116], [119, 195, 114, 114, 121, 44, 32, 98, 101]],
         [[0x1F604], []]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [
                          u"Don't".encode(encoding),
                          u"w\xc3rry, be".encode(encoding)
                      ], [u"\U0001f604".encode(encoding), u"".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrixWithRagged2ndAnd3rdDim(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]], [],
         [[0x1F604]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [], [u"\U0001f604".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test4DimRaggedMatrix(self, encoding):
    test_value = ragged_factory_ops.constant(
        [[[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]]],
         [[[]], [[72, 121, 112, 101]]]], np.int32)
    expected_value = [[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                      [[u"".encode(encoding)], [u"Hype".encode(encoding)]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testRaggedMatrixWithMultiDimensionInnerValues(self, encoding):
    test_flat_values = constant_op.constant([[[72, 101, 108, 108, 111],
                                              [87, 111, 114, 108, 100]],
                                             [[102, 105, 120, 101, 100],
                                              [119, 111, 114, 100, 115]],
                                             [[72, 121, 112, 101, 114],
                                              [99, 117, 98, 101, 46]]])
    test_row_splits = [
        constant_op.constant([0, 2, 3], dtype=np.int64),
        constant_op.constant([0, 1, 1, 3], dtype=np.int64)
    ]
    test_value = ragged_tensor.RaggedTensor.from_nested_row_splits(
        test_flat_values, test_row_splits)
    expected_value = [[[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                       []],
                      [[[u"fixed".encode(encoding), u"words".encode(encoding)],
                        [u"Hyper".encode(encoding),
                         u"cube.".encode(encoding)]]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  def testUnknownInputRankError(self):
    # Use a tf.function that erases shape information.
    @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
    def f(v):
      return ragged_string_ops.unicode_encode(v, "UTF-8")

    with self.assertRaisesRegex(
        ValueError, "Rank of input_tensor must be statically known."):
      f([72, 101, 108, 108, 111])


if __name__ == "__main__":
  test.main()

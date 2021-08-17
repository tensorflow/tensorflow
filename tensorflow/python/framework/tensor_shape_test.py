# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Functional tests for shape inference helper classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DimensionTest(test_util.TensorFlowTestCase):

  def testDimension(self):
    dim = tensor_shape.Dimension(12)
    self.assertEqual(12, dim.value)
    self.assertEqual(12, int(dim))
    self.assertEqual(dim, tensor_shape.Dimension(12))
    self.assertEqual(tensor_shape.Dimension(15),
                     dim + tensor_shape.Dimension(3))
    self.assertEqual(tensor_shape.Dimension(15), dim + 3)
    self.assertEqual(tensor_shape.Dimension(15), 3 + dim)
    self.assertEqual(tensor_shape.Dimension(9), dim - 3)
    self.assertEqual(tensor_shape.Dimension(1), 13 - dim)
    self.assertEqual(tensor_shape.Dimension(24),
                     dim * tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(24), dim * 2)
    self.assertEqual(tensor_shape.Dimension(24), 2 * dim)
    self.assertEqual([4] * 12, [4] * dim)
    self.assertEqual(12 * [4], dim * [4])
    self.assertEqual(tensor_shape.Dimension(24), 2 * dim)
    self.assertEqual(
        tensor_shape.Dimension(6), dim // tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(6), dim // 2)
    self.assertEqual(tensor_shape.Dimension(0), 2 // dim)
    self.assertEqual(tensor_shape.Dimension(12),
                     dim.merge_with(tensor_shape.Dimension(12)))
    self.assertEqual(tensor_shape.Dimension(12), dim.merge_with(12))
    self.assertLess(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertGreater(tensor_shape.Dimension(13), tensor_shape.Dimension(12))
    self.assertLessEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertLessEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertGreater(tensor_shape.Dimension(13), tensor_shape.Dimension(12))
    self.assertGreaterEqual(tensor_shape.Dimension(12),
                            tensor_shape.Dimension(12))
    self.assertGreaterEqual(tensor_shape.Dimension(13),
                            tensor_shape.Dimension(12))
    self.assertNotEqual(dim, (12,))
    with self.assertRaises(ValueError):
      dim.merge_with(tensor_shape.Dimension(13))

  def testUnknownDimension(self):
    dim = tensor_shape.Dimension(None)
    self.assertIsNone(dim.value)
    self.assertEqual(dim.value, tensor_shape.Dimension(None).value)
    self.assertEqual(tensor_shape.Dimension(None).value,
                     (dim + tensor_shape.Dimension(None)).value)
    self.assertEqual(tensor_shape.Dimension(None).value,
                     (dim * tensor_shape.Dimension(None)).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        (dim // tensor_shape.Dimension(None)).value)
    self.assertEqual(tensor_shape.Dimension(None).value,
                     dim.merge_with(tensor_shape.Dimension(None)).value)
    self.assertIsNone(
        tensor_shape.Dimension(None) < tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) <= tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) > tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) >= tensor_shape.Dimension(None))

  def testKnownAndUnknownDimensions(self):
    known = tensor_shape.Dimension(12)
    unknown = tensor_shape.Dimension(None)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known + unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown + known).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known * unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown * known).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known // unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown // known).value)
    self.assertEqual(
        tensor_shape.Dimension(12), known.merge_with(unknown))
    self.assertEqual(
        tensor_shape.Dimension(12), unknown.merge_with(known))
    self.assertIsNone(tensor_shape.Dimension(12) < tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(12) <= tensor_shape.Dimension(None))
    self.assertIsNone(tensor_shape.Dimension(12) > tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(12) >= tensor_shape.Dimension(None))
    self.assertIsNone(tensor_shape.Dimension(None) < tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) <= tensor_shape.Dimension(12))
    self.assertIsNone(tensor_shape.Dimension(None) > tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) >= tensor_shape.Dimension(12))

  def testAsDimension(self):
    self.assertEqual(tensor_shape.Dimension(12),
                     tensor_shape.as_dimension(tensor_shape.Dimension(12)))
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.as_dimension(12))
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        tensor_shape.as_dimension(tensor_shape.Dimension(None)).value)
    self.assertEqual(tensor_shape.Dimension(None).value,
                     tensor_shape.as_dimension(None).value)

  def testEquality(self):
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertNotEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertIsNone(
        tensor_shape.Dimension(12) == tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) == tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) == tensor_shape.Dimension(None))

    # None indicates ambiguous comparison, but comparison vs the wrong type
    # is unambiguously False.
    self.assertIsNotNone(tensor_shape.Dimension(None) == 12.99)
    self.assertNotEqual(tensor_shape.Dimension(None), 12.99)

    # pylint: disable=singleton-comparison, g-equals-none
    self.assertIsNone(tensor_shape.Dimension(None) == None)
    # pylint: enable=singleton-comparison, g-equals-none
    self.assertNotEqual(tensor_shape.Dimension(12), 12.99)

  def testInequality(self):
    self.assertNotEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(12) != tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) != tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) != tensor_shape.Dimension(None))

    # None indicates ambiguous comparison, but comparison vs the wrong type
    # is unambiguously False.
    self.assertIsNotNone(tensor_shape.Dimension(None) != 12.99)
    self.assertNotEqual(tensor_shape.Dimension(None), 12.99)

    self.assertIsNone(tensor_shape.Dimension(None) != None)  # pylint: disable=g-equals-none
    self.assertNotEqual(tensor_shape.Dimension(12), 12.99)

  def testIsCompatibleWithError(self):
    with self.assertRaisesRegex(TypeError, "must be integer or None"):
      tensor_shape.Dimension(42).is_compatible_with([])

    with self.assertRaisesRegex(ValueError, "must be >= 0"):
      tensor_shape.Dimension(42).is_compatible_with(-1)

  def testMergeWithError(self):
    with self.assertRaisesRegex(TypeError, "must be integer or None"):
      tensor_shape.Dimension(42).merge_with([])

    with self.assertRaisesRegex(ValueError, "must be >= 0"):
      tensor_shape.Dimension(42).merge_with(-1)

  def testRepr(self):
    self.assertEqual(repr(tensor_shape.Dimension(7)), "Dimension(7)")
    self.assertEqual(repr(tensor_shape.Dimension(None)), "Dimension(None)")

  def testStr(self):
    self.assertEqual(str(tensor_shape.Dimension(7)), "7")
    self.assertEqual(str(tensor_shape.Dimension(None)), "?")

  def testUnsupportedType(self):
    with self.assertRaises(TypeError):
      tensor_shape.Dimension(dtypes.string)

  def testBool(self):
    one = tensor_shape.Dimension(1)
    zero = tensor_shape.Dimension(0)
    has_none = tensor_shape.Dimension(None)
    self.assertTrue(one)
    self.assertFalse(zero)
    self.assertFalse(has_none)

  def testMod(self):
    four = tensor_shape.Dimension(4)
    nine = tensor_shape.Dimension(9)
    self.assertEqual(nine % four, 1)
    # test both __mod__ and __rmod__.
    self.assertEqual(nine % 4, 1)
    self.assertEqual(4 % nine, 4)

  def testReduce(self):
    dim = tensor_shape.Dimension(5)
    ctor, args = dim.__reduce__()
    self.assertEqual(ctor, tensor_shape.Dimension)
    self.assertEqual(args, (5,))
    reconstructed = ctor(*args)
    self.assertEqual(reconstructed, dim)

  def testDiv(self):
    # Note: This test is related to GitHub issue 25790.
    six = tensor_shape.Dimension(6)
    two = tensor_shape.Dimension(2)
    message = (r"unsupported operand type\(s\) for /: "
               r"'Dimension' and 'Dimension', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = six / two
    message = (r"unsupported operand type\(s\) for /: "
               r"'Dimension' and 'int', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = six / 2
    message = (r"unsupported operand type\(s\) for /: "
               r"'int' and 'Dimension', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = 6 / two


class ShapeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testUnknownShape(self):
    s = tensor_shape.TensorShape(None)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(ValueError, "Shape .+ is not fully defined"):
      s.assert_is_fully_defined()
    # pylint: enable=g-error-prone-assert-raises
    self.assertIsNone(s.rank)
    with self.assertRaisesRegex(
        ValueError, "Cannot take the length of shape with unknown rank."):
      len(s)
    self.assertFalse(s)
    self.assertIsNone(s.dims)
    with self.assertRaisesRegex(
        ValueError, "Cannot iterate over a shape with unknown rank."):
      for _ in tensor_shape.TensorShape(None):
        pass

  def testFullyDefinedShape(self):
    s = tensor_shape.TensorShape([tensor_shape.Dimension(
        3), tensor_shape.Dimension(4), tensor_shape.Dimension(7)])
    s.assert_is_fully_defined()
    self.assertEqual(s.rank, 3)
    self.assertLen(s, 3)
    self.assertTrue(s)
    s.assert_has_rank(3)
    self.assertEqual([tensor_shape.Dimension(3),
                      tensor_shape.Dimension(4),
                      tensor_shape.Dimension(7)], s.dims)
    self.assertEqual(tensor_shape.Dimension(3), s[0])
    self.assertEqual(tensor_shape.Dimension(4), s[1])
    self.assertEqual(tensor_shape.Dimension(7), s[2])
    self.assertEqual([3, 4, 7], s.as_list())
    s.assert_is_compatible_with([3, 4, 7])
    s.assert_same_rank([6, 3, 7])
    for d1, d2 in zip(s, [3, 4, 7]):
      assert tensor_shape.dimension_value(d1) == d2

  def testPartiallyDefinedShape(self):
    s = tensor_shape.TensorShape([tensor_shape.Dimension(
        3), tensor_shape.Dimension(None), tensor_shape.Dimension(7)])
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(ValueError, "Shape .+ is not fully defined"):
      s.assert_is_fully_defined()
    # pylint: enable=g-error-prone-assert-raises
    self.assertEqual(s.rank, 3)
    self.assertLen(s, 3)
    self.assertTrue(s)
    s.assert_has_rank(3)
    self.assertEqual(tensor_shape.Dimension(3), s[0])
    self.assertEqual(tensor_shape.Dimension(None).value, s.dims[1].value)
    self.assertEqual(tensor_shape.Dimension(7), s.dims[2])
    s.assert_same_rank([6, 3, 7])
    for d1, d2 in zip(s, [3, None, 7]):
      assert tensor_shape.dimension_value(d1) == d2

  def testMergeFullShapes(self):
    self.assertEqual([3, 4, 7],
                     tensor_shape.TensorShape([3, 4, 7]).merge_with(
                         tensor_shape.TensorShape([3, 4, 7])).as_list())
    with self.assertRaises(ValueError):
      tensor_shape.TensorShape([3, 4, 7]).merge_with(
          tensor_shape.TensorShape([6, 3, 7]))

  def testMergePartialShapes(self):
    s1 = tensor_shape.TensorShape([tensor_shape.Dimension(
        3), tensor_shape.Dimension(None), tensor_shape.Dimension(7)])
    s2 = tensor_shape.TensorShape([tensor_shape.Dimension(
        None), tensor_shape.Dimension(4), tensor_shape.Dimension(7)])
    self.assertEqual([3, 4, 7], s1.merge_with(s2).as_list())

  def testMergeFullAndUnknownShape(self):
    self.assertEqual([3, 4, 7],
                     tensor_shape.TensorShape([3, 4, 7]).merge_with(
                         tensor_shape.TensorShape(None)).as_list())

  def testSlice(self):
    known = tensor_shape.TensorShape([0, 1, 2, 3, 4])
    self.assertEqual(tensor_shape.Dimension(2), known[2])
    tensor_shape.TensorShape([1, 2, 3]).assert_is_compatible_with(known[1:4])

    unknown = tensor_shape.TensorShape(None)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        tensor_shape.dimension_value(unknown[2]))
    tensor_shape.TensorShape(
        [None, None, None]).assert_is_compatible_with(unknown[1:4])

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y)),
      ("Add", lambda x, y: x + y),
      ("RAdd", lambda x, y: y.__radd__(x)))
  def testConcatenate(self, concatenate_fn):
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]),
            tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]),
            tensor_shape.TensorShape(None)))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape(None),
            tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape(None),
            tensor_shape.TensorShape(None)))

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y)),
      ("Add", lambda x, y: x + y))
  def testConcatenateWithDimension(self, concatenate_fn):
    tensor_shape.TensorShape([1, 2, 3]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]),
            tensor_shape.Dimension(3)))

  @parameterized.named_parameters(
      ("List", [3, 4, 5]),
      ("Tuple", (3, 4, 5)))
  def testAdd_nonTensorShape(self, addend):
    two = tensor_shape.TensorShape([2])
    result = two + addend
    self.assertIsInstance(result, tensor_shape.TensorShape)
    tensor_shape.TensorShape([2, 3, 4, 5]).assert_is_compatible_with(result)

  @parameterized.named_parameters(
      ("List", [2, 3, 4]),
      ("Tuple", (2, 3, 4)))
  def testRAdd_nonTensorShape(self, addend):
    five = tensor_shape.TensorShape([5])
    result = addend + five
    self.assertIsInstance(result, tensor_shape.TensorShape)
    tensor_shape.TensorShape([2, 3, 4, 5]).assert_is_compatible_with(result)

  def _testMostSpecificCompatibleShapeHelper(self, x, y, expected):
    mcs = tensor_shape.TensorShape(x).most_specific_compatible_shape(
        tensor_shape.TensorShape(y))
    mcs_dims = mcs.dims
    if expected is None or mcs_dims is None:
      self.assertIs(expected, mcs_dims)
    else:
      self.assertEqual(expected, mcs.as_list())

  def testMostSpecificCompatibleShape(self):
    self._testMostSpecificCompatibleShapeHelper([1, 2], None, None)
    self._testMostSpecificCompatibleShapeHelper(None, [1, 2], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2], [1, 2, 3, 4], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2, 3, 4], [1, 2], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2], [1, 2], [1, 2])
    self._testMostSpecificCompatibleShapeHelper([None, 2, 3], [1, 1, 3],
                                                [None, None, 3])
    self._testMostSpecificCompatibleShapeHelper([1, 1, 3], [None, 2, 3],
                                                [None, None, 3])

  def testTruedivFails(self):
    unknown = tensor_shape.Dimension(None)
    self.assertEqual((unknown // unknown).value, None)
    with self.assertRaisesRegex(TypeError, r"unsupported operand type"):
      unknown / unknown  # pylint: disable=pointless-statement

  def testConvertFromProto(self):
    def make_tensor_shape_proto(shape):
      return tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=x) for x in shape])
    proto = make_tensor_shape_proto([])
    self.assertEqual(tensor_shape.TensorShape([]),
                     tensor_shape.TensorShape(proto))
    self.assertEqual(tensor_shape.TensorShape([]),
                     tensor_shape.as_shape(proto))

    proto = make_tensor_shape_proto([1, 37, 42])
    self.assertEqual(tensor_shape.TensorShape([1, 37, 42]),
                     tensor_shape.TensorShape(proto))
    self.assertEqual(tensor_shape.TensorShape([1, 37, 42]),
                     tensor_shape.as_shape(proto))

    partial_proto_shape = tensor_shape.as_shape(
        make_tensor_shape_proto([-1, 37, 42]))
    partial_shape = tensor_shape.TensorShape([None, 37, 42])
    self.assertNotEqual(partial_proto_shape, partial_shape)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[0]), None)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[1]), 37)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[2]), 42)
    self.assertTrue(partial_shape.is_compatible_with(partial_proto_shape))

  def testStr(self):
    self.assertEqual("<unknown>", str(tensor_shape.unknown_shape()))
    self.assertEqual(
        "(None,)",
        str(tensor_shape.unknown_shape(rank=1)).replace("?", "None"))
    self.assertEqual(
        "(None, None)",
        str(tensor_shape.unknown_shape(rank=2)).replace("?", "None"))
    self.assertEqual(
        "(None, None, None)",
        str(tensor_shape.unknown_shape(rank=3)).replace("?", "None"))
    self.assertEqual(
        "(32, None, 1, 9)",
        str(tensor_shape.TensorShape([32, None, 1, 9])).replace("?", "None"))
    self.assertEqual("()", str(tensor_shape.TensorShape([])))
    self.assertEqual("(7,)", str(tensor_shape.TensorShape([7])))
    self.assertEqual("(3, 8)", str(tensor_shape.TensorShape([3, 8])))
    self.assertEqual("(4, 5, 2)", str(tensor_shape.TensorShape([4, 5, 2])))

  def testAsProto(self):
    self.assertTrue(tensor_shape.unknown_shape().as_proto().unknown_rank)
    self.assertFalse(
        tensor_shape.unknown_shape(rank=3).as_proto().unknown_rank)
    self.assertFalse(
        tensor_shape.TensorShape([1, 2, 3]).as_proto().unknown_rank)
    self.assertFalse(
        tensor_shape.TensorShape([1, None, 3]).as_proto().unknown_rank)

  def testEquality(self):
    s1 = tensor_shape.TensorShape([tensor_shape.Dimension(
        3), tensor_shape.Dimension(4), tensor_shape.Dimension(7)])
    s2 = tensor_shape.TensorShape([tensor_shape.Dimension(
        3), tensor_shape.Dimension(4), tensor_shape.Dimension(7)])
    s3 = tensor_shape.TensorShape([tensor_shape.Dimension(3),
                                   tensor_shape.Dimension(4), None])

    self.assertEqual(s1, s2)
    self.assertEqual(s1, s2)

    # Test with an unknown shape in s3
    self.assertNotEqual(s1, s3)

    # eq and neq are not symmetric for unknown shapes.
    # pylint: disable=g-generic-assert
    unk0 = tensor_shape.unknown_shape()
    self.assertFalse(unk0 == s1)
    self.assertFalse(s1 == unk0)
    # pylint: enable=g-generic-assert
    with self.assertRaises(ValueError):
      _ = unk0 != s1
    with self.assertRaises(ValueError):
      _ = s1 != unk0
    unk1 = tensor_shape.unknown_shape()
    # pylint: disable=g-generic-assert
    self.assertTrue(unk0 == unk1)
    self.assertTrue(unk1 == unk0)
    # pylint: enable=g-generic-assert
    with self.assertRaises(ValueError):
      _ = unk0 != unk1
    with self.assertRaises(ValueError):
      _ = unk1 != unk0

  def testAsList(self):
    with self.assertRaisesRegex(ValueError,
                                "not defined on an unknown TensorShape"):
      tensor_shape.unknown_shape().as_list()
    self.assertAllEqual([None, None], tensor_shape.unknown_shape(2).as_list())
    self.assertAllEqual([2, None, 4], tensor_shape.TensorShape(
        (2, None, 4)).as_list())

  def testReduce(self):
    shape = tensor_shape.TensorShape([2, 3])
    ctor, args = shape.__reduce__()
    self.assertEqual(ctor, tensor_shape.TensorShape)
    self.assertEqual(args,
                     ([tensor_shape.Dimension(2),
                       tensor_shape.Dimension(3)],))
    reconstructed = ctor(*args)
    self.assertEqual(reconstructed, shape)


if __name__ == "__main__":
  googletest.main()

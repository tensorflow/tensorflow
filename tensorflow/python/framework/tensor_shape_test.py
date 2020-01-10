"""Functional tests for shape inference helper classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

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
    self.assertEqual(tensor_shape.Dimension(24),
                     dim * tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(24), dim * 2)
    self.assertEqual(
        tensor_shape.Dimension(6), dim // tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(6), dim // 2)
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
    with self.assertRaises(ValueError):
      dim.merge_with(tensor_shape.Dimension(13))

  def testUnknownDimension(self):
    dim = tensor_shape.Dimension(None)
    self.assertIs(None, dim.value)
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
    self.assertIs(None,
                  tensor_shape.Dimension(None) < tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(None) <= tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(None) > tensor_shape.Dimension(None))
    self.assertIs(None,
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
    self.assertIs(None,
                  tensor_shape.Dimension(12) < tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(12) <= tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(12) > tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(12) >= tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(None) < tensor_shape.Dimension(12))
    self.assertIs(None,
                  tensor_shape.Dimension(None) <= tensor_shape.Dimension(12))
    self.assertIs(None,
                  tensor_shape.Dimension(None) > tensor_shape.Dimension(12))
    self.assertIs(None,
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
    self.assertTrue(tensor_shape.Dimension(12) == tensor_shape.Dimension(12))
    self.assertFalse(tensor_shape.Dimension(12) == tensor_shape.Dimension(13))
    self.assertIs(None,
                  tensor_shape.Dimension(12) == tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(None) == tensor_shape.Dimension(12))
    self.assertIs(None,
                  tensor_shape.Dimension(None) == tensor_shape.Dimension(None))

  def testInequality(self):
    self.assertTrue(tensor_shape.Dimension(12) != tensor_shape.Dimension(13))
    self.assertFalse(tensor_shape.Dimension(12) != tensor_shape.Dimension(12))
    self.assertIs(None,
                  tensor_shape.Dimension(12) != tensor_shape.Dimension(None))
    self.assertIs(None,
                  tensor_shape.Dimension(None) != tensor_shape.Dimension(12))
    self.assertIs(None,
                  tensor_shape.Dimension(None) != tensor_shape.Dimension(None))


class ShapeTest(test_util.TensorFlowTestCase):

  def testUnknownShape(self):
    s = tensor_shape.TensorShape(None)
    with self.assertRaises(ValueError):
      s.assert_is_fully_defined()
    self.assertIs(None, s.ndims)
    with self.assertRaises(ValueError):
      len(s)
    self.assertFalse(s)
    self.assertIs(None, s.dims)

  def testFullyDefinedShape(self):
    s = tensor_shape.TensorShape([tensor_shape.Dimension(3),
                     tensor_shape.Dimension(4),
                     tensor_shape.Dimension(7)])
    s.assert_is_fully_defined()
    self.assertEqual(3, s.ndims)
    self.assertEqual(3, len(s))
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

  def testPartiallyDefinedShape(self):
    s = tensor_shape.TensorShape([tensor_shape.Dimension(3),
                     tensor_shape.Dimension(None),
                     tensor_shape.Dimension(7)])
    with self.assertRaises(ValueError):
      s.assert_is_fully_defined()
    self.assertEqual(3, s.ndims)
    self.assertEqual(3, len(s))
    self.assertTrue(s)
    s.assert_has_rank(3)
    self.assertEqual(tensor_shape.Dimension(3), s[0])
    self.assertEqual(tensor_shape.Dimension(None).value, s[1].value)
    self.assertEqual(tensor_shape.Dimension(7), s[2])
    s.assert_same_rank([6, 3, 7])

  def testMergeFullShapes(self):
    self.assertEqual([3, 4, 7],
                     tensor_shape.TensorShape([3, 4, 7]).merge_with(
                         tensor_shape.TensorShape([3, 4, 7])).as_list())
    with self.assertRaises(ValueError):
      tensor_shape.TensorShape([3, 4, 7]).merge_with(
          tensor_shape.TensorShape([6, 3, 7]))

  def testMergePartialShapes(self):
    s1 = tensor_shape.TensorShape([tensor_shape.Dimension(3),
                      tensor_shape.Dimension(None),
                      tensor_shape.Dimension(7)])
    s2 = tensor_shape.TensorShape([tensor_shape.Dimension(None),
                      tensor_shape.Dimension(4),
                      tensor_shape.Dimension(7)])
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
    self.assertEqual(tensor_shape.Dimension(None).value, unknown[2].value)
    tensor_shape.TensorShape(
        [None, None, None]).assert_is_compatible_with(unknown[1:4])

  def testConcatenate(self):
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        tensor_shape.TensorShape([1, 2]).concatenate(
            tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        tensor_shape.TensorShape([1, 2]).concatenate(
            tensor_shape.TensorShape(None)))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        tensor_shape.TensorShape(None).concatenate(
            tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        tensor_shape.TensorShape(None).concatenate(
            tensor_shape.TensorShape(None)))
    tensor_shape.TensorShape([1, 2, 3]).assert_is_compatible_with(
        tensor_shape.TensorShape([1, 2]).concatenate(
            tensor_shape.Dimension(3)))

  def testHelpers(self):
    tensor_shape.TensorShape([]).assert_is_compatible_with(
        tensor_shape.scalar())
    tensor_shape.TensorShape([37]).assert_is_compatible_with(
        tensor_shape.vector(37))
    tensor_shape.TensorShape(
        [94, 43]).assert_is_compatible_with(tensor_shape.matrix(94, 43))

  def testTruedivFails(self):
    unknown = tensor_shape.Dimension(None)
    self.assertEqual((unknown // unknown).value, None)
    with self.assertRaisesRegexp(TypeError, r"unsupported operand type"):
      unknown / unknown  # pylint: disable=pointless-statement


if __name__ == "__main__":
  googletest.main()

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Tests for third_party.tensorflow.compiler.xla.python_api.xla_literal."""

import collections
import itertools
import operator

from absl.testing import absltest
import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import xla_literal


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
  return np.array(*args, dtype=bool, **kwargs)


# To facilitate iteration over different test cases, we collect similar array
# creation functions and identify each with its corresponding XLA element type
# (a field in xla_data_pb2) and literal field attribute getter (from types.py).

ArrayCreatorRecord = collections.namedtuple('ArrayCreatorRecord',
                                            ['array_fun', 'etype', 'pbfield'])

float_arrays = [ArrayCreatorRecord(NumpyArrayF32, xla_data_pb2.F32,
                                   operator.attrgetter('f32s')),
                ArrayCreatorRecord(NumpyArrayF64, xla_data_pb2.F64,
                                   operator.attrgetter('f64s'))]

int_arrays = [ArrayCreatorRecord(NumpyArrayS32, xla_data_pb2.S32,
                                 operator.attrgetter('s32s')),
              ArrayCreatorRecord(NumpyArrayS64, xla_data_pb2.S64,
                                 operator.attrgetter('f64s'))]


class XlaLiteralTest(absltest.TestCase):

  def assertShape(self, shape, expected_dimensions, expected_element_type):
    self.assertEqual(shape.element_type, expected_element_type)
    self.assertEqual(shape.dimensions, expected_dimensions)

  def assertLayout(self, layout, expected_minor_to_major):
    self.assertEqual(layout.minor_to_major, expected_minor_to_major)

  def assertTupleShape(self, shape, expected):
    self.assertEqual(shape.element_type, xla_data_pb2.TUPLE)
    for sub_shape, sub_expected in zip(
        shape.tuple_shapes, expected):
      if sub_shape.element_type == xla_data_pb2.TUPLE:
        self.assertTupleShape(sub_shape, sub_expected)
      else:
        expected_dimensions, expected_element_types = sub_expected
        self.assertShape(
            sub_shape, expected_dimensions, expected_element_types)

  def testConvertNumpyScalar1DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(array(1.1))
      self.assertShape(literal.shape, [], etype)
      self.assertLayout(literal.shape.layout, [])
      np.testing.assert_allclose(pbfield(literal), [1.1])

  def testConvertNumpyArray1DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          array([1.1, 2.2, 3.3]))
      self.assertShape(literal.shape, [3], etype)
      self.assertLayout(literal.shape.layout, [0])
      np.testing.assert_allclose(pbfield(literal), [1.1, 2.2, 3.3])

  def testConvertNumpyArray2DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          array([[1, 2, 3], [4, 5, 6]]))
      self.assertShape(literal.shape, [2, 3], etype)
      # By default the layout is row-major ('C' order).
      self.assertLayout(literal.shape.layout, [1, 0])
      np.testing.assert_allclose(pbfield(literal), [1, 2, 3, 4, 5, 6])

  def testConvertNumpyArray2DToLiteralColumnMajor(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          array(
              [[1, 2, 3], [4, 5, 6]], order='F'))
      self.assertShape(literal.shape, [2, 3], etype)
      self.assertLayout(literal.shape.layout, [0, 1])
      np.testing.assert_allclose(pbfield(literal), [1, 4, 2, 5, 3, 6])

  def testConvertNumpyArray3DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          array([[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]], [[
              100, 200, 300
          ], [400, 500, 600]], [[1000, 2000, 3000], [4000, 5000, 6000]]]))
      self.assertShape(literal.shape, [4, 2, 3], etype)
      self.assertLayout(literal.shape.layout, [2, 1, 0])
      np.testing.assert_allclose(pbfield(literal), [
          1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500,
          600, 1000, 2000, 3000, 4000, 5000, 6000
      ])

  def testConvertTupleOfNumpyArray3DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      inner_array = array([
          [[1, 2, 3], [4, 5, 6]],
          [[10, 20, 30], [40, 50, 60]],
          [[100, 200, 300], [400, 500, 600]],
          [[1000, 2000, 3000], [4000, 5000, 6000]]])
      inner_spec = ([4, 2, 3], etype)
      inner_flat = [
          1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500,
          600, 1000, 2000, 3000, 4000, 5000, 6000
      ]
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          (inner_array, inner_array, inner_array))

      self.assertTupleShape(
          literal.shape,
          (inner_spec, inner_spec, inner_spec))

      for subliteral in literal.tuple_literals:
        self.assertLayout(subliteral.shape.layout, [2, 1, 0])
        np.testing.assert_allclose(pbfield(subliteral), inner_flat)

  def testConvertNestedTupleOfNumpyArray3DToLiteral(self):
    for array, etype, pbfield in float_arrays:
      inner_array = array([
          [[1, 2, 3], [4, 5, 6]],
          [[10, 20, 30], [40, 50, 60]],
          [[100, 200, 300], [400, 500, 600]],
          [[1000, 2000, 3000], [4000, 5000, 6000]]])
      inner_spec = ([4, 2, 3], etype)
      inner_flat = [
          1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500,
          600, 1000, 2000, 3000, 4000, 5000, 6000
      ]
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          (inner_array, (inner_array, inner_array), inner_array))

      self.assertTupleShape(
          literal.shape,
          (inner_spec, (inner_spec, inner_spec), inner_spec))

      leaf_literals = (
          literal.tuple_literals[0],
          literal.tuple_literals[1].tuple_literals[0],
          literal.tuple_literals[1].tuple_literals[1],
          literal.tuple_literals[2])

      for leaf_literal in leaf_literals:
        self.assertLayout(leaf_literal.shape.layout, [2, 1, 0])
        np.testing.assert_allclose(pbfield(leaf_literal), inner_flat)

  def testConvertNumpyArray3DToLiteralColumnMajor(self):
    for array, etype, pbfield in float_arrays:
      literal = xla_literal.ConvertNumpyArrayToLiteral(
          array(
              [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]], [[
                  100, 200, 300
              ], [400, 500, 600]], [[1000, 2000, 3000], [4000, 5000, 6000]]],
              order='F'))
      self.assertShape(literal.shape, [4, 2, 3], etype)
      self.assertLayout(literal.shape.layout, [0, 1, 2])
      np.testing.assert_allclose(pbfield(literal), [
          1, 10, 100, 1000, 4, 40, 400, 4000, 2, 20, 200, 2000, 5, 50, 500,
          5000, 3, 30, 300, 3000, 6, 60, 600, 6000
      ])

  def testNumpyToLiteralToNumpyRoundtrip(self):

    def _DoRoundtripTest(ndarray_in):
      literal = xla_literal.ConvertNumpyArrayToLiteral(ndarray_in)
      ndarray_out = xla_literal.ConvertLiteralToNumpyArray(literal)
      np.testing.assert_allclose(ndarray_in, ndarray_out)

    _DoRoundtripTest(NumpyArrayBool([False, True, True, False]))

    for array, _, _ in itertools.chain(float_arrays, int_arrays):
      ## Scalars
      _DoRoundtripTest(array(42))
      _DoRoundtripTest(array(42, order='F'))

      ## 1D
      _DoRoundtripTest(array([42, 52]))
      _DoRoundtripTest(array([42, 52], order='F'))

      ## 2D
      _DoRoundtripTest(array([[1, 2, 3], [10, 20, 30]]))
      _DoRoundtripTest(array([[1, 2, 3], [10, 20, 30]], order='F'))

      ## 3D
      _DoRoundtripTest(array([[[1, 2, 3], [4, 5, 6]],
                              [[7, 8, 9], [10, 11, 12]]]))
      _DoRoundtripTest(array([[[1, 2, 3], [4, 5, 6]],
                              [[7, 8, 9], [10, 11, 12]]], order='F'))


if __name__ == '__main__':
  absltest.main()

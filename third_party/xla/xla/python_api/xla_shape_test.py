# Copyright 2023 The OpenXLA Authors.
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
"""Tests for third_party.tensorflow.compiler.xla.python_api.xla_shape."""

from absl.testing import absltest
import numpy as np
from local_xla.xla import xla_data_pb2
from xla.python_api import xla_shape


def NumpyArrayF32(*args, **kwargs):
  """Convenience wrapper to create Numpy arrays with a np.float32 dtype."""
  return np.array(*args, dtype=np.float32, **kwargs)


class CreateShapeFromNumpyTest(absltest.TestCase):

  def assertShape(self, shape, expected_dimensions, expected_element_type):
    self.assertEqual(shape.element_type(), expected_element_type)
    self.assertEqual(shape.dimensions(), expected_dimensions)

  def assertLayout(self, layout, expected_minor_to_major):
    self.assertEqual(layout.minor_to_major, expected_minor_to_major)

  def assertTupleShape(self, shape, expected):
    self.assertEqual(shape.element_type(), xla_data_pb2.TUPLE)
    for sub_shape, sub_message, sub_expected in zip(
        shape.tuple_shapes(),
        shape.message.tuple_shapes,
        expected):
      self.assertEqual(sub_shape.element_type(), sub_message.element_type)
      if sub_shape.is_tuple():
        self.assertTupleShape(sub_shape, sub_expected)
      else:
        expected_dimensions, expected_element_types = sub_expected
        self.assertShape(
            sub_shape, expected_dimensions, expected_element_types)

  def testCreateShapeFromNumpy1D(self):
    shape = xla_shape.CreateShapeFromNumpy(NumpyArrayF32([1.1, 2.2]))
    self.assertShape(shape, [2], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0])

  def testCreateShapeFromNumpy2DRowMajor(self):
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], order='C'))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [1, 0])

  def testCreateShapeFromNumpy2DColumnMajor(self):
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], order='F'))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0, 1])

  def testCreateShapeFromNumpy2DDefaultIsRowMajor(self):
    # The default layout in Numpy is C (row major)
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    self.assertShape(shape, [2, 3], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [1, 0])

  def testCreateShapeFromNumpy3DRowMajor(self):
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]], order='C'))
    self.assertShape(shape, [2, 3, 1], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [2, 1, 0])

  def testCreateShapeFromNumpy3DColumnMajor(self):
    shape = xla_shape.CreateShapeFromNumpy(
        NumpyArrayF32(
            [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]], order='F'))
    self.assertShape(shape, [2, 3, 1], xla_data_pb2.F32)
    self.assertLayout(shape.layout(), [0, 1, 2])

  def testCreateShapeFromTupleOfNumpy3D(self):
    inner_array = NumpyArrayF32(
        [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]])
    inner_spec = ([2, 3, 1], xla_data_pb2.F32)

    shape = xla_shape.CreateShapeFromNumpy(
        (inner_array, inner_array, inner_array))
    self.assertTupleShape(
        shape,
        (inner_spec, inner_spec, inner_spec))

  def testCreateShapeFromNestedTupleOfNumpy3D(self):
    inner_array = NumpyArrayF32(
        [[[1.1], [2.2], [3.3]], [[4.4], [5.5], [6.6]]])
    inner_spec = ([2, 3, 1], xla_data_pb2.F32)

    shape = xla_shape.CreateShapeFromNumpy(
        (inner_array, (inner_array, inner_array), inner_array))
    self.assertTupleShape(
        shape,
        (inner_spec, (inner_spec, inner_spec), inner_spec))


if __name__ == '__main__':
  absltest.main()

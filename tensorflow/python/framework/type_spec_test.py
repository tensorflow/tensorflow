# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the TypeSpec base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.platform import googletest


class TwoTensors(object):
  """A simple value type to test TypeSpec.

  Contains two tensors (x, y) and a string (color).  The color value is a
  stand-in for any extra type metadata we might need to store.
  """

  def __init__(self, x, y, color="red"):
    assert isinstance(color, str)
    self.x = ops.convert_to_tensor(x)
    self.y = ops.convert_to_tensor(y)
    self.color = color


class TwoTensorsSpec(type_spec.TypeSpec):
  """A TypeSpec for the TwoTensors value type."""

  def __init__(self, x_shape, x_dtype, y_shape, y_dtype, color="red"):
    self.x_shape = tensor_shape.as_shape(x_shape)
    self.x_dtype = dtypes.as_dtype(x_dtype)
    self.y_shape = tensor_shape.as_shape(y_shape)
    self.y_dtype = dtypes.as_dtype(y_dtype)
    self.color = color

  value_type = property(lambda self: TwoTensors)

  @property
  def _component_specs(self):
    return (tensor_spec.TensorSpec(self.x_shape, self.x_dtype),
            tensor_spec.TensorSpec(self.y_shape, self.y_dtype))

  def _to_components(self, value):
    return (value.x, value.y)

  def _from_components(self, components):
    return TwoTensors(*components)

  def _serialize(self):
    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)

  @classmethod
  def from_value(cls, value):
    return cls(value.x.shape, value.x.dtype, value.y.shape, value.y.dtype,
               value.color)


type_spec.register_type_spec_from_value_converter(
    TwoTensors, TwoTensorsSpec.from_value)


class TypeSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("FullySpecified",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("Metadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("NumpyMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 4]])),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 4]]))),
      )
  def testEquality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertEqual(v1, v2)
    self.assertEqual(v2, v1)
    self.assertFalse(v1 != v2)
    self.assertFalse(v2 != v1)
    self.assertEqual(hash(v1), hash(v2))

  @parameterized.named_parameters(
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("IncompatibleDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      ("DiffMetadataNumpy",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 4]])),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 8]]))),
      ("DiffMetadataTensorSpecName",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="a")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="b"))),
      ("Non-TypeSpec",
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool), 5),
      )
  def testInequality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertNotEqual(v1, v2)
    self.assertNotEqual(v2, v1)
    self.assertFalse(v1 == v2)
    self.assertFalse(v2 == v1)

  @parameterized.named_parameters(
      ("SameValue",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      )
  def testIsCompatibleWith(self, v1, v2):
    self.assertTrue(v1.is_compatible_with(v2))
    self.assertTrue(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("IncompatibleDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      )
  def testIsNotCompatibleWith(self, v1, v2):
    self.assertFalse(v1.is_compatible_with(v2))
    self.assertFalse(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ("EqualTypes",
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("DiffRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)),
      ("DiffDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("DiffMetadataTensorSpecName",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="a")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="b")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name=None))),
      )
  def testMostSpecificCompatibleType(self, v1, v2, expected):
    self.assertEqual(v1.most_specific_compatible_type(v2), expected)
    self.assertEqual(v2.most_specific_compatible_type(v1), expected)

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool, "blue")),
      )
  def testMostSpecificCompatibleTypeException(self, v1, v2):
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  def toTensorList(self):
    value = TwoTensors([1, 2, 3], [1.0, 2.0], "red")
    spec = TwoTensorsSpec.from_value(value)
    tensor_list = spec._to_tensor_list(value)
    self.assertLen(tensor_list, 2)
    self.assertIs(tensor_list[0], value.x)
    self.assertIs(tensor_list[1], value.y)

  def fromTensorList(self):
    x = ops.convert_to_tensor([1, 2, 3])
    y = ops.convert_to_tensor([1.0, 2.0])
    color = "green"
    spec = TwoTensorsSpec(x.shape, x.dtype, y.shape, y.dtype, color)
    value = spec._from_tensor_list([x, y])
    self.assertIs(value.x, x)
    self.assertIs(value.y, y)
    self.assertEqual(value.color, color)

  def fromIncompatibleTensorList(self):
    x = ops.convert_to_tensor([1, 2, 3])
    y = ops.convert_to_tensor([1.0, 2.0])
    spec1 = TwoTensorsSpec([100], x.dtype, y.shape, y.dtype, "green")
    spec2 = TwoTensorsSpec(x.shape, x.dtype, y.shape, dtypes.bool, "green")
    with self.assertRaises(ValueError):
      spec1._from_tensor_list([x, y])  # shape mismatch
    with self.assertRaises(ValueError):
      spec2._from_tensor_list([x, y])  # dtype mismatch

  def testFlatTensorSpecs(self):
    spec = TwoTensorsSpec([5], dtypes.int32, [5, 8], dtypes.float32, "red")
    self.assertEqual(spec._flat_tensor_specs,
                     [tensor_spec.TensorSpec([5], dtypes.int32),
                      tensor_spec.TensorSpec([5, 8], dtypes.float32)])

  def testRepr(self):
    spec = TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)
    self.assertEqual(
        repr(spec),
        "TwoTensorsSpec(%r, %r, %r, %r, %r)" %
        (tensor_shape.TensorShape([5, 3]), dtypes.int32,
         tensor_shape.TensorShape(None), dtypes.bool, "red"))

  def testFromValue(self):
    value = TwoTensors([1, 2, 3], [1.0, 2.0], "red")
    spec = type_spec.type_spec_from_value(value)
    self.assertEqual(spec, TwoTensorsSpec.from_value(value))

if __name__ == "__main__":
  googletest.main()

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

import collections

from absl.testing import parameterized

import numpy as np
import six

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


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


@type_spec.register("tf.TwoTensorsSpec")
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
    x, y = components
    return TwoTensors(x, y, self.color)

  def _serialize(self):
    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)

  @classmethod
  def from_value(cls, value):
    return cls(value.x.shape, value.x.dtype, value.y.shape, value.y.dtype,
               value.color)


@type_spec.register("tf.TwoTensorsSpecTwin")
class TwoTensorsSpecTwin(TwoTensorsSpec):
  pass


@type_spec.register("tf.TwoTensorsSpecVariableSerialize")
class TwoTensorsSpecVariableSerialize(TwoTensorsSpec):

  def _serialize(self):
    if self.color == "smaller_tuple":
      return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype)
    elif self.color == "different_order":
      return (self.y_shape, self.x_shape, self.y_dtype, self.color,
              self.x_dtype)

    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)


type_spec.register_type_spec_from_value_converter(
    TwoTensors, TwoTensorsSpec.from_value)


class TwoComposites(object):
  """A simple value type to test TypeSpec.

  Contains two composite tensorstensors (x, y) and a string (color).
  """

  def __init__(self, x, y, color="red"):
    assert isinstance(color, str)
    self.x = ops.convert_to_tensor_or_composite(x)
    self.y = ops.convert_to_tensor_or_composite(y)
    self.color = color


@type_spec.register("tf.TwoCompositesSpec")
class TwoCompositesSpec(type_spec.BatchableTypeSpec):
  """A TypeSpec for the TwoTensors value type."""

  def __init__(self, x_spec, y_spec, color="red"):
    self.x_spec = x_spec
    self.y_spec = y_spec
    self.color = color

  value_type = property(lambda self: TwoComposites)

  @property
  def _component_specs(self):
    return (self.x_spec, self.y_spec)

  def _to_components(self, value):
    return (value.x, value.y)

  def _from_components(self, components):
    x, y = components
    return TwoComposites(x, y, self.color)

  def _serialize(self):
    return (self.x_spec, self.y_spec, self.color)

  @classmethod
  def from_value(cls, value):
    return cls(type_spec.type_spec_from_value(value.x),
               type_spec.type_spec_from_value(value.y),
               value.color)

  def _batch(self, batch_size):
    return TwoCompositesSpec(
        self.x_spec._batch(batch_size), self.y_spec._batch(batch_size),
        self.color)

  def _unbatch(self):
    return TwoCompositesSpec(self.x_spec._unbatch(), self.y_spec._unbatch(),
                             self.color)


type_spec.register_type_spec_from_value_converter(
    TwoComposites, TwoCompositesSpec.from_value)


class NestOfTensors(object):
  """CompositeTensor containing a nest of tensors."""

  def __init__(self, x):
    self.nest = x


@type_spec.register("tf.NestOfTensorsSpec")
class NestOfTensorsSpec(type_spec.TypeSpec):
  """A TypeSpec for the NestOfTensors value type."""

  def __init__(self, spec):
    self.spec = spec

  value_type = property(lambda self: NestOfTensors)
  _component_specs = property(lambda self: self.spec)

  def _to_components(self, value):
    return nest.flatten(value)

  def _from_components(self, components):
    return nest.pack_sequence_as(self.spec, components)

  def _serialize(self):
    return self.spec

  def __repr__(self):
    if hasattr(self.spec, "_fields") and isinstance(
        self.spec._fields, collections_abc.Sequence) and all(
            isinstance(f, six.string_types) for f in self.spec._fields):
      return "%s(%r)" % (type(self).__name__, self._serialize())
    return super(type_spec.TypeSpec, self).__repr__()  # pylint: disable=bad-super-call

  @classmethod
  def from_value(cls, value):
    return cls(nest.map_structure(type_spec.type_spec_from_value, value.nest))

  @classmethod
  def _deserialize(cls, spec):
    return cls(spec)


type_spec.register_type_spec_from_value_converter(
    NestOfTensors, NestOfTensorsSpec.from_value)

_TestNamedTuple = collections.namedtuple("NamedTuple", ["a", "b"])
_TestNamedTuple2 = collections.namedtuple("NamedTuple", ["a", "b"])
_TestNamedTupleSingleField = collections.namedtuple("SingleField", ["a"])
_TestNamedTupleDifferentField = collections.namedtuple("DifferentField",
                                                       ["a", "c"])


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
                      (np.int32(1), np.float32(1.),
                       np.array([[1, 2], [3, 4]]))),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      (np.int32(1), np.float32(1.),
                       np.array([[1, 2], [3, 4]])))),
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
      ("SameValue", TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("UnknownDim", TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank", TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("NamedTuple",
       NestOfTensorsSpec(
           _TestNamedTuple(
               a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
               b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
       NestOfTensorsSpec(
           _TestNamedTuple(
               a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
               b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
      (
          "NamedTupleRedefined",
          NestOfTensorsSpec(
              _TestNamedTuple2(  # Separate but equivalent type.
                  a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
  )
  def testIsSubtypeOf(self, v1, v2):
    self.assertTrue(v1.is_subtype_of(v2))

  @parameterized.named_parameters(
      ("DifferentType",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpecTwin([5, 3], dtypes.int32, [None], dtypes.bool),
      ),
      ("DifferentDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("DifferentRank", TwoTensorsSpec([5, 3], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("DifferentDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("DifferentMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues", TwoTensorsSpec([5, 3], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      ("SwappedDimensions",
       TwoTensorsSpec([3, 5], dtypes.int32, [None], dtypes.int32),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.int32)),
      ("Supertype", TwoTensorsSpec([5, None], dtypes.int32, [None],
                                   dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("SerializeDifferentStructure",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "smaller_tuple")),
      ("SerializeDifferentOrder",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "different_order")),
  )
  def testIsNotSubtypeOf(self, v1, v2):
    self.assertFalse(v1.is_subtype_of(v2))

  @parameterized.named_parameters(
      ("SameValue", TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("DifferentValue",
       TwoTensorsSpec([2, 1], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None, None], dtypes.int32, [None], dtypes.bool)),
      ("DifferentRank",
       TwoTensorsSpec([3, 2, 1], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, [None], dtypes.bool)),
      (
          "NamedTupleRedefined",
          NestOfTensorsSpec(
              _TestNamedTuple2(  # Separate but equivalent type.
                  a=tensor_spec.TensorSpec([8, 3], dtypes.int32),
                  b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([7, None], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, None], dtypes.int32),
                  b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
  )
  def testMostSpecificCommonSupertype(self, v1, v2, result):
    self.assertEqual(v1.most_specific_common_supertype([v2]), result)
    self.assertEqual(v2.most_specific_common_supertype([v1]), result)

  @parameterized.named_parameters(
      ("DifferentType",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpecTwin([5, 3], dtypes.int32, [None], dtypes.bool),
      ),
      ("DifferentDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("DifferentMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SerializeDifferentStructure",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "smaller_tuple")),
      ("SerializeDifferentOrder",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "different_order")),
  )
  def testNoCommonSupertype(self, v1, v2):
    self.assertIsNone(v1.most_specific_common_supertype([v2]))
    self.assertIsNone(v2.most_specific_common_supertype([v1]))

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
      ("NamedTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([None, None], dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([8, 12], dtypes.int32)))),
      ("NamedTupleRedefined",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([None, None], dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple2(  # Separate but equivalent type.
           a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([8, 12], dtypes.int32)))),
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
      ("NamedTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("NamedTupleRedefined",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple2(  # Separate but equivalent type.
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
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
      ("IncompatibleTensorSpecName",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="a")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="b"))),
      ("IncompatibleNestType",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(dict(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      )
  def testMostSpecificCompatibleTypeException(self, v1, v2):
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  def testMostSpecificCompatibleTypeNamedTupleIsNotTuple(self):
    named_tuple_spec_a = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=1, b="aaa")))
    named_tuple_spec_b = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=2, b="bbb")))
    named_tuple_spec_c = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=3, b="ccc")))
    normal_tuple_spec = NestOfTensorsSpec.from_value(NestOfTensors((2, "bbb")))
    result_a_b = named_tuple_spec_a.most_specific_compatible_type(
        named_tuple_spec_b)
    result_b_a = named_tuple_spec_b.most_specific_compatible_type(
        named_tuple_spec_a)
    self.assertEqual(repr(result_a_b), repr(named_tuple_spec_c))
    self.assertEqual(repr(result_b_a), repr(named_tuple_spec_c))
    # Test that spec of named tuple is not equal to spec of normal tuple.
    self.assertNotEqual(repr(result_a_b), repr(normal_tuple_spec))

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.bool))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("DifferentTupleSize",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.bool))),
       NestOfTensorsSpec(_TestNamedTupleSingleField(
           a=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("DifferentFieldName",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTupleDifferentField(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           c=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("NamedTupleAndTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec((
           tensor_spec.TensorSpec((), dtypes.int32),
           tensor_spec.TensorSpec((), dtypes.int32)))),
      )
  def testMostSpecificCompatibleTypeForNamedTuplesException(self, v1, v2):
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

  def testFullTypesForFlatTensors(self):
    spec = TwoTensorsSpec([5], dtypes.int32, [5, 8], dtypes.float32, "red")
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET),
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)
    ]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

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

  def testNestedRagged(self):
    # Check that TwoCompositeSpecs are compatible if one has a nested
    # RaggedTensorSpec w/ ragged_rank=0 and the other has a corresponding
    # nested TensorSpec.
    spec1 = TwoCompositesSpec(
        ragged_tensor.RaggedTensorSpec([10], dtypes.int32, ragged_rank=0),
        tensor_spec.TensorSpec(None, dtypes.int32))
    spec2 = TwoCompositesSpec(
        tensor_spec.TensorSpec([10], dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32))
    spec3 = TwoCompositesSpec(
        tensor_spec.TensorSpec([12], dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32))
    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))

  def testRegistry(self):
    self.assertEqual("tf.TwoCompositesSpec",
                     type_spec.get_name(TwoCompositesSpec))
    self.assertEqual("tf.TwoTensorsSpec", type_spec.get_name(TwoTensorsSpec))
    self.assertEqual(TwoCompositesSpec,
                     type_spec.lookup("tf.TwoCompositesSpec"))
    self.assertEqual(TwoTensorsSpec, type_spec.lookup("tf.TwoTensorsSpec"))

  def testRegistryTypeErrors(self):
    with self.assertRaisesRegex(TypeError, "Expected `name` to be a string"):
      type_spec.register(None)

    with self.assertRaisesRegex(TypeError, "Expected `name` to be a string"):
      type_spec.register(TwoTensorsSpec)

    with self.assertRaisesRegex(TypeError, "Expected `cls` to be a TypeSpec"):
      type_spec.register("tf.foo")(None)

    with self.assertRaisesRegex(TypeError, "Expected `cls` to be a TypeSpec"):
      type_spec.register("tf.foo")(ragged_tensor.RaggedTensor)

  def testRegistryDuplicateErrors(self):
    with self.assertRaisesRegex(
        ValueError, "Name tf.TwoCompositesSpec has already been registered "
        "for class __main__.TwoCompositesSpec."):

      @type_spec.register("tf.TwoCompositesSpec")  # pylint: disable=unused-variable
      class NewTypeSpec(TwoCompositesSpec):
        pass

    with self.assertRaisesRegex(
        ValueError, "Class __main__.TwoCompositesSpec has already been "
        "registered with name tf.TwoCompositesSpec"):
      type_spec.register("tf.NewName")(TwoCompositesSpec)

  def testRegistryNameErrors(self):
    for bad_name in ["foo", "", "hello world"]:
      with self.assertRaises(ValueError):
        type_spec.register(bad_name)

  def testRegistryLookupErrors(self):
    with self.assertRaises(TypeError):
      type_spec.lookup(None)
    with self.assertRaisesRegex(
        ValueError, "No TypeSpec has been registered with name 'foo.bar'"):
      type_spec.lookup("foo.bar")

  def testRegistryGetNameErrors(self):
    with self.assertRaises(TypeError):
      type_spec.get_name(None)

    class Foo(TwoCompositesSpec):
      pass

    with self.assertRaisesRegex(
        ValueError, "TypeSpec __main__.Foo has not been registered."):
      type_spec.get_name(Foo)


class BatchableTypeSpecTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  @parameterized.parameters([
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([]), tensor_spec.TensorSpec([8])),
          "batch_size":
              5,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([5]), tensor_spec.TensorSpec([5, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec(None), tensor_spec.TensorSpec([8])),
          "batch_size":
              None,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec(None),
                  tensor_spec.TensorSpec([None, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([3, None]),
                  tensor_spec.TensorSpec([8])),
          "batch_size":
              12,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([12, 3, None]),
                  tensor_spec.TensorSpec([12, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  ragged_tensor.RaggedTensorSpec([3, None]),
                  tensor_spec.TensorSpec([8])),
          "batch_size":
              12,
          "batched":
              TwoCompositesSpec(
                  ragged_tensor.RaggedTensorSpec([12, 3, None]),
                  tensor_spec.TensorSpec([12, 8]))
      },
  ])
  def testBatch(self, unbatched, batch_size, batched):
    self.assertEqual(unbatched._batch(batch_size), batched)
    self.assertEqual(batched._unbatch(), unbatched)

  def testFlatTensorSpecs(self):
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    flat_specs = a_spec._flat_tensor_specs
    self.assertEqual(flat_specs, [
        tensor_spec.TensorSpec(None, dtypes.variant),
        tensor_spec.TensorSpec(None, dtypes.variant)
    ])

  def testFullTypesForFlatTensors(self):
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    full_type_list = fulltypes_for_flat_tensors(a_spec)
    expect = [
        full_type_pb2.FullTypeDef(
            type_id=full_type_pb2.TFT_RAGGED,
            args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_INT32)]),
        full_type_pb2.FullTypeDef(
            type_id=full_type_pb2.TFT_RAGGED,
            args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_INT32)]),
    ]
    self.assertEqual(len(a_spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  def testToTensorList(self):
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    tensor_list = a_spec._to_tensor_list(a)
    self.assertLen(tensor_list, 2)
    self.assertEqual(tensor_list[0].dtype, dtypes.variant)
    self.assertEqual(tensor_list[1].dtype, dtypes.variant)
    self.assertEqual(tensor_list[0].shape.rank, 0)
    self.assertEqual(tensor_list[1].shape.rank, 0)

    b = a_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(a.x, b.x)
    self.assertAllEqual(a.y, b.y)
    self.assertEqual(a.color, b.color)

    c = a_spec._from_compatible_tensor_list(tensor_list)
    self.assertAllEqual(a.x, c.x)
    self.assertAllEqual(a.y, c.y)
    self.assertEqual(a.color, c.color)

  def testToBatchedTensorList(self):
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    tensor_list = a_spec._to_batched_tensor_list(a)
    self.assertLen(tensor_list, 2)
    self.assertEqual(tensor_list[0].dtype, dtypes.variant)
    self.assertEqual(tensor_list[1].dtype, dtypes.variant)
    self.assertEqual(tensor_list[0].shape.rank, 1)
    self.assertEqual(tensor_list[1].shape.rank, 1)

    b = a_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(a.x, b.x)
    self.assertAllEqual(a.y, b.y)
    self.assertEqual(a.color, b.color)

    c = a_spec._from_compatible_tensor_list(tensor_list)
    self.assertAllEqual(a.x, c.x)
    self.assertAllEqual(a.y, c.y)
    self.assertEqual(a.color, c.color)


if __name__ == "__main__":
  googletest.main()

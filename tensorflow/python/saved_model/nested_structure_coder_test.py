# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for nested structure coding."""

import collections
import typing
import warnings

from google.protobuf import text_format
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal


class NestedStructureCoderTest(test.TestCase):

  def testEncodeDecodeList(self):
    structure = [1.5, 2.5, 3.0]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().float64_value = 1.5
    expected.list_value.values.add().float64_value = 2.5
    expected.list_value.values.add().float64_value = 3.0
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTuple(self):
    structure = ("hello", [3, (2, 1)])
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.tuple_value.values.add().string_value = "hello"
    list_value = expected.tuple_value.values.add().list_value
    list_value.values.add().int64_value = 3
    tuple_value = list_value.values.add().tuple_value
    tuple_value.values.add().int64_value = 2
    tuple_value.values.add().int64_value = 1
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeDict(self):
    structure = dict(a=3, b=[7, 2.5])
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.dict_value.fields["a"].int64_value = 3
    list_value = expected.dict_value.fields["b"].list_value
    list_value.values.add().int64_value = 7
    list_value.values.add().float64_value = 2.5
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertIsInstance(decoded["a"], int)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorShape(self):
    structure = [tensor_shape.TensorShape([1, 2, 3]), "hello"]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_shape = expected_list.values.add().tensor_shape_value
    expected_tensor_shape.dim.add().size = 1
    expected_tensor_shape.dim.add().size = 2
    expected_tensor_shape.dim.add().size = 3
    expected_tensor_shape = expected_list.values.add().string_value = "hello"
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeNamedTuple(self):
    named_tuple_type = collections.namedtuple("NamedTuple", ["x", "y"])
    named_tuple = named_tuple_type(x=[1, 2], y="hello")
    self.assertTrue(nested_structure_coder.can_encode(named_tuple))
    encoded = nested_structure_coder.encode_structure(named_tuple)
    expected = struct_pb2.StructuredValue()
    expected_named_tuple = expected.named_tuple_value
    expected_named_tuple.name = "NamedTuple"
    key_value_pair = expected_named_tuple.values.add()
    key_value_pair.key = "x"
    list_value = key_value_pair.value.list_value
    list_value.values.add().int64_value = 1
    list_value.values.add().int64_value = 2
    key_value_pair = expected_named_tuple.values.add()
    key_value_pair.key = "y"
    key_value_pair.value.string_value = "hello"
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(named_tuple._asdict(), decoded._asdict())
    self.assertEqual(named_tuple.__class__.__name__, decoded.__class__.__name__)

  def testNone(self):
    structure = [1.0, None]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().float64_value = 1.0
    expected.list_value.values.add().none_value.CopyFrom(struct_pb2.NoneValue())
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testBool(self):
    structure = [False]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().bool_value = False
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEmptyStructures(self):
    structure = [list(), dict(), tuple()]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().list_value.CopyFrom(struct_pb2.ListValue())
    expected.list_value.values.add().dict_value.CopyFrom(struct_pb2.DictValue())
    expected.list_value.values.add().tuple_value.CopyFrom(
        struct_pb2.TupleValue())
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testDtype(self):
    structure = [dtypes.int64]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    list_value = expected.list_value.values.add()
    list_value.tensor_dtype_value = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorSpec(self):
    structure = [tensor_spec.TensorSpec([1, 2, 3], dtypes.int64, "hello")]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = "hello"
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorSpecWithNoName(self):
    structure = [tensor_spec.TensorSpec([1, 2, 3], dtypes.int64)]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = ""
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeRaggedTensorSpec(self):
    structure = [
        ragged_tensor.RaggedTensorSpec([1, 2, 3], dtypes.int64, 2, dtypes.int32)
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      list_value {
        values {
          type_spec_value {
            type_spec_class: RAGGED_TENSOR_SPEC
            type_spec_class_name: 'RaggedTensorSpec'
            num_flat_components: 3
            type_state {
              tuple_value {
                # spec._shape
                values {
                  tensor_shape_value {
                    dim { size: 1 }
                    dim { size: 2 }
                    dim { size: 3 }
                  }
                }
                # spec._dtype
                values { tensor_dtype_value: DT_INT64 }
                # spec._ragged_rank
                values { int64_value: 2 }
                # spec._row_splits_dtype
                values { tensor_dtype_value: DT_INT32 }
              }
            }
          }
        }
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeSparseTensorSpec(self):
    structure = [sparse_tensor.SparseTensorSpec([10, 20], dtypes.float32)]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      list_value {
        values {
          type_spec_value {
            type_spec_class: SPARSE_TENSOR_SPEC
            type_spec_class_name: 'SparseTensorSpec'
            num_flat_components: 3
            type_state {
              tuple_value {
                # spec._shape
                values {
                  tensor_shape_value {
                    dim { size: 10 }
                    dim { size: 20 }
                  }
                }
                # spec._dtype
                values { tensor_dtype_value: DT_FLOAT }
              }
            }
          }
        }
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeExtensionTypeSpec(self):

    class Zoo(extension_type.ExtensionType):
      __name__ = "tf.nested_structure_coder_test.Zoo"
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, ops.Tensor]

    structure = [
        Zoo.Spec(
            zookeepers=["Zoey", "Zack"],
            animals={"tiger": tensor_spec.TensorSpec([16])})
    ]

    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
    list_value {
      values {
        type_spec_value {
          type_spec_class: EXTENSION_TYPE_SPEC
          type_spec_class_name: "tf.nested_structure_coder_test.Zoo.Spec"
          num_flat_components: 1
          type_state {
            tuple_value {
              values {
                tuple_value {
                  values { string_value: "zookeepers" }
                  values { tuple_value {
                    values { string_value: "Zoey" }
                    values { string_value: "Zack" } } } } }
              values {
                tuple_value {
                  values { string_value: "animals" }
                  values { dict_value {
                    fields {
                      key: "tiger"
                      value { tensor_spec_value {
                        shape { dim { size: 16 } }
                        dtype: DT_FLOAT } } } } } } } } } } } }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testDecodeUnknownTensorSpec(self):
    encoded = struct_pb2.StructuredValue()
    encoded.type_spec_value.type_spec_class = 0
    encoded.type_spec_value.type_spec_class_name = "FutureTensorSpec"
    with self.assertRaisesRegex(ValueError,
                                "The type 'FutureTensorSpec' is not supported"):
      nested_structure_coder.decode_proto(encoded)

  def testEncodeDecodeBoundedTensorSpec(self):
    structure = [
        tensor_spec.BoundedTensorSpec([1, 2, 3], dtypes.int64, 0, 10,
                                      "hello_0_10")
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().bounded_tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = "hello_0_10"
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    expected_tensor_spec.minimum.CopyFrom(
        tensor_util.make_tensor_proto([0], dtype=dtypes.int64, shape=[]))
    expected_tensor_spec.maximum.CopyFrom(
        tensor_util.make_tensor_proto([10], dtype=dtypes.int64, shape=[]))
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeBoundedTensorSpecNoName(self):
    structure = [
        tensor_spec.BoundedTensorSpec((28, 28, 3), dtypes.float64, -2,
                                      (1, 1, 20))
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().bounded_tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 28
    expected_tensor_spec.shape.dim.add().size = 28
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = ""
    expected_tensor_spec.dtype = dtypes.float64.as_datatype_enum
    expected_tensor_spec.minimum.CopyFrom(
        tensor_util.make_tensor_proto([-2], dtype=dtypes.float64, shape=[]))
    expected_tensor_spec.maximum.CopyFrom(
        tensor_util.make_tensor_proto([1, 1, 20],
                                      dtype=dtypes.float64,
                                      shape=[3]))
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDataSetSpec(self):
    structure = [
        dataset_ops.DatasetSpec({
            "rt": ragged_tensor.RaggedTensorSpec([10, None], dtypes.int32),
            "st": sparse_tensor.SparseTensorSpec([10, 20], dtypes.float32),
            "t": tensor_spec.TensorSpec([10, 8], dtypes.string)
        })
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  @test_util.run_in_graph_and_eager_modes
  def testEncodeDecodeTensor(self):
    structure = constant_op.constant(1)
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      tensor_value {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertAllEqual(structure, decoded)

  def testNotEncodable(self):

    class NotEncodable(object):
      pass

    self.assertFalse(nested_structure_coder.can_encode([NotEncodable()]))

  def testRegisterCustomCodec(self):

    class MyObject(object):
      pass

    class MyObjectCodec(object):
      """Codec for MyObject."""

      def can_encode(self, pyobj):
        return isinstance(pyobj, MyObject)

      def do_encode(self, array, encode_fn):
        del array, encode_fn
        return struct_pb2.StructuredValue()

      def can_decode(self, value):
        del value
        return False

      def do_decode(self, value, decode_fn):
        raise NotImplementedError("Test only.")

    nested_structure_coder.register_codec(MyObjectCodec())
    my_object = MyObject()
    self.assertTrue(nested_structure_coder.can_encode(my_object))

  def testRegisteredTypeSpec(self):
    expected_warning = ("Encoding a StructuredValue with type "
                        "NestedStructureTest.RegisteredTypeSpec; loading "
                        "this StructuredValue will require that this type "
                        "be imported and registered")
    structure = {"x": RegisteredTypeSpec()}

    self.assertTrue(nested_structure_coder.can_encode(structure))
    with warnings.catch_warnings(record=True) as w:
      encoded = nested_structure_coder.encode_structure(structure)
      self.assertLen(w, 1)
      self.assertIn(expected_warning, str(w[0].message))
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testUnregisteredTypeSpec(self):
    structure = {"x": UnregisteredTypeSpec()}
    self.assertFalse(nested_structure_coder.can_encode(structure))
    with self.assertRaises(nested_structure_coder.NotEncodableError):
      nested_structure_coder.encode_structure(structure)

  def testBuiltInTypeSpecCodecInvalidInputs(self):
    class Foo:
      pass

    class Bar(internal.TypeSpec):
      pass

    with self.assertRaisesRegex(
        ValueError, "The type '(.*?)' does not subclass tf.TypeSpec."):
      nested_structure_coder.BuiltInTypeSpecCodec(Foo, 0)
    with self.assertRaisesRegex(
        ValueError, "The type '(.*?)' already has an instantiated codec."):
      nested_structure_coder.BuiltInTypeSpecCodec(
          dataset_ops.DatasetSpec, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is already registered."):
      nested_structure_coder.BuiltInTypeSpecCodec(
          Bar, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 0)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 11)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 12)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 13)


# Trivial TypeSpec class for testing.
class UnregisteredTypeSpec(type_spec.TypeSpec):
  value_type = property(lambda self: None)
  _component_specs = property(lambda self: ())
  _to_components = lambda self, v: ()
  _from_components = classmethod(lambda cls, c: cls())
  _serialize = lambda self: ()


# Trivial TypeSpec class for testing.
@type_spec_registry.register("NestedStructureTest.RegisteredTypeSpec")
class RegisteredTypeSpec(type_spec.TypeSpec):
  value_type = property(lambda self: None)
  _component_specs = property(lambda self: ())
  _to_components = lambda self, v: ()
  _from_components = classmethod(lambda cls, c: cls())
  _serialize = lambda self: ()


if __name__ == "__main__":
  test.main()

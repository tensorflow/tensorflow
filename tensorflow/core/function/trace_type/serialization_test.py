# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for serialization."""

from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import serialization_test_pb2
from tensorflow.python.platform import test


class MyCustomClass(serialization.Serializable):

  def __init__(self, index, name):
    self.index = index
    self.name = name

  @classmethod
  def experimental_type_proto(cls):
    return serialization_test_pb2.MyCustomRepresentation

  @classmethod
  def experimental_from_proto(cls, proto):
    return MyCustomClass(proto.index, proto.name)

  def experimental_as_proto(self):
    proto = serialization_test_pb2.MyCustomRepresentation(
        index=self.index, name=self.name)
    return proto


serialization.register_serializable(MyCustomClass)


class MyCompositeClass(serialization.Serializable):

  def __init__(self, *elements):
    self.elements = elements

  @classmethod
  def experimental_type_proto(cls):
    return serialization_test_pb2.MyCompositeRepresentation

  @classmethod
  def experimental_from_proto(cls, proto):
    return MyCompositeClass(
        *[serialization.deserialize(element) for element in proto.elements])

  def experimental_as_proto(self):
    serialized_elements = [
        serialization.serialize(element) for element in self.elements
    ]
    proto = serialization_test_pb2.MyCompositeRepresentation(
        elements=serialized_elements)
    return proto


serialization.register_serializable(MyCompositeClass)


class SerializeTest(test.TestCase):

  def testCustomClassSerialization(self):
    my_custom = MyCustomClass(1234, "my_name")
    serialized = serialization.serialize(my_custom)

    self.assertTrue(
        serialized.representation.Is(
            serialization_test_pb2.MyCustomRepresentation.DESCRIPTOR))

    proto = serialization_test_pb2.MyCustomRepresentation()
    serialized.representation.Unpack(proto)
    self.assertEqual(proto.index, my_custom.index)
    self.assertEqual(proto.name, my_custom.name)

  def testCustomClassDeserialization(self):
    original = MyCustomClass(1234, "my_name")
    serialized = serialization.serialize(original)
    deserialized = serialization.deserialize(serialized)

    self.assertIsInstance(deserialized, MyCustomClass)
    self.assertEqual(deserialized.index, original.index)
    self.assertEqual(deserialized.name, original.name)

  def testCompositeClassSerialization(self):
    my_composite = MyCompositeClass(
        MyCustomClass(1, "name_1"), MyCustomClass(2, "name_2"),
        MyCustomClass(3, "name_3"))
    serialized = serialization.serialize(my_composite)

    self.assertTrue(
        serialized.representation.Is(
            serialization_test_pb2.MyCompositeRepresentation.DESCRIPTOR))

    proto = serialization_test_pb2.MyCompositeRepresentation()
    serialized.representation.Unpack(proto)

    self.assertEqual(proto.elements[0],
                     serialization.serialize(MyCustomClass(1, "name_1")))
    self.assertEqual(proto.elements[1],
                     serialization.serialize(MyCustomClass(2, "name_2")))
    self.assertEqual(proto.elements[2],
                     serialization.serialize(MyCustomClass(3, "name_3")))

  def testCompositeClassDeserialization(self):
    original = MyCompositeClass(
        MyCustomClass(1, "name_1"), MyCustomClass(2, "name_2"),
        MyCustomClass(3, "name_3"))
    serialized = serialization.serialize(original)
    deserialized = serialization.deserialize(serialized)

    self.assertIsInstance(deserialized, MyCompositeClass)

    self.assertEqual(deserialized.elements[0].index, 1)
    self.assertEqual(deserialized.elements[1].index, 2)
    self.assertEqual(deserialized.elements[2].index, 3)

    self.assertEqual(deserialized.elements[0].name, "name_1")
    self.assertEqual(deserialized.elements[1].name, "name_2")
    self.assertEqual(deserialized.elements[2].name, "name_3")

  def testNonUniqueProto(self):
    class ClassThatReusesProto(serialization.Serializable):

      @classmethod
      def experimental_type_proto(cls):
        return serialization_test_pb2.MyCustomRepresentation

      @classmethod
      def experimental_from_proto(cls, proto):
        raise NotImplementedError

      def experimental_as_proto(self):
        raise NotImplementedError

    with self.assertRaisesRegex(
        ValueError,
        ("Existing Python class MyCustomClass already has "
         "MyCustomRepresentation as its associated proto representation. "
         "Please ensure ClassThatReusesProto has a unique proto representation."
        )):
      serialization.register_serializable(ClassThatReusesProto)

  def testWrongProto(self):

    class ClassReturningWrongProto(serialization.Serializable):

      @classmethod
      def experimental_type_proto(cls):
        return serialization.SerializedTraceType

      @classmethod
      def experimental_from_proto(cls, proto):
        raise NotImplementedError

      def experimental_as_proto(self):
        return serialization_test_pb2.MyCustomRepresentation()

    with self.assertRaisesRegex(
        ValueError,
        ("ClassReturningWrongProto returned different type of proto than "
         "specified by experimental_type_proto()")):
      serialization.serialize(ClassReturningWrongProto())


if __name__ == "__main__":
  test.main()

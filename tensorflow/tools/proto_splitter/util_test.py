# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests util functions."""

from typing import Iterable

from tensorflow.python.platform import test
from tensorflow.tools.proto_splitter import util
from tensorflow.tools.proto_splitter.testdata import test_message_pb2


class UtilTest(test.TestCase):

  def test_format_bytes(self):
    self.assertEqual(util.format_bytes(1024), "1KiB")
    self.assertEqual(util.format_bytes(5632), "5.5KiB")
    self.assertEqual(util.format_bytes(53432), "52.18KiB")

    self.assertEqual(util.format_bytes(76493281), "72.95MiB")
    self.assertEqual(util.format_bytes(5.977e7), "57MiB")
    self.assertEqual(util.format_bytes(1.074e9), "1GiB")
    self.assertEqual(util.format_bytes(16493342281), "15.36GiB")

  def test_get_field_tag(self):
    proto = test_message_pb2.ManyFields()

    # proto.field_one.repeated_field[15].map_field_uint32[10]
    ret = util.get_field_tag(
        proto, ["field_one", "repeated_field", 15, "map_field_uint32", 10]
    )
    self.assertLen(ret, 5)
    self.assertEqual(1, ret[0].field)
    self.assertEqual(2, ret[1].field)
    self.assertEqual(15, ret[2].index)
    self.assertEqual(5, ret[3].field)
    self.assertEqual(10, ret[4].map_key.ui32)
    self.assertFalse(ret[4].map_key.HasField("i32"))

    # proto.nested_map_bool[False].map_field_int64
    ret = util.get_field_tag(
        proto, ["nested_map_bool", False, "map_field_int64"]
    )

    self.assertLen(ret, 3)
    self.assertEqual(7, ret[0].field)
    self.assertEqual(False, ret[1].map_key.boolean)
    self.assertEqual(6, ret[2].field)

    # proto.repeated_field[55].nested_map_bool[True].string_field
    ret = util.get_field_tag(proto, [2, 55, 7, True, 3])
    self.assertLen(ret, 5)
    self.assertEqual(2, ret[0].field)
    self.assertEqual(55, ret[1].index)
    self.assertEqual(7, ret[2].field)
    self.assertEqual(True, ret[3].map_key.boolean)
    self.assertEqual(3, ret[4].field)

  def test_get_field_tag_invalid(self):
    proto = test_message_pb2.ManyFields()
    with self.assertRaisesRegex(KeyError, "Unable to find field 'not_a_field'"):
      util.get_field_tag(proto, ["field_one", "not_a_field"])
    with self.assertRaisesRegex(KeyError, "Unable to find field number 10000"):
      util.get_field_tag(proto, [1, 10000])
    with self.assertRaisesRegex(ValueError, "Unable to find fields.*in proto"):
      util.get_field_tag(proto, ["string_field", 1])

  def test_get_field_and_desc(self):
    proto = test_message_pb2.ManyFields(
        field_one=test_message_pb2.ManyFields(
            repeated_field=[
                test_message_pb2.ManyFields(),
                test_message_pb2.ManyFields(
                    string_field="inner_inner_string",
                    map_field_uint32={
                        324: "map_value_324",
                        543: "map_value_543",
                    },
                ),
            ]
        ),
        map_field_int64={
            -1345: "map_value_-1345",
        },
        nested_map_bool={
            True: test_message_pb2.ManyFields(string_field="string_true"),
            False: test_message_pb2.ManyFields(string_field="string_false"),
        },
    )

    field, field_desc = util.get_field(proto, [])
    self.assertIs(proto, field)
    self.assertIsNone(field_desc)

    field, field_desc = util.get_field(proto, ["field_one", "repeated_field"])
    self.assertIsInstance(field, Iterable)
    self.assertLen(field, 2)
    self.assertEqual("repeated_field", field_desc.name)
    self.assertEqual(2, field_desc.number)
    self.assertProtoEquals(proto.field_one.repeated_field, field)

    field, field_desc = util.get_field(proto, ["field_one", 2, 1])
    self.assertIsInstance(field, test_message_pb2.ManyFields)
    self.assertEqual("repeated_field", field_desc.name)
    self.assertEqual(2, field_desc.number)
    self.assertProtoEquals(proto.field_one.repeated_field[1], field)

    field, _ = util.get_field(proto, ["field_one", 2, 1, "string_field"])
    self.assertEqual("inner_inner_string", field)

    field, _ = util.get_field(
        proto, ["field_one", 2, 1, "map_field_uint32", 324]
    )
    self.assertEqual("map_value_324", field)

    field, _ = util.get_field(proto, ["nested_map_bool", False, "string_field"])
    self.assertEqual("string_false", field)

    field, _ = util.get_field(proto, ["nested_map_bool", True, "string_field"])
    self.assertEqual("string_true", field)


if __name__ == "__main__":
  test.main()

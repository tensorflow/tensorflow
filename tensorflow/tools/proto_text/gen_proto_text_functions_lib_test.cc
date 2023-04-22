/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/tools/proto_text/gen_proto_text_functions_lib.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/proto_text/test.pb_text.h"
#include "tensorflow/tools/proto_text/test.pb.h"

namespace tensorflow {
namespace test {
namespace {

// Convert <input> to text depending on <short_debug>, then parse that into a
// new message using the generated parse function. Return the new message.
template <typename T>
T RoundtripParseProtoOrDie(const T& input, bool short_debug) {
  const string s = short_debug ? input.ShortDebugString() : input.DebugString();
  T t;
  EXPECT_TRUE(ProtoParseFromString(s, &t)) << "Failed to parse " << s;
  return t;
}

// Macro that takes <proto> and verifies the proto text string output
// matches DebugString calls on the proto, and verifies parsing the
// DebugString output works. It does this for regular and short
// debug strings.
#define EXPECT_TEXT_TRANSFORMS_MATCH()                               \
  EXPECT_EQ(proto.DebugString(), ProtoDebugString(proto));           \
  EXPECT_EQ(proto.ShortDebugString(), ProtoShortDebugString(proto)); \
  EXPECT_EQ(proto.DebugString(),                                     \
            RoundtripParseProtoOrDie(proto, true).DebugString());    \
  EXPECT_EQ(proto.DebugString(),                                     \
            RoundtripParseProtoOrDie(proto, false).DebugString());

// Macro for failure cases. Verifies both protobuf and proto_text to
// make sure they match.
#define EXPECT_PARSE_FAILURE(str)                  \
  EXPECT_FALSE(ProtoParseFromString(str, &proto)); \
  EXPECT_FALSE(protobuf::TextFormat::ParseFromString(str, &proto))

// Macro for success cases parsing from a string. Verifies protobuf and
// proto_text cases match.
#define EXPECT_PARSE_SUCCESS(expected, str)                          \
  do {                                                               \
    EXPECT_TRUE(ProtoParseFromString(str, &proto));                  \
    string proto_text_str = ProtoShortDebugString(proto);            \
    EXPECT_TRUE(protobuf::TextFormat::ParseFromString(str, &proto)); \
    string protobuf_str = ProtoShortDebugString(proto);              \
    EXPECT_EQ(proto_text_str, protobuf_str);                         \
    EXPECT_EQ(expected, proto_text_str);                             \
  } while (false)

// Test different cases of numeric values, including repeated values.
TEST(CreateProtoDebugStringLibTest, ValidSimpleTypes) {
  TestAllTypes proto;

  // Note that this also tests that output of fields matches tag number order,
  // since some of these fields have high tag numbers.
  proto.Clear();
  proto.set_optional_int32(-1);
  proto.set_optional_int64(-2);
  proto.set_optional_uint32(3);
  proto.set_optional_uint64(4);
  proto.set_optional_sint32(-5);
  proto.set_optional_sint64(-6);
  proto.set_optional_fixed32(-7);
  proto.set_optional_fixed64(-8);
  proto.set_optional_sfixed32(-9);
  proto.set_optional_sfixed64(-10);
  proto.set_optional_float(-12.34);
  proto.set_optional_double(-5.678);
  proto.set_optional_bool(true);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Max numeric values.
  proto.Clear();
  proto.set_optional_int32(std::numeric_limits<int32>::max());
  proto.set_optional_int64(std::numeric_limits<protobuf_int64>::max());
  proto.set_optional_uint32(std::numeric_limits<uint32>::max());
  proto.set_optional_uint64(std::numeric_limits<uint64>::max());
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::max());
  proto.set_optional_double(std::numeric_limits<double>::max());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Least positive numeric values.
  proto.Clear();
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::min());
  proto.set_optional_double(std::numeric_limits<double>::min());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Lowest numeric values.
  proto.Clear();
  proto.set_optional_int32(std::numeric_limits<int32>::lowest());
  proto.set_optional_int64(std::numeric_limits<protobuf_int64>::lowest());
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::lowest());
  proto.set_optional_double(std::numeric_limits<double>::lowest());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // inf and -inf for float and double.
  proto.Clear();
  proto.set_optional_double(std::numeric_limits<double>::infinity());
  proto.set_optional_float(std::numeric_limits<float>::infinity());
  EXPECT_TEXT_TRANSFORMS_MATCH();
  proto.set_optional_double(-1 * std::numeric_limits<double>::infinity());
  proto.set_optional_float(-1 * std::numeric_limits<float>::infinity());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // String and bytes values.
  proto.Clear();
  for (int i = 0; i < 256; ++i) {
    proto.mutable_optional_string()->push_back(static_cast<char>(i));
    proto.mutable_optional_bytes()->push_back(static_cast<char>(i));
  }
  strings::StrAppend(proto.mutable_optional_string(), "¢€𐍈");
  proto.set_optional_cord(proto.optional_string());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Repeated values. Include zero values to show they are retained in
  // repeateds.
  proto.Clear();
  proto.add_repeated_int32(-1);
  proto.add_repeated_int32(0);
  proto.add_repeated_int64(0);
  proto.add_repeated_int64(1);
  proto.add_repeated_uint32(-10);
  proto.add_repeated_uint32(0);
  proto.add_repeated_uint32(10);
  proto.add_repeated_uint64(-20);
  proto.add_repeated_uint64(0);
  proto.add_repeated_uint64(20);
  proto.add_repeated_sint32(-30);
  proto.add_repeated_sint32(0);
  proto.add_repeated_sint32(30);
  proto.add_repeated_sint64(-40);
  proto.add_repeated_sint64(0);
  proto.add_repeated_sint64(40);
  proto.add_repeated_fixed32(-50);
  proto.add_repeated_fixed32(0);
  proto.add_repeated_fixed32(50);
  proto.add_repeated_fixed64(-60);
  proto.add_repeated_fixed64(0);
  proto.add_repeated_fixed64(60);
  proto.add_repeated_sfixed32(-70);
  proto.add_repeated_sfixed32(0);
  proto.add_repeated_sfixed32(70);
  proto.add_repeated_sfixed64(-80);
  proto.add_repeated_sfixed64(0);
  proto.add_repeated_sfixed64(80);
  proto.add_repeated_float(-1.2345);
  proto.add_repeated_float(0);
  proto.add_repeated_float(-2.3456);
  proto.add_repeated_double(-10.2345);
  proto.add_repeated_double(0);
  proto.add_repeated_double(-20.3456);
  proto.add_repeated_bool(false);
  proto.add_repeated_bool(true);
  proto.add_repeated_bool(false);
  proto.add_repeated_string("abc");
  proto.add_repeated_string("");
  proto.add_repeated_string("def");
  proto.add_repeated_cord("abc");
  proto.add_repeated_cord("");
  proto.add_repeated_cord("def");
  proto.add_packed_repeated_int64(-1000);
  proto.add_packed_repeated_int64(0);
  proto.add_packed_repeated_int64(1000);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Proto supports [] for list values as well.
  EXPECT_PARSE_SUCCESS("repeated_int32: 1 repeated_int32: 2 repeated_int32: 3",
                       "repeated_int32: [1, 2 , 3]");

  // Test [] and also interesting bool values.
  EXPECT_PARSE_SUCCESS(("repeated_bool: false repeated_bool: false "
                        "repeated_bool: true repeated_bool: true "
                        "repeated_bool: false repeated_bool: true"),
                       "repeated_bool: [false, 0, 1, true, False, True]");

  EXPECT_PARSE_SUCCESS(("repeated_string: \"a,b\" "
                        "repeated_string: \"cdef\""),
                       "repeated_string:   [  'a,b', 'cdef'  ]  ");

  // Proto supports ' as quote character.
  EXPECT_PARSE_SUCCESS("optional_string: \"123\\\" \\'xyz\"",
                       "optional_string: '123\\\" \\'xyz'  ");

  EXPECT_PARSE_SUCCESS("optional_double: 10000", "optional_double: 1e4");

  // Error cases.
  EXPECT_PARSE_FAILURE("optional_string: '1' optional_string: '2'");
  EXPECT_PARSE_FAILURE("optional_double: 123 optional_double: 456");
  EXPECT_PARSE_FAILURE("optional_double: 0001");
  EXPECT_PARSE_FAILURE("optional_double: 000.1");
  EXPECT_PARSE_FAILURE("optional_double: a");
  EXPECT_PARSE_FAILURE("optional_double: x123");
  EXPECT_PARSE_FAILURE("optional_double: '123'");
  EXPECT_PARSE_FAILURE("optional_double: --111");
  EXPECT_PARSE_FAILURE("optional_string: 'abc\"");
  EXPECT_PARSE_FAILURE("optional_bool: truE");
  EXPECT_PARSE_FAILURE("optional_bool: FALSE");
}

TEST(CreateProtoDebugStringLibTest, NestedMessages) {
  TestAllTypes proto;

  proto.Clear();
  // Test empty message.
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_foreign_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Empty messages.
  proto.Clear();
  proto.mutable_optional_nested_message();
  proto.mutable_optional_foreign_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1);
  proto.mutable_optional_foreign_message()->set_c(-1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1234);
  proto.mutable_optional_nested_message()
      ->mutable_msg();  // empty double-nested
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1234);
  proto.mutable_optional_nested_message()->mutable_msg()->set_optional_string(
      "abc");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->mutable_msg()->set_optional_string(
      "abc");
  proto.mutable_optional_nested_message()->set_optional_int64(1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  auto* nested = proto.add_repeated_nested_message();
  nested = proto.add_repeated_nested_message();
  nested->set_optional_int32(123);
  nested->mutable_msg();
  nested = proto.add_repeated_nested_message();
  nested->mutable_msg();
  nested->mutable_msg()->set_optional_string("abc");
  nested->set_optional_int64(1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // text format allows use of <> for messages.
  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message: < optional_int32: 123   >");

  // <> and {} must use same style for closing.
  EXPECT_PARSE_FAILURE("optional_nested_message: < optional_int32: 123   }");
  EXPECT_PARSE_FAILURE("optional_nested_message: { optional_int32: 123   >");

  // colon after identifier is optional for messages.
  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message < optional_int32: 123   >");

  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message{ optional_int32: 123   }  ");

  // Proto supports [] for list values as well.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 }"),
      "repeated_nested_message: [ { }, { optional_int32: 123  } ]");

  // Colon after repeated_nested_message is optional.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 }"),
      "repeated_nested_message [ { }, { optional_int32: 123  } ]");

  // Using the list format a:[..] twice, like a:[..] a:[..] joins the two
  // arrays.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 } "
       "repeated_nested_message { optional_int32: 456 }"),
      ("repeated_nested_message [ { }, { optional_int32: 123  } ]"
       "repeated_nested_message [ { optional_int32: 456  } ]"));

  // Parse errors on nested messages.
  EXPECT_PARSE_FAILURE("optional_nested_message: {optional_int32: 'abc' }");

  // Optional_nested_message appearing twice is an error.
  EXPECT_PARSE_FAILURE(
      ("optional_nested_message { optional_int32: 123 } "
       "optional_nested_message { optional_int64: 456 }"));
}

TEST(CreateProtoDebugStringLibTest, RecursiveMessage) {
  NestedTestAllTypes proto;

  NestedTestAllTypes* cur = &proto;
  for (int depth = 0; depth < 20; ++depth) {
    cur->mutable_payload()->set_optional_int32(1000 + depth);
    cur = cur->mutable_child();
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();
}

template <typename T>
T ParseProto(const string& value_text_proto) {
  T value;
  EXPECT_TRUE(protobuf::TextFormat::ParseFromString(value_text_proto, &value))
      << value_text_proto;
  return value;
}

TestAllTypes::NestedMessage ParseNestedMessage(const string& value_text_proto) {
  return ParseProto<TestAllTypes::NestedMessage>(value_text_proto);
}

TEST(CreateProtoDebugStringLibTest, Map) {
  TestAllTypes proto;

  std::vector<TestAllTypes::NestedMessage> msg_values;
  msg_values.push_back(ParseNestedMessage("optional_int32: 345"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 123"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 234"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 0"));

  // string->message map
  proto.Clear();
  {
    auto& map = *proto.mutable_map_string_to_message();
    map["def"] = msg_values[0];
    map["abc"] = msg_values[1];
    map["cde"] = msg_values[2];
    map[""] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int32->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int32_to_message();
    map[20] = msg_values[0];
    map[10] = msg_values[1];
    map[15] = msg_values[2];
    map[0] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int64->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_message();
    map[20] = msg_values[0];
    map[10] = msg_values[1];
    map[15] = msg_values[2];
    map[0] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // bool->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_message();
    map[true] = msg_values[0];
    map[false] = msg_values[1];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // string->int64 map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_string_to_int64();
    map["def"] = 0;
    map["abc"] = std::numeric_limits<protobuf_int64>::max();
    map[""] = 20;
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int64->string map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_string();
    map[0] = "def";
    map[std::numeric_limits<protobuf_int64>::max()] = "";
    map[20] = "abc";
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Test a map with the same key multiple times.
  EXPECT_PARSE_SUCCESS(("map_string_to_int64 { key: \"abc\" value: 5 } "
                        "map_string_to_int64 { key: \"def\" value: 2 } "
                        "map_string_to_int64 { key: \"ghi\" value: 4 }"),
                       ("map_string_to_int64: { key: 'abc' value: 1 } "
                        "map_string_to_int64: { key: 'def' value: 2 } "
                        "map_string_to_int64: { key: 'ghi' value: 3 } "
                        "map_string_to_int64: { key: 'ghi' value: 4 } "
                        "map_string_to_int64: { key: 'abc' value: 5 } "));
}

TEST(CreateProtoDebugStringLibTest, Enums) {
  TestAllTypes proto;

  proto.Clear();
  proto.set_optional_nested_enum(TestAllTypes::ZERO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.set_optional_nested_enum(TestAllTypes::FOO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.add_repeated_nested_enum(TestAllTypes::FOO);
  proto.add_repeated_nested_enum(TestAllTypes::ZERO);
  proto.add_repeated_nested_enum(TestAllTypes::BAR);
  proto.add_repeated_nested_enum(TestAllTypes::NEG);
  proto.add_repeated_nested_enum(TestAllTypes::ZERO);
  proto.set_optional_foreign_enum(ForeignEnum::FOREIGN_BAR);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Parsing from numbers works as well.
  EXPECT_PARSE_SUCCESS(
      "optional_nested_enum: BAR "   // 2
      "repeated_nested_enum: BAR "   // 2
      "repeated_nested_enum: ZERO "  // 0
      "repeated_nested_enum: FOO",   // 1
      ("repeated_nested_enum: 2 "
       "repeated_nested_enum: 0 "
       "optional_nested_enum: 2 "
       "repeated_nested_enum: 1"));

  EXPECT_PARSE_SUCCESS("", "optional_nested_enum: -0");
  // TODO(amauryfa): restore the line below when protobuf::TextFormat also
  // supports unknown enum values.
  // EXPECT_PARSE_SUCCESS("optional_nested_enum: 6", "optional_nested_enum: 6");
  EXPECT_PARSE_FAILURE("optional_nested_enum: 2147483648");  // > INT32_MAX
  EXPECT_PARSE_FAILURE("optional_nested_enum: BARNONE");
  EXPECT_PARSE_FAILURE("optional_nested_enum: 'BAR'");
  EXPECT_PARSE_FAILURE("optional_nested_enum: \"BAR\" ");

  EXPECT_EQ(string("BAR"),
            string(EnumName_TestAllTypes_NestedEnum(TestAllTypes::BAR)));
  // out of range - returns empty string (see NameOfEnum in proto library).
  EXPECT_EQ(string(""), string(EnumName_TestAllTypes_NestedEnum(
                            static_cast<TestAllTypes_NestedEnum>(123))));
}

TEST(CreateProtoDebugStringLibTest, Oneof) {
  TestAllTypes proto;

  proto.Clear();
  proto.set_oneof_string("abc");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Empty oneof_string is printed, as the setting of the value in the oneof is
  // meaningful.
  proto.Clear();
  proto.set_oneof_string("");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.set_oneof_string("abc");
  proto.set_oneof_uint32(123);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Zero uint32 is printed, as the setting of the value in the oneof is
  // meaningful.
  proto.Clear();
  proto.set_oneof_uint32(0);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Zero enum value is meaningful.
  proto.Clear();
  proto.set_oneof_enum(TestAllTypes::ZERO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Parse a text format that lists multiple members of the oneof.
  EXPECT_PARSE_FAILURE("oneof_string: \"abc\" oneof_uint32: 13 ");
  EXPECT_PARSE_FAILURE("oneof_string: \"abc\" oneof_string: \"def\" ");
}

TEST(CreateProtoDebugStringLibTest, Comments) {
  TestAllTypes proto;

  EXPECT_PARSE_SUCCESS("optional_int64: 123 optional_string: \"#text\"",
                       ("#leading comment \n"
                        "optional_int64# comment\n"
                        ":# comment\n"
                        "123# comment\n"
                        "optional_string  # comment\n"
                        ":   # comment\n"
                        "\"#text\"#comment####\n"));

  EXPECT_PARSE_FAILURE("optional_int64:// not a valid comment\n123");
  EXPECT_PARSE_FAILURE("optional_int64:/* not a valid comment */\n123");
}

}  // namespace
}  // namespace test
}  // namespace tensorflow

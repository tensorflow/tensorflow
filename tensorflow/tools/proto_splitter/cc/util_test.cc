/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/tools/proto_splitter/cc/util.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/proto_splitter/cc/test_util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {
namespace {

using ::tensorflow::proto_splitter::ChunkedField;
using ::tensorflow::proto_splitter_testdata::ManyFields;
using ::testing::HasSubstr;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

// Required in OSS to prevent string to bool conversion in FieldType variant.
using namespace std::string_literals;  // NOLINT

absl::StatusOr<ManyFields> MakeManyFields() {
  return ParseTextProto<ManyFields>(
      R"pb(field_one {
             repeated_field {}
             repeated_field {
               string_field: "inner_inner_string"
               map_field_uint32 { key: 324 value: "map_value_324" }
               map_field_uint32 { key: 543 value: "map_value_543" }
             }
           }
           map_field_int64 { key: -1345 value: "map_value_-1345" }
           nested_map_bool {
             key: false
             value { string_field: "string_false" }
           }
           nested_map_bool {
             key: true
             value { string_field: "string_true" }
           })pb");
}

absl::StatusOr<
    tsl::protobuf::RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
MakeFieldTags() {
  TF_ASSIGN_OR_RETURN(auto ret, ParseTextProto<ChunkedField>(R"pb(
                        field_tag { field: 2 }
                        field_tag { index: 1505 }
                        field_tag { field: 5 }
                        field_tag { map_key { ui32: 123 } }
                      )pb"));
  return ret.field_tag();
}

absl::StatusOr<
    tsl::protobuf::RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
MakeFieldTagsTooManyIndices() {
  TF_ASSIGN_OR_RETURN(auto ret, ParseTextProto<ChunkedField>(R"pb(
                        field_tag { field: 2 }
                        field_tag { index: 1505 }
                        field_tag { index: 1506 }
                        field_tag { field: 5 }
                        field_tag { map_key { ui32: 123 } }
                      )pb"));
  return ret.field_tag();
}

absl::StatusOr<
    tsl::protobuf::RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
MakeFieldTagsTooManyMapKeys() {
  TF_ASSIGN_OR_RETURN(auto ret, ParseTextProto<ChunkedField>(R"pb(
                        field_tag { field: 2 }
                        field_tag { index: 1505 }
                        field_tag { field: 5 }
                        field_tag { map_key { ui32: 123 } }
                        field_tag { map_key: { ui32: 124 } }
                      )pb"));
  return ret.field_tag();
}

absl::StatusOr<
    tsl::protobuf::RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
MakeFieldTagsMisplacedIndex() {
  TF_ASSIGN_OR_RETURN(auto ret, ParseTextProto<ChunkedField>(R"pb(
                        field_tag { field: 2 }
                        field_tag { index: 1505 }
                        field_tag { field: 5 }
                        field_tag { map_key { ui32: 123 } }
                        field_tag { index: 1504 }
                      )pb"));
  return ret.field_tag();
}

absl::StatusOr<
    tsl::protobuf::RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
MakeFieldTagsMisplacedMapKey() {
  TF_ASSIGN_OR_RETURN(auto ret, ParseTextProto<ChunkedField>(R"pb(
                        field_tag { field: 2 }
                        field_tag { index: 1505 }
                        field_tag { map_key: { ui32: 321 } }
                        field_tag { field: 5 }
                        field_tag { map_key { ui32: 123 } }
                      )pb"));
  return ret.field_tag();
}

TEST(UtilTest, TestFieldTag) {
  ManyFields message;
  ChunkedField field;
  std::vector<FieldType> fields = {"nested_map_bool"s, true, 2, 50, 7, false};
  TF_ASSERT_OK(AddFieldTag(*message.descriptor(), fields, field));

  EXPECT_THAT(field,
              EqualsProto(R"pb(field_tag { field: 7 }
                               field_tag { map_key { boolean: true } }
                               field_tag { field: 2 }
                               field_tag { index: 50 }
                               field_tag { field: 7 }
                               field_tag { map_key { boolean: false } })pb"));
}
TEST(UtilTest, TestFieldTagWithInt64MapKey) {
  ManyFields message;
  ChunkedField field;
  std::vector<FieldType> fields = {"map_field_int64"s, -4234};
  TF_ASSERT_OK(AddFieldTag(*message.descriptor(), fields, field));

  EXPECT_THAT(field, EqualsProto(R"pb(field_tag { field: 6 }
                                      field_tag { map_key { i64: -4234 } }
              )pb"));
}

TEST(UtilTest, TestFieldTagWithUInt32MapKey) {
  ManyFields message;
  ChunkedField field;
  std::vector<FieldType> fields = {"repeated_field"s, 1505, "map_field_uint32"s,
                                   123};
  TF_ASSERT_OK(AddFieldTag(*message.descriptor(), fields, field));

  EXPECT_THAT(field, EqualsProto(R"pb(field_tag { field: 2 }
                                      field_tag { index: 1505 }
                                      field_tag { field: 5 }
                                      field_tag { map_key { ui32: 123 } }
              )pb"));
}

TEST(UtilTest, TestFieldTagConversion) {
  ManyFields message;
  ChunkedField field;
  std::vector<FieldType> fields = {"repeated_field"s, "1505"s,
                                   "map_field_uint32"s, 123};
  TF_ASSERT_OK(AddFieldTag(*message.descriptor(), fields, field));

  EXPECT_THAT(field, EqualsProto(R"pb(field_tag { field: 2 }
                                      field_tag { index: 1505 }
                                      field_tag { field: 5 }
                                      field_tag { map_key { ui32: 123 } }
              )pb"));
}

TEST(UtilTest, TestGetFieldTypes) {
  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTags());
  EXPECT_THAT(
      GetFieldTypes(tags),
      absl_testing::IsOkAndHolds(std::vector<Field>{{2, 1505}, {5, 123}}));
}

TEST(UtilTest, TestGetFieldTypesThenAddFieldTags) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  ChunkedField chunked_field;

  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTags());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Field> fields, GetFieldTypes(tags));
  for (const auto& field : fields) {
    TF_ASSERT_OK(AddFieldTag(*message.descriptor(), field, chunked_field));
  }

  EXPECT_THAT(chunked_field,
              EqualsProto(R"pb(field_tag { field: 2 }
                               field_tag { index: 1505 }
                               field_tag { field: 5 }
                               field_tag { map_key { ui32: 123 } }
              )pb"));
}

TEST(UtilTest, TestGetFieldTypesThenAddFieldTagsTooManyIndices) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  ChunkedField chunked_field;

  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTagsTooManyIndices());
  EXPECT_THAT(
      GetFieldTypes(tags),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("Index doesn't belong to any field")));
}

TEST(UtilTest, TestGetFieldTypesThenAddFieldTagsTooManyMapKeys) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  ChunkedField chunked_field;

  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTagsTooManyMapKeys());
  EXPECT_THAT(
      GetFieldTypes(tags),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("Map key doesn't belong to any field")));
}

TEST(UtilTest, TestGetFieldTypesThenAddFieldTagsMisplacedIndex) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  ChunkedField chunked_field;

  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTagsMisplacedIndex());
  EXPECT_THAT(
      GetFieldTypes(tags),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("Index doesn't belong to any field")));
}

TEST(UtilTest, TestGetFieldTypesThenAddFieldTagsMisplacedMapKey) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  ChunkedField chunked_field;

  TF_ASSERT_OK_AND_ASSIGN(auto tags, MakeFieldTagsMisplacedMapKey());
  EXPECT_THAT(
      GetFieldTypes(tags),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("Map key doesn't belong to any field")));
}

TEST(UtilTest, TestSetRepeatedFieldElement) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message,
      ParseTextProto<ManyFields>(R"pb(repeated_string_field: ""
                                      repeated_string_field: "")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("repeated_string_field");
  const std::vector<std::string> chunks = {"repeated_string_one",
                                           "repeated_string_two"};
  TF_ASSERT_OK(SetRepeatedFieldElement(
      &message, field_desc, 0, chunks[0],
      []() -> absl::Status { return absl::OkStatus(); }));
  TF_ASSERT_OK(SetRepeatedFieldElement(
      &message, field_desc, 1, chunks[1],
      []() -> absl::Status { return absl::OkStatus(); }));

  EXPECT_FALSE(message.repeated_string_field().empty());
  EXPECT_EQ(message.repeated_string_field().at(0), "repeated_string_one");
  EXPECT_EQ(message.repeated_string_field().at(1), "repeated_string_two");
}

TEST(UtilTest, TestSetRepeatedFieldElementInvalidIndex) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message,
      ParseTextProto<ManyFields>(R"pb(repeated_string_field: "")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("repeated_string_field");

  EXPECT_THAT(SetRepeatedFieldElement(
                  &message, field_desc, 1, "",
                  []() -> absl::Status { return absl::OkStatus(); }),
              absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                     HasSubstr("Field index out of range")));
}

TEST(UtilTest, TestSetRepeatedFieldElementAlreadyExists) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message, ParseTextProto<ManyFields>(
                        R"pb(repeated_string_field: "existing_string")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("repeated_string_field");
  const std::string chunk = "inner_inner_string_v2";

  TF_EXPECT_OK(SetRepeatedFieldElement(
      &message, field_desc, 0, chunk,
      []() -> absl::Status { return absl::OkStatus(); }));

  EXPECT_FALSE(message.repeated_string_field().empty());
  EXPECT_NE(message.repeated_string_field().at(0), "existing_string");
  EXPECT_EQ(message.repeated_string_field().at(0), "inner_inner_string_v2");
}

TEST(UtilTest, TestSetRepeatedFieldElementBadFieldDesc) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message, ParseTextProto<ManyFields>(
                        R"pb(repeated_string_field: "existing_string")pb"));

  const tsl::protobuf::FieldDescriptor* singular_field_desc =
      message.GetDescriptor()->FindFieldByName("string_field");
  EXPECT_DEATH(
      auto status = SetRepeatedFieldElement(
          &message, singular_field_desc, 0, "",
          []() -> absl::Status { return absl::OkStatus(); }),
      HasSubstr("Field is singular; the method requires a repeated field"));

  const tsl::protobuf::FieldDescriptor* map_field_desc =
      message.GetDescriptor()->FindFieldByName("map_field_uint32");
  EXPECT_THAT(SetRepeatedFieldElement(
                  &message, map_field_desc, 0, "",
                  []() -> absl::Status { return absl::OkStatus(); }),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     HasSubstr("Field is a map")));
}

TEST(UtilTest, TestSetRepeatedFieldElementMessage) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message, ParseTextProto<ManyFields>(R"pb(repeated_field {})pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("repeated_field");

  TF_ASSERT_OK_AND_ASSIGN(
      auto chunk_proto, ParseTextProto<ManyFields>(R"pb(string_field: "")pb"));

  const std::string chunk = chunk_proto.SerializeAsString();

  TF_ASSERT_OK(SetRepeatedFieldElement(
      &message, field_desc, 0, chunk,
      [&message, &field_desc]() -> absl::Status {
        tsl::protobuf::Message* inner_message =
            message.GetReflection()->MutableRepeatedMessage(&message,
                                                            field_desc, 0);
        inner_message->GetReflection()->SetString(
            inner_message,
            field_desc->message_type()->FindFieldByName("string_field"),
            "inner_string");
        return absl::OkStatus();
      }));

  EXPECT_FALSE(message.repeated_field().empty());
  EXPECT_EQ(message.repeated_field().at(0).string_field(), "inner_string");
}

TEST(UtilTest, TestSetFieldElement) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message, ParseTextProto<ManyFields>(R"pb(string_field: "")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("string_field");
  const std::string chunk = "string_field_v2";
  TF_ASSERT_OK(
      SetFieldElement(&message, field_desc, chunk,
                      []() -> absl::Status { return absl::OkStatus(); }));

  EXPECT_EQ(message.string_field(), "string_field_v2");
}

TEST(UtilTest, TestSetFieldElementInvalidRepeated) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message,
      ParseTextProto<ManyFields>(R"pb(repeated_string_field: "")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("repeated_string_field");

  EXPECT_DEATH(
      auto status =
          SetFieldElement(&message, field_desc, "",
                          []() -> absl::Status { return absl::OkStatus(); }),
      HasSubstr("Field is repeated; the method requires a singular field"));
}

TEST(UtilTest, TestSetFieldElementAlreadyExists) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto message,
      ParseTextProto<ManyFields>(R"pb(string_field: "existing_string")pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("string_field");
  const std::string chunk = "inner_inner_string_v2";

  TF_EXPECT_OK(
      SetFieldElement(&message, field_desc, chunk,
                      []() -> absl::Status { return absl::OkStatus(); }));

  EXPECT_EQ(message.string_field(), "inner_inner_string_v2");
}

TEST(UtilTest, TestSetFieldElementMessage) {
  TF_ASSERT_OK_AND_ASSIGN(auto message,
                          ParseTextProto<ManyFields>(R"pb(field_one {})pb"));
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("field_one");

  TF_ASSERT_OK_AND_ASSIGN(
      auto chunk_proto, ParseTextProto<ManyFields>(R"pb(string_field: "")pb"));

  const std::string chunk = chunk_proto.SerializeAsString();

  TF_ASSERT_OK(SetFieldElement(
      &message, field_desc, chunk, [&message, &field_desc]() -> absl::Status {
        tsl::protobuf::Message* inner_message =
            message.GetReflection()->MutableMessage(&message, field_desc);
        inner_message->GetReflection()->SetString(
            inner_message,
            field_desc->message_type()->FindFieldByName("string_field"),
            "inner_string");
        return absl::OkStatus();
      }));

  EXPECT_TRUE(message.has_field_one());
  EXPECT_EQ(message.field_one().string_field(), "inner_string");
}

TEST(UtilTest, TestAddMapEntry) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  const tsl::protobuf::FieldDescriptor* field_desc =
      message.GetDescriptor()->FindFieldByName("map_field_int64");
  FieldType map_key = -4234;
  TF_ASSERT_OK(AddMapEntry(&message, field_desc, map_key));

  TF_ASSERT_OK_AND_ASSIGN(int map_entry_index,
                          FindMapKey(message, *field_desc, nullptr, map_key));
  tsl::protobuf::Message* map_entry =
      message.GetReflection()->MutableRepeatedMessage(&message, field_desc,
                                                      map_entry_index);
  map_entry->GetReflection()->SetString(
      map_entry, field_desc->message_type()->FindFieldByNumber(2),
      "map_value_-4234");

  EXPECT_EQ(message.map_field_int64().at(-4234), "map_value_-4234");
}

TEST(UtilTest, GetFieldInvalidIndex) {
  std::vector<FieldType> fields = {"field_one"s, "repeated_field"s, 100};

  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  EXPECT_THAT(GetField(message, fields),
              absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                     HasSubstr("Can't access index 100")));
}

TEST(UtilTest, GetFieldInvalidField) {
  std::vector<FieldType> fields = {"field_one"s, "INVALID"s};
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  EXPECT_THAT(GetField(message, fields),
              absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                     HasSubstr("Field not found: INVALID")));
}

TEST(UtilTest, GetFieldInvalidMapKey) {
  std::vector<FieldType> fields = {"map_field_int64"s, 10000};
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  EXPECT_THAT(GetField(message, fields),
              absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                     HasSubstr("couldn't find key: 10000")));
}
TEST(UtilTest, GetField) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  std::vector<FieldType> fields = {"field_one"s, "repeated_field"s};

  TF_ASSERT_OK_AND_ASSIGN(auto ret, GetField(message, fields));
  EXPECT_EQ(-1, ret.index);
  EXPECT_EQ(2, ret.parent->GetReflection()->FieldSize(*ret.parent, ret.field));

  fields = {"nested_map_bool"s, "true"s, "string_field"s};
  TF_ASSERT_OK_AND_ASSIGN(auto ret2, GetField(message, fields));
  EXPECT_EQ(-1, ret2.index);
  EXPECT_EQ("string_true",
            ret2.parent->GetReflection()->GetString(*ret2.parent, ret2.field));
}

TEST(UtilTest, FindMapKey) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  const tsl::protobuf::FieldDescriptor* map_field =
      message.GetDescriptor()->FindFieldByName("nested_map_bool");
  TF_ASSERT_OK_AND_ASSIGN(int i,
                          FindMapKey(message, *map_field, nullptr, true));
  EXPECT_EQ(1, i);
  TF_ASSERT_OK_AND_ASSIGN(i, FindMapKey(message, *map_field, nullptr, false));
  EXPECT_EQ(0, i);

  map_field = message.GetDescriptor()->FindFieldByName("map_field_int64");
  TF_ASSERT_OK_AND_ASSIGN(i, FindMapKey(message, *map_field, nullptr, -1345));
  EXPECT_EQ(0, i);
  TF_ASSERT_OK_AND_ASSIGN(i, FindMapKey(message, *map_field, nullptr, 1345));
  EXPECT_EQ(-1, i);
}

TEST(UtilTest, FindMapKeyInvalid) {
  TF_ASSERT_OK_AND_ASSIGN(auto message, MakeManyFields());
  const tsl::protobuf::FieldDescriptor* not_map_field =
      message.GetDescriptor()->FindFieldByName("string_field");
  EXPECT_THAT(FindMapKey(message, *not_map_field, nullptr, -1345),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("was given a non map field")));
}

TEST(UtilTest, TestHumanReadableBytes) {
  EXPECT_EQ("1.0KiB", HumanReadableBytes(1024));
  EXPECT_EQ("5.5KiB", HumanReadableBytes(5632));
  EXPECT_EQ("52.2KiB", HumanReadableBytes(53432));
  EXPECT_EQ("72.9MiB", HumanReadableBytes(76493281));
  EXPECT_EQ("57.0MiB", HumanReadableBytes(5.977e7));
  EXPECT_EQ("1.0GiB", HumanReadableBytes(1.074e9));
  EXPECT_EQ("15.4GiB", HumanReadableBytes(16493342281));
}

TEST(UtilTest, TestHumanReadableDuration) {
  EXPECT_EQ("534 microseconds", HumanReadableDuration(534));
  EXPECT_EQ("1.00 ms", HumanReadableDuration(1000));
  EXPECT_EQ("14.33 ms", HumanReadableDuration(14328));
  EXPECT_EQ("95.83 s", HumanReadableDuration(95825433));
}

TEST(UtilTest, TestReadChunk) {
  std::string cpb_file =
      io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-standard.cpb");

  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ::tensorflow::proto_splitter::ChunkMetadata metadata = read_metadata.value();
  std::vector<::tensorflow::proto_splitter::ChunkInfo> chunks_info(
      metadata.chunks().begin(), metadata.chunks().end());

  for (const auto& chunk_info : chunks_info) {
    TF_ASSERT_OK_AND_ASSIGN(std::string chunk, ReadChunk(reader, chunk_info));
    ASSERT_EQ(chunk.size(), chunk_info.size());
  }
}

}  // namespace
}  // namespace tools::proto_splitter
}  // namespace tensorflow

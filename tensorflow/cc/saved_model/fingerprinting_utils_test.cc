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

#include "tensorflow/cc/saved_model/fingerprinting_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
// IWYU pragma: no_include "third_party/protobuf/io/zero_copy_stream_impl_lite.h"
// IWYU pragma: no_include "third_party/protobuf/util/message_differencer.h"

namespace tensorflow::saved_model::fingerprinting {

namespace {

using fingerprinting_utils_internal::fieldTagMatches;
using fingerprinting_utils_internal::HashFields;
using fingerprinting_utils_internal::HashGraphDef;
using fingerprinting_utils_internal::HashSavedObjectGraph;
using fingerprinting_utils_internal::HashSignatureDef;
using fingerprinting_utils_internal::PruneChunkedMessage;
using fingerprinting_utils_internal::SerializeProto;
using ::tensorflow::proto_splitter::ChunkedField;
using ::tensorflow::proto_splitter::ChunkedMessage;
using ::tensorflow::proto_splitter::ChunkInfo;
using ::tensorflow::proto_splitter::ChunkMetadata;
using ::tensorflow::proto_splitter::FieldIndex;
using ::tensorflow::proto_splitter_testdata::ManyFields;
using ::tensorflow::protobuf::Message;
using ::tensorflow::protobuf::RepeatedPtrField;
using ::tensorflow::protobuf::TextFormat;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::tensorflow::protobuf::io::ArrayInputStream;
// NOLINTNEXTLINE: clang-tidy missing-includes false positive
using ::tensorflow::protobuf::util::MessageDifferencer;
using tools::proto_splitter::GetChunkMetadata;
using tools::proto_splitter::GetRiegeliReader;
using tsl::testing::IsOkAndHolds;
using tsl::testing::TensorFlowSrcRoot;

absl::Status ParseTextProto(absl::string_view text_proto,
                            Message* parsed_proto) {
  TextFormat::Parser parser;
  // Attempt to parse as text.
  ArrayInputStream input_stream(text_proto.data(), text_proto.size());
  if (parser.Parse(&input_stream, parsed_proto)) {
    return absl::OkStatus();
  }
  parsed_proto->Clear();
  return absl::InvalidArgumentError(
      absl::StrCat("Could not parse text proto: ", text_proto));
}

absl::StatusOr<RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
ExtractFieldTags(absl::string_view chunked_field_text_proto) {
  ChunkedField chunked_field;
  TF_RETURN_IF_ERROR(ParseTextProto(chunked_field_text_proto, &chunked_field));
  return chunked_field.field_tag();
}

TEST(FingerprintingTest, TestFieldTagMatchesInitialSubsequence) {
  TF_ASSERT_OK_AND_ASSIGN(RepeatedPtrField<FieldIndex> field_tags,
                          ExtractFieldTags(R"pb(
                            field_tag { field: 2 }
                            field_tag { index: 1505 }
                            field_tag { field: 5 }
                            field_tag { map_key { ui32: 123 } }
                          )pb"));
  RepeatedPtrField<FieldIndex> field_tags_sub;
  field_tags_sub.CopyFrom(field_tags);
  field_tags_sub.DeleteSubrange(2, 2);

  EXPECT_THAT(fieldTagMatches(field_tags_sub, field_tags),
              absl_testing::IsOkAndHolds(2));
}

TEST(FingerprintingTest, TestFieldTagMatchesNoninitialSubsequence) {
  TF_ASSERT_OK_AND_ASSIGN(RepeatedPtrField<FieldIndex> field_tags,
                          ExtractFieldTags(R"pb(
                            field_tag { field: 2 }
                            field_tag { index: 1505 }
                            field_tag { field: 5 }
                            field_tag { map_key { ui32: 123 } }
                          )pb"));
  RepeatedPtrField<FieldIndex> field_tags_sub;
  field_tags_sub.CopyFrom(field_tags);
  field_tags_sub.DeleteSubrange(0, 2);

  EXPECT_THAT(fieldTagMatches(field_tags_sub, field_tags),
              absl_testing::IsOkAndHolds(0));
}

TEST(FingerprintingTest, TestFieldTagMatchesIdenticalSubsequence) {
  TF_ASSERT_OK_AND_ASSIGN(RepeatedPtrField<FieldIndex> field_tags,
                          ExtractFieldTags(R"pb(
                            field_tag { field: 2 }
                            field_tag { index: 1505 }
                            field_tag { field: 5 }
                            field_tag { map_key { ui32: 123 } }
                          )pb"));
  RepeatedPtrField<FieldIndex> field_tags_sub;
  field_tags_sub.CopyFrom(field_tags);

  EXPECT_THAT(fieldTagMatches(field_tags_sub, field_tags),
              absl_testing::IsOkAndHolds(4));
}

TEST(FingerprintingTest, TestFieldTagMatchesSuperSubsequence) {
  TF_ASSERT_OK_AND_ASSIGN(RepeatedPtrField<FieldIndex> field_tags,
                          ExtractFieldTags(R"pb(
                            field_tag { field: 2 }
                            field_tag { index: 1505 }
                            field_tag { field: 5 }
                            field_tag { map_key { ui32: 123 } }
                          )pb"));
  RepeatedPtrField<FieldIndex> field_tags_sub;
  field_tags_sub.CopyFrom(field_tags);
  field_tags_sub.Add()->set_field(6);

  EXPECT_THAT(fieldTagMatches(field_tags_sub, field_tags),
              absl_testing::IsOkAndHolds(4));
}

TEST(FingerprintingTest, TestPruneChunkedMessageSingleTarget) {
  std::string cpb_file = io::JoinPath(
      TensorFlowSrcRoot(), "tools/proto_splitter/testdata", "many-field.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  FieldIndex field_one_field_tag;
  field_one_field_tag.set_field(1);
  FieldIndex repeated_field_field_tag;
  repeated_field_field_tag.set_field(2);
  FieldIndex repeated_field_index_field_tag;
  repeated_field_index_field_tag.set_index(1);
  RepeatedPtrField<FieldIndex> target_field_tags;
  target_field_tags.Add(FieldIndex(field_one_field_tag));
  target_field_tags.Add(FieldIndex(repeated_field_field_tag));
  target_field_tags.Add(FieldIndex(repeated_field_index_field_tag));

  ChunkedMessage pruned_chunked_message;
  TF_ASSERT_OK_AND_ASSIGN(
      pruned_chunked_message,
      PruneChunkedMessage(chunk_metadata.message(), reader, chunks_info,
                          {target_field_tags}));

  std::string expected_pruned_chunked_message_text_proto = R"pb(
    chunk_index: 0
    chunked_fields {
      field_tag { field: 1 }
      message { chunk_index: 1 }
    }
  )pb";
  ChunkedMessage expected_pruned_chunked_message;
  TF_ASSERT_OK(ParseTextProto(expected_pruned_chunked_message_text_proto,
                              &expected_pruned_chunked_message));
  ASSERT_TRUE(MessageDifferencer::Equals(pruned_chunked_message,
                                         expected_pruned_chunked_message));
}

TEST(FingerprintingTest, TestPruneChunkedMessageMultiTarget) {
  std::string cpb_file = io::JoinPath(
      TensorFlowSrcRoot(), "tools/proto_splitter/testdata", "many-field.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  // ManyFields.field_one.repeated_field[1]
  FieldIndex field_one_field_tag;
  field_one_field_tag.set_field(1);
  FieldIndex repeated_field_field_tag;
  repeated_field_field_tag.set_field(2);
  FieldIndex repeated_field_index_field_tag;
  repeated_field_index_field_tag.set_index(1);
  RepeatedPtrField<FieldIndex> target_one_field_tags;
  target_one_field_tags.Add(FieldIndex(field_one_field_tag));
  target_one_field_tags.Add(FieldIndex(repeated_field_field_tag));
  target_one_field_tags.Add(FieldIndex(repeated_field_index_field_tag));

  // ManyFields.nested_map_bool[true].string_field
  FieldIndex nested_map_bool_field_tag;
  nested_map_bool_field_tag.set_field(7);
  FieldIndex nested_map_bool_mapkey_field_tag;
  nested_map_bool_mapkey_field_tag.mutable_map_key()->set_boolean(true);
  FieldIndex string_field_field_tag;
  string_field_field_tag.set_field(3);
  RepeatedPtrField<FieldIndex> target_two_field_tags;
  target_two_field_tags.Add(FieldIndex(nested_map_bool_field_tag));
  target_two_field_tags.Add(FieldIndex(nested_map_bool_mapkey_field_tag));
  target_two_field_tags.Add(FieldIndex(string_field_field_tag));

  ChunkedMessage pruned_chunked_message;
  TF_ASSERT_OK_AND_ASSIGN(
      pruned_chunked_message,
      PruneChunkedMessage(chunk_metadata.message(), reader, chunks_info,
                          {target_one_field_tags, target_two_field_tags}));

  std::string expected_pruned_chunked_message_text_proto = R"pb(
    chunk_index: 0
    chunked_fields {
      field_tag { field: 1 }
      message { chunk_index: 1 }
    }
    chunked_fields {
      field_tag { field: 7 }
      field_tag { map_key { boolean: true } }
      message { chunk_index: 2 }
    }
  )pb";
  ChunkedMessage expected_pruned_chunked_message;
  TF_ASSERT_OK(ParseTextProto(expected_pruned_chunked_message_text_proto,
                              &expected_pruned_chunked_message));
  ASSERT_TRUE(MessageDifferencer::Equals(pruned_chunked_message,
                                         expected_pruned_chunked_message));
}

TEST(FingerprintingTest, TestPruneChunkedMessageNoTarget) {
  std::string cpb_file = io::JoinPath(
      TensorFlowSrcRoot(), "tools/proto_splitter/testdata", "many-field.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  ChunkedMessage pruned_chunked_message;
  TF_ASSERT_OK_AND_ASSIGN(
      pruned_chunked_message,
      PruneChunkedMessage(chunk_metadata.message(), reader, chunks_info, {}));

  std::string expected_pruned_chunked_message_text_proto = R"pb(
    chunk_index: 0
  )pb";
  ChunkedMessage expected_pruned_chunked_message;
  TF_ASSERT_OK(ParseTextProto(expected_pruned_chunked_message_text_proto,
                              &expected_pruned_chunked_message));
  ASSERT_TRUE(MessageDifferencer::Equals(pruned_chunked_message,
                                         expected_pruned_chunked_message));
}

TEST(FingerprintingTest, TestSerializeProto) {
  std::string many_fields_text_proto = R"pb(
    string_field: "abc123"
  )pb";
  ManyFields many_fields;
  TF_ASSERT_OK(ParseTextProto(many_fields_text_proto, &many_fields));
  ASSERT_EQ(SerializeProto(many_fields), many_fields.SerializeAsString());
}

TEST(FingerprintingTest, TestHashFieldsV2) {
  std::string cpb_file = io::JoinPath(
      TensorFlowSrcRoot(), "tools/proto_splitter/testdata", "many-field.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  ManyFields many_fields;
  TF_ASSERT_OK_AND_ASSIGN(uint64_t many_fields_hash,
                          HashFields(chunk_metadata.message(), reader,
                                     chunks_info, {}, &many_fields));
  ASSERT_EQ(many_fields_hash, 14850154939410192811U);
}

TEST(FingerprintingTest, TestHashGraphDef) {
  std::string cpb_file =
      io::JoinPath(TensorFlowSrcRoot(), "tools/proto_splitter/testdata",
                   "split-standard.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  GraphDef graph_def;
  EXPECT_THAT(
      HashGraphDef(&graph_def, chunk_metadata.message(), reader, chunks_info),
      absl_testing::IsOkAndHolds(16782272393894422524U));
}

TEST(FingerprintingTest, TestHashSignatureDef) {
  std::string cpb_file =
      io::JoinPath(TensorFlowSrcRoot(), "tools/proto_splitter/testdata",
                   "split-standard.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  ::tensorflow::protobuf::Map<std::string, SignatureDef> signature_def_map;
  SignatureDef signature_def;
  EXPECT_THAT(HashSignatureDef(signature_def_map, chunk_metadata.message(),
                               reader, chunks_info),
              absl_testing::IsOkAndHolds(0));
}

TEST(FingerprintingTest, TestHashSavedObjectGraph) {
  std::string cpb_file =
      io::JoinPath(TensorFlowSrcRoot(), "tools/proto_splitter/testdata",
                   "split-standard.cpb");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, GetRiegeliReader(cpb_file));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  SavedObjectGraph saved_object_graph;
  EXPECT_THAT(
      HashSavedObjectGraph(&saved_object_graph, chunk_metadata.message(),
                           reader, chunks_info),
      absl_testing::IsOkAndHolds(17454850744699451884U));
}

}  // namespace

}  // namespace tensorflow::saved_model::fingerprinting

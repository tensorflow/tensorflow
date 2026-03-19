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
#include "tensorflow/tools/proto_splitter/merge.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/proto_splitter/cc/test_util.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::tools::proto_splitter {

namespace {

inline constexpr std::array kDFSplitTreeChunks = {
    "val: \"0\"",       "val: \"010\"",     "val: \"01020\"",
    "val: \"0102030\"", "val: \"0102031\"", "val: \"0102032\"",
    "val: \"01021\"",   "val: \"0102130\"", "val: \"0102131\"",
    "val: \"0102132\""};

inline constexpr std::array kBFSplitTreeChunks = {
    "val: \"0\"",       "val: \"010\"",     "val: \"01020\"",
    "val: \"01021\"",   "val: \"0102030\"", "val: \"0102031\"",
    "val: \"0102032\"", "val: \"0102130\"", "val: \"0102131\"",
    "val: \"0102132\""};

TEST(MergeTest, TestReadRiegeliTreeDepthFirst) {
  // TODO(b/282779639): Use test data.
  const std::string cpb_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "df-split-tree");
  ::tensorflow::proto_splitter_testdata::StringNode merged_tree;
  TF_ASSERT_OK(Merger::Read(cpb_path, &merged_tree));

  const std::string pbtxt_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-tree");
  ::tensorflow::proto_splitter_testdata::StringNode test_proto;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(pbtxt_path, ".pbtxt"), &test_proto));

  ASSERT_THAT(merged_tree, EqualsProto(test_proto));
}

TEST(MergeTest, TestReadRiegeliTreeBreadthFirst) {
  const std::string cpb_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "bf-split-tree");
  ::tensorflow::proto_splitter_testdata::StringNode merged_tree;
  TF_ASSERT_OK(Merger::Read(cpb_path, &merged_tree));

  const std::string pbtxt_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-tree");

  ::tensorflow::proto_splitter_testdata::StringNode test_proto;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(pbtxt_path, ".pbtxt"), &test_proto));

  ASSERT_THAT(merged_tree, EqualsProto(test_proto));
}

TEST(MergeTest, TestMergeTreeChunksDepthFirst) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "df-split-tree");
  std::vector<std::unique_ptr<::tsl::protobuf::Message>> chunks;
  for (const auto& chunk : kDFSplitTreeChunks) {
    ::tensorflow::proto_splitter_testdata::StringNode string_node;
    ::tsl::protobuf::TextFormat::ParseFromString(chunk, &string_node);
    std::unique_ptr<::tsl::protobuf::Message> node =
        std::make_unique<::tensorflow::proto_splitter_testdata::StringNode>(
            string_node);
    chunks.push_back(std::move(node));
  }

  std::string split_tree_metadata;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &split_tree_metadata));
  ::tensorflow::proto_splitter::ChunkedMessage chunked_message;
  ::tsl::protobuf::TextFormat::ParseFromString(split_tree_metadata,
                                               &chunked_message);

  ::tensorflow::proto_splitter_testdata::StringNode merged_tree;
  TF_ASSERT_OK(Merger::Merge(chunks, chunked_message, &merged_tree));

  const std::string pbtxt_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-tree");

  ::tensorflow::proto_splitter_testdata::StringNode test_proto;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(pbtxt_path, ".pbtxt"), &test_proto));

  ASSERT_THAT(merged_tree, EqualsProto(test_proto));
}

TEST(MergeTest, TestMergeTreeChunksBreadthFirst) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "bf-split-tree");
  std::vector<std::unique_ptr<::tsl::protobuf::Message>> chunks;
  for (const auto& chunk : kBFSplitTreeChunks) {
    ::tensorflow::proto_splitter_testdata::StringNode string_node;
    ::tsl::protobuf::TextFormat::ParseFromString(chunk, &string_node);
    std::unique_ptr<::tsl::protobuf::Message> node =
        std::make_unique<::tensorflow::proto_splitter_testdata::StringNode>(
            string_node);
    chunks.push_back(std::move(node));
  }

  std::string split_tree_metadata;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &split_tree_metadata));
  ::tensorflow::proto_splitter::ChunkedMessage chunked_message;
  ::tsl::protobuf::TextFormat::ParseFromString(split_tree_metadata,
                                               &chunked_message);

  ::tensorflow::proto_splitter_testdata::StringNode merged_tree;
  TF_ASSERT_OK(Merger::Merge(chunks, chunked_message, &merged_tree));

  const std::string pbtxt_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-tree");

  ::tensorflow::proto_splitter_testdata::StringNode test_proto;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(pbtxt_path, ".pbtxt"), &test_proto));

  ASSERT_THAT(merged_tree, EqualsProto(test_proto));
}

TEST(MergeTest, TestReadGraphDefLotsNodes) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-lots-nodes");
  GraphDef merged_graph_def;
  TF_ASSERT_OK(Merger::Read(path, &merged_graph_def));

  GraphDef test_graph_def;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_graph_def));

  ASSERT_THAT(merged_graph_def, EqualsProto(test_graph_def));
}

TEST(MergeTest, TestReadGraphDefLargeNodes) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-large-nodes");
  GraphDef merged_graph_def;
  TF_ASSERT_OK(Merger::Read(path, &merged_graph_def));

  GraphDef test_graph_def;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_graph_def));

  ASSERT_THAT(merged_graph_def, EqualsProto(test_graph_def));
}

TEST(MergeTest, TestReadGraphDefLargeConstant) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-large-constant");
  GraphDef merged_graph_def;
  TF_ASSERT_OK(Merger::Read(path, &merged_graph_def));

  GraphDef test_graph_def;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_graph_def));

  ASSERT_THAT(merged_graph_def, EqualsProto(test_graph_def));
}

TEST(MergeTest, TestReadManyField) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "many-field");
  ::tensorflow::proto_splitter_testdata::ManyFields merged_many_field;
  TF_ASSERT_OK(Merger::Read(path, &merged_many_field));

  ::tensorflow::proto_splitter_testdata::ManyFields test_many_field;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_many_field));

  ASSERT_THAT(merged_many_field, EqualsProto(test_many_field));
}

TEST(MergeTest, TestReadSavedModel) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-standard");
  SavedModel merged_saved_model;
  TF_ASSERT_OK(Merger::Read(path, &merged_saved_model));

  SavedModel test_saved_model;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_saved_model));

  ASSERT_THAT(merged_saved_model, EqualsProto(test_saved_model));
}

TEST(MergeTest, TestReadChunkedModel) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "chunked_saved_model/chunked_model/saved_model");
  SavedModel merged_saved_model;
  TF_ASSERT_OK(Merger::Read(path, &merged_saved_model));

  SavedModel test_saved_model;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_saved_model));

  ASSERT_THAT(merged_saved_model, EqualsProto(test_saved_model));
}

TEST(MergeTest, TestReadPartial) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "many-field");
  TF_ASSERT_OK_AND_ASSIGN(auto reader, tools::proto_splitter::GetRiegeliReader(
                                           absl::StrCat(path, ".cpb")));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    TF_ASSERT_OK(read_metadata.status());
  }
  ::tensorflow::proto_splitter::ChunkMetadata chunk_metadata =
      read_metadata.value();
  ::tensorflow::proto_splitter::ChunkMetadata partial_chunk_metadata;
  partial_chunk_metadata.mutable_chunks()->CopyFrom(chunk_metadata.chunks());
  partial_chunk_metadata.mutable_message()->set_chunk_index(
      chunk_metadata.message().chunk_index());

  proto_splitter_testdata::ManyFields merged_many_fields;
  TF_ASSERT_OK(
      Merger::ReadPartial(path, partial_chunk_metadata, &merged_many_fields));
  ASSERT_THAT(merged_many_fields, EqualsProto(R"pb(
                map_field_int64 { key: -1345 value: "map_value_-1345" }
              )pb"));
}

TEST(MergeTest, TestReadChunkedFromString) {
  const std::string path =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "chunked_saved_model/chunked_model/saved_model");
  std::string data;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(),
                                     absl::StrCat(path, ".cpb"), &data));

  SavedModel merged_saved_model;
  TF_ASSERT_OK(Merger::ReadChunkedFromString(data, &merged_saved_model));

  SavedModel test_saved_model;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(), absl::StrCat(path, ".pbtxt"), &test_saved_model));

  ASSERT_THAT(merged_saved_model, EqualsProto(test_saved_model));
}

TEST(MergeTest, TestProcessFieldReturnsErrorOnInvalidFieldNumber) {
  ::tensorflow::proto_splitter::ChunkedMessage chunked_message;

  auto* chunk_field = chunked_message.add_chunked_fields();

  auto* tag = chunk_field->add_field_tag();
  tag->set_field(99999);

  chunk_field->mutable_message()->set_chunk_index(0);

  std::vector<std::unique_ptr<tsl::protobuf::Message>> chunks;
  chunks.push_back(
      std::make_unique<::tensorflow::proto_splitter_testdata::ManyFields>());

  ::tensorflow::proto_splitter_testdata::ManyFields merged;
  absl::Status status = Merger::Merge(chunks, chunked_message, &merged);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not found in message descriptor"));
}

}  // namespace

}  // namespace tensorflow::tools::proto_splitter

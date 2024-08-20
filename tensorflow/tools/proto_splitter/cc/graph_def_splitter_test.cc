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
#include "tensorflow/tools/proto_splitter/cc/graph_def_splitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/test_util.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {
namespace {

using ::tensorflow::proto_splitter::ChunkedMessage;

TEST(GraphDefSplitterTest, TestLargeConstant) {
  GraphDef proto;
  const std::string graph_def_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-large-constant.pb");
  int64_t max_size = 500;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSizeLong(), GetMaxSize());
  std::string large_constant_1, large_constant_2;
  const std::variant<std::string, absl::Cord>& tensor_constant_1 =
      proto.node(2).attr().at("value").tensor().tensor_content();
  const std::variant<std::string, absl::Cord>& tensor_constant_2 =
      proto.node(4).attr().at("value").tensor().tensor_content();
  if (std::holds_alternative<std::string>(tensor_constant_1)) {
    large_constant_1 = std::get<std::string>(tensor_constant_1);
  } else {
    absl::CopyCordToString(std::get<absl::Cord>(tensor_constant_1),
                           &large_constant_1);
  }
  if (std::holds_alternative<std::string>(tensor_constant_2)) {
    large_constant_2 = std::get<std::string>(tensor_constant_2);
  } else {
    absl::CopyCordToString(std::get<absl::Cord>(tensor_constant_2),
                           &large_constant_2);
  }

  GraphDefSplitter splitter(&proto);
  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  ChunkedMessage* chunked_message = x.chunked_message;
  ASSERT_NE(chunked_message, nullptr);
  EXPECT_THAT(*chunked_message,
              EqualsProto(R"pb(chunk_index: 0
                               chunked_fields {
                                 field_tag { field: 1 }
                                 field_tag { index: 2 }
                                 field_tag { field: 5 }
                                 field_tag { map_key { s: "value" } }
                                 field_tag { field: 8 }
                                 field_tag { field: 4 }
                                 message { chunk_index: 1 }
                               }
                               chunked_fields {
                                 field_tag { field: 1 }
                                 field_tag { index: 4 }
                                 field_tag { field: 5 }
                                 field_tag { map_key { s: "value" } }
                                 field_tag { field: 8 }
                                 field_tag { field: 4 }
                                 message { chunk_index: 2 }
                               })pb"));

  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);

  EXPECT_THAT((*chunks)[1],
              ::testing::VariantWith<std::string>(large_constant_1));
  EXPECT_THAT((*chunks)[2],
              ::testing::VariantWith<std::string>(large_constant_2));
}

TEST(GraphDefSplitterTest, TestLargeNodes) {
  GraphDef proto;
  const std::string graph_def_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-large-nodes.pb");
  int64_t max_size = 200;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSize(), GetMaxSize());

  // Get the large nodes that should be extracted for later comparison.
  NodeDef node_1 = proto.node(1);
  NodeDef node_2 = proto.node(2);
  NodeDef node_3 = proto.node(3);
  NodeDef node_5 = proto.node(5);

  GraphDefSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  ChunkedMessage* chunked_message = x.chunked_message;
  ASSERT_NE(chunked_message, nullptr);
  EXPECT_THAT(*chunked_message, EqualsProto(R"pb(chunk_index: 0
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 1 }
                                                   message { chunk_index: 1 }
                                                 }
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 2 }
                                                   message { chunk_index: 2 }
                                                 }
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 3 }
                                                   message { chunk_index: 3 }
                                                 }
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 5 }
                                                   message { chunk_index: 4 }
                                                 })pb"));
  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);

  EXPECT_TRUE(std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
      (*chunks)[1]));
  EXPECT_TRUE(std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
      (*chunks)[2]));
  EXPECT_TRUE(std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
      (*chunks)[3]));
  EXPECT_TRUE(std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
      (*chunks)[4]));

  EXPECT_THAT(
      *std::get<std::shared_ptr<tsl::protobuf::Message>>((*chunks)[1]).get(),
      EqualsProto(node_1));
  EXPECT_THAT(
      *std::get<std::shared_ptr<tsl::protobuf::Message>>((*chunks)[2]).get(),
      EqualsProto(node_2));
  EXPECT_THAT(
      *std::get<std::shared_ptr<tsl::protobuf::Message>>((*chunks)[3]).get(),
      EqualsProto(node_3));
  EXPECT_THAT(
      *std::get<std::shared_ptr<tsl::protobuf::Message>>((*chunks)[4]).get(),
      EqualsProto(node_5));
}
TEST(GraphDefSplitterTest, TestLotsNodes) {
  GraphDef proto;
  const std::string graph_def_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "split-lots-nodes.pb");

  // split-lots-nodes.pb has 15 nodes that are 95 or 96 bytes each. The max size
  // is set to "exactly" the size of 5 nodes, but with the extra encoding bytes,
  // only 4 nodes should fit in each chunk. Thus, there should be exactly 4
  // chunks created for all 15 nodes.
  int64_t max_size = 96 * 5;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSize(), GetMaxSize());
  int expected_node_size = proto.node_size();

  GraphDefSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());

  ChunkedMessage* chunked_message = x.chunked_message;
  ASSERT_NE(chunked_message, nullptr);
  EXPECT_THAT(
      *chunked_message,
      EqualsProto(R"pb(chunk_index: 0
                       chunked_fields { message { chunk_index: 1 } }
                       chunked_fields { message { chunk_index: 2 } }
                       chunked_fields { message { chunk_index: 3 } }
                       chunked_fields { message { chunk_index: 4 } })pb"));

  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);

  int actual_node_size = 0;
  for (MessageBytes& chunk : *chunks) {
    GraphDef* message = nullptr;
    if (std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
            chunk)) {
      message = tsl::protobuf::DynamicCastToGenerated<GraphDef>(
          std::get<std::shared_ptr<tsl::protobuf::Message>>(chunk).get());
    } else if (std::holds_alternative<tsl::protobuf::Message*>(chunk)) {
      message = tsl::protobuf::DynamicCastToGenerated<GraphDef>(
          std::get<tsl::protobuf::Message*>(chunk));
    } else {
      EXPECT_FALSE(std::holds_alternative<std::string>(chunk));
    }
    actual_node_size += message->node_size();
  }
  EXPECT_EQ(actual_node_size, expected_node_size);
}

TEST(GraphDefSplitterTest, TestFunctionLotsOfNodes) {
  GraphDef proto;
  const std::string graph_def_path = io::JoinPath(
      testing::TensorFlowSrcRoot(), "tools/proto_splitter/testdata",
      "function-lots-of-nodes.pb");
  int64_t max_size = 500;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSize(), GetMaxSize());

  GraphDefSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);
}

TEST(GraphDefSplitterTest, TestFunctionLargeNodes) {
  GraphDef proto;
  const std::string graph_def_path =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "tools/proto_splitter/testdata", "function-large-nodes.pb");
  int64_t max_size = 200;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSize(), GetMaxSize());

  GraphDefSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);
}

TEST(GraphDefSplitterTest, TestGraphAndFunction) {
  GraphDef proto;
  const std::string graph_def_path = io::JoinPath(
      testing::TensorFlowSrcRoot(), "tools/proto_splitter/testdata",
      "graph-def-and-function.pb");
  int64_t max_size = 200;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           graph_def_path, &proto));
  EXPECT_GE(proto.ByteSize(), GetMaxSize());

  GraphDefSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);
  EXPECT_CHUNK_SIZES(chunks, max_size);

  TF_ASSERT_OK(splitter.Write("/tmp/hoi"));
}

}  // namespace
}  // namespace tools::proto_splitter
}  // namespace tensorflow

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
#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tools/proto_splitter/cc/test_util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {
namespace {

using ::proto_splitter::ChunkedMessage;
using ::proto_splitter::ChunkMetadata;
using ::proto_splitter_testdata::RepeatedRepeatedString;
using ::proto_splitter_testdata::RepeatedString;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using tsl::testing::StatusIs;

// Required in OSS to prevent string to bool conversion in FieldType variant.
using namespace std::string_literals;  // NOLINT

// Splits each string in a RepeatedString into separate chunks.
class RepeatedStringSplitter : public ComposableSplitter {
  friend class ComposableSplitter;

 public:
  using ComposableSplitter::ComposableSplitter;

  absl::Status BuildChunks() override {
    RepeatedString* repeated_string =
        tsl::protobuf::DynamicCastToGenerated<RepeatedString>(message());
    auto strings = repeated_string->strings();

    if (strings.empty()) {
      TF_RETURN_IF_ERROR(SetMessageAsBaseChunk());
      return absl::OkStatus();
    }
    for (int i = 0; i < strings.size(); i++) {
      auto s = std::make_unique<MessageBytes>(strings[i]);
      std::vector<FieldType> fields = {"strings"s, i};
      TF_RETURN_IF_ERROR(AddChunk(std::move(s), &fields));
    }
    return absl::OkStatus();
  }
};

RepeatedString SetUpRepeatedString(std::vector<string> strings) {
  RepeatedString message;
  *message.mutable_strings() = {strings.begin(), strings.end()};
  return message;
}

TEST(RepeatedStringSplitterTest, TestSplitChunks) {
  std::vector<string> strings = {"piece-1", "piece-2", "piece-3"};
  auto message = SetUpRepeatedString(strings);
  RepeatedStringSplitter splitter = RepeatedStringSplitter(&message);
  TF_ASSERT_OK_AND_ASSIGN(auto ret, splitter.Split());
  auto chunks = ret.first;
  auto chunked_message = ret.second;

  for (int i = 0; i < chunks->size(); i++) {
    auto chunk = chunks->at(i);
    EXPECT_TRUE(std::holds_alternative<std::string>(chunk));
    EXPECT_EQ(strings[i], std::get<std::string>(chunk));
  }
  EXPECT_THAT(*chunked_message, EqualsProto(R"pb(chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 0 }
                                                   message { chunk_index: 0 }
                                                 }
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 1 }
                                                   message { chunk_index: 1 }
                                                 }
                                                 chunked_fields {
                                                   field_tag { field: 1 }
                                                   field_tag { index: 2 }
                                                   message { chunk_index: 2 }
                                                 })pb"));

  // Calling split again should return the same chunks/ChunkedMessage.
  TF_ASSERT_OK_AND_ASSIGN(auto ret2, splitter.Split());
  auto chunks2 = ret2.first;
  auto chunked_message2 = ret2.second;
  EXPECT_EQ(chunks2, chunks);
  EXPECT_EQ(chunked_message2, chunked_message);
}

TEST(RepeatedStringSplitterTest, TestWrite) {
  std::vector<string> strings = {"piece-1", "piece-2", "piece-3"};
  auto message = SetUpRepeatedString(strings);
  RepeatedStringSplitter splitter = RepeatedStringSplitter(&message);

  std::string output_prefix = tensorflow::io::GetTempFilename("");
  TF_ASSERT_OK(splitter.Write(output_prefix));
  std::string expected_file = absl::StrCat(output_prefix, ".cpb");

  TF_ASSERT_OK_AND_ASSIGN(auto exists,
                          internal::FileExists(Env::Default(), expected_file));
  EXPECT_TRUE(exists);

  // Look for the last chunk, which should contain a ChunkMetadata proto.
  riegeli::RecordReader<riegeli::FdReader<>> reader(
      (riegeli::FdReader(expected_file)));

  ChunkMetadata chunk_metadata;
  reader.Seek(reader.Size().value());
  reader.SeekBack();
  reader.ReadRecord(chunk_metadata);

  auto chunk_info = chunk_metadata.chunks();
  EXPECT_EQ(chunk_info.size(), strings.size());
  for (int i = 0; i < chunk_info.size(); i++) {
    reader.Seek(chunk_info[i].offset());
    absl::string_view chunk;
    reader.ReadRecord(chunk);
    EXPECT_EQ(strings[i], std::string(chunk));
  }

  EXPECT_THAT(chunk_metadata.message(),
              EqualsProto(R"pb(chunked_fields {
                                 field_tag { field: 1 }
                                 field_tag { index: 0 }
                                 message { chunk_index: 0 }
                               }
                               chunked_fields {
                                 field_tag { field: 1 }
                                 field_tag { index: 1 }
                                 message { chunk_index: 1 }
                               }
                               chunked_fields {
                                 field_tag { field: 1 }
                                 field_tag { index: 2 }
                                 message { chunk_index: 2 }
                               })pb"));
}

TEST(RepeatedStringSplitterTest, TestNoSplit) {
  RepeatedString message;  // No strings
  RepeatedStringSplitter splitter = RepeatedStringSplitter(&message);
  TF_ASSERT_OK_AND_ASSIGN(auto ret, splitter.Split());
  auto chunks = ret.first;
  auto chunked_message = ret.second;

  EXPECT_THAT(*chunks, SizeIs(1));
  EXPECT_THAT(*std::get<tsl::protobuf::Message*>(chunks->at(0)),
              EqualsProto(""));
  EXPECT_THAT(*chunked_message, EqualsProto(R"pb(chunk_index: 0)pb"));
}

// Splits each string in a RepeatedString into separate chunks.
class RepeatedRepeatedStringSplitter : public ComposableSplitter {
 public:
  using ComposableSplitter::ComposableSplitter;

  absl::Status BuildChunks() override {
    TF_RETURN_IF_ERROR(SetMessageAsBaseChunk());
    RepeatedRepeatedString* msg =
        tsl::protobuf::DynamicCastToGenerated<RepeatedRepeatedString>(
            message());
    auto repeated_strings = msg->rs();
    for (int i = 0; i < repeated_strings.size(); i++) {
      std::vector<FieldType> fields = {"rs"s, i};
      auto splitter =
          RepeatedStringSplitter(&repeated_strings[i], this, &fields);
      TF_RETURN_IF_ERROR(splitter.BuildChunks());
    }
    return absl::OkStatus();
  }
};

TEST(ComposableTest, RepeatedRepeatedStringTest) {
  std::vector<string> strings1 = {"piece-1", "piece-2", "piece-3"};
  auto rs1 = SetUpRepeatedString(strings1);
  std::vector<string> strings2 = {"new-strings-1"};
  auto rs2 = SetUpRepeatedString(strings2);
  std::vector<string> strings3 = {"foo-1", "foo-2"};
  auto rs3 = SetUpRepeatedString(strings3);

  std::vector<RepeatedString> rs = {rs1, rs2, rs3};

  RepeatedRepeatedString message;
  message.mutable_rs()->Add(rs.begin(), rs.end());

  RepeatedRepeatedStringSplitter splitter =
      RepeatedRepeatedStringSplitter(&message);
  TF_ASSERT_OK_AND_ASSIGN(auto ret, splitter.Split());
  auto chunks = ret.first;
  auto chunked_message = ret.second;

  std::vector<string> expected_chunks = {"piece-1",       "piece-2", "piece-3",
                                         "new-strings-1", "foo-1",   "foo-2"};

  // RepeatedRepeatedStringSplitter sets the first chunk as the user-provided
  // message, so the expected size is 7.
  EXPECT_THAT(*chunks, SizeIs(7));
  EXPECT_THAT(*std::get<tsl::protobuf::Message*>(chunks->at(0)),
              EqualsProto(message));

  for (int i = 1; i < chunks->size(); i++) {
    auto chunk = chunks->at(i);
    EXPECT_TRUE(std::holds_alternative<std::string>(chunk));
    EXPECT_EQ(expected_chunks[i - 1], std::get<std::string>(chunk));
  }

  // message.rs[2].strings[0] (value = "foo-1") should be the chunk at index 5.
  EXPECT_THAT(chunked_message->chunked_fields()[4],
              EqualsProto(R"pb(field_tag { field: 2 }
                               field_tag { index: 2 }
                               field_tag { field: 1 }
                               field_tag { index: 0 }
                               message { chunk_index: 5 })pb"));
}

TEST(ComposableTest, ChildSplitterTest) {
  std::vector<string> strings1 = {"piece-1", "piece-2", "piece-3"};
  auto message1 = SetUpRepeatedString(strings1);
  RepeatedStringSplitter splitter(&message1);
  std::vector<FieldType> fields = {};

  std::vector<string> strings2 = {"s1", "s2"};
  auto message2 = SetUpRepeatedString(strings2);
  RepeatedStringSplitter child(&message2, &splitter, &fields);

  TF_EXPECT_OK(child.BuildChunks());
  TF_ASSERT_OK_AND_ASSIGN(auto ret, splitter.Split());
  auto chunks = ret.first;
  EXPECT_THAT(*chunks, SizeIs(5));  // Total 5 chunks should be generated.
}

TEST(ComposableTest, ChildSplitterUnimplementedTest) {
  RepeatedString message;
  RepeatedStringSplitter splitter(&message);
  std::vector<FieldType> fields = {};
  RepeatedStringSplitter child(&message, &splitter, &fields);

  EXPECT_THAT(child.Split(), StatusIs(absl::StatusCode::kUnimplemented,
                                      HasSubstr("`Split` function behavior")));
  EXPECT_THAT(child.Write("str"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("`Write` function behavior")));
}

class NoOpSplitter : public ComposableSplitter {
 public:
  using ComposableSplitter::ComposableSplitter;

  absl::Status BuildChunks() override { return absl::OkStatus(); }
};

TEST(NoOpSplitterTest, TestWrite) {
  std::vector<string> strings = {"piece-1", "piece-2", "piece-3"};
  auto message = SetUpRepeatedString(strings);
  NoOpSplitter splitter(&message);

  std::string output_prefix = tensorflow::io::GetTempFilename("");
  TF_ASSERT_OK(splitter.Write(output_prefix));
  std::string expected_file = absl::StrCat(output_prefix, ".pb");

  TF_ASSERT_OK_AND_ASSIGN(auto exists,
                          internal::FileExists(Env::Default(), expected_file));
  EXPECT_TRUE(exists);

  RepeatedString read_message;
  auto status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                            expected_file, &read_message);

  EXPECT_THAT(read_message, EqualsProto(message));
}

}  // namespace
}  // namespace tools::proto_splitter
}  // namespace tensorflow

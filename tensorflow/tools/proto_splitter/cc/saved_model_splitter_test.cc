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
#include "tensorflow/tools/proto_splitter/cc/saved_model_splitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/testdata/test_message.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {
namespace {

// Ensures that all Messages are less than the max size. std::string chunks are
// not limited by the max size, so they are ignored in this check.
#define EXPECT_CHUNK_SIZES(chunks, max_size)                                \
  do {                                                                      \
    for (auto chunk : *chunks) {                                            \
      if (std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(  \
              chunk)) {                                                     \
        EXPECT_LE(std::get<std::shared_ptr<tsl::protobuf::Message>>(chunk)  \
                      ->ByteSizeLong(),                                     \
                  max_size);                                                \
      } else if (std::holds_alternative<tsl::protobuf::Message*>(chunk)) {  \
        EXPECT_LE(std::get<tsl::protobuf::Message*>(chunk)->ByteSizeLong(), \
                  max_size);                                                \
      }                                                                     \
    }                                                                       \
  } while (0)

std::string NonChunkedSavedModel() {
  return io::JoinPath(testing::TensorFlowSrcRoot(), "cc", "saved_model",
                      "testdata", "chunked_saved_model", "non_chunked_model",
                      "saved_model.pb");
}

TEST(SavedModelSplitterTest, TestSplit) {
  SavedModel proto;
  int64_t max_size = 80000;
  DebugSetMaxSize(max_size);

  TF_EXPECT_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                           NonChunkedSavedModel(), &proto));
  EXPECT_GE(proto.ByteSizeLong(), GetMaxSize());

  SavedModelSplitter splitter(&proto);

  TF_ASSERT_OK_AND_ASSIGN(auto x, splitter.Split());
  std::vector<MessageBytes>* chunks = x.chunks;
  ASSERT_NE(chunks, nullptr);

  // Should create a new chunk with the single large constant.
  EXPECT_EQ(2, chunks->size());
  EXPECT_CHUNK_SIZES(chunks, max_size);
}

}  // namespace
}  // namespace tools::proto_splitter
}  // namespace tensorflow

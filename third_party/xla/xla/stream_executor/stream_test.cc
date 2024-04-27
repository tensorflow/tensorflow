/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>

#include "absl/log/check.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

class StreamTest : public ::testing::Test {
 protected:
  std::unique_ptr<StreamExecutor> NewStreamExecutor() {
    Platform* platform = PlatformManager::PlatformWithName("Host").value();
    StreamExecutorConfig config(/*ordinal=*/0);
    return platform->GetUncachedExecutor(config).value();
  }
};

TEST_F(StreamTest, InitOk) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
}

TEST_F(StreamTest, InitWithIntPriorityOk) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream(1));
}

TEST_F(StreamTest, InitWithStreamPriorityOk) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream,
                          executor->CreateStream(StreamPriority::Highest));
}

TEST_F(StreamTest, OneSubStream) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Get and return a sub-stream. Sub-streams are always initialized.
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream1, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream1->ok());
  stream->ReturnSubStream(sub_stream1);

  // Get and return another sub-stream.
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream2, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream2->ok());
  stream->ReturnSubStream(sub_stream1);

  // The underlying sub-streams should be the same, since sub_stream1
  // was returned before we tried to get sub_stream2.
  EXPECT_EQ(sub_stream1, sub_stream2);
}

TEST_F(StreamTest, TwoSubStreams) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Get two sub-streams.
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream1, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream1->ok());
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream2, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream2->ok());

  // The underlying sub-streams should be different, since neither
  // sub-stream has been returned.
  EXPECT_NE(sub_stream1, sub_stream2);

  // Return sub_stream1 and get sub_stream3, which should be the same.
  stream->ReturnSubStream(sub_stream1);
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream3, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream3->ok());
  EXPECT_EQ(sub_stream1, sub_stream3);
  EXPECT_NE(sub_stream2, sub_stream3);

  // Return sub_stream2 and get sub_stream4, which should be the same.
  stream->ReturnSubStream(sub_stream2);
  TF_ASSERT_OK_AND_ASSIGN(Stream * sub_stream4, stream->GetOrCreateSubStream());
  EXPECT_TRUE(sub_stream4->ok());
  EXPECT_EQ(sub_stream2, sub_stream4);
  EXPECT_NE(sub_stream3, sub_stream4);
}

}  // namespace
}  // namespace stream_executor

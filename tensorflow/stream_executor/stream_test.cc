/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/stream_executor.h"

#include "tensorflow/core/platform/test.h"

namespace stream_executor {
namespace {

class StreamTest : public ::testing::Test {
 protected:
  std::unique_ptr<StreamExecutor> NewStreamExecutor() {
    Platform* platform =
        MultiPlatformManager::PlatformWithName("Host").ConsumeValueOrDie();
    StreamExecutorConfig config(/*ordinal=*/0);
    return platform->GetUncachedExecutor(config).ConsumeValueOrDie();
  }
};

TEST_F(StreamTest, NoInitNotOk) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  Stream stream(executor.get());
  EXPECT_FALSE(stream.ok());
}

TEST_F(StreamTest, InitOk) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  Stream stream(executor.get());
  stream.Init();
  EXPECT_TRUE(stream.ok());
}

TEST_F(StreamTest, OneSubStream) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  Stream stream(executor.get());
  stream.Init();
  EXPECT_TRUE(stream.ok());

  // Get and return a sub-stream. Sub-streams are always initialized.
  Stream* sub_stream1 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream1->ok());
  stream.ReturnSubStream(sub_stream1);

  // Get and return another sub-stream.
  Stream* sub_stream2 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream2->ok());
  stream.ReturnSubStream(sub_stream1);

  // The underlying sub-streams should be the same, since sub_stream1
  // was returned before we tried to get sub_stream2.
  EXPECT_EQ(sub_stream1, sub_stream2);
}

TEST_F(StreamTest, TwoSubStreams) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  Stream stream(executor.get());
  stream.Init();
  EXPECT_TRUE(stream.ok());

  // Get two sub-streams.
  Stream* sub_stream1 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream1->ok());
  Stream* sub_stream2 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream2->ok());

  // The underlying sub-streams should be different, since neither
  // sub-stream has been returned.
  EXPECT_NE(sub_stream1, sub_stream2);

  // Return sub_stream1 and get sub_stream3, which should be the same.
  stream.ReturnSubStream(sub_stream1);
  Stream* sub_stream3 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream3->ok());
  EXPECT_EQ(sub_stream1, sub_stream3);
  EXPECT_NE(sub_stream2, sub_stream3);

  // Return sub_stream2 and get sub_stream4, which should be the same.
  stream.ReturnSubStream(sub_stream2);
  Stream* sub_stream4 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream4->ok());
  EXPECT_EQ(sub_stream2, sub_stream4);
  EXPECT_NE(sub_stream3, sub_stream4);
}

TEST_F(StreamTest, FailedSubStreamNotReused) {
  std::unique_ptr<StreamExecutor> executor = NewStreamExecutor();
  Stream stream(executor.get());
  stream.Init();
  EXPECT_TRUE(stream.ok());

  // Get a sub-stream.
  Stream* sub_stream1 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream1->ok());

  // Force an error on the stream; here we call a method that requires
  // DNN support, which we know the Host platform doesn't support.
  sub_stream1->ThenDepthConcatenate({}, {}, nullptr);
  EXPECT_FALSE(sub_stream1->ok());

  // Return sub_stream1 and get sub_stream2.
  stream.ReturnSubStream(sub_stream1);
  Stream* sub_stream2 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream2->ok());

  // The underlying streams should be different. They would have been
  // the same, but since we forced an error on sub_stream1, it will
  // not be re-used. Sadly we can't just check:
  //   EXPECT_NE(sub_stream1, sub_stream2);
  //
  // The above should hold logically, but it may fail if the new
  // stream instance allocated for sub_stream2 happens to reside in
  // the same memory address as sub_stream1.
  //
  // The check that sub_stream2->ok() serves as a good-enough check.

  // Return sub_stream2 and get sub_stream3. The previous error on
  // sub_stream1 has no effect on these streams, and they are the
  // same.
  stream.ReturnSubStream(sub_stream2);
  Stream* sub_stream3 = stream.GetOrCreateSubStream();
  EXPECT_TRUE(sub_stream3->ok());
  EXPECT_EQ(sub_stream2, sub_stream3);
}

}  // namespace
}  // namespace stream_executor

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

#include "tensorflow/compiler/xla/service/stream_pool.h"

#include <memory>

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace {

class StreamPoolTest : public ::testing::Test {
 protected:
  std::unique_ptr<se::StreamExecutor> NewStreamExecutor() {
    se::Platform* platform =
        se::MultiPlatformManager::PlatformWithName("Host").ConsumeValueOrDie();
    se::StreamExecutorConfig config(/*ordinal=*/0);
    return platform->GetUncachedExecutor(config).ConsumeValueOrDie();
  }
};

TEST_F(StreamPoolTest, EmptyPool) { StreamPool pool; }

TEST_F(StreamPoolTest, OneStreamPool) {
  std::unique_ptr<se::StreamExecutor> executor = NewStreamExecutor();
  StreamPool pool;

  // Borrow and return a stream.
  StreamPool::Ptr stream1 = pool.BorrowStream(executor.get());
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  stream1 = nullptr;

  // Borrow and return another stream.
  StreamPool::Ptr stream2 = pool.BorrowStream(executor.get());
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());
  stream2 = nullptr;

  // The underlying streams should be the same, since stream1 was the
  // only stream available in the pool when stream2 was borrowed.
  EXPECT_EQ(stream1_ptr, stream2_ptr);
}

TEST_F(StreamPoolTest, TwoStreamPool) {
  std::unique_ptr<se::StreamExecutor> executor = NewStreamExecutor();
  StreamPool pool;

  // Borrow two streams.
  StreamPool::Ptr stream1 = pool.BorrowStream(executor.get());
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  StreamPool::Ptr stream2 = pool.BorrowStream(executor.get());
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());

  // The underlying streams should be different, since we haven't
  // returned either of them yet.
  EXPECT_NE(stream1_ptr, stream2_ptr);

  // Return stream1 and borrow stream3.
  stream1 = nullptr;
  StreamPool::Ptr stream3 = pool.BorrowStream(executor.get());
  se::Stream* stream3_ptr = stream3.get();
  EXPECT_TRUE(stream3->ok());

  // stream1 and stream3 should be the same.
  EXPECT_EQ(stream1_ptr, stream3_ptr);
  EXPECT_NE(stream2_ptr, stream3_ptr);

  // Return stream2, and borrow stream4.
  stream2 = nullptr;
  StreamPool::Ptr stream4 = pool.BorrowStream(executor.get());
  se::Stream* stream4_ptr = stream4.get();
  EXPECT_TRUE(stream4->ok());

  // Stream2 and stream4 should be the same.
  EXPECT_EQ(stream2_ptr, stream4_ptr);
  EXPECT_NE(stream3_ptr, stream4_ptr);
}

TEST_F(StreamPoolTest, BadStreamDiscarded) {
  std::unique_ptr<se::StreamExecutor> executor = NewStreamExecutor();
  StreamPool pool;

  // Borrow a stream.
  StreamPool::Ptr stream1 = pool.BorrowStream(executor.get());
  EXPECT_TRUE(stream1->ok());

  // Force an error on the stream; here we call a method that requires
  // DNN support, which we know the Host platform doesn't support.
  stream1->ThenDepthConcatenate({}, {}, nullptr);
  EXPECT_FALSE(stream1->ok());

  // Return stream1 and borrow stream2.
  stream1 = nullptr;
  StreamPool::Ptr stream2 = pool.BorrowStream(executor.get());
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());

  // The underlying streams should be different. They would have been
  // the same, but since we forced an error on stream1, it cannot be
  // put back into the pool. Sadly we can't just check:
  //    EXPECT_NE(stream1_ptr, stream2_ptr);
  //
  // The above should hold logically, but it may fail if the new
  // stream instance allocated for stream2 happens to reside in the
  // same memory address as stream1, which has been deleted.
  //
  // The check that stream2->ok() serves as a good-enough check.

  // Return stream2 and borrow stream3. The previous error on stream1
  // has no effect on these streams, and they are the same.
  stream2 = nullptr;
  StreamPool::Ptr stream3 = pool.BorrowStream(executor.get());
  se::Stream* stream3_ptr = stream3.get();
  EXPECT_TRUE(stream3->ok());
  EXPECT_EQ(stream2_ptr, stream3_ptr);
}

TEST_F(StreamPoolTest, BadStreamAfterReturnDiscarded) {
  std::unique_ptr<se::StreamExecutor> executor = NewStreamExecutor();
  StreamPool pool;

  // Borrow a stream.
  StreamPool::Ptr stream1 = pool.BorrowStream(executor.get());
  EXPECT_TRUE(stream1->ok());

  // Return the stream, but hold a handle to it.
  se::Stream* stream1_ptr = stream1.get();
  stream1 = nullptr;

  // Now stream1 is back in the pool, force an error on the stream. Here we call
  // a method that requires DNN support, which we know the Host platform doesn't
  // support.
  stream1_ptr->ThenDepthConcatenate({}, {}, nullptr);
  EXPECT_FALSE(stream1_ptr->ok());

  // Borrow stream2.
  StreamPool::Ptr stream2 = pool.BorrowStream(executor.get());
  EXPECT_TRUE(stream2->ok());

  // The underlying streams should be different. They would have been
  // the same, but since we forced an error on stream1, it cannot be
  // put back into the pool. Sadly we can't just check:
  //    EXPECT_NE(stream1_ptr, stream2_ptr);
  //
  // The above should hold logically, but it may fail if the new
  // stream instance allocated for stream2 happens to reside in the
  // same memory address as stream1, which has been deleted.
  //
  // The check that stream2->ok() serves as a good-enough check.
}

}  // namespace
}  // namespace xla

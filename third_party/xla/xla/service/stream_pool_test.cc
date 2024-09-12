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

#include "xla/service/stream_pool.h"

#include <memory>

#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/test_helpers.h"

namespace xla {
namespace {

class StreamPoolTest : public ::testing::Test {
 protected:
  se::StreamExecutor* NewStreamExecutor() {
    se::Platform* platform =
        se::PlatformManager::PlatformWithName("Host").value();
    return platform->ExecutorForDevice(/*ordinal=*/0).value();
  }
};

TEST_F(StreamPoolTest, EmptyPool) {
  se::StreamExecutor* executor = NewStreamExecutor();
  StreamPool pool(executor);
}

TEST_F(StreamPoolTest, OneStreamPool) {
  se::StreamExecutor* executor = NewStreamExecutor();
  StreamPool pool(executor);

  // Borrow and return a stream.
  StreamPool::Ptr stream1 = pool.BorrowStream();
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  stream1 = nullptr;

  // Borrow and return another stream.
  StreamPool::Ptr stream2 = pool.BorrowStream();
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());
  stream2 = nullptr;

  // The underlying streams should be the same, since stream1 was the
  // only stream available in the pool when stream2 was borrowed.
  EXPECT_EQ(stream1_ptr, stream2_ptr);
}

TEST_F(StreamPoolTest, TwoStreamPool) {
  se::StreamExecutor* executor = NewStreamExecutor();
  StreamPool pool(executor);

  // Borrow two streams.
  StreamPool::Ptr stream1 = pool.BorrowStream();
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  StreamPool::Ptr stream2 = pool.BorrowStream();
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());

  // The underlying streams should be different, since we haven't
  // returned either of them yet.
  EXPECT_NE(stream1_ptr, stream2_ptr);

  // Return stream1 and borrow stream3.
  stream1 = nullptr;
  StreamPool::Ptr stream3 = pool.BorrowStream();
  se::Stream* stream3_ptr = stream3.get();
  EXPECT_TRUE(stream3->ok());

  // stream1 and stream3 should be the same.
  EXPECT_EQ(stream1_ptr, stream3_ptr);
  EXPECT_NE(stream2_ptr, stream3_ptr);

  // Return stream2, and borrow stream4.
  stream2 = nullptr;
  StreamPool::Ptr stream4 = pool.BorrowStream();
  se::Stream* stream4_ptr = stream4.get();
  EXPECT_TRUE(stream4->ok());

  // Stream2 and stream4 should be the same.
  EXPECT_EQ(stream2_ptr, stream4_ptr);
  EXPECT_NE(stream3_ptr, stream4_ptr);
}

}  // namespace
}  // namespace xla

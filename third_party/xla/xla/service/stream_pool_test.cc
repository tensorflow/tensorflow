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
#include <string>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace {

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

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
  ASSERT_OK_AND_ASSIGN(auto stream1, pool.BorrowStream());
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  stream1 = nullptr;

  // Borrow and return another stream.
  ASSERT_OK_AND_ASSIGN(auto stream2, pool.BorrowStream());
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
  ASSERT_OK_AND_ASSIGN(auto stream1, pool.BorrowStream());
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_TRUE(stream1->ok());
  ASSERT_OK_AND_ASSIGN(auto stream2, pool.BorrowStream());
  se::Stream* stream2_ptr = stream2.get();
  EXPECT_TRUE(stream2->ok());

  // The underlying streams should be different, since we haven't
  // returned either of them yet.
  EXPECT_NE(stream1_ptr, stream2_ptr);

  // Return stream1 and borrow stream3.
  stream1 = nullptr;
  ASSERT_OK_AND_ASSIGN(auto stream3, pool.BorrowStream());
  se::Stream* stream3_ptr = stream3.get();
  EXPECT_TRUE(stream3->ok());

  // stream1 and stream3 should be the same.
  EXPECT_EQ(stream1_ptr, stream3_ptr);
  EXPECT_NE(stream2_ptr, stream3_ptr);

  // Return stream2, and borrow stream4.
  stream2 = nullptr;
  ASSERT_OK_AND_ASSIGN(auto stream4, pool.BorrowStream());
  se::Stream* stream4_ptr = stream4.get();
  EXPECT_TRUE(stream4->ok());

  // Stream2 and stream4 should be the same.
  EXPECT_EQ(stream2_ptr, stream4_ptr);
  EXPECT_NE(stream3_ptr, stream4_ptr);
}

TEST_F(StreamPoolTest, CustomFactoryPreservesLeaseExclusivity) {
  se::StreamExecutor* executor = NewStreamExecutor();
  int created_streams = 0;
  StreamPool pool([&](se::StreamPriority priority) {
    ++created_streams;
    return executor->CreateStream(priority);
  });

  ASSERT_OK_AND_ASSIGN(auto stream1, pool.BorrowStream());
  ASSERT_OK_AND_ASSIGN(auto stream2, pool.BorrowStream());
  se::Stream* stream1_ptr = stream1.get();
  EXPECT_NE(stream1.get(), stream2.get());
  EXPECT_EQ(created_streams, 2);

  stream1 = nullptr;
  ASSERT_OK_AND_ASSIGN(auto stream3, pool.BorrowStream());
  EXPECT_EQ(stream3.get(), stream1_ptr);
  EXPECT_EQ(created_streams, 2);
}

TEST_F(StreamPoolTest, NamesUnnamedFactoryStreams) {
  se::StreamExecutor* executor = NewStreamExecutor();
  StreamPool pool(executor);

  ASSERT_OK_AND_ASSIGN(auto stream, pool.BorrowStream());
  EXPECT_EQ(
      stream->GetName(),
      se::StreamPriorityToString(se::StreamPriority::Default) + " pool stream");
}

TEST_F(StreamPoolTest, AppendsPoolLabelToFactoryAssignedName) {
  se::StreamExecutor* executor = NewStreamExecutor();
  StreamPool pool([&](se::StreamPriority priority) {
    auto stream = executor->CreateStream(priority);
    if (stream.ok()) {
      (*stream)->SetName("factory stream");
    }
    return stream;
  });

  ASSERT_OK_AND_ASSIGN(auto stream, pool.BorrowStream());
  EXPECT_EQ(stream->GetName(),
            "factory stream " +
                se::StreamPriorityToString(se::StreamPriority::Default) +
                " pool stream");
}

TEST_F(StreamPoolTest, ReusesStreamsUnderBorrowedPriority) {
  se::StreamExecutor* executor = NewStreamExecutor();
  int created_streams = 0;
  StreamPool pool([&](se::StreamPriority) {
    ++created_streams;
    return executor->CreateStream(se::StreamPriority::Default);
  });

  ASSERT_OK_AND_ASSIGN(auto stream1,
                       pool.BorrowStream(se::StreamPriority::Highest));
  se::Stream* stream1_ptr = stream1.get();
  stream1 = nullptr;

  ASSERT_OK_AND_ASSIGN(auto stream2,
                       pool.BorrowStream(se::StreamPriority::Highest));
  EXPECT_EQ(stream2.get(), stream1_ptr);
  EXPECT_EQ(created_streams, 1);

  ASSERT_OK_AND_ASSIGN(auto stream3,
                       pool.BorrowStream(se::StreamPriority::Default));
  EXPECT_NE(stream3.get(), stream1_ptr);
  EXPECT_EQ(created_streams, 2);
}

TEST_F(StreamPoolTest, ReturnsStreamsWithIntegerPriority) {
  const std::string empty_name;
  int created_streams = 0;
  StreamPool pool(
      [&](se::StreamPriority) -> absl::StatusOr<std::unique_ptr<se::Stream>> {
        ++created_streams;
        auto stream = std::make_unique<NiceMock<se::MockStream>>();
        ON_CALL(*stream, ok()).WillByDefault(Return(true));
        ON_CALL(*stream, priority())
            .WillByDefault(Return(std::variant<se::StreamPriority, int>(0)));
        ON_CALL(*stream, GetName()).WillByDefault(ReturnRef(empty_name));
        return std::unique_ptr<se::Stream>(std::move(stream));
      });

  ASSERT_OK_AND_ASSIGN(auto stream1,
                       pool.BorrowStream(se::StreamPriority::Highest));
  se::Stream* stream1_ptr = stream1.get();
  stream1 = nullptr;

  ASSERT_OK_AND_ASSIGN(auto stream2,
                       pool.BorrowStream(se::StreamPriority::Highest));
  EXPECT_EQ(stream2.get(), stream1_ptr);
  EXPECT_EQ(created_streams, 1);
}

}  // namespace
}  // namespace xla

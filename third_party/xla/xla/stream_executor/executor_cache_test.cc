/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/executor_cache.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace {

TEST(ExecutorCacheTest, GetOnEmptyCacheFails) {
  ExecutorCache cache;
  StreamExecutorConfig config;
  config.ordinal = 0;
  EXPECT_FALSE(cache.Get(config).ok());
}

TEST(ExecutorCacheTest, GetViaStreamOnEmptyCacheFails) {
  ExecutorCache cache;
  StreamExecutorConfig config;
  config.ordinal = 0;
  config.gpu_stream = reinterpret_cast<void *>(0x1234);
  EXPECT_FALSE(cache.Get(config).ok());
}

TEST(ExecutorCacheTest, GetOrCreateConstructsAndRepeatedlyReturns) {
  ExecutorCache cache;
  StreamExecutorConfig config;
  config.ordinal = 0;
  StreamExecutor *created = nullptr;
  auto factory = [&created]() {
    auto executor = std::make_unique<MockStreamExecutor>();
    created = executor.get();
    return executor;
  };
  TF_ASSERT_OK_AND_ASSIGN(auto executor, cache.GetOrCreate(config, factory));
  EXPECT_EQ(executor, created);
  TF_ASSERT_OK_AND_ASSIGN(auto found, cache.GetOrCreate(config, factory));
  EXPECT_EQ(found, created);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.Get(config));
  EXPECT_EQ(found, created);
}

TEST(ExecutorCacheTest, GetViaStreamFailsIfNotFound) {
  ExecutorCache cache;
  StreamExecutorConfig config;
  config.ordinal = 0;
  StreamExecutor *created = nullptr;
  void *expected_stream = reinterpret_cast<void *>(0x1234);
  auto factory = [&created, &expected_stream]() {
    auto executor = std::make_unique<MockStreamExecutor>();
    EXPECT_CALL(*executor, FindAllocatedStream(expected_stream))
        .WillRepeatedly(testing::Return(nullptr));
    created = executor.get();
    return executor;
  };

  // Create the executor.
  TF_ASSERT_OK_AND_ASSIGN(auto executor, cache.GetOrCreate(config, factory));
  EXPECT_EQ(executor, created);
  // Now look for the expected stream, and don't expected to find it.
  config.gpu_stream = expected_stream;
  EXPECT_FALSE(cache.Get(config).ok());
}

TEST(ExecutorCacheTest, GetViaStreamWorksOnSecondStream) {
  ExecutorCache cache;
  StreamExecutorConfig config;
  config.ordinal = 0;
  StreamExecutor *created = nullptr;
  Stream *expected_stream = reinterpret_cast<Stream *>(0x1234);

  // Create a factory that will make the second StreamExecutor find the
  // expected_stream.
  auto factory = [&created, &expected_stream]() {
    static int count = 0;
    auto executor = std::make_unique<MockStreamExecutor>();
    if (count != 1) {
      EXPECT_CALL(*executor, FindAllocatedStream(expected_stream))
          .WillRepeatedly(testing::Return(nullptr));
    } else {
      created = executor.get();
      EXPECT_CALL(*executor, FindAllocatedStream(expected_stream))
          .WillRepeatedly(testing::Invoke(
              [expected_stream](void *stream) { return expected_stream; }));
    }
    ++count;
    return executor;
  };

  // Create four executors.
  std::vector<StreamExecutor *> created_executors;
  for (int i = 0; i < 4; ++i) {
    config.ordinal = i;
    TF_ASSERT_OK_AND_ASSIGN(auto executor, cache.GetOrCreate(config, factory));
    EXPECT_NE(executor, nullptr);
    created_executors.push_back(executor);
  }
  EXPECT_EQ(created_executors.size(), 4);
  // Now look for the expected stream, and expect to find it on the second
  // stream.
  config.gpu_stream = expected_stream;
  TF_ASSERT_OK_AND_ASSIGN(auto found, cache.Get(config));
  EXPECT_EQ(found, created);
}

}  // namespace
}  // namespace stream_executor

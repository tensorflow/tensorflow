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

#include "absl/log/log.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(ExecutorCacheTest, GetOnEmptyCacheFails) {
  ExecutorCache cache;
  EXPECT_FALSE(cache.Get(0).ok());
}

TEST(ExecutorCacheTest, GetReturnsExpectedExecutor) {
  ExecutorCache cache;
  StreamExecutor *executor0 = nullptr;
  StreamExecutor *executor1 = nullptr;
  auto factory = [&executor0, &executor1]() {
    auto executor = std::make_unique<MockStreamExecutor>();
    if (executor0 == nullptr) {
      executor0 = executor.get();
    } else if (executor1 == nullptr) {
      executor1 = executor.get();
    } else {
      LOG(FATAL) << "Bad call to factory.";
    }
    return executor;
  };
  TF_ASSERT_OK_AND_ASSIGN(auto found, cache.GetOrCreate(0, factory));
  EXPECT_EQ(found, executor0);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.GetOrCreate(1, factory));
  EXPECT_EQ(found, executor1);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.GetOrCreate(0, factory));
  EXPECT_EQ(found, executor0);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.GetOrCreate(1, factory));
  EXPECT_EQ(found, executor1);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.Get(0));
  EXPECT_EQ(found, executor0);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.Get(1));
  EXPECT_EQ(found, executor1);
}

}  // namespace
}  // namespace stream_executor

/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/async_execution.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

// A minimal thunk for testing AsyncExecution.
class TestThunk : public Thunk {
 public:
  explicit TestThunk(Thunk::ThunkInfo thunk_info)
      : Thunk(Thunk::kAllReduce, std::move(thunk_info)) {}
  absl::Status ExecuteOnStream(const ExecuteParams&) override {
    return absl::OkStatus();
  }
};

static absl::StatusOr<se::StreamExecutor*> CreateExecutor() {
  ASSIGN_OR_RETURN(std::string platform_name,
                   xla::PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName(platform_name));
  return platform->ExecutorForDevice(0);
}

TEST(AsyncExecutionTest, InitializeStartDone) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, CreateExecutor());

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto async_stream, executor->CreateStream());

  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = ThunkId(1);
  thunk_info.profile_annotation = "test-thunk";
  TestThunk thunk(thunk_info);

  AsyncExecution async_execution(&thunk);
  Thunk::ExecutionScopedState state;

  // Initialize creates an event in the execution scoped state.
  ASSERT_OK(async_execution.Initialize(&state, executor));

  {  // Start creates a dependency from stream to async_stream.
    ASSERT_OK_AND_ASSIGN(auto guard, async_execution.Start(&state, stream.get(),
                                                           async_stream.get()));
  }  // ExecutionGuard destructor records the completion event on async_stream.

  // Done waits for the event recorded by the guard.
  ASSERT_OK(async_execution.Done(&state, stream.get()));
}

TEST(AsyncExecutionTest, DoneWithoutStartFails) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, CreateExecutor());

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = ThunkId(1);
  thunk_info.profile_annotation = "test-thunk";
  TestThunk thunk(thunk_info);

  AsyncExecution async_execution(&thunk);
  Thunk::ExecutionScopedState state;

  // Done without Initialize should fail because event is not in state.
  EXPECT_THAT(async_execution.Done(&state, stream.get()),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST(AsyncExecutionTest, DoubleInitializeFails) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, CreateExecutor());

  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = ThunkId(1);
  thunk_info.profile_annotation = "test-thunk";
  TestThunk thunk(thunk_info);

  AsyncExecution async_execution(&thunk);
  Thunk::ExecutionScopedState state;

  ASSERT_OK(async_execution.Initialize(&state, executor));
  EXPECT_THAT(async_execution.Initialize(&state, executor),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla::gpu

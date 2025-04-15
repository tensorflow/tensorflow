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

#include "xla/backends/gpu/runtime/custom_call_thunk.h"

#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  TF_ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  TF_ASSIGN_OR_RETURN(auto* platform,
                      se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

TEST(CustomCallThunkTest, SimpleCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  bool was_called = false;

  CustomCallThunk::CustomCallTarget target =
      [&](se::Stream* stream_in_callback, void** args, const char* target_name,
          size_t num_args, XlaCustomCallStatus* status) {
        was_called = true;
        EXPECT_THAT(stream_in_callback, ::testing::Eq(stream.get()));
      };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CustomCallThunk::Create(Thunk::ThunkInfo(), "target_name",
                                          target, {}, {}, ""));
  se::StreamExecutorMemoryAllocator allocator(executor);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(Thunk::ExecuteParams(params)),
              ::tsl::testing::IsOk());
  EXPECT_TRUE(was_called);
}

TEST(CustomCallThunkTest, CustomCallOnCustomStream) {
  // Whitebox test to ensure that custom calls respect execution_stream_id
  // assignments.
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> extra_stream,
                          executor->CreateStream());
  // Setup the additional streams.
  Thunk::ExecutionStreamIdMap additional_compute_streams = {};
  additional_compute_streams[ExecutionStreamId(1)] = extra_stream.get();
  se::StreamExecutorMemoryAllocator allocator(executor);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr, additional_compute_streams);

  CustomCallThunk::CustomCallTarget target =
      [&](se::Stream* stream_in_callback, void** args, const char* target_name,
          size_t num_args, XlaCustomCallStatus* status) {
        // We should be launching on the extra stream and not the default one.
        EXPECT_THAT(stream_in_callback, ::testing::Eq(extra_stream.get()));
      };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CustomCallThunk::Create(Thunk::ThunkInfo(), "target_name",
                                          target, {}, {}, ""));
  // Setting this tells the thunk to dispatch on one of the additional streams.
  thunk->set_execution_stream_id(ExecutionStreamId(1));
  EXPECT_THAT(thunk->ExecuteOnStream(Thunk::ExecuteParams(params)),
              ::tsl::testing::IsOk());
}

}  // namespace
}  // namespace xla::gpu

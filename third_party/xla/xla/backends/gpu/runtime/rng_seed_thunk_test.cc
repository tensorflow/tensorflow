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

#include "xla/backends/gpu/runtime/rng_seed_thunk.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(auto* platform, se::PlatformManager::PlatformWithName(
                                       absl::AsciiStrToUpper(name)));
  return platform->ExecutorForDevice(0);
}

TEST(RngSeedThunkTest, ExecuteExplicitSeed) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint64_t> dest = executor->AllocateArray<uint64_t>(1, 0);
  ASSERT_OK(stream->MemZero(&dest, sizeof(uint64_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint64_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint64_t));

  RngSeedThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  const int kExplicitSeed = 42;
  execute_params.rng_seed = kExplicitSeed;

  ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  ASSERT_OK(stream->BlockHostUntilDone());

  uint64_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint64_t)));
  EXPECT_EQ(result, kExplicitSeed);
}

TEST(RngSeedThunkTest, ExecuteRandomSeed) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint64_t> dest = executor->AllocateArray<uint64_t>(1, 0);
  ASSERT_OK(stream->MemZero(&dest, sizeof(uint64_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint64_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint64_t));

  RngSeedThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  execute_params.rng_seed = 0;  // Triggers internal generator.

  ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  ASSERT_OK(stream->BlockHostUntilDone());

  uint64_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint64_t)));
  EXPECT_NE(result, 0);
}

TEST(RngSeedThunkTest, BufferUses) {
  BufferAllocation alloc(/*index=*/0, sizeof(uint64_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint64_t));
  RngSeedThunk thunk(Thunk::ThunkInfo{}, slice);

  auto uses = thunk.buffer_uses();
  ASSERT_EQ(uses.size(), 1);
  EXPECT_EQ(uses[0].access(), BufferUse::MemoryAccess::kWrite);
}

}  // namespace
}  // namespace xla::gpu

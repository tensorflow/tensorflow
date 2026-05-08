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

#include "xla/backends/gpu/runtime/async_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

static absl::StatusOr<se::StreamExecutor*> CreateExecutor() {
  ASSIGN_OR_RETURN(std::string platform_name,
                   xla::PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName(platform_name));
  return platform->ExecutorForDevice(0);
}

// Test that 4 async start thunks each memset a quarter of a buffer on separate
// streams, and 4 async done thunks synchronize back to the main stream.
TEST(AsyncThunkTest, ConcurrentMemsets) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int32_t kNumChunks = 4;
  constexpr int32_t kChunkLength = 256;
  constexpr int32_t kTotalLength = kNumChunks * kChunkLength;
  constexpr int64_t kChunkBytes = kChunkLength * sizeof(int32_t);
  constexpr int64_t kTotalBytes = kTotalLength * sizeof(int32_t);

  // Create 4 async streams, one per chunk.
  std::vector<std::unique_ptr<se::Stream>> async_streams;
  std::vector<se::Stream*> additional_streams;
  for (int i = 0; i < kNumChunks; ++i) {
    ASSERT_OK_AND_ASSIGN(async_streams.emplace_back(),
                         executor->CreateStream());
    additional_streams.push_back(async_streams.back().get());
  }

  // Allocate device buffer and zero it.
  se::StreamExecutorAddressAllocator allocator(executor);
  ASSERT_OK_AND_ASSIGN(
      auto buf, allocator.Allocate(executor->device_ordinal(), kTotalBytes));
  ASSERT_OK(stream->MemZero(buf.ptr(), kTotalBytes));

  // Single buffer allocation covering the entire buffer.
  BufferAllocation alloc(/*index=*/0, kTotalBytes, /*color=*/0);

  // Build a thunk sequence: 4 async starts followed by 4 async dones. Each
  // async start contains a nested Memset32BitValueThunk that writes a distinct
  // value to a different quarter of the buffer.
  ThunkSequence thunks;

  for (int i = 0; i < kNumChunks; ++i) {
    BufferAllocation::Slice slice(&alloc, i * kChunkBytes, kChunkBytes);
    uint32_t value = static_cast<uint32_t>(100 + i);

    ThunkSequence nested;
    nested.push_back(std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(),
                                                             value, slice));

    Thunk::ThunkInfo start_info;
    start_info.profile_annotation = absl::StrCat("start#", i);
    start_info.thunk_id = ThunkId(i + 1);

    thunks.push_back(std::make_unique<AsyncStartThunk>(
        std::move(start_info), ComputationStreamId(i), std::move(nested)));
  }

  for (int i = 0; i < kNumChunks; ++i) {
    auto* start = static_cast<AsyncStartThunk*>(thunks[i].get());
    thunks.push_back(std::make_unique<AsyncDoneThunk>(
        Thunk::ThunkInfo(), start->async_execution()));
  }

  // Build a ThunkExecutor to execute all thunks.
  ThunkExecutor thunk_executor(std::move(thunks));

  // Set up execute params.
  BufferAllocations allocations({buf.cref()}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, additional_streams, &state);

  // Initialize and execute all thunks via ThunkExecutor.
  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.execution_scoped_state = &state;
  ASSERT_OK(thunk_executor.Initialize(init_params));
  ASSERT_OK(thunk_executor.ExecuteOnStream(params));

  // Wait for everything to complete.
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy result back to host and verify.
  std::vector<int32_t> result(kTotalLength, 0);
  ASSERT_OK(stream->Memcpy(result.data(), buf.cref(), kTotalBytes));

  for (int i = 0; i < kNumChunks; ++i) {
    int32_t expected = 100 + i;
    for (int j = 0; j < kChunkLength; ++j) {
      ASSERT_EQ(result[i * kChunkLength + j], expected)
          << "Mismatch at chunk " << i << " element " << j;
    }
  }
}

// Test that AsyncDoneThunk::Record() creates an empty command buffer node and
// is a no-op on update.
TEST(AsyncThunkTest, AsyncDoneRecordCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Create a paired start/done to obtain a valid AsyncExecution.
  AsyncStartThunk start(Thunk::ThunkInfo(), ComputationStreamId(0),
                        ThunkSequence());
  AsyncDoneThunk done(Thunk::ThunkInfo(), start.async_execution());

  // Set up minimal execute params (not used by AsyncDoneThunk::Record()).
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState thunk_state;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, /*additional_streams=*/{}, &thunk_state);

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  CommandStateManager cmd_state;
  Command::RecordParams record_params = {cmd_state};

  // RecordCreate: should produce a non-null empty command node.
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       done.Record(execute_params, record_params,
                                   Command::RecordCreate{/*.dependencies=*/{}},
                                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  // RecordUpdate: empty command is not updatable; same pointer returned.
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* updated_cmd,
                       done.Record(execute_params, record_params,
                                   Command::RecordUpdate{/*.command=*/cmd},
                                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);
}

}  // namespace
}  // namespace xla::gpu

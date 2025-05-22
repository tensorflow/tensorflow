/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_cmd.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {

using BufferUseVector = CommandBufferCmd::BufferUseVector;
using MemoryAccess = BufferUse::MemoryAccess;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Give a short alias to execution thread.
static constexpr auto s0 = ExecutionStreamId(0);

// Give a short alias to synchronization mode.
static constexpr auto serialize =
    CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

// A command buffer cmd for testing automatic barriers insertion by the command
// buffer cmd commands. We never execute this command, we need it only to pass
// buffer usage vector to the command buffer cmd commands.
struct TestOnlyCommandBufferCmd : public CommandBufferCmd {
  TestOnlyCommandBufferCmd(ExecutionStreamId execution_stream_id,
                           BufferUseVector buffer_usage)
      : CommandBufferCmd(CommandBufferCmdType::kUnknownCmd,
                         execution_stream_id),
        buffer_usage(buffer_usage) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction,
      se::CommandBuffer*) override {
    return nullptr;
  }

  BufferUseVector buffers() const override { return buffer_usage; }

  BufferUseVector buffer_usage;
};

class FakeCmd : public CommandBufferCmd {
 public:
  explicit FakeCmd(ExecutionStreamId execution_stream_id)
      : CommandBufferCmd(CommandBufferCmdType::kTracedCommandBufferCmd,
                         execution_stream_id) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction,
      se::CommandBuffer*) override {
    return nullptr;
  }

  BufferUseVector buffers() const override { return BufferUseVector{}; }
};

TEST(CommandBufferCmdStateManageTest, GetOrCreateState) {
  struct StateA : public CommandBufferCmd::State {
    int32_t value = 0;
  };

  struct StateB : public CommandBufferCmd::State {
    float value = 0;
  };

  // We need a fake command buffer pointer to use as a key.
  auto* cmd =
      tsl::safe_reinterpret_cast<CommandBufferCmd*>(std::intptr_t{0x1234567});
  auto* command_buffer =
      tsl::safe_reinterpret_cast<se::CommandBuffer*>(std::intptr_t{0x1234567});

  CommandBufferCmd::StateManager state_manager;

  // Create a state of type StateA.
  auto* stateA0 = state_manager.GetOrNull<StateA>(cmd, command_buffer);
  ASSERT_EQ(stateA0, nullptr);

  auto* stateA1 = state_manager.GetOrCreate<StateA>(cmd, command_buffer);
  ASSERT_EQ(stateA1->value, 0);
  stateA1->value += 42;

  auto* stateA2 = state_manager.GetOrCreate<StateA>(cmd, command_buffer);
  ASSERT_EQ(stateA2->value, 42);
  ASSERT_EQ(stateA1, stateA2);

  // StateB has a different type, and has no connection to StateA created above.
  auto* stateB0 = state_manager.GetOrNull<StateB>(cmd, command_buffer);
  ASSERT_EQ(stateB0, nullptr);

  auto* stateB1 = state_manager.GetOrCreate<StateB>(cmd, command_buffer);
  ASSERT_EQ(stateB1->value, 0);
  stateB1->value += 42.0;

  auto* stateB2 = state_manager.GetOrCreate<StateB>(cmd, command_buffer);
  ASSERT_EQ(stateB2->value, 42.0);
  ASSERT_EQ(stateB1, stateB2);
}

TEST(CommandBufferCmdTest, SerializeExecution) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers by default.
  auto use0 = BufferUse(slice0, BufferUse::kRead);
  auto use1 = BufferUse(slice1, BufferUse::kRead);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, NoReadBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers.
  auto use0 = BufferUse(slice0, BufferUse::kRead);
  auto use1 = BufferUse(slice1, BufferUse::kRead);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, NoWriteBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  // Writes to non-overlapping slices do not require barriers.
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 200, 100);

  auto use0 = BufferUse(slice0, BufferUse::kWrite);
  auto use1 = BufferUse(slice1, BufferUse::kWrite);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, WriteConflictBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices can be done in parallel, and before a write
  // into overlapping slice we need to insert a barrier.
  auto use0 = BufferUse(slice0, BufferUse::kRead);
  auto use1 = BufferUse(slice0, BufferUse::kRead);
  auto use2 = BufferUse(slice1, BufferUse::kWrite);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use1});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUseVector{use2});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, MemcpyCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_b, slice_a, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  CommandBufferCmd::StateManager state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  CommandBufferCmd::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      stream_executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK(executor.Record(params, record_params, command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferCmdTest, LaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};  // b = a + a
  auto args_access = {BufferUse::kRead, MemoryAccess::kRead, BufferUse::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // Initialize command commands and load device kernels.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> fatbin,
                          se::gpu::GetGpuTestKernelsFatbin());
  Thunk::ExecutableSource source = {/*text=*/{},
                                    /*binary=*/fatbin};

  CommandBufferCmd::StateManager state;
  TF_ASSERT_OK(executor.Initialize({stream_executor, source}, state));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  CommandBufferCmd::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      stream_executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK(executor.Record(params, record_params, command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(TracedCommandBuffer, GetOrUpdateCommandBuffer) {
  auto run_traced_test = [](int trace_cache_size) {
    se::StreamExecutor* executor = GpuExecutor();

    auto stream = executor->CreateStream().value();
    auto traced_cmd = FakeCmd(ExecutionStreamId(0));
    BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
    BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

    CommandBufferCmd::BufferUseVector buffers = {
        {BufferAllocation::Slice(&alloc0, 0, 1024), BufferUse::kRead},
        {BufferAllocation::Slice(&alloc1, 0, 1024), BufferUse::kWrite}};

    TracedCommandBuffer traced_cmd_buffer(&traced_cmd, buffers,
                                          /*capacity=*/trace_cache_size);

    se::DeviceMemoryBase mem0(reinterpret_cast<void*>(0x01234567));
    se::DeviceMemoryBase mem1(reinterpret_cast<void*>(0x12345670));

    se::StreamExecutorMemoryAllocator allocator(executor);
    BufferAllocations allocations({mem0, mem1}, 0, &allocator);

    se::DeviceMemory<int32_t> mem = executor->AllocateArray<int32_t>(16, 0);

    // Count how many times trace callback was called. We also need to record
    // something on the given stream because we can't leave traced command
    // buffer empty.
    int64_t num_calls = 0;
    auto trace = [&](se::Stream* stream) -> absl::Status {
      TF_RETURN_IF_ERROR(stream->Memset32(&mem, 42, 16));
      num_calls++;
      return absl::OkStatus();
    };

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer0,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer1,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));

    // Check that command buffer was reused as buffer allocations didn't
    // change.
    ASSERT_EQ(command_buffer0, command_buffer1);
    EXPECT_EQ(num_calls, 1);

    // Check that when memory address changes we re-trace the command
    // buffer.
    se::DeviceMemoryBase mem2(reinterpret_cast<void*>(0x23456701));
    allocations = BufferAllocations({mem0, mem2}, 0, &allocator);

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer2,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));

    ASSERT_NE(command_buffer0, command_buffer2);
    EXPECT_EQ(num_calls, 2);

    // Check that we keep first command buffer in cache.
    allocations = BufferAllocations({mem0, mem1}, 0, &allocator);

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer3,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));
    ASSERT_EQ(command_buffer0, command_buffer3);
    EXPECT_EQ(num_calls, 2);

    // Check that we trace a new graph when buffer allocation pattern is
    // new.
    allocations = BufferAllocations({mem0, mem0}, 0, &allocator);

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer4,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));
    ASSERT_NE(command_buffer4, command_buffer3);
    ASSERT_NE(command_buffer4, command_buffer2);
    EXPECT_EQ(num_calls, 3);

    // Check that we still keep the previous graph in cache.
    allocations = BufferAllocations({mem0, mem1}, 0, &allocator);

    TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer5,
                            traced_cmd_buffer.GetOrTraceCommandBuffer(
                                &allocations, executor, stream.get(), trace));
    ASSERT_EQ(command_buffer0, command_buffer5);
    EXPECT_EQ(num_calls, 3);
  };
  run_traced_test(2);
  run_traced_test(3);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_GetOrTraceCommandBuffer(benchmark::State& state) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

  CommandBufferCmd::BufferUseVector buffers = {
      {BufferAllocation::Slice(&alloc0, 0, 1024), BufferUse::kRead},
      {BufferAllocation::Slice(&alloc1, 0, 1024), BufferUse::kWrite}};

  se::DeviceMemoryBase mem0(reinterpret_cast<void*>(0x01234567));
  se::DeviceMemoryBase mem1(reinterpret_cast<void*>(0x12345670));
  se::StreamExecutorMemoryAllocator allocator(executor);

  std::array<BufferAllocations, 4> allocations = {
      BufferAllocations({mem0, mem1}, 0, &allocator),
      BufferAllocations({mem1, mem0}, 0, &allocator),
      BufferAllocations({mem0, mem0}, 0, &allocator),
      BufferAllocations({mem1, mem1}, 0, &allocator),
  };

  int32_t index = 0;
  auto traced_cmd = FakeCmd(ExecutionStreamId(0));
  TracedCommandBuffer traced_cmd_buffer(&traced_cmd, buffers);

  auto trace = [](se::Stream*) { return absl::OkStatus(); };
  absl::FunctionRef<absl::Status(se::Stream*)> trace_ref(trace);

  for (auto s : state) {
    TF_CHECK_OK(traced_cmd_buffer
                    .GetOrTraceCommandBuffer(&allocations[index++ % 4],
                                             executor, stream.get(), trace_ref)
                    .status());
  }
}

BENCHMARK(BM_GetOrTraceCommandBuffer);

}  // namespace xla::gpu

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

#include "xla/service/gpu/runtime/command_buffer_cmd.h"

#include <array>
#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::gpu {

using BufferUsage = CommandBufferCmd::BufferUsage;
using BufferUsageVector = CommandBufferCmd::BufferUsageVector;
using MemoryAccess = CommandBufferCmd::MemoryAccess;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Give a short aliases to execution threads.
static constexpr auto s0 = ExecutionStreamId(0);
static constexpr auto s1 = ExecutionStreamId(1);

// A command buffer cmd for testing automatic barriers insertion by the command
// buffer cmd sequence. We never execute this command, we need it only to pass
// buffer usage vector to the command buffer cmd sequence.
struct TestOnlyCommandBufferCmd : public CommandBufferCmd {
  TestOnlyCommandBufferCmd(ExecutionStreamId execution_stream_id,
                           BufferUsageVector buffer_usage)
      : CommandBufferCmd(execution_stream_id), buffer_usage(buffer_usage) {}

  absl::Status Record(const Thunk::ExecuteParams&, const RecordParams&,
                      se::CommandBuffer*) override {
    return absl::OkStatus();
  }

  BufferUsageVector buffers() override { return buffer_usage; }

  BufferUsageVector buffer_usage;
};

TEST(CommandBufferCmdTest, SerializeExecution) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers by default.
  auto use0 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use1 = BufferUsage(slice1, MemoryAccess::kRead);

  CommandBufferCmdSequence commands(
      CommandBufferCmdSequence::SynchronizationMode::kSerialize);
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use1});

  ASSERT_EQ(commands.barriers().size(), 2);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), true);
}

TEST(CommandBufferCmdTest, NoReadBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers.
  auto use0 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use1 = BufferUsage(slice1, MemoryAccess::kRead);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use1});

  ASSERT_EQ(commands.barriers().size(), 2);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), false);
}

TEST(CommandBufferCmdTest, NoWriteBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  // Writes to non-overlapping slices do not require barriers.
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 200, 100);

  auto use0 = BufferUsage(slice0, MemoryAccess::kWrite);
  auto use1 = BufferUsage(slice1, MemoryAccess::kWrite);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use1});

  ASSERT_EQ(commands.barriers().size(), 2);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), false);
}

TEST(CommandBufferCmdTest, WriteConflictBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices can be done in parallel, and before a write
  // into overlapping slice we need to insert a barrier.
  auto use0 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use1 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use2 = BufferUsage(slice1, MemoryAccess::kWrite);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use1});
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use2});

  ASSERT_EQ(commands.barriers().size(), 3);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), false);
  EXPECT_EQ(commands.barriers().at(2), true);
}

TEST(CommandBufferCmdTest, NoWriteConflictsAcrossStreams) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Read and write happens on different execution streams and we do not insert
  // any automatic barriers between streams.
  auto use0 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use1 = BufferUsage(slice1, MemoryAccess::kWrite);

  CommandBufferCmdSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(s0, BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(s1, BufferUsageVector{use1});

  ASSERT_EQ(commands.barriers().size(), 2);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), false);
}

TEST(CommandBufferCmdTest, MemcpyCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  auto stream = executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

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

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  CommandBufferCmd::StateManager state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  CommandBufferCmd::RecordParams record_params = {state};

  auto command_buffer = se::CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(commands.Record(params, record_params, command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(executor->Submit(stream.get(), *command_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferCmdTest, BarrierCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  auto stream = executor->CreateStream().value();

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> e = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));
  TF_ASSERT_OK(stream->MemZero(&e, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation alloc_e(/*index=*/4, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);
  BufferAllocation::Slice slice_d(&alloc_d, 0, byte_length);
  BufferAllocation::Slice slice_e(&alloc_e, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_b, slice_a, byte_length);
  commands.Emplace<BarrierCmd>(s1, s0);
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s1, slice_c, slice_b, byte_length);
  commands.Emplace<BarrierCmd>(s0, s1);
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_d, slice_c, byte_length);
  commands.Emplace<BarrierCmd>(s1, s0);
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s1, slice_e, slice_d, byte_length);

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b, c, d, e}, 0, &allocator);

  CommandBufferCmd::StateManager state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  CommandBufferCmd::RecordParams record_params = {state};

  auto command_buffer = se::CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(commands.Record(params, record_params, command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(executor->Submit(stream.get(), *command_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferCmdTest, LaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  auto stream = executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Initialize command sequence and load device kernels.
  Thunk::ExecutableSource source = {
#if defined(GOOGLE_CUDA)
      /*text=*/se::gpu::internal::kAddI32Kernel,
      /*binary=*/{}
#elif defined(TENSORFLOW_USE_ROCM)
      /*text=*/{},
      /*binary=*/se::gpu::internal::kAddI32KernelModule
#endif
  };

  CommandBufferCmd::StateManager state;
  TF_ASSERT_OK(commands.Initialize({executor, source}, state));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  CommandBufferCmd::RecordParams record_params = {state};

  auto command_buffer = se::CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(commands.Record(params, record_params, command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(executor->Submit(stream.get(), *command_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferCmdStateManageTest, GetOrCreateState) {
  struct TestState : public CommandBufferCmd::State {
    int32_t value = 0;
  };

  // We need a fake command buffer pointer to use as a key.
  CommandBufferCmd* cmd = reinterpret_cast<CommandBufferCmd*>(0x1234567);

  CommandBufferCmd::StateManager state_manager;

  auto* state0 = state_manager.GetOrNull<TestState>(cmd);
  ASSERT_EQ(state0, nullptr);

  auto* state1 = state_manager.GetOrCreate<TestState>(cmd);
  ASSERT_EQ(state1->value, 0);
  state1->value += 42;

  auto* state2 = state_manager.GetOrCreate<TestState>(cmd);
  ASSERT_EQ(state2->value, 42);
  ASSERT_EQ(state1, state2);
}

TEST(TracedCommandBuffer, GetOrUpdateCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();

  auto stream = executor->CreateStream().value();
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

  CommandBufferCmd::BufferUsageVector buffers = {
      {BufferAllocation::Slice(&alloc0, 0, 1024), MemoryAccess::kRead},
      {BufferAllocation::Slice(&alloc1, 0, 1024), MemoryAccess::kWrite}};

  TracedCommandBuffer traced_cmd_buffer(buffers, /*capacity=*/2);

  se::DeviceMemoryBase mem0(reinterpret_cast<void*>(0x01234567));
  se::DeviceMemoryBase mem1(reinterpret_cast<void*>(0x12345670));

  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({mem0, mem1}, 0, &allocator);

  // No-op trace callback to count how many times it was called.
  int64_t num_calls = 0;
  auto trace = [&](se::Stream*) {
    num_calls++;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer0,
                          traced_cmd_buffer.GetOrTraceCommandBuffer(
                              &allocations, executor, stream.get(), trace));

  TF_ASSERT_OK_AND_ASSIGN(auto* command_buffer1,
                          traced_cmd_buffer.GetOrTraceCommandBuffer(
                              &allocations, executor, stream.get(), trace));

  // Check that command buffer was reused as buffer allocations didn't change.
  ASSERT_EQ(command_buffer0, command_buffer1);
  EXPECT_EQ(num_calls, 1);

  // Check that when memory address changes we re-trace the command buffer.
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

  // Check that we trace a new graph when buffer allocation pattern is new.
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
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_GetOrTraceCommandBuffer(benchmark::State& state) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

  CommandBufferCmd::BufferUsageVector buffers = {
      {BufferAllocation::Slice(&alloc0, 0, 1024), MemoryAccess::kRead},
      {BufferAllocation::Slice(&alloc1, 0, 1024), MemoryAccess::kWrite}};

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
  TracedCommandBuffer traced_cmd_buffer(buffers);

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

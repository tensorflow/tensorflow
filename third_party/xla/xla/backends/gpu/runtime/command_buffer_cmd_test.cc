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

namespace xla::gpu {

using BufferUseVector = CommandBufferCmd::BufferUseVector;
using MemoryAccess = BufferUse::MemoryAccess;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Some of the tests rely on CUDA 12.9+ features.
bool IsAtLeastCuda12900(const se::StreamExecutor* stream_executor) {
  const auto& device_description = stream_executor->GetDeviceDescription();
  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &device_description.gpu_compute_capability());
  if (cuda_cc != nullptr) {
    // We need a recent driver to support the feature at runtime and we need a
    // recent version of the toolkit at compile time, so that we have access to
    // the driver's headers.
    if (std::min(device_description.driver_version(),
                 device_description.compile_time_toolkit_version()) >=
        stream_executor::SemanticVersion(12, 9, 0)) {
      return true;
    }
  }
  return false;
}

// Give a short alias to synchronization mode.
static constexpr auto serialize =
    CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

// A command buffer cmd for testing automatic barriers insertion by the command
// buffer cmd commands. We never execute this command, we need it only to pass
// buffer usage vector to the command buffer cmd commands.
struct TestOnlyCommandBufferCmd : public CommandBufferCmd {
  TestOnlyCommandBufferCmd(BufferUseVector buffer_usage)
      : CommandBufferCmd(CommandBufferCmdType::kUnknownCmd, {}),
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
  explicit FakeCmd()
      : CommandBufferCmd(CommandBufferCmdType::kTracedCommandBufferCmd, {}) {}

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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use1});
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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use1});
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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use1});
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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use1});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUseVector{use2});
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
  commands.Emplace<MemcpyDeviceToDeviceCmd>(slice_b, slice_a, byte_length);
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
  TF_ASSERT_OK(executor.Record(params, record_params,
                               CommandBufferCmd::RecordCreate{},
                               command_buffer.get()));

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
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // Initialize command commands and load device kernels.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
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
  TF_ASSERT_OK(executor.Record(params, record_params,
                               CommandBufferCmd::RecordCreate{},
                               command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferCmdTest, LaunchCmdWithPriority) {
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
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  commands.back()->set_priority(se::StreamPriority::Highest);

  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // Initialize command commands and load device kernels.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
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
  TF_ASSERT_OK(executor.Record(params, record_params,
                               CommandBufferCmd::RecordCreate{},
                               command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferCmdTest, DynamicSliceCopyFusionCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 8;
  int64_t byte_length = sizeof(int32_t) * length;

  std::vector<int32_t> a_data = {40, 41, 42, 43, 44, 45, 46, 47};

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memcpy(&a, a_data.data(), byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<DynamicSliceCopyFusionCmd>(
      slice_a, slice_b, 16, DynamicMemcpyThunk::Offsets{false, {16}, {16}});
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
  TF_ASSERT_OK(executor.Record(params, record_params,
                               CommandBufferCmd::RecordCreate{},
                               command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(8, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>({0, 0, 0, 0, 44, 45, 46, 47}));
}

TEST(TracedCommandBuffer, GetOrUpdateCommandBuffer) {
  auto run_traced_test = [](int trace_cache_size) {
    se::StreamExecutor* executor = GpuExecutor();

    auto stream = executor->CreateStream().value();
    auto traced_cmd = FakeCmd();
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

TEST(CommandBufferCmdTest, RecordExecutorsWithDependencies) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Device buffers: a, b, c
  se::DeviceMemory<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);

  // Initialize to zero.
  TF_ASSERT_OK(stream->MemZero(&a, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Buffer allocations for recording.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);

  // Executor A: a = 1 (memset)
  CommandBufferCmdSequence seq_a;
  seq_a.Emplace<Memset32Cmd>(slice_a, /*bit_pattern=*/1);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor exec_a,
      CommandBufferCmdExecutor::Create(std::move(seq_a), serialize));

  // Executor B: b = a + a (launch kernel AddI32)
  CommandBufferCmdSequence seq_b;
  {
    auto args = {slice_a, slice_a, slice_b};
    auto args_access = {BufferUse::kRead, MemoryAccess::kRead,
                        BufferUse::kWrite};
    seq_b.Emplace<LaunchCmd>("AddI32", args, args_access,
                             LaunchDimensions(1, 4), /*shmem_bytes=*/0);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor exec_b,
      CommandBufferCmdExecutor::Create(std::move(seq_b), serialize));

  // Executor C: c = b (memcpy)
  CommandBufferCmdSequence seq_c;
  seq_c.Emplace<MemcpyDeviceToDeviceCmd>(slice_c, slice_b, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor exec_c,
      CommandBufferCmdExecutor::Create(std::move(seq_c), serialize));

  // Initialize executors (B needs kernel fatbin).
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
  Thunk::ExecutableSource source_empty = {/*text=*/{}, /*binary=*/{}};
  Thunk::ExecutableSource source_fatbin = {/*text=*/{}, /*binary=*/fatbin};

  CommandBufferCmd::StateManager state;
  TF_ASSERT_OK(exec_a.Initialize({stream_executor, source_empty}, state));
  TF_ASSERT_OK(exec_b.Initialize({stream_executor, source_fatbin}, state));
  TF_ASSERT_OK(exec_c.Initialize({stream_executor, source_empty}, state));

  // Execute params and allocations mapping indices 0=a,1=b,2=c
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c}, 0, &allocator);

  Thunk::ExecuteParams exec_params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);
  CommandBufferCmd::RecordParams record_params = {state};

  // Create a primary command buffer and record A -> B -> C with dependencies.
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      stream_executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  // Record A (no deps)
  // Record A, B, C with dependencies using the Record API; finalize on B.
  TF_ASSERT_OK(exec_a.Record(exec_params, record_params,
                             CommandBufferCmd::RecordCreate{},
                             command_buffer.get(), /*finalize=*/false));

  auto a_sinks = exec_a.SinkCommands(record_params, command_buffer.get(),
                                     /*unroll_iteration=*/0);
  TF_ASSERT_OK(
      exec_b.Record(exec_params, record_params,
                    CommandBufferCmd::RecordCreate{absl::MakeSpan(a_sinks)},
                    command_buffer.get(), /*finalize=*/false));

  auto b_sinks = exec_b.SinkCommands(record_params, command_buffer.get(),
                                     /*unroll_iteration=*/0);
  TF_ASSERT_OK(
      exec_c.Record(exec_params, record_params,
                    CommandBufferCmd::RecordCreate{absl::MakeSpan(b_sinks)},
                    command_buffer.get(), /*finalize=*/true));

  // Submit and verify c == 2 for all elements.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  std::vector<int32_t> dst(length, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(length, 2));
}

TEST(CommandBufferCmdTest, NestedChildCmdCreateAndUpdate) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  if (!IsAtLeastCuda12900(stream_executor)) {
    GTEST_SKIP() << "Child command is not supported for CUDA < 12.9";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  // Prepare device memory for three buffers.
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  se::DeviceMemory<int32_t> a = stream_executor->AllocateArray<int32_t>(length);
  se::DeviceMemory<int32_t> b = stream_executor->AllocateArray<int32_t>(length);
  se::DeviceMemory<int32_t> c = stream_executor->AllocateArray<int32_t>(length);

  // Initialize a = 1s, b = 0s, c = 0s.
  TF_ASSERT_OK(stream->Memset32(&a, /*pattern=*/1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Buffer allocations.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);

  // Inner child: c = a (device-to-device memcpy)
  CommandBufferCmdSequence inner_seq;
  inner_seq.Emplace<MemcpyDeviceToDeviceCmd>(slice_c, slice_a, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor inner_executor,
      CommandBufferCmdExecutor::Create(std::move(inner_seq), serialize));

  // Middle child wraps inner.
  CommandBufferCmdSequence middle_seq;
  middle_seq.Emplace<ChildCmd>(std::move(inner_executor));
  // Add a couple of extra commands that don't affect `c`.
  middle_seq.Emplace<Memset32Cmd>(slice_b, /*bit_pattern=*/3);
  middle_seq.Emplace<MemcpyDeviceToDeviceCmd>(slice_b, slice_b, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor middle_executor,
      CommandBufferCmdExecutor::Create(std::move(middle_seq), serialize));

  // Outer child wraps middle.
  CommandBufferCmdSequence outer_seq;
  outer_seq.Emplace<ChildCmd>(std::move(middle_executor));
  // Add a couple more commands at the outer level that still don't affect `c`.
  outer_seq.Emplace<MemzeroCmd>(slice_b);
  outer_seq.Emplace<EmptyCmd>();
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor outer_executor,
      CommandBufferCmdExecutor::Create(std::move(outer_seq), serialize));

  // Prepare state and params; ChildCmd requires initialization to create a
  // nested buffer.
  CommandBufferCmd::StateManager state;
  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c}, 0, &allocator);
  TF_ASSERT_OK(outer_executor.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()},
      state));

  // allocations already created above
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams exec_params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);
  CommandBufferCmd::RecordParams record_params = {state};

  // Create a command buffer and record the nested ChildCmd (Create).
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      stream_executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK(outer_executor.Record(exec_params, record_params,
                                     CommandBufferCmd::RecordCreate{},
                                     command_buffer.get(), /*finalize=*/true));
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Verify c == a (all ones).
  std::vector<int32_t> dst(length, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(length, 1));

  // Also verify a == 1s and b == 0s.
  {
    std::vector<int32_t> a_host(length, 0);
    std::vector<int32_t> b_host(length, 0);
    TF_ASSERT_OK(stream->Memcpy(a_host.data(), a, byte_length));
    TF_ASSERT_OK(stream->Memcpy(b_host.data(), b, byte_length));
    ASSERT_EQ(a_host, std::vector<int32_t>(length, 1));
    ASSERT_EQ(b_host, std::vector<int32_t>(length, 0));
  }

  // Now update: change a and c buffers and record an update on the same command
  // buffer.
  se::DeviceMemory<int32_t> a2 =
      stream_executor->AllocateArray<int32_t>(length);
  se::DeviceMemory<int32_t> c2 =
      stream_executor->AllocateArray<int32_t>(length);
  TF_ASSERT_OK(stream->Memset32(&a2, /*pattern=*/7, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c2, byte_length));

  BufferAllocations allocations2({a2, b, c2}, 0, &allocator);
  Thunk::ExecuteParams exec_params2 = Thunk::ExecuteParams::Create(
      run_options, allocations2, stream.get(), stream.get(), nullptr, nullptr);

  // Indicate which allocations changed to ensure update is not skipped.
  std::vector<BufferAllocation::Index> updated_allocs = {0, 2};
  CommandBufferCmd::RecordParams record_params2 = {state,
                                                   std::move(updated_allocs)};

  TF_ASSERT_OK(outer_executor.Record(exec_params2, record_params2,
                                     CommandBufferCmd::RecordCreate{},
                                     command_buffer.get(), /*finalize=*/true));
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  // Verify c2 == a2 (all sevens).
  std::vector<int32_t> dst2(length, 0);
  TF_ASSERT_OK(stream->Memcpy(dst2.data(), c2, byte_length));
  ASSERT_EQ(dst2, std::vector<int32_t>(length, 7));

  // Also verify a2 == 7s and b == 0s.
  {
    std::vector<int32_t> a2_host(length, 0);
    std::vector<int32_t> b_host(length, 0);
    TF_ASSERT_OK(stream->Memcpy(a2_host.data(), a2, byte_length));
    TF_ASSERT_OK(stream->Memcpy(b_host.data(), b, byte_length));
    ASSERT_EQ(a2_host, std::vector<int32_t>(length, 7));
    ASSERT_EQ(b_host, std::vector<int32_t>(length, 0));
  }
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
  auto traced_cmd = FakeCmd();
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

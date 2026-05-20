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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/traced_command_buffer.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Give a short alias to synchronization mode.
static constexpr auto serialize =
    CommandExecutor::SynchronizationMode::kSerialize;

// A command buffer cmd for testing automatic barriers insertion by the command
// buffer cmd commands. We never execute this command, we need it only to pass
// buffer usage vector to the command buffer cmd commands.
struct TestOnlyCommandBufferCmd : public Command {
  explicit TestOnlyCommandBufferCmd(Command::BufferUses buffers)
      : Command(CommandType::kUnknownCmd, {}), buffers(buffers) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction,
      se::CommandBuffer*) override {
    return nullptr;
  }

  BufferUses buffer_uses() const override { return buffers; }

  BufferUses buffers;
};

class FakeCmd : public Command {
 public:
  explicit FakeCmd() : Command(CommandType::kUnknownCmd, {}) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction,
      se::CommandBuffer*) override {
    return nullptr;
  }

  BufferUses buffer_uses() const override { return BufferUses{}; }
};

TEST(CommandBufferCmdStateManageTest, GetOrCreateState) {
  struct StateA : public CommandState {
    int32_t value = 0;
  };

  struct StateB : public CommandState {
    float value = 0;
  };

  // We need a fake command and command buffer pointer to use as a key.
  auto* cmd = tsl::safe_reinterpret_cast<Command*>(std::intptr_t{0x1234567});
  auto* command_buffer =
      tsl::safe_reinterpret_cast<se::CommandBuffer*>(std::intptr_t{0x1234567});

  CommandStateManager state_manager;

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

  Shape shape = ShapeUtil::MakeShape(U8, {100});
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers by default.
  auto use0 = BufferUse::Read(slice0, shape);
  auto use1 = BufferUse::Read(slice1, shape);

  CommandSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, NoReadBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  Shape shape = ShapeUtil::MakeShape(U8, {100});
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers.
  auto use0 = BufferUse::Read(slice0, shape);
  auto use1 = BufferUse::Read(slice1, shape);

  CommandSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, NoWriteBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  // Writes to non-overlapping slices do not require barriers.
  Shape shape = ShapeUtil::MakeShape(U8, {100});
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 200, 100);

  auto use0 = BufferUse::Write(slice0, shape);
  auto use1 = BufferUse::Write(slice1, shape);

  CommandSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use1});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, WriteConflictBarrier) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  Shape shape = ShapeUtil::MakeShape(U8, {100});
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices can be done in parallel, and before a write
  // into overlapping slice we need to insert a barrier.
  auto use0 = BufferUse::Read(slice0, shape);
  auto use1 = BufferUse::Read(slice0, shape);
  auto use2 = BufferUse::Write(slice1, shape);

  CommandSequence commands;
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use1});
  commands.Emplace<TestOnlyCommandBufferCmd>(Command::BufferUses{use2});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // TODO(ezhulenev): Check that executor correctly infer dependencies.
}

TEST(CommandBufferCmdTest, LaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  Shape shape = ShapeUtil::MakeShape(S32, {1});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> args{
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  auto args_access = {BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Append(KernelThunk::MakeKernelThunk(
      "AddI32", absl::MakeConstSpan(args), args_access, LaunchDimensions(1, 4),
      /*shmem_bytes=*/0));
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Initialize command commands and load device kernels.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
  Thunk::ExecutableSource source = {/*text=*/{},
                                    /*binary=*/fatbin};

  TF_ASSERT_OK(executor.Initialize({stream_executor, source}));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

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

TEST(CommandBufferCmdTest, LaunchCmdWithPriority) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  auto stream = stream_executor->CreateStream().value();
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {1});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> args{
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  auto args_access = {BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Append(KernelThunk::MakeKernelThunk(
      "AddI32", absl::MakeConstSpan(args), args_access, LaunchDimensions(1, 4),
      /*shmem_bytes=*/0));
  commands.back()->set_priority(se::StreamPriority::Highest);

  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Initialize command commands and load device kernels.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
  Thunk::ExecutableSource source = {/*text=*/{},
                                    /*binary=*/fatbin};

  TF_ASSERT_OK(executor.Initialize({stream_executor, source}));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

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
    auto traced_cmd = FakeCmd();
    BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
    BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

    Shape shape = ShapeUtil::MakeShape(U8, {1024});
    Command::BufferUses buffers = {
        BufferUse::Read(BufferAllocation::Slice(&alloc0, 0, 1024), shape),
        BufferUse::Write(BufferAllocation::Slice(&alloc1, 0, 1024), shape)};

    TracedCommandBuffer traced_cmd_buffer(&traced_cmd, buffers,
                                          /*capacity=*/trace_cache_size);

    se::DeviceAddressBase mem0(reinterpret_cast<void*>(0x01234567));
    se::DeviceAddressBase mem1(reinterpret_cast<void*>(0x12345670));

    se::StreamExecutorAddressAllocator allocator(executor);
    BufferAllocations allocations({mem0, mem1}, 0, &allocator);

    se::DeviceAddress<int32_t> mem = executor->AllocateArray<int32_t>(16, 0);

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

    // Check that command buffer was reused as buffer allocations didn't change.
    ASSERT_EQ(command_buffer0, command_buffer1);
    EXPECT_EQ(num_calls, 1);

    // Check that when memory address changes we re-trace the command
    // buffer.
    se::DeviceAddressBase mem2(reinterpret_cast<void*>(0x23456701));
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
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Device buffers: a, b, c
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> c =
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
  Memset32BitValueThunk memset_a_thunk(Thunk::ThunkInfo(), /*value=*/1,
                                       slice_a);
  CommandSequence seq_a;
  seq_a.Append(&memset_a_thunk);
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor exec_a,
                          CommandExecutor::Create(std::move(seq_a), serialize));

  // Executor B: b = a + a (launch kernel AddI32)
  CommandSequence seq_b;
  {
    std::vector<ShapedSlice> args{
        {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};
    auto args_access = {BufferUse::MemoryAccess::kRead,
                        BufferUse::MemoryAccess::kRead,
                        BufferUse::MemoryAccess::kWrite};
    seq_b.Append(
        KernelThunk::MakeKernelThunk("AddI32", absl::MakeConstSpan(args),
                                     args_access, LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0));
  }
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor exec_b,
                          CommandExecutor::Create(std::move(seq_b), serialize));

  // Executor C: c = b (memcpy)
  CommandSequence seq_c;
  seq_c.Emplace<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice_b, shape},
      ShapedSlice{slice_c, shape}, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor exec_c,
                          CommandExecutor::Create(std::move(seq_c), serialize));

  // Initialize executors (B needs kernel fatbin).
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(stream_executor->GetPlatform()->Name()));
  Thunk::ExecutableSource source_empty = {/*text=*/{}, /*binary=*/{}};
  Thunk::ExecutableSource source_fatbin = {/*text=*/{}, /*binary=*/fatbin};

  TF_ASSERT_OK(exec_a.Initialize({stream_executor, source_empty}));
  TF_ASSERT_OK(exec_b.Initialize({stream_executor, source_fatbin}));
  TF_ASSERT_OK(exec_c.Initialize({stream_executor, source_empty}));

  // Execute params and allocations mapping indices 0=a,1=b,2=c
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c}, 0, &allocator);

  Thunk::ExecuteParams exec_params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // Create a primary command buffer and record A -> B -> C with dependencies.
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      stream_executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  // Record A (no deps)
  // Record A, B, C with dependencies using the Record API; finalize on B.
  TF_ASSERT_OK_AND_ASSIGN(auto a_sinks,
                          exec_a.RecordCreate(exec_params, record_params,
                                              command_buffer.get(), {}));

  TF_ASSERT_OK_AND_ASSIGN(auto b_sinks,
                          exec_b.RecordCreate(exec_params, record_params,
                                              command_buffer.get(), a_sinks));

  TF_ASSERT_OK_AND_ASSIGN(auto c_sinks,
                          exec_c.RecordCreate(exec_params, record_params,
                                              command_buffer.get(), b_sinks))

  // Finalize command buffer after recording multiple iterations.
  TF_ASSERT_OK(command_buffer->Finalize());

  // Submit and verify c == 2 for all elements.
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  std::vector<int32_t> dst(length, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(length, 2));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_GetOrTraceCommandBuffer(benchmark::State& state) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);

  Shape shape = ShapeUtil::MakeShape(U8, {1024});
  Command::BufferUses buffers = {
      BufferUse::Read(BufferAllocation::Slice(&alloc0, 0, 1024), shape),
      BufferUse::Write(BufferAllocation::Slice(&alloc1, 0, 1024), shape)};

  se::DeviceAddressBase mem0(reinterpret_cast<void*>(0x01234567));
  se::DeviceAddressBase mem1(reinterpret_cast<void*>(0x12345670));
  se::StreamExecutorAddressAllocator allocator(executor);

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
    CHECK_OK(traced_cmd_buffer
                 .GetOrTraceCommandBuffer(&allocations[index++ % 4], executor,
                                          stream.get(), trace_ref)
                 .status());
  }
}

BENCHMARK(BM_GetOrTraceCommandBuffer);

}  // namespace xla::gpu

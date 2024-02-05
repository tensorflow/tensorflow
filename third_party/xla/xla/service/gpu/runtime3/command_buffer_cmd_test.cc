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

#include "xla/service/gpu/runtime3/command_buffer_cmd.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/platform_util.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

using BufferUsage = CommandBufferCmd::BufferUsage;
using BufferUsageVector = CommandBufferCmd::BufferUsageVector;
using MemoryAccess = CommandBufferCmd::MemoryAccess;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::MultiPlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// A command buffer cmd for testing automatic barriers insertion by the command
// buffer cmd sequence. We never execute this command, we need it only to pass
// buffer usage vector to the command buffer cmd sequence.
struct TestOnlyCommandBufferCmd : public CommandBufferCmd {
  explicit TestOnlyCommandBufferCmd(BufferUsageVector buffer_usage)
      : buffer_usage(buffer_usage) {}

  absl::Status Record(const RecordParams&, se::CommandBuffer*) override {
    return absl::OkStatus();
  }

  BufferUsageVector buffers() override { return buffer_usage; }

  BufferUsageVector buffer_usage;
};

TEST(CommandBufferCmdTest, ForceBarriers) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);

  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 100);
  auto slice1 = BufferAllocation::Slice(&alloc0, 50, 100);

  // Reads from overlapping slices do not require barriers by default.
  auto use0 = BufferUsage(slice0, MemoryAccess::kRead);
  auto use1 = BufferUsage(slice1, MemoryAccess::kRead);

  CommandBufferCmdSequence commands(/*force_barriers=*/true);
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use1});

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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use1});

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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use1});

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
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use0});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use1});
  commands.Emplace<TestOnlyCommandBufferCmd>(BufferUsageVector{use2});

  ASSERT_EQ(commands.barriers().size(), 3);
  EXPECT_EQ(commands.barriers().at(0), false);
  EXPECT_EQ(commands.barriers().at(1), false);
  EXPECT_EQ(commands.barriers().at(2), true);
}

TEST(CommandBufferCmdTest, MemcpyCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  se::Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 42, byte_length);
  stream.ThenMemZero(&b, byte_length);

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemcpyDeviceToDeviceCmd>(slice_b, slice_a, byte_length);

  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());

  auto command_buffer = se::CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(commands.Record({executor, &stream, &stream, &allocations},
                               command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(executor->Submit(&stream, *command_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  stream.ThenMemcpy(dst.data(), b, byte_length);

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferCmdTest, LaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  se::Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 42, byte_length);
  stream.ThenMemZero(&b, byte_length);

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
  commands.Emplace<LaunchCmd>("add", args, args_access, LaunchDimensions(1, 4),
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
  TF_ASSERT_OK(commands.Initialize({executor, source}));

  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());

  auto command_buffer = se::CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(commands.Record({executor, &stream, &stream, &allocations},
                               command_buffer.get()));

  // Execute command buffer and verify that it copied the memory.
  TF_ASSERT_OK(executor->Submit(&stream, *command_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  stream.ThenMemcpy(dst.data(), b, byte_length);

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

}  // namespace xla::gpu

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/test.h"

namespace xla::gpu {

static se::StreamExecutor* CudaExecutor() {
  auto* platform = se::MultiPlatformManager::PlatformWithName("CUDA").value();
  return platform->ExecutorForDevice(0).value();
}

TEST(CommandBufferCmdTest, MemcpyCmd) {
  se::StreamExecutor* executor = CudaExecutor();

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

  // Record commands into the command buffer.
  auto commands = CommandBufferCmdSequence::Create(executor).value();
  commands.Emplace<MemcpyDeviceToDeviceCmd>(slice_b, slice_a, byte_length);

  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());
  ASSERT_TRUE(commands.Record({&allocations}).ok());

  // Execute command buffer and verify that it copied the memory.
  ASSERT_TRUE(executor->Submit(&stream, commands.command_buffer()).ok());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  stream.ThenMemcpy(dst.data(), b, byte_length);

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

}  // namespace xla::gpu

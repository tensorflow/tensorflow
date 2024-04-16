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

#include "xla/service/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/command_buffer_allocations.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace xla::gpu {

using MemoryAccess = CommandBufferCmd::MemoryAccess;
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

static Thunk::ExecutableSource ExecutableSource() {
  Thunk::ExecutableSource source = {
#if defined(GOOGLE_CUDA)
      /*text=*/se::gpu::internal::kAddI32Kernel,
      /*binary=*/{}
#elif defined(TENSORFLOW_USE_ROCM)
      /*text=*/{},
      /*binary=*/se::gpu::internal::kAddI32KernelModule
#endif
  };
  return source;
}

static KernelArgsPacking CreateDefaultArgsPacking() {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    return se::PackKernelArgs(mem_args->device_memory_args(),
                              args.number_of_shared_bytes());
  };
}

// Some of the tests rely on CUDA 12.3+ features.
static bool IsAtLeastCuda12300() {
#if defined(TENSORFLOW_USE_ROCM)
  return false;
#endif
#if CUDA_VERSION >= 12030
  return true;
#endif
  return false;
}

// Give a short aliases to execution threads.
static constexpr auto s0 = ExecutionStreamId(0);
static constexpr auto s1 = ExecutionStreamId(1);

TEST(CommandBufferThunkTest, MemcpyCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

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

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it copied the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferThunkTest, MemzeroCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemzeroCmd>(s0, slice_a);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it zeroes the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 0));
}

TEST(CommandBufferThunkTest, Memset32Cmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{84});

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdOnDifferentStreams) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(2, 0);
  TF_ASSERT_OK(stream->MemZero(&a, 2 * sizeof(int32_t)));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc(/*index=*/0, a.size(), /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0 * sizeof(int32_t), sizeof(int32_t));
  BufferAllocation::Slice slice1(&alloc, 1 * sizeof(int32_t), sizeof(int32_t));

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice0, int32_t{12});
  commands.Emplace<Memset32Cmd>(s1, slice1, int32_t{34});

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(2, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, a.size()));

  ASSERT_EQ(dst, std::vector<int32_t>({12, 34}));
}

// This test does the following operations:
// 1. Allocates memory region "a" and "c" outside command buffer.
// 2. Allocates memory region "b" inside command buffer.
// 3. MemCopyDeviceToDevice from "a" to "b" inside command buffer.

// 4. MemCopyDeviceToDevice from "b" to "c" inside command buffer.
// 5. Free memory region "b" inside command buffer.
// 6. Verify that region "c" has the same content as "a".
TEST(CommandBufferThunkTest, MemallocFreeCmdSameThunk) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Prepare arguments:
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<AllocateCmd>(s0, alloc_b);
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_b, slice_a, byte_length);
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_c, slice_b, byte_length);
  commands.Emplace<FreeCmd>(s0, alloc_b);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  se::DeviceMemory<int32_t> b(se::DeviceMemoryBase(
      reinterpret_cast<int32_t*>(BufferAllocations::kExternalAllocationMarker),
      byte_length));
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  auto external_allocation = std::make_unique<CommandBufferAllocations>();

  BufferAllocations allocations({a, b, c}, 0, executor->GetAllocator(),
                                external_allocation.get());

  ServiceExecutableRunOptions run_options;

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it copied the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(
      dst.data(), allocations.GetMutableDeviceAddress(2), byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

// This test does the following operations:
// 1. Allocates memory region "a" and "c" outside command buffer.
// 2. Allocates memory region "b" inside command buffer thunk 1.
// 3. MemCopyDeviceToDevice from "a" to "b" inside command buffer 1.
// 4. MemCopyDeviceToDevice from "b" to "c" inside command buffer 2.
// 5. Free memory region "b" inside command buffer 2.
// 6. Verify that region "c" has the same content as "a".
TEST(CommandBufferThunkTest, MemallocFreeCmdAcrossThunk) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Prepare arguments:
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);

  // =================Thunk 1=================================
  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands1;
  commands1.Emplace<AllocateCmd>(s0, alloc_b);
  commands1.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_b, slice_a, byte_length);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk1(std::move(commands1), Thunk::ThunkInfo(nullptr));

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  se::DeviceMemory<int32_t> b(se::DeviceMemoryBase(
      reinterpret_cast<int32_t*>(BufferAllocations::kExternalAllocationMarker),
      byte_length));
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  auto external_allocation = std::make_unique<CommandBufferAllocations>();

  BufferAllocations allocations({a, b, c}, 0, executor->GetAllocator(),
                                external_allocation.get());

  ServiceExecutableRunOptions run_options;

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  // Execute command buffer thunk and verify that it copied the memory.
  TF_ASSERT_OK(thunk1.ExecuteOnStream(params));

  // =================Thunk 2=================================
  CommandBufferCmdSequence commands2;
  commands2.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_c, slice_b, byte_length);
  commands2.Emplace<FreeCmd>(s0, alloc_b);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk2(std::move(commands2), Thunk::ThunkInfo(nullptr));

  // Execute command buffer thunk and verify that it copied the memory.
  TF_ASSERT_OK(thunk2.ExecuteOnStream(params));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(
      dst.data(), allocations.GetMutableDeviceAddress(2), byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferThunkTest, LaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

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

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, c}, 0, executor->GetAllocator());

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, CustomAddKernelLaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  auto packing = CreateDefaultArgsPacking();

  se::MultiKernelLoaderSpec spec(/*arity=*/3, std::move(packing));
  spec.AddInProcessSymbol(se::gpu::internal::GetAddI32Kernel(), "add");

  auto custom_kernel =
      CustomKernel("add", std::move(spec), se::BlockDim(),
                   se::ThreadDim(4, 1, 1), /*shared_memory_bytes=*/0);

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

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, c}, 0, executor->GetAllocator());

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, GemmCmd) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 3;

  // Prepare arguments:
  // lhs = [1.0, 2.0, 3.0, 4.0
  //        5.0, 6.0, 7.0, 8.0]
  // rhs = [1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, 1024 * 1024, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<GemmCmd>(s0, config.value(), slice_lhs, slice_rhs, slice_out,
                            slice_workspace,
                            /*deterministic=*/true);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0,
                                executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(6, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceMemory<float> updated_out = executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Update buffer allocation to updated `out` buffer.
  allocations = BufferAllocations({lhs, rhs, updated_out, workspace}, 0,
                                  executor->GetAllocator());

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));
}

TEST(CommandBufferThunkTest, MultipleLaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));
  TF_ASSERT_OK(stream->Memset32(&c, 21, byte_length));
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);
  BufferAllocation::Slice slice_d(&alloc_d, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};    // b = a + a
  auto args_1 = {slice_c, slice_c, slice_d};  // d = c + c
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  commands.Emplace<LaunchCmd>(s0, "add", args_1, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b, c, d}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  BufferAllocation alloc_e(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_e(&alloc_e, 0, byte_length);

  // Prepare buffer allocation for updating command buffer: e=0
  se::DeviceMemory<int32_t> e = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&e, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, b, c, e}, 0, executor->GetAllocator());

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&e, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));
}

TEST(CommandBufferThunkTest, IfCmd) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: pred=true, a=42, b=0
  se::DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_p(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_p(&alloc_p, 0, 1);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for `then` branch.
  CommandBufferCmdSequence then_commands;
  then_commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<IfCmd>(s0, slice_p, std::move(then_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({pred, a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #2 to buffer `c`.
  allocations = BufferAllocations({pred, a, c}, 0, executor->GetAllocator());

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, IfElseCmd) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: pred=true, a=42, b=0
  se::DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_p(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_p(&alloc_p, 0, 1);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for `then` & `else` branches.
  CommandBufferCmdSequence then_commands;
  CommandBufferCmdSequence else_commands;

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Then: b = a + a
    auto args = {slice_a, slice_a, slice_b};
    then_commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                                     LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0);
  }

  {  // Else: b = b + b
    auto args = {slice_b, slice_b, slice_b};
    else_commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                                     LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0);
  }

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<IfElseCmd>(s0, slice_p, std::move(then_commands),
                              std::move(else_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({pred, a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Change branch to `else` and check that it updated the `b` buffer.
  constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));

  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 2 * (42 + 42)));
}

TEST(CommandBufferThunkTest, CaseCmd) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: index=0, a=42, b=0
  se::DeviceMemory<int32_t> index = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_i(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_i(&alloc_i, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for branches.
  std::vector<CommandBufferCmdSequence> branches(2);

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Case 0: b = a + a
    auto args = {slice_a, slice_a, slice_b};
    branches[0].Emplace<LaunchCmd>(s0, "add", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  {  // Case 1: b = b + b
    auto args = {slice_b, slice_b, slice_b};
    branches[1].Emplace<LaunchCmd>(s0, "add", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<CaseCmd>(s0, slice_i, std::move(branches));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({index, a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Change `index` to `1` and check that it updated the `b` buffer.
  TF_ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 2 * (42 + 42)));
}

TEST(CommandBufferThunkTest, ForCmd) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: loop_cnt=0, a=1, b=0
  se::DeviceMemory<int32_t> loop_cnt = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&loop_cnt, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_cnt(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_cnt(&alloc_cnt, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_b, slice_b};  // b = a + b
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for loop `body`.
  CommandBufferCmdSequence body_commands;
  body_commands.Emplace<LaunchCmd>(s0, "add", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<ForCmd>(s0, /*num_iterations=*/10, slice_cnt,
                           std::move(body_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(nullptr));

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({loop_cnt, a, b}, 0, executor->GetAllocator());

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), {}, nullptr, nullptr);

  Thunk::ExecutableSource source = ExecutableSource();
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value 10 times.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 10));
}

TEST(CommandBufferThunkTest, WhileCmd) {
  // TODO(ezhulenev): Find a way to test WhileCmd: add a test only TraceCmd that
  // could allow us trace custom kernels to update while loop iterations. Or
  // maybe add a CustomLaunchCmd and wrap loop update into custom kernel.
}

}  // namespace xla::gpu

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

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_lock.h"

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace xla::gpu {

using MemoryAccess = BufferUse::MemoryAccess;
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

namespace {
se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

struct OwningExecutableSource {
  std::string text;
  std::vector<uint8_t> binary;

  explicit operator Thunk::ExecutableSource() const { return {text, binary}; }
};

absl::StatusOr<OwningExecutableSource> ExecutableSource() {
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> fatbin,
                      se::gpu::GetGpuTestKernelsFatbin());
  return OwningExecutableSource{/*text=*/{},
                                /*binary=*/fatbin};
}

KernelArgsPacking CreateDefaultArgsPacking() {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    return se::PackKernelArgs(mem_args->device_memory_args(),
                              args.number_of_shared_bytes());
  };
}

// Some of the tests rely on CUDA 12.3+ features.
bool IsAtLeastCuda12300(const se::StreamExecutor* executor) {
  const auto& device_description = executor->GetDeviceDescription();
  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &device_description.gpu_compute_capability());
  if (cuda_cc != nullptr) {
    if (device_description.driver_version() >=
        stream_executor::SemanticVersion(12, 3, 0)) {
      return true;
    }
  }

  return false;
}

// Give a short aliases to execution threads.
constexpr auto s0 = ExecutionStreamId(0);
constexpr auto s1 = ExecutionStreamId(1);
}  // namespace

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
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  se::StreamExecutorMemoryAllocator allocator(executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

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
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

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
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersDisabledDuringProfiling) {
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

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{12});

  constexpr bool kProfileCommandBuffersEnabled = false;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                          tsl::profiler::ProfilerLock::Acquire());
  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersEnabledDuringProfiling) {
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

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{12});

  constexpr bool kProfileCommandBuffersEnabled = true;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                          tsl::profiler::ProfilerLock::Acquire());
  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 12));
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
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(2, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, a.size()));

  ASSERT_EQ(dst, std::vector<int32_t>({12, 34}));
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
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

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
  allocations = BufferAllocations({a, c}, 0, &allocator);

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
      CustomKernel("AddI32", std::move(spec), se::BlockDim(),
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
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

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
  allocations = BufferAllocations({a, c}, 0, &allocator);

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
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

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

  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<GemmCmd>(s0, config.value(), slice_lhs, slice_rhs, slice_out,
                            slice_workspace,
                            /*deterministic=*/true);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

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
  allocations =
      BufferAllocations({lhs, rhs, updated_out, workspace}, 0, &allocator);

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

TEST(CommandBufferThunkTest, DynamicSliceFusionCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 4 * 4;
  int64_t fake_lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 3;

  // Prepare arguments:
  // lhs = [1.0, 2.0, 3.0, 4.0
  //        5.0, 6.0, 7.0, 8.0]
  // rhs = [1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(4 * 4);
  std::vector<float> lhs_arr{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8};
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
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);
  fake_allocations[0] = std::make_unique<BufferAllocation>(
      /*index=*/0, fake_lhs_length, /*color=*/0);
  fake_allocations[1] =
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0);
  fake_allocations[2] =
      std::make_unique<BufferAllocation>(/*index=*/2, out_length,
                                         /*color=*/0);

  fake_allocations[3] =
      std::make_unique<BufferAllocation>(/*index=*/3, 1024 * 1024,
                                         /*color=*/0);
  BufferAllocation::Slice fake_slice_lhs(fake_allocations[0].get(), 0,
                                         fake_lhs_length);
  BufferAllocation::Slice slice_rhs(fake_allocations[1].get(), 0, rhs_length);
  BufferAllocation::Slice slice_out(fake_allocations[2].get(), 0, out_length);
  BufferAllocation::Slice slice_workspace(fake_allocations[3].get(), 0,
                                          1024 * 1024);
  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  std::unique_ptr<CommandBufferCmdSequence> embed_commands =
      std::make_unique<CommandBufferCmdSequence>();
  embed_commands->Emplace<GemmCmd>(s0, config.value(), fake_slice_lhs,
                                   slice_rhs, slice_out, slice_workspace,
                                   /*deterministic=*/true);

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);

  std::vector<DynamicSliceThunk::Offset> lhs_offsets = {
      DynamicSliceThunk::Offset(2l), DynamicSliceThunk::Offset(0l)};

  std::vector<std::optional<BufferAllocation::Slice>> arguments = {
      std::optional<BufferAllocation::Slice>(slice_lhs),
      std::optional<BufferAllocation::Slice>(slice_rhs),
      std::optional<BufferAllocation::Slice>(slice_out),
      std::optional<BufferAllocation::Slice>(slice_workspace)};

  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets = {
      lhs_offsets, std::nullopt, std::nullopt, std::nullopt};

  std::vector<std::optional<Shape>> orig_shapes = {
      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 4}), std::nullopt,
      std::nullopt, std::nullopt};
  std::vector<std::optional<Shape>> sliced_shapes = {
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
      std::nullopt, std::nullopt};
  std::vector<std::optional<uint64_t>> offset_byte_sizes = {
      sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt};

  CommandBufferCmdSequence commands;
  commands.Emplace<DynamicSliceFusionCmd>(
      s0, std::move(embed_commands), arguments, std::move(fake_allocations),
      offsets, orig_shapes, sliced_shapes, offset_byte_sizes);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

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
  allocations =
      BufferAllocations({lhs, rhs, updated_out, workspace}, 0, &allocator);

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

TEST(CommandBufferThunkTest, CublasLtCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream1, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto stream2, executor->CreateStream());

  // CublasLt formula: D = alpha*(A*B) + beta*(C),

  int64_t a_length = sizeof(float) * 2 * 4;
  int64_t b_length = sizeof(float) * 4 * 3;
  int64_t c_length = sizeof(float) * 2 * 3;
  int64_t d_length = sizeof(float) * 2 * 3;

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, a_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, b_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, c_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, d_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/4, 1024 * 1024, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, a_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, b_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, c_length);
  BufferAllocation::Slice slice_d(&alloc_d, 0, d_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  auto config = GemmConfig::For(
      /*lhs_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
      /*lhs_batch_dims*/ {}, /*lhs_contracting_dims*/ {1},
      /*rhs_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}),
      /*rhs_batch_dims*/ {}, /*rhs_contracting_dims*/ {0},
      /*c_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}),
      /*bias_shape_ptr*/ nullptr,
      /*output_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}),
      /*alpha_real*/ 1.0, /*alpha_imag*/ 0,
      /*beta*/ 1.0,
      /*precision_algorithm*/ PrecisionConfig::ALG_UNSET,
      /*algorithm*/ std::nullopt,
      /*compute_precision*/ se::blas::kDefaultComputePrecision,
      /*grad_x*/ false, /*grad_y*/ false,
      executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<CublasLtCmd>(
      s0, config.value(), se::gpu::BlasLt::Epilogue::kDefault, 0, slice_a,
      slice_b, slice_c, slice_d, BufferAllocation::Slice(),
      BufferAllocation::Slice(), BufferAllocation::Slice(),
      BufferAllocation::Slice(), BufferAllocation::Slice(),
      BufferAllocation::Slice(), BufferAllocation::Slice(), slice_workspace);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  std::vector<float> a_arr_1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> a_arr_2{2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> result_1{11, 11, 11, 27, 27, 27};
  std::vector<float> result_2{15, 15, 15, 31, 31, 31};

  auto run_cublaslt_test = [&](std::unique_ptr<se::Stream>& stream,
                               std::vector<float> a_arr,
                               std::vector<float> result) {
    se::DeviceMemory<float> a = executor->AllocateArray<float>(2 * 4);
    TF_ASSERT_OK(stream->Memcpy(&a, a_arr.data(), a_length));

    se::DeviceMemory<float> b = executor->AllocateArray<float>(4 * 3);
    std::vector<float> b_arr(12, 1);
    TF_ASSERT_OK(stream->Memcpy(&b, b_arr.data(), b_length));

    se::DeviceMemory<float> c = executor->AllocateArray<float>(2 * 3);
    std::vector<float> c_arr(6, 1);
    TF_ASSERT_OK(stream->Memcpy(&c, c_arr.data(), c_length));

    se::DeviceMemory<float> d = executor->AllocateArray<float>(2 * 3);
    TF_ASSERT_OK(stream->MemZero(&d, d_length));

    se::DeviceMemory<float> workspace =
        executor->AllocateArray<float>(1024 * 1024);
    TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

    ServiceExecutableRunOptions run_options;
    se::StreamExecutorMemoryAllocator allocator(executor);
    BufferAllocations allocations({a, b, c, d, workspace}, 0, &allocator);

    Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
        run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    TF_ASSERT_OK(thunk.Initialize(
        {executor, source, &allocations, stream.get(), stream.get()}));

    // Execute command buffer thunk and verify that it executed a GEMM.
    TF_ASSERT_OK(thunk.ExecuteOnStream(params));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `out` data back to host.
    std::vector<float> dst(6, 0);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), d, d_length));

    ASSERT_EQ(dst, result);

    // Prepare buffer allocation for updating command buffer.
    se::DeviceMemory<float> updated_d = executor->AllocateArray<float>(2 * 3);
    TF_ASSERT_OK(stream->MemZero(&updated_d, d_length));

    // Update buffer allocation to updated `d` buffer.
    allocations =
        BufferAllocations({a, b, c, updated_d, workspace}, 0, &allocator);

    // Thunk execution should automatically update underlying command
    // buffer.
    TF_ASSERT_OK(thunk.ExecuteOnStream(params));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `updated_out` data back to host.
    std::fill(dst.begin(), dst.end(), 0);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_d, d_length));

    ASSERT_EQ(dst, result);

    // Try to update the command buffer with the same buffers.
    TF_ASSERT_OK(stream->MemZero(&updated_d, d_length));

    // Thunk execution should automatically update underlying command
    // buffer.
    TF_ASSERT_OK(thunk.ExecuteOnStream(params));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `updated_out` data back to host.
    std::fill(dst.begin(), dst.end(), 0);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_d, d_length));

    ASSERT_EQ(dst, result);
  };
  std::thread t1(run_cublaslt_test, std::ref(stream1), a_arr_1, result_1);
  std::thread t2(run_cublaslt_test, std::ref(stream2), a_arr_2, result_2);
  t1.join();
  t2.join();
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
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  commands.Emplace<LaunchCmd>(s0, "AddI32", args_1, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b, c, d}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

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
  allocations = BufferAllocations({a, b, c, e}, 0, &allocator);

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

TEST(CommandBufferThunkTest, CaseCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

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
    branches[0].Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  {  // Case 1: b = b + b
    auto args = {slice_b, slice_b, slice_b};
    branches[1].Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<CaseCmd>(s0, slice_i, false, std::move(branches));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({index, a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

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

TEST(CommandBufferThunkTest, WhileCmd) {
  // TODO(ezhulenev): Find a way to test WhileCmd: add a test only TraceCmd that
  // could allow us trace custom kernels to update while loop iterations. Or
  // maybe add a CustomLaunchCmd and wrap loop update into custom kernel.
}

class CmdBufferTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
    debug_options.add_xla_gpu_enable_command_buffer(
        DebugOptions::DYNAMIC_SLICE_FUSION);
    return debug_options;
  }
};

TEST_F(CmdBufferTest, DynamicSliceFusionCmd) {
  // Hlo generated by below jax code
  // def scan_body(carry, x):
  //     sliced_x = lax.slice(x, (0, 0), (128, 128))
  //     result = jnp.dot(carry, sliced_x)
  //     new_carry = result
  //     return new_carry, result
  // @jax.jit
  // def run_scan(initial_carry, xs):
  //     final_carry, outputs = lax.scan(scan_body, initial_carry, xs, length=2)
  //     return final_carry, outputs

  const char* module_str = R"(
HloModule jit_run_scan

None.7 {
  Arg_0.8 = f32[128,128]{1,0} parameter(0)
  Arg_1.9 = f32[128,128]{1,0} parameter(1)
  dot.10 = f32[128,128]{1,0} dot(Arg_0.8, Arg_1.9), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.11 = (f32[128,128]{1,0}, f32[128,128]{1,0}) tuple(dot.10, dot.10)
}

region_0.12 {
  arg_tuple.13 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) parameter(0)
  get-tuple-element.14 = s32[] get-tuple-element(arg_tuple.13), index=0
  constant.18 = s32[] constant(1)
  add.34 = s32[] add(get-tuple-element.14, constant.18)
  get-tuple-element.15 = f32[128,128]{1,0} get-tuple-element(arg_tuple.13), index=1
  get-tuple-element.17 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.13), index=3
  constant.20 = s32[] constant(0)
  compare.21 = pred[] compare(get-tuple-element.14, constant.20), direction=LT
  constant.19 = s32[] constant(2)
  add.22 = s32[] add(get-tuple-element.14, constant.19)
  select.23 = s32[] select(compare.21, add.22, get-tuple-element.14)
  dynamic-slice.24 = f32[1,128,128]{2,1,0} dynamic-slice(get-tuple-element.17, select.23, constant.20, constant.20), dynamic_slice_sizes={1,128,128}
  reshape.25 = f32[128,128]{1,0} reshape(dynamic-slice.24)
  call.26 = (f32[128,128]{1,0}, f32[128,128]{1,0}) call(get-tuple-element.15, reshape.25), to_apply=None.7
  get-tuple-element.27 = f32[128,128]{1,0} get-tuple-element(call.26), index=0
  get-tuple-element.16 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.13), index=2
  get-tuple-element.28 = f32[128,128]{1,0} get-tuple-element(call.26), index=1
  reshape.29 = f32[1,128,128]{2,1,0} reshape(get-tuple-element.28)
  compare.30 = pred[] compare(get-tuple-element.14, constant.20), direction=LT
  add.31 = s32[] add(get-tuple-element.14, constant.19)
  select.32 = s32[] select(compare.30, add.31, get-tuple-element.14)
  dynamic-update-slice.33 = f32[2,128,128]{2,1,0} dynamic-update-slice(get-tuple-element.16, reshape.29, select.32, constant.20, constant.20)
  ROOT tuple.35 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) tuple(add.34, get-tuple-element.27, dynamic-update-slice.33, get-tuple-element.17)
} // region_0.12

region_1.36 {
  arg_tuple.37 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) parameter(0)
  get-tuple-element.39 = f32[128,128]{1,0} get-tuple-element(arg_tuple.37), index=1
  get-tuple-element.40 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.37), index=2
  get-tuple-element.41 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.37), index=3
  get-tuple-element.38 = s32[] get-tuple-element(arg_tuple.37), index=0
  constant.42 = s32[] constant(2)
  ROOT compare.43 = pred[] compare(get-tuple-element.38, constant.42), direction=LT
} // region_1.36

ENTRY main.49 {
  constant.3 = s32[] constant(0)
  Arg_0.1 = f32[128,128]{1,0} parameter(0)
  constant.4 = f32[] constant(0)
  broadcast.5 = f32[2,128,128]{2,1,0} broadcast(constant.4), dimensions={}
  Arg_1.2 = f32[2,128,128]{2,1,0} parameter(1)
  tuple.6 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) tuple(constant.3, Arg_0.1, broadcast.5, Arg_1.2)
  while.44 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) while(tuple.6), condition=region_1.36, body=region_0.12
  get-tuple-element.45 = s32[] get-tuple-element(while.44), index=0
  get-tuple-element.46 = f32[128,128]{1,0} get-tuple-element(while.44), index=1
  get-tuple-element.47 = f32[2,128,128]{2,1,0} get-tuple-element(while.44), index=2
  ROOT tuple.48 = (f32[128,128]{1,0}, f32[2,128,128]{2,1,0}) tuple(get-tuple-element.46, get-tuple-element.47)
}
)";

  // running with module without exclusive lock on GpuExecutable
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_require_exclusive_lock(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 2e-3}));

  // running with module with exclusive lock on GpuExecutable
  debug_options.set_xla_gpu_require_exclusive_lock(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(module_str, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 2e-3}));
}

}  // namespace xla::gpu

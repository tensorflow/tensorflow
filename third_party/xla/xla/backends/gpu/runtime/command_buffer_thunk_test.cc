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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/profiler_lock.h"

namespace xla::gpu {
using ::testing::HasSubstr;

using MemoryAccess = BufferUse::MemoryAccess;
using KernelArgsPacking = se::KernelLoaderSpec::KernelArgsPacking;

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
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(GpuExecutor()->GetPlatform()->Name()));
  return OwningExecutableSource{/*text=*/{},
                                /*binary=*/fatbin};
}

KernelArgsPacking CreateDefaultArgsPacking() {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceAddressArray>(&args);

    return se::PackKernelArgs(mem_args->device_addr_args(),
                              args.number_of_shared_bytes());
  };
}

// Some of the tests rely on CUDA 12.3+ features.
bool IsAtLeastCuda12300(const se::StreamExecutor* stream_executor) {
  const auto& device_description = stream_executor->GetDeviceDescription();
  const auto* cuda_cc =
      device_description.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc != nullptr) {
    // We need a recent driver to support the feature at runtime and we need a
    // recent version of the toolkit at compile time, so that we have access to
    // the driver's headers.
    if (std::min(device_description.driver_version(),
                 device_description.compile_time_toolkit_version()) >=
        stream_executor::SemanticVersion(12, 3, 0)) {
      return true;
    }
  }

  return false;
}

bool IsAtLeastCuda12900(const se::StreamExecutor* stream_executor) {
  const auto& device_description = stream_executor->GetDeviceDescription();
  const auto* cuda_cc =
      device_description.gpu_compute_capability().cuda_compute_capability();
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
    CommandExecutor::SynchronizationMode::kSerialize;

}  // namespace

TEST(CommandBufferThunkTest, MemcpyCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});
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

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<MemcpyDeviceToDeviceCmd>(
      ShapedSlice{slice_b, shape}, ShapedSlice{slice_a, shape}, byte_length);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

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
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<MemzeroCmd>(ShapedSlice{slice_a, shape});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  // Execute command buffer thunk and verify that it zeroes the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 0));
}

TEST(CommandBufferThunkTest, Memset32Cmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<Memset32Cmd>(slice_a, int32_t{84});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersDisabledDuringProfiling) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  ThunkSequence thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandSequence commands;
  commands.Emplace<Memset32Cmd>(slice_a, int32_t{12});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  constexpr bool kProfileCommandBuffersEnabled = false;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

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
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  ThunkSequence thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandSequence commands;
  commands.Emplace<Memset32Cmd>(slice_a, int32_t{12});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  constexpr bool kProfileCommandBuffersEnabled = true;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                          tsl::profiler::ProfilerLock::Acquire());

  // skip warm up iteration
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 12));
}

TEST(CommandBufferThunkTest, Memset32CmdOnDifferentStreams) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  se::DeviceAddress<int32_t> a = stream_executor->AllocateArray<int32_t>(2, 0);
  TF_ASSERT_OK(stream->MemZero(&a, 2 * sizeof(int32_t)));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc(/*index=*/0, a.size(), /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0 * sizeof(int32_t), sizeof(int32_t));
  BufferAllocation::Slice slice1(&alloc, 1 * sizeof(int32_t), sizeof(int32_t));

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<Memset32Cmd>(slice0, int32_t{12});
  commands.Emplace<Memset32Cmd>(slice1, int32_t{34});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(2, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, a.size()));

  ASSERT_EQ(dst, std::vector<int32_t>({12, 34}));
}

TEST(CommandBufferThunkTest, LaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

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

  std::vector<ShapedSlice> args = {
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(thunk.Initialize({stream_executor,
                                 static_cast<Thunk::ExecutableSource>(source),
                                 &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  auto packing = CreateDefaultArgsPacking();

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::KernelLoaderSpec spec,
                          stream_executor::gpu::GetAddI32TestKernelSpec(
                              stream_executor->GetPlatform()->id()));

  auto custom_kernel =
      CustomKernel("AddI32", std::move(spec), se::BlockDim(),
                   se::ThreadDim(4, 1, 1), /*shared_memory_bytes=*/0);

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

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
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(thunk.Initialize({stream_executor,
                                 static_cast<Thunk::ExecutableSource>(source),
                                 &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

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
  se::DeviceAddress<float> lhs = stream_executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = stream_executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = stream_executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      stream_executor->AllocateArray<float>(1024 * 1024);
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
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<GemmCmd>(config.value(), slice_lhs, slice_rhs, slice_out,
                            slice_workspace,
                            /*deterministic=*/true);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(6, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceAddress<float> updated_out =
      stream_executor->AllocateArray<float>(2 * 3);
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

TEST(CommandBufferThunkTest, ChildGemmCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12900(stream_executor)) {
    GTEST_SKIP() << "Child command is not supported for CUDA < 12.9";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

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
  se::DeviceAddress<float> lhs = stream_executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = stream_executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = stream_executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      stream_executor->AllocateArray<float>(1024 * 1024);
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
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence child_commands;
  child_commands.Emplace<GemmCmd>(config.value(), slice_lhs, slice_rhs,
                                  slice_out, slice_workspace,
                                  /*deterministic=*/true);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor child_executor,
      CommandExecutor::Create(std::move(child_commands), serialize));

  CommandSequence commands;
  commands.Emplace<ChildCmd>(std::move(child_executor));

  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(6, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceAddress<float> updated_out =
      stream_executor->AllocateArray<float>(2 * 3);
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

TEST(CommandBufferThunkTest, DISABLED_DynamicSliceFusionCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

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
  se::DeviceAddress<float> lhs = stream_executor->AllocateArray<float>(4 * 4);
  std::vector<float> lhs_arr{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = stream_executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = stream_executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      stream_executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  // Prepare buffer allocations for recording command buffer.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(
      /*index=*/0, fake_lhs_length, /*color=*/0);
  fake_allocations.emplace_back(
      /*index=*/1, rhs_length, /*color=*/0);
  fake_allocations.emplace_back(/*index=*/2, out_length,
                                /*color=*/0);

  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024,
                                /*color=*/0);
  BufferAllocation::Slice fake_slice_lhs(&fake_allocations[0], 0,
                                         fake_lhs_length);
  BufferAllocation::Slice slice_rhs(&fake_allocations[1], 0, rhs_length);
  BufferAllocation::Slice slice_out(&fake_allocations[2], 0, out_length);
  BufferAllocation::Slice slice_workspace(&fake_allocations[3], 0, 1024 * 1024);
  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence embed_commands;
  embed_commands.Emplace<GemmCmd>(config.value(), fake_slice_lhs, slice_rhs,
                                  slice_out, slice_workspace,
                                  /*deterministic=*/true);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor embed_executor,
      CommandExecutor::Create(std::move(embed_commands), serialize));

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
  std::vector<std::optional<PrimitiveType>> offset_primitive_types = {
      S64, std::nullopt, std::nullopt, std::nullopt};

  CommandSequence commands;
  commands.Emplace<DynamicSliceFusionCmd>(
      std::move(embed_executor), arguments, std::move(fake_allocations),
      offsets, orig_shapes, sliced_shapes, offset_primitive_types);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(6, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceAddress<float> updated_out =
      stream_executor->AllocateArray<float>(2 * 3);
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream1, stream_executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto stream2, stream_executor->CreateStream());

  // CublasLt formula: D = alpha*(A*B) + beta*(C),

  Shape a_shape = ShapeUtil::MakeShape(F32, {2, 4});
  int64_t a_length = sizeof(float) * 2 * 4;
  Shape b_shape = ShapeUtil::MakeShape(F32, {4, 3});
  int64_t b_length = sizeof(float) * 4 * 3;
  Shape c_shape = ShapeUtil::MakeShape(F32, {2, 3});
  int64_t c_length = sizeof(float) * 2 * 3;
  Shape d_shape = ShapeUtil::MakeShape(F32, {2, 3});
  int64_t d_length = sizeof(float) * 2 * 3;

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, a_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, b_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, c_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, d_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/4, 1024 * 1024, /*color=*/0);

  ShapedSlice slice_a{BufferAllocation::Slice{&alloc_a, 0, a_length}, a_shape};
  ShapedSlice slice_b{BufferAllocation::Slice(&alloc_b, 0, b_length), b_shape};
  ShapedSlice slice_c{BufferAllocation::Slice(&alloc_c, 0, c_length), c_shape};
  ShapedSlice slice_d{BufferAllocation::Slice(&alloc_d, 0, d_length), d_shape};
  ShapedSlice slice_workspace{
      BufferAllocation::Slice(&alloc_workspace, 0, 1024 * 1024),
      ShapeUtil::MakeShape(U8, {1024, 1024})};

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
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<CublasLtCmd>(CublasLtMatmulThunk(
      Thunk::ThunkInfo(), /*canonical_hlo=*/"", config.value(),
      se::gpu::BlasLt::Epilogue::kDefault, /*algorithm_idx=*/0,
      /*autotune_workspace_size=*/0, slice_a, slice_b, slice_c, slice_d,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, slice_workspace));
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  std::vector<float> a_arr_1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> a_arr_2{2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> result_1{11, 11, 11, 27, 27, 27};
  std::vector<float> result_2{15, 15, 15, 31, 31, 31};

  auto run_cublaslt_test = [&](std::unique_ptr<se::Stream>& stream,
                               std::vector<float> a_arr,
                               std::vector<float> result) {
    se::DeviceAddress<float> a = stream_executor->AllocateArray<float>(2 * 4);
    TF_ASSERT_OK(stream->Memcpy(&a, a_arr.data(), a_length));

    se::DeviceAddress<float> b = stream_executor->AllocateArray<float>(4 * 3);
    std::vector<float> b_arr(12, 1);
    TF_ASSERT_OK(stream->Memcpy(&b, b_arr.data(), b_length));

    se::DeviceAddress<float> c = stream_executor->AllocateArray<float>(2 * 3);
    std::vector<float> c_arr(6, 1);
    TF_ASSERT_OK(stream->Memcpy(&c, c_arr.data(), c_length));

    se::DeviceAddress<float> d = stream_executor->AllocateArray<float>(2 * 3);
    TF_ASSERT_OK(stream->MemZero(&d, d_length));

    se::DeviceAddress<float> workspace =
        stream_executor->AllocateArray<float>(1024 * 1024);
    TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

    ServiceExecutableRunOptions run_options;
    stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
    BufferAllocations allocations({a, b, c, d, workspace}, 0, &allocator);

    Thunk::ExecuteParams params =
        Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                     stream.get(), nullptr, nullptr, nullptr);

    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    TF_ASSERT_OK(thunk.Initialize(
        {stream_executor, source, &allocations, stream.get(), stream.get()}));

    // Execute command buffer thunk and verify that it executed a GEMM.
    TF_ASSERT_OK(thunk.ExecuteOnStream(params));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `out` data back to host.
    std::vector<float> dst(6, 0);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), d, d_length));

    ASSERT_EQ(dst, result);

    // Prepare buffer allocation for updating command buffer.
    se::DeviceAddress<float> updated_d =
        stream_executor->AllocateArray<float>(2 * 3);
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> d =
      stream_executor->AllocateArray<int32_t>(length, 0);

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

  std::vector<ShapedSlice> args{
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  std::vector<ShapedSlice> args_1{
      {slice_c, shape}, {slice_c, shape}, {slice_d, shape}};  // d = c + c
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  commands.Emplace<LaunchCmd>("AddI32", args_1, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c, d}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(thunk.Initialize({stream_executor,
                                 static_cast<Thunk::ExecutableSource>(source),
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
  se::DeviceAddress<int32_t> e =
      stream_executor->AllocateArray<int32_t>(length, 0);
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: index=0, a=42, b=0
  se::DeviceAddress<int32_t> index =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_i(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_i(&alloc_i, 0, sizeof(int32_t));
  Shape i_shape = ShapeUtil::MakeShape(S32, {});

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for branches.
  std::vector<CommandSequence> branches_sequence(2);

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Case 0: b = a + a
    std::vector<ShapedSlice> args{
        {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};
    branches_sequence[0].Emplace<LaunchCmd>("AddI32", args, args_access,
                                            LaunchDimensions(1, 4),
                                            /*shmem_bytes=*/0);
  }

  {  // Case 1: b = b + b
    std::vector<ShapedSlice> args{
        {slice_b, shape}, {slice_b, shape}, {slice_b, shape}};
    branches_sequence[1].Emplace<LaunchCmd>("AddI32", args, args_access,
                                            LaunchDimensions(1, 4),
                                            /*shmem_bytes=*/0);
  }

  std::vector<CommandExecutor> branches(2);
  TF_ASSERT_OK_AND_ASSIGN(
      branches[0],
      CommandExecutor::Create(std::move(branches_sequence[0]), serialize));
  TF_ASSERT_OK_AND_ASSIGN(
      branches[1],
      CommandExecutor::Create(std::move(branches_sequence[1]), serialize));

  // Prepare commands sequence for thunk.
  CommandSequence commands;
  commands.Emplace<CaseCmd>(ShapedSlice{slice_i, i_shape}, std::move(branches));
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({index, a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(thunk.Initialize({stream_executor,
                                 static_cast<Thunk::ExecutableSource>(source),
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
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: loop_cnt=0, num_iters=10, a=1, b=0
  se::DeviceAddress<bool> pred = stream_executor->AllocateArray<bool>(1, 0);
  se::DeviceAddress<int32_t> loop_cnt =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> num_iters =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&loop_cnt, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&num_iters, 10, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_pred(/*index=*/0, sizeof(bool), /*color=*/0);
  BufferAllocation alloc_loop_cnt(/*index=*/1, sizeof(int32_t), /*color=*/0);
  BufferAllocation alloc_num_iters(/*index=*/2, sizeof(int32_t), /*color=*/0);
  BufferAllocation alloc_a(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/4, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_pred(&alloc_pred, 0, sizeof(bool));
  BufferAllocation::Slice slice_loop_cnt(&alloc_loop_cnt, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_num_iters(&alloc_num_iters, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> cond_args{
      {slice_loop_cnt, shape}, {slice_pred, shape}, {slice_num_iters, shape}};
  auto cond_args_access = {MemoryAccess::kWrite, MemoryAccess::kWrite,
                           MemoryAccess::kRead};

  std::vector<ShapedSlice> body_args{
      {slice_a, shape}, {slice_b, shape}, {slice_b, shape}};  // b = a + b
  auto body_args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                           MemoryAccess::kWrite};

  // Prepare commands sequence for loop `cond`.
  CommandSequence cond_commands;
  cond_commands.Emplace<LaunchCmd>("IncAndCmp", cond_args, cond_args_access,
                                   LaunchDimensions(1, 1),
                                   /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor cond_executor,
      CommandExecutor::Create(std::move(cond_commands), serialize));

  // Prepare commands sequence for loop `body`.
  CommandSequence body_commands;
  body_commands.Emplace<LaunchCmd>("AddI32", body_args, body_args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor body_executor,
      CommandExecutor::Create(std::move(body_commands), serialize));

  // Prepare commands sequence for thunk.
  CommandSequence commands;
  commands.Emplace<WhileCmd>(slice_pred, std::move(cond_executor),
                             std::move(body_executor));
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({pred, loop_cnt, num_iters, a, b}, 0,
                                &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(thunk.Initialize({stream_executor,
                                 static_cast<Thunk::ExecutableSource>(source),
                                 &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value 10 times.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 10));

  // Initialize `loop_cnt` to `5` and check that we run only 5 iterations.
  TF_ASSERT_OK(stream->Memset32(&loop_cnt, 5, sizeof(int32_t)));

  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 15));
}

TEST(CommandBufferThunkTest, ToStringPrintsNestedThunks) {
  BufferAllocation alloc_a(/*index=*/0, /*size=*/4, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, /*offset=*/0, /*size=*/4);
  CommandSequence commands;
  commands.Emplace<Memset32Cmd>(slice_a, int32_t{42});
  TF_ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      CommandExecutor::Create(std::move(commands), serialize));
  ThunkSequence thunks;
  thunks.emplace_back(
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 42, slice_a));
  CommandBufferThunk thunk(
      std::move(executor), Thunk::ThunkInfo(),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks)));
  EXPECT_THAT(thunk.ToString(/*indent=*/1),
              HasSubstr("    000: kMemset32BitValue"));
}

}  // namespace xla::gpu

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

#include "xla/stream_executor/gpu/gpu_command_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace stream_executor::gpu {

using ExecutionScopeId = CommandBuffer::ExecutionScopeId;

static Platform* GpuPlatform() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  return PlatformManager::PlatformWithName(name).value();
}

static MultiKernelLoaderSpec GetAddI32KernelSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/3);
#if defined(GOOGLE_CUDA)
  spec.AddCudaPtxInMemory(internal::kAddI32Kernel, "add");
#elif defined(TENSORFLOW_USE_ROCM)
  spec.AddCudaCubinInMemory(internal::kAddI32KernelModule, "add");
#endif
  return spec;
}

using AddI32Kernel =
    TypedKernelFactory<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                       DeviceMemory<int32_t>>;
using MulI32Kernel =
    TypedKernelFactory<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                       DeviceMemory<int32_t>>;
using IncAndCmpKernel =
    TypedKernelFactory<DeviceMemory<int32_t>, DeviceMemory<bool>, int32_t>;

using AddI32Ptrs3 = TypedKernelFactory<internal::Ptrs3<int32_t>>;

static constexpr auto nested = CommandBuffer::Mode::kNested;    // NOLINT
static constexpr auto primary = CommandBuffer::Mode::kPrimary;  // NOLINT

template <typename Info>
static std::vector<GpuGraphNodeHandle> Deps(Info info) {
  if (auto deps = GpuDriver::GraphNodeGetDependencies(info.handle); deps.ok()) {
    return *deps;
  }
  return {GpuGraphNodeHandle(0xDEADBEEF)};
}

template <typename... Infos>
static std::vector<GpuGraphNodeHandle> ExpectedDeps(Infos... info) {
  return {info.handle...};
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

TEST(GpuCommandBufferTest, LaunchSingleKernel) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Create a command buffer with a single kernel launch.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->Launch(add, ThreadDim(), BlockDim(4), a, b, c));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // Update command buffer to write into `d` buffer.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->Launch(add, ThreadDim(), BlockDim(4), a, b, d));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(CudaCommandBufferTest, TraceSingleKernel) {
#if defined(TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Not supported on ROCM";
#endif
#if CUDA_VERSION < 12030
  GTEST_SKIP() << "Command buffer tracing is not supported";
#endif
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Register a kernel with a custom arguments packing function that packs
  // device memory arguments into a struct with pointers.
  MultiKernelLoaderSpec spec(/*arity=*/1, [&](const Kernel& kernel,
                                              const KernelArgs& args) {
    auto bufs = Cast<KernelArgsDeviceMemoryArray>(&args)->device_memory_args();
    auto cast = [](auto m) { return reinterpret_cast<int32_t*>(m.opaque()); };
    return PackKernelArgs(/*shmem_bytes=*/0, internal::Ptrs3<int32_t>{
                                                 cast(bufs[0]),
                                                 cast(bufs[1]),
                                                 cast(bufs[2]),
                                             });
  });
  spec.AddInProcessSymbol(internal::GetAddI32Ptrs3Kernel(), "add");

  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Ptrs3::Create(executor, spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Use an array of device memory base pointers as argument to test packing.
  KernelArgsDeviceMemoryArray args({a, b, c}, 0);

  // Create a command buffer by tracing kernel launch operations.
  auto cmd_buffer = TraceCommandBufferFactory::Create(
      executor,
      [&](Stream* stream) {
        return executor->Launch(stream, ThreadDim(), BlockDim(4), *add, args);
      },
      primary);

  TF_ASSERT_OK(cmd_buffer.status());
  TF_ASSERT_OK(executor->Submit(stream.get(), **cmd_buffer));

  // Copy data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, LaunchNestedCommandBuffer) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec = GetAddI32KernelSpec();
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Create a command buffer with a single kernel launch.
  auto primary_cmd = executor->CreateCommandBuffer(primary).value();
  auto nested_cmd = executor->CreateCommandBuffer(nested).value();
  TF_ASSERT_OK(nested_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c));
  TF_ASSERT_OK(primary_cmd->AddNestedCommandBuffer(*nested_cmd));
  TF_ASSERT_OK(primary_cmd->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *primary_cmd));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // Update command buffer to write into `d` buffer by creating a new nested
  // command buffer.
  nested_cmd = executor->CreateCommandBuffer(nested).value();
  TF_ASSERT_OK(nested_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, d));
  TF_ASSERT_OK(primary_cmd->Update());
  TF_ASSERT_OK(primary_cmd->AddNestedCommandBuffer(*nested_cmd));
  TF_ASSERT_OK(primary_cmd->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *primary_cmd));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, MemcpyDeviceToDevice) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=uninitialized
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Create a command buffer with a single a to b memcpy command.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->MemcpyDeviceToDevice(&b, a, byte_length));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  ASSERT_EQ(dst, expected);

  // Update command buffer to swap the memcpy direction.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->MemcpyDeviceToDevice(&a, b, byte_length));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  // Clear destination to test that command buffer actually copied memory.
  TF_ASSERT_OK(stream->Memset32(&a, 0, byte_length));

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `a` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, Memset) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  // Create a command buffer with a single memset command.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->Memset(&a, uint32_t{42}, length));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  ASSERT_EQ(dst, expected);

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->Memset(&a, uint32_t{43}, length));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  expected = {43, 43, 43, 43};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, Barriers) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Allocate device buffers for memset operations.
  std::vector<DeviceMemory<int32_t>> buffers;
  for (size_t i = 0; i < 6; ++i) {
    buffers.push_back(executor->AllocateArray<int32_t>(1, 0));
  }

  // Transfer buffers data back to host.
  auto transfer_buffers = [&]() -> std::vector<int32_t> {
    std::vector<int32_t> dst(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); ++i) {
      TF_CHECK_OK(stream->Memcpy(dst.data() + i, buffers[i], sizeof(int32_t)));
    }
    return dst;
  };

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern) {
    // Check that root barrier ignored.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[0], bit_pattern + 0, 1));
    // Check barrier after a single command.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[1], bit_pattern + 1, 1));
    // Check that repeated barriers are no-op.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[2], bit_pattern + 2, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[3], bit_pattern + 3, 1));
    // Check that barrier can have multiple dependencies.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[4], bit_pattern + 4, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(&buffers[5], bit_pattern + 5, 1));
    // Check that barrier can be that last command.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier());
    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a DAG of memset commands.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());
  ASSERT_EQ(gpu_cmd_buffer->nodes().size(), 6);
  ASSERT_EQ(gpu_cmd_buffer->barriers().size(), 6);

  auto nodes = gpu_cmd_buffer->nodes();
  auto barriers = gpu_cmd_buffer->barriers();

  // First barrier does not have any dependencies.
  EXPECT_TRUE(barriers[0].is_barrier_node);
  EXPECT_TRUE(Deps(barriers[0]).empty());

  // Second barrier reuses first memset node.
  EXPECT_FALSE(barriers[1].is_barrier_node);
  EXPECT_EQ(barriers[1].handle, nodes[0].handle);

  // Third and fourth barriers reuse second memset node.
  EXPECT_FALSE(barriers[2].is_barrier_node);
  EXPECT_FALSE(barriers[3].is_barrier_node);
  EXPECT_EQ(barriers[2].handle, nodes[1].handle);
  EXPECT_EQ(barriers[3].handle, nodes[1].handle);

  // Fifth and sixth barriers are barrier nodes.
  EXPECT_TRUE(barriers[4].is_barrier_node);
  EXPECT_TRUE(barriers[5].is_barrier_node);

  EXPECT_EQ(Deps(barriers[4]), ExpectedDeps(nodes[2], nodes[3]));
  EXPECT_EQ(Deps(barriers[5]), ExpectedDeps(nodes[4], nodes[5]));

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  expected = {43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, IndependentExecutionScopes) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  CommandBuffer::ExecutionScopeId s0 = CommandBuffer::ExecutionScopeId(0);
  CommandBuffer::ExecutionScopeId s1 = CommandBuffer::ExecutionScopeId(1);

  // Allocate device buffers for memset operations.
  std::vector<DeviceMemory<int32_t>> buffers;
  for (size_t i = 0; i < 4; ++i) {
    buffers.push_back(executor->AllocateArray<int32_t>(1, 0));
  }

  // Transfer buffers data back to host.
  auto transfer_buffers = [&]() -> std::vector<int32_t> {
    std::vector<int32_t> dst(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); ++i) {
      TF_CHECK_OK(stream->Memcpy(dst.data() + i, buffers[i], sizeof(int32_t)));
    }
    return dst;
  };

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern) {
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[0], bit_pattern + 0, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[1], bit_pattern + 1, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[2], bit_pattern + 2, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[3], bit_pattern + 3, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier(s0));
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier(s1));
    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a DAG of memset commands.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  std::vector<int32_t> expected = {42, 43, 44, 45};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());

  auto nodes0 = gpu_cmd_buffer->nodes(s0);
  auto nodes1 = gpu_cmd_buffer->nodes(s1);
  auto barriers0 = gpu_cmd_buffer->barriers(s0);
  auto barriers1 = gpu_cmd_buffer->barriers(s1);

  ASSERT_EQ(nodes0.size(), 2);
  ASSERT_EQ(nodes1.size(), 2);
  ASSERT_EQ(barriers0.size(), 1);
  ASSERT_EQ(barriers1.size(), 1);

  EXPECT_TRUE(barriers0[0].is_barrier_node);
  EXPECT_TRUE(barriers1[0].is_barrier_node);

  EXPECT_EQ(Deps(barriers0[0]), ExpectedDeps(nodes0[0], nodes0[1]));
  EXPECT_EQ(Deps(barriers1[0]), ExpectedDeps(nodes1[0], nodes1[1]));

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  expected = {43, 44, 45, 46};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, ExecutionScopeBarriers) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  CommandBuffer::ExecutionScopeId s0 = CommandBuffer::ExecutionScopeId(0);
  CommandBuffer::ExecutionScopeId s1 = CommandBuffer::ExecutionScopeId(1);
  CommandBuffer::ExecutionScopeId s2 = CommandBuffer::ExecutionScopeId(2);

  // Allocate device buffers for memset operations.
  std::vector<DeviceMemory<int32_t>> buffers;
  for (size_t i = 0; i < 7; ++i) {
    buffers.push_back(executor->AllocateArray<int32_t>(1, 0));
  }

  // Transfer buffers data back to host.
  auto transfer_buffers = [&]() -> std::vector<int32_t> {
    std::vector<int32_t> dst(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); ++i) {
      TF_CHECK_OK(stream->Memcpy(dst.data() + i, buffers[i], sizeof(int32_t)));
    }
    return dst;
  };

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern) {
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[0], bit_pattern + 0, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[1], bit_pattern + 1, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[2], bit_pattern + 2, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[3], bit_pattern + 3, 1));
    // This will synchronize scopes 0 and 1 and also create an empty scope 2.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier({s0, s1, s2}));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[4], bit_pattern + 4, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[5], bit_pattern + 5, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s2, &buffers[6], bit_pattern + 6, 1));
    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a DAG of memset commands.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());

  auto nodes0 = gpu_cmd_buffer->nodes(s0);
  auto nodes1 = gpu_cmd_buffer->nodes(s1);
  auto nodes2 = gpu_cmd_buffer->nodes(s2);
  auto barriers0 = gpu_cmd_buffer->barriers(s0);
  auto barriers1 = gpu_cmd_buffer->barriers(s1);
  auto barriers2 = gpu_cmd_buffer->barriers(s2);

  ASSERT_EQ(nodes0.size(), 3);
  ASSERT_EQ(nodes1.size(), 3);
  ASSERT_EQ(nodes2.size(), 1);
  ASSERT_EQ(barriers0.size(), 2);
  ASSERT_EQ(barriers1.size(), 2);
  ASSERT_EQ(barriers2.size(), 2);

  // All barriers are real barrier nodes.
  EXPECT_TRUE(barriers0[0].is_barrier_node && barriers0[1].is_barrier_node);
  EXPECT_TRUE(barriers1[0].is_barrier_node && barriers1[1].is_barrier_node);
  EXPECT_TRUE(barriers2[0].is_barrier_node && barriers2[1].is_barrier_node);

  // All scopes share a broadcasted barrier.
  EXPECT_TRUE(barriers0[1].handle == barriers1[1].handle);
  EXPECT_TRUE(barriers1[1].handle == barriers2[1].handle);

  EXPECT_EQ(Deps(barriers0[0]), ExpectedDeps(nodes0[0], nodes0[1]));
  EXPECT_EQ(Deps(barriers1[0]), ExpectedDeps(nodes1[0], nodes1[1]));

  EXPECT_TRUE(Deps(barriers2[0]).empty());
  EXPECT_EQ(Deps(barriers2[1]),
            ExpectedDeps(barriers0[0], barriers1[0], barriers2[0]));

  EXPECT_EQ(Deps(nodes0[2]), ExpectedDeps(barriers0[1]));
  EXPECT_EQ(Deps(nodes1[2]), ExpectedDeps(barriers1[1]));
  EXPECT_EQ(Deps(nodes2[0]), ExpectedDeps(barriers2[1]));

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  expected = {43, 44, 45, 46, 47, 48, 49};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, ExecutionScopeOneDirectionalBarriers) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  CommandBuffer::ExecutionScopeId s0 = CommandBuffer::ExecutionScopeId(0);
  CommandBuffer::ExecutionScopeId s1 = CommandBuffer::ExecutionScopeId(1);

  // Allocate device buffers for memset operations.
  std::vector<DeviceMemory<int32_t>> buffers;
  for (size_t i = 0; i < 6; ++i) {
    buffers.push_back(executor->AllocateArray<int32_t>(1, 0));
  }

  // Transfer buffers data back to host.
  auto transfer_buffers = [&]() -> std::vector<int32_t> {
    std::vector<int32_t> dst(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); ++i) {
      TF_CHECK_OK(stream->Memcpy(dst.data() + i, buffers[i], sizeof(int32_t)));
    }
    return dst;
  };

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern) {
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[0], bit_pattern + 0, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[1], bit_pattern + 1, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[2], bit_pattern + 2, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[3], bit_pattern + 3, 1));
    // This will synchronize scopes 0 and 1.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier(s0, s1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[4], bit_pattern + 4, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s1, &buffers[5], bit_pattern + 5, 1));
    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a DAG of memset commands.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());

  auto nodes0 = gpu_cmd_buffer->nodes(s0);
  auto nodes1 = gpu_cmd_buffer->nodes(s1);
  auto barriers0 = gpu_cmd_buffer->barriers(s0);
  auto barriers1 = gpu_cmd_buffer->barriers(s1);

  ASSERT_EQ(nodes0.size(), 3);
  ASSERT_EQ(nodes1.size(), 3);
  ASSERT_EQ(barriers0.size(), 1);
  ASSERT_EQ(barriers1.size(), 2);

  // All barriers are real barrier nodes.
  EXPECT_TRUE(barriers0[0].is_barrier_node);
  EXPECT_TRUE(barriers1[0].is_barrier_node && barriers1[1].is_barrier_node);

  EXPECT_EQ(Deps(barriers0[0]), ExpectedDeps(nodes0[0], nodes0[1]));
  EXPECT_EQ(Deps(barriers1[0]), ExpectedDeps(nodes1[0], nodes1[1]));
  EXPECT_EQ(Deps(barriers1[1]), ExpectedDeps(barriers0[0], barriers1[0]));
  EXPECT_EQ(Deps(nodes0[2]), ExpectedDeps(barriers0[0]));
  EXPECT_EQ(Deps(nodes1[2]), ExpectedDeps(barriers1[1]));

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  expected = {43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, ConditionalIf) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0, pred=true
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // if (pred == true) c = a + b
  CommandBuffer::Builder then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->If(pred, then_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Reset predicate to false and clear output buffer.
  constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Submit the same command buffer, but this time it should not execute
  // conditional branch as conditional handle should be updated to false.
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  std::vector<int32_t> zeroes = {0, 0, 0, 0};
  ASSERT_EQ(dst, zeroes);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // Set predicate buffer to true to run conditional command buffer.
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));

  // if (pred == true) d = a + b (write to a new location).
  then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, d);
  };

  // Update command buffer with a conditional to use new builder.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->If(pred, then_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIfElse) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load multiplication kernel.
  MultiKernelLoaderSpec mul_spec(/*arity=*/3);
  mul_spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "mul");
  TF_ASSERT_OK_AND_ASSIGN(auto mul, MulI32Kernel::Create(executor, mul_spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=2, b=3, c=0, pred=true
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 2, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 3, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // if (pred == true) c = a + b
  CommandBuffer::Builder then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // if (pred == false) c = a * b
  CommandBuffer::Builder else_builder = [&](CommandBuffer* else_cmd) {
    return else_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->IfElse(pred, then_builder, else_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Reset predicate to false.
  constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));

  // Submit the same command buffer, but this time it should execute `else`
  // branch and multiply inputs.
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  std::vector<int32_t> expected_mul = {6, 6, 6, 6};
  ASSERT_EQ(dst, expected_mul);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // if (pred == false) d = a * b (write to a new location).
  else_builder = [&](CommandBuffer* else_cmd) {
    return else_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, d);
  };

  // Update command buffer with a conditional to use new `else` builder.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->IfElse(pred, then_builder, else_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalCase) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load multiplication kernel.
  MultiKernelLoaderSpec mul_spec(/*arity=*/3);
  mul_spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "mul");
  TF_ASSERT_OK_AND_ASSIGN(auto mul, MulI32Kernel::Create(executor, mul_spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=2, b=3, c=0, index=0
  DeviceMemory<int32_t> index = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 2, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, 3, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // if (index == 0) c = a + b
  CommandBuffer::Builder branch0 = [&](CommandBuffer* branch0_cmd) {
    return branch0_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // if (index == 1) c = a * b
  CommandBuffer::Builder branch1 = [&](CommandBuffer* branch1_cmd) {
    return branch1_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->Case(index, {branch0, branch1}));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Set index to `1`
  TF_ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  // Submit the same command buffer, but this time it should multiply inputs.
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  std::vector<int32_t> expected_mul = {6, 6, 6, 6};
  ASSERT_EQ(dst, expected_mul);

  // Set index to `-1` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, -1, sizeof(int32_t)));

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_mul);

  // Set index to `2` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, 2, sizeof(int32_t)));

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalFor) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=0, loop_counter=100
  DeviceMemory<int32_t> loop_counter = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  // Set loop counter to 100 to check that command buffer resets it.
  TF_ASSERT_OK(stream->Memset32(&loop_counter, 100, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Loop body: b = a + b
  CommandBuffer::Builder body_builder = [&](CommandBuffer* body_cmd) {
    return body_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, b);
  };

  int32_t num_iters = 10;

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->For(num_iters, loop_counter, body_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalWhile) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load inc_and_cmp kernel.
  MultiKernelLoaderSpec icmp_spec(/*arity=*/3);
  icmp_spec.AddInProcessSymbol(internal::GetIncAndCmpKernel(), "inc_and_cmp");
  TF_ASSERT_OK_AND_ASSIGN(auto inc_and_cmp,
                          IncAndCmpKernel::Create(executor, icmp_spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=0, loop_counter=0, pred=false
  // Value of `pred` is not important, as it will be updated by `cond_builder`
  // below.
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> loop_counter = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  static constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));
  TF_ASSERT_OK(stream->Memset32(&loop_counter, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  int32_t num_iters = 10;

  // Loop cond: loop_counter++ < num_iters;
  CommandBuffer::ExecutionScopeBuilder cond_builder =
      [&](ExecutionScopeId id, CommandBuffer* cond_cmd) {
        return cond_cmd->Launch(inc_and_cmp, id, ThreadDim(), BlockDim(),
                                loop_counter, pred, num_iters);
      };

  // Loop body: b = a + b
  CommandBuffer::Builder body_builder = [&](CommandBuffer* body_cmd) {
    return body_cmd->Launch(add, ThreadDim(), BlockDim(length), a, b, b);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->While(pred, cond_builder, body_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIfInExecutionScope) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  CommandBuffer::ExecutionScopeId s0 = CommandBuffer::ExecutionScopeId(0);
  CommandBuffer::ExecutionScopeId s1 = CommandBuffer::ExecutionScopeId(1);

  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));

  // Allocate device buffers for memset operations.
  std::vector<DeviceMemory<int32_t>> buffers;
  for (size_t i = 0; i < 3; ++i) {
    buffers.push_back(executor->AllocateArray<int32_t>(1, 0));
  }

  // Transfer buffers back to host.
  auto transfer_buffers = [&]() -> std::vector<int32_t> {
    std::vector<int32_t> dst(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); ++i) {
      stream->Memcpy(dst.data() + i, buffers[i], sizeof(int32_t)).IgnoreError();
    }
    return dst;
  };

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern) {
    // Record memsets in execution scope #0
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[0], bit_pattern + 0, 1));
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &buffers[1], bit_pattern + 1, 1));

    // Record If in execution scope #1
    TF_RETURN_IF_ERROR(cmd_buffer->If(s1, pred, [&](CommandBuffer* then_cmd) {
      return then_cmd->Memset(&buffers[2], bit_pattern + 2, 1);
    }));

    // Create a barrier in execution scope #0.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier(s0));

    // Create a barrier between two execution scopes.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier({s0, s1}));

    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a DAG of memset commands.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  std::vector<int32_t> expected = {42, 43, 44};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());

  auto nodes0 = gpu_cmd_buffer->nodes(s0);
  auto nodes1 = gpu_cmd_buffer->nodes(s1);
  auto barriers0 = gpu_cmd_buffer->barriers(s0);
  auto barriers1 = gpu_cmd_buffer->barriers(s1);

  ASSERT_EQ(nodes0.size(), 2);
  ASSERT_EQ(nodes1.size(), 2);
  ASSERT_EQ(barriers0.size(), 3);
  ASSERT_EQ(barriers1.size(), 3);

  EXPECT_EQ(Deps(barriers0[0]), ExpectedDeps(nodes0[0], nodes0[1]));
  EXPECT_EQ(barriers0[0].handle, barriers0[1].handle);

  EXPECT_EQ(barriers1[0].handle, nodes1[0].handle);
  EXPECT_EQ(barriers1[1].handle, nodes1[1].handle);

  // s0 and s1 share broadcasted barrier.
  EXPECT_TRUE(barriers0[2].handle == barriers1[2].handle);
  EXPECT_EQ(Deps(barriers0[2]), ExpectedDeps(barriers0[1], nodes1[1]));

  // TODO(b/326284532): Add a test for bit pattern update.

  // Disable conditional branch.
  constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));
  TF_ASSERT_OK(stream->MemZero(&buffers[2], sizeof(int32_t)));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  expected = {42, 43, 0};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, ConditionalWhileInExecutionScope) {
  if (!IsAtLeastCuda12300()) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  CommandBuffer::ExecutionScopeId s0 = CommandBuffer::ExecutionScopeId(0);
  CommandBuffer::ExecutionScopeId s1 = CommandBuffer::ExecutionScopeId(1);

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load inc_and_cmp kernel.
  MultiKernelLoaderSpec icmp_spec(/*arity=*/3);
  icmp_spec.AddInProcessSymbol(internal::GetIncAndCmpKernel(), "inc_and_cmp");
  TF_ASSERT_OK_AND_ASSIGN(auto inc_and_cmp,
                          IncAndCmpKernel::Create(executor, icmp_spec));

  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> loop_counter = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(1, 0);

  TF_ASSERT_OK(stream->MemZero(&loop_counter, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, sizeof(int32_t)));
  TF_ASSERT_OK(stream->MemZero(&b, sizeof(int32_t)));

  auto record = [&](CommandBuffer* cmd_buffer, uint32_t bit_pattern,
                    int32_t num_iters) {
    // Record memset in execution scope #0
    TF_RETURN_IF_ERROR(cmd_buffer->Memset(s0, &c, bit_pattern, 1));

    // Record While in execution scope #1
    TF_RETURN_IF_ERROR(cmd_buffer->While(
        s1, pred,
        // Loop cond: loop_counter++ < num_iters;
        [&](ExecutionScopeId id, CommandBuffer* cond_cmd) {
          return cond_cmd->Launch(inc_and_cmp, id, ThreadDim(), BlockDim(),
                                  loop_counter, pred, num_iters);
        },
        // Loop body: b = a + b
        [&](CommandBuffer* body_cmd) {
          return body_cmd->Launch(add, ThreadDim(), BlockDim(), a, b, b);
        }));

    // Create a barrier between two execution scopes.
    TF_RETURN_IF_ERROR(cmd_buffer->Barrier({s0, s1}));

    return cmd_buffer->Finalize();
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(record(cmd_buffer.get(), 42, 10));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  // Copy `b` and `c` data back to host.
  int32_t b_dst, c_dst;
  TF_ASSERT_OK(stream->Memcpy(&b_dst, b, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memcpy(&c_dst, c, sizeof(int32_t)));

  EXPECT_EQ(b_dst, 10);
  EXPECT_EQ(c_dst, 42);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = GpuCommandBuffer::Cast(cmd_buffer.get());

  auto nodes0 = gpu_cmd_buffer->nodes(s0);
  auto nodes1 = gpu_cmd_buffer->nodes(s1);
  auto barriers0 = gpu_cmd_buffer->barriers(s0);
  auto barriers1 = gpu_cmd_buffer->barriers(s1);

  // s0 should have only one real barrier joining while op and memset.
  ASSERT_EQ(nodes0.size(), 1);
  ASSERT_EQ(nodes1.size(), 3);
  ASSERT_EQ(barriers0.size(), 2);
  ASSERT_EQ(barriers1.size(), 4);

  // The final barrier that joins while and memset.
  EXPECT_EQ(Deps(barriers0[1]), ExpectedDeps(nodes0[0], nodes1[2]));

  // Update bit pattern and number of iterations.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43, 20));

  TF_ASSERT_OK(stream->MemZero(&loop_counter, sizeof(int32_t)));
  TF_ASSERT_OK(stream->MemZero(&b, sizeof(int32_t)));
  TF_ASSERT_OK(executor->Submit(stream.get(), *cmd_buffer));

  TF_ASSERT_OK(stream->Memcpy(&b_dst, b, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memcpy(&c_dst, c, sizeof(int32_t)));

  EXPECT_EQ(b_dst, 20);
  EXPECT_EQ(c_dst, 43);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

#define BENCHMARK_SIZES(NAME) \
  BENCHMARK(NAME)->Arg(8)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);

// In benchmarks we construct command buffers in nested mode when we
// do not want to measure graph executable instantiation overhead.
static void BM_CreateCommandBuffer(benchmark::State& state) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  for (auto s : state) {
    auto cmd_buffer = executor->CreateCommandBuffer(nested).value();
    for (int i = 1; i < state.range(0); ++i) {
      CHECK_OK(cmd_buffer->Launch(add, ThreadDim(), BlockDim(4), b, b, b));
    }
    CHECK_OK(cmd_buffer->Finalize());
  }
}

BENCHMARK_SIZES(BM_CreateCommandBuffer);

static void BM_TraceCommandBuffer(benchmark::State& state) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  for (auto s : state) {
    auto launch_kernels = [&](Stream* stream) {
      for (int i = 1; i < state.range(0); ++i) {
        CHECK_OK(stream->ThenLaunch(ThreadDim(), BlockDim(4), add, b, b, b));
      }
      return absl::OkStatus();
    };

    CHECK_OK(
        TraceCommandBufferFactory::Create(executor, launch_kernels, nested));
  }
}

BENCHMARK_SIZES(BM_TraceCommandBuffer);

static void BM_UpdateCommandBuffer(benchmark::State& state) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, spec));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  for (int i = 1; i < state.range(0); ++i) {
    CHECK_OK(cmd_buffer->Launch(add, ThreadDim(), BlockDim(4), b, b, b));
  }
  CHECK_OK(cmd_buffer->Finalize());

  for (auto s : state) {
    CHECK_OK(cmd_buffer->Update());
    for (int i = 1; i < state.range(0); ++i) {
      CHECK_OK(cmd_buffer->Launch(add, ThreadDim(), BlockDim(4), b, b, b));
    }
    CHECK_OK(cmd_buffer->Finalize());
  }
}

BENCHMARK_SIZES(BM_UpdateCommandBuffer);

}  // namespace stream_executor::gpu

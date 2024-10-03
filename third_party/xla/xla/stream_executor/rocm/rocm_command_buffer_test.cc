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

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace stream_executor::gpu {

using ExecutionScopeId = CommandBuffer::ExecutionScopeId;

static Platform* GpuPlatform() {
  return PlatformManager::PlatformWithName("ROCM").value();
}

static MultiKernelLoaderSpec GetAddI32KernelSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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
static std::vector<hipGraphNode_t> Deps(Info info) {
  if (auto deps = GpuDriver::GraphNodeGetDependencies(info.handle); deps.ok()) {
    return *deps;
  }
  return {hipGraphNode_t(0xDEADBEEF)};
}

template <typename... Infos>
static std::vector<hipGraphNode_t> ExpectedDeps(Infos... info) {
  return {info.handle...};
}

TEST(RocmCommandBufferTest, LaunchSingleKernel) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(RocmCommandBufferTest, LaunchNestedCommandBuffer) {
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

  TF_ASSERT_OK(primary_cmd->Submit(stream.get()));

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

  TF_ASSERT_OK(primary_cmd->Submit(stream.get()));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(RocmCommandBufferTest, MemcpyDeviceToDevice) {
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `a` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(RocmCommandBufferTest, Memset) {
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  ASSERT_EQ(dst, expected);

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->Memset(&a, uint32_t{43}, length));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  expected = {43, 43, 43, 43};
  ASSERT_EQ(dst, expected);
}

TEST(RocmCommandBufferTest, Barriers) {
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  RocmCommandBuffer* gpu_cmd_buffer = RocmCommandBuffer::Cast(cmd_buffer.get());
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  expected = {43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(RocmCommandBufferTest, IndependentExecutionScopes) {
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  std::vector<int32_t> expected = {42, 43, 44, 45};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  RocmCommandBuffer* gpu_cmd_buffer = RocmCommandBuffer::Cast(cmd_buffer.get());

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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  expected = {43, 44, 45, 46};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(RocmCommandBufferTest, ExecutionScopeBarriers) {
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  RocmCommandBuffer* gpu_cmd_buffer = RocmCommandBuffer::Cast(cmd_buffer.get());

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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  expected = {43, 44, 45, 46, 47, 48, 49};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(RocmCommandBufferTest, ExecutionScopeOneDirectionalBarriers) {
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  RocmCommandBuffer* gpu_cmd_buffer = RocmCommandBuffer::Cast(cmd_buffer.get());

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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  expected = {43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);
}

class RocmCommandBufferCaseTest : public testing::TestWithParam<int> {
 protected:
  int GetNumCases() { return GetParam(); }

  int GetEffectiveIndex(int i) {
    return (i < 0 || i >= GetNumCases()) ? GetNumCases() - 1 : i;
  }
};

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
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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

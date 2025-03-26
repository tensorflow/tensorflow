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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace stream_executor::gpu {

using testing::ElementsAre;
using testing::IsEmpty;
using tsl::testing::IsOkAndHolds;

static GpuCommandBuffer* CastToGpuCommandBuffer(CommandBuffer* command_buffer) {
  return static_cast<GpuCommandBuffer*>(command_buffer);
}

static Platform* GpuPlatform() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  return PlatformManager::PlatformWithName(name).value();
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

// Some of the tests rely on CUDA 12.3+ features.
static bool IsAtLeastCuda12300(
    const stream_executor::StreamExecutor* executor) {
  if (executor->GetPlatform()->id() != cuda::kCudaPlatformId) {
    return false;
  }
  if (std::min({executor->GetDeviceDescription().runtime_version(),
                executor->GetDeviceDescription().driver_version()}) <
      SemanticVersion{12, 3, 0}) {
    return false;
  }
  return true;
}

TEST(GpuCommandBufferTest, LaunchSingleKernel) {
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

TEST(CudaCommandBufferTest, TraceSingleKernel) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (platform->id() == rocm::kROCmPlatformId) {
    GTEST_SKIP() << "Not supported on ROCM";
  }

  if (platform->id() == cuda::kCudaPlatformId &&
      executor->GetDeviceDescription().runtime_version() <
          SemanticVersion{12, 3, 0}) {
    GTEST_SKIP() << "Command buffer tracing is not supported";
  }

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
  spec.AddInProcessSymbol(internal::GetAddI32Ptrs3Kernel(), "AddI32Ptrs3");

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
  TF_ASSERT_OK_AND_ASSIGN(auto cmd_buffer, TraceCommandBufferFactory::Create(
                                               executor,
                                               [&](Stream* stream) {
                                                 return add->Launch(
                                                     ThreadDim(), BlockDim(4),
                                                     stream, args);
                                               },
                                               primary));

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  std::vector<int32_t> expected = {42, 43, 44, 45, 46, 47};
  ASSERT_EQ(transfer_buffers(), expected);

  // Check the command buffer structure.
  GpuCommandBuffer* gpu_cmd_buffer = CastToGpuCommandBuffer(cmd_buffer.get());
  ASSERT_EQ(gpu_cmd_buffer->nodes().size(), 6);
  ASSERT_EQ(gpu_cmd_buffer->barriers().size(), 6);

  auto nodes = gpu_cmd_buffer->nodes();
  auto barriers = gpu_cmd_buffer->barriers();

  // First barrier does not have any dependencies.
  EXPECT_TRUE(barriers[0].is_barrier_node);
  EXPECT_THAT(gpu_cmd_buffer->GetNodeDependencies(barriers[0].handle),
              IsOkAndHolds(IsEmpty()));

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

  EXPECT_THAT(gpu_cmd_buffer->GetNodeDependencies(barriers[4].handle),
              IsOkAndHolds(ElementsAre(nodes[2].handle, nodes[3].handle)));
  EXPECT_THAT(gpu_cmd_buffer->GetNodeDependencies(barriers[5].handle),
              IsOkAndHolds(ElementsAre(nodes[4].handle, nodes[5].handle)));

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(record(cmd_buffer.get(), 43));
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  expected = {43, 44, 45, 46, 47, 48};
  ASSERT_EQ(transfer_buffers(), expected);
}

TEST(GpuCommandBufferTest, ConditionalIf) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIfWithMemset) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (platform->id() == rocm::kROCmPlatformId) {
    GTEST_SKIP() << "Not supported on ROCM";
  }

  if (platform->id() == cuda::kCudaPlatformId &&
      executor->GetDeviceDescription().driver_version() <
          SemanticVersion{12, 4, 0}) {
    GTEST_SKIP() << "ConditionalsWithMemset are not supported before 12.4.";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=0, pred=true
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 0, byte_length));

  // if (pred == true) memset(&a, ...);
  CommandBuffer::Builder then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Memset(&a, uint8_t{1}, byte_length);
  };

  // Create a command buffer with a single conditional operation.
  TF_ASSERT_OK_AND_ASSIGN(auto cmd_buffer,
                          executor->CreateCommandBuffer(primary));
  TF_ASSERT_OK(cmd_buffer->If(pred, then_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `a` data back to host.
  std::vector<int32_t> dst(length, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  std::vector<int32_t> expected(length, 1 << 24 | 1 << 16 | 1 << 8 | 1);
  ASSERT_EQ(dst, expected);

  // Prepare argument for graph update: b = 0
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&a, byte_length));

  // if (pred == true) memset(&b, ...);
  then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Memset(&b, uint8_t{1}, byte_length);
  };

  // Update command buffer with a conditional to use new builder.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->If(pred, then_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIfElse) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load multiplication kernel.
  MultiKernelLoaderSpec mul_spec(/*arity=*/3);
  mul_spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "MulI32");
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
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
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalCaseEmptyGraph) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  // See b/362769658.
  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

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
    return absl::OkStatus();
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->Case(index, {branch0, branch1}));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Set index to `1`
  TF_ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  // Submit the same command buffer, but this time it should take the empty path
  // and do nothing.
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_add);

  // Set index to `-1` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, -1, sizeof(int32_t)));

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_add);

  // Set index to `2` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, 2, sizeof(int32_t)));

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_add);
}

class GpuCommandBufferCaseTest : public testing::TestWithParam<int> {
 protected:
  int GetNumCases() { return GetParam(); }

  int GetEffectiveIndex(int i) {
    return (i < 0 || i >= GetNumCases()) ? GetNumCases() - 1 : i;
  }
};

TEST_P(GpuCommandBufferCaseTest, ConditionalMultiCase) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load multiplication kernel.
  MultiKernelLoaderSpec mul_spec(/*arity=*/3);
  mul_spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "MulI32");
  TF_ASSERT_OK_AND_ASSIGN(auto mul, MulI32Kernel::Create(executor, mul_spec));

  constexpr int64_t kLength = 1;
  int64_t byte_length = sizeof(int32_t) * kLength;

  // Prepare arguments: index=0
  DeviceMemory<int32_t> index = executor->AllocateArray<int32_t>(1, 0);
  TF_ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));

  const int kNumCases = GetNumCases();
  std::vector<DeviceMemory<int32_t>> values;
  std::vector<DeviceMemory<int32_t>> results;
  std::vector<CommandBuffer::Builder> branches;
  values.resize(kNumCases);
  results.resize(kNumCases);
  branches.resize(kNumCases);
  for (int i = 0; i < kNumCases; ++i) {
    values[i] = executor->AllocateArray<int32_t>(kLength, 0);
    TF_ASSERT_OK(stream->Memset32(&values[i], i, byte_length));
    results[i] = executor->AllocateArray<int32_t>(kLength, 0);
    TF_ASSERT_OK(stream->Memset32(&results[i], 0, byte_length));
    branches[i] = [&, i](CommandBuffer* branch_cmd) {
      // result = i * i;
      return branch_cmd->Launch(mul, ThreadDim(), BlockDim(kLength), values[i],
                                values[i], results[i]);
    };
  }

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->Case(index, branches));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  // We test the out of bounds cases as well ( i < 0, i >= kNumCases).
  for (int i = -1; i <= kNumCases; ++i) {
    // Set index.
    TF_ASSERT_OK(stream->Memset32(&index, i, sizeof(int32_t)));

    // Submit case.
    TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    int effective_index = GetEffectiveIndex(i);

    // Check all results are 0 except case index submitted.
    for (int z = 0; z < kNumCases; ++z) {
      std::vector<int32_t> dst(kLength, 42);
      TF_ASSERT_OK(stream->Memcpy(dst.data(), results[z], byte_length));

      // Build expected result vector.
      std::vector<int32_t> expected;
      expected.resize(kLength);
      for (int p = 0; p < kLength; ++p) {
        if (effective_index == z) {
          expected[p] = effective_index * effective_index;
        } else {
          expected[p] = 0;
        }
      }

      ASSERT_EQ(dst, expected)
          << "For result " << z << " after running case " << i;
      TF_ASSERT_OK(stream->Memset32(&results[z], 0, byte_length));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ConditionalMultipleCaseTest, GpuCommandBufferCaseTest,
                         testing::Range(1, 32),
                         testing::PrintToStringParamName());

TEST(GpuCommandBufferTest, ConditionalCase) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load multiplication kernel.
  MultiKernelLoaderSpec mul_spec(/*arity=*/3);
  mul_spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "MulI32");
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Set index to `1`
  TF_ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  // Submit the same command buffer, but this time it should multiply inputs.
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  std::vector<int32_t> expected_mul = {6, 6, 6, 6};
  ASSERT_EQ(dst, expected_mul);

  // Set index to `-1` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, -1, sizeof(int32_t)));

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_mul);

  // Set index to `2` (out of bound index value).
  TF_ASSERT_OK(stream->Memset32(&index, 2, sizeof(int32_t)));

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalFor) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
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

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalWhile) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load inc_and_cmp kernel.
  MultiKernelLoaderSpec icmp_spec(/*arity=*/3);
  icmp_spec.AddInProcessSymbol(internal::GetIncAndCmpKernel(), "IncAndCmp");
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
  CommandBuffer::Builder cond_builder = [&](CommandBuffer* cond_cmd) {
    return cond_cmd->Launch(inc_and_cmp, ThreadDim(), BlockDim(), loop_counter,
                            pred, num_iters);
  };

  // Loop body: b = a + b
  CommandBuffer::Builder body_builder = [&](CommandBuffer* body_cmd) {
    return body_cmd->Launch(add, ThreadDim(), BlockDim(length), a, b, b);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->While(pred, cond_builder, body_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
}

// TODO(b/339653343): Re-enable when not failing.
TEST(GpuCommandBufferTest, DISABLED_WhileNestedConditional) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load addition kernel.
  MultiKernelLoaderSpec add_spec(/*arity=*/3);
  add_spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "AddI32");
  TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor, add_spec));

  // Load inc_and_cmp kernel.
  MultiKernelLoaderSpec icmp_spec(/*arity=*/3);
  icmp_spec.AddInProcessSymbol(internal::GetIncAndCmpKernel(), "IncAndCmp");
  TF_ASSERT_OK_AND_ASSIGN(auto inc_and_cmp,
                          IncAndCmpKernel::Create(executor, icmp_spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=0, loop_counter=0, pred=false
  // Value of `pred` is not important, as it will be updated by `cond_builder`
  // below.
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<bool> pred_then = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> loop_counter = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  static constexpr bool kFalse = false;
  static constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));
  TF_ASSERT_OK(stream->Memcpy(&pred_then, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&loop_counter, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  int32_t num_iters = 10;

  CommandBuffer::Builder then_builder =
      // Then body: b = a + b
      [&](CommandBuffer* then_cmd) {
        return then_cmd->Launch(add, ThreadDim(), BlockDim(length), a, b, b);
      };

  auto nested_cmd = executor->CreateCommandBuffer(nested).value();
  // TODO(b/339653343): Adding this If condition causes AddNestedCommandBuffer
  // to fail.
  TF_ASSERT_OK(nested_cmd->If(pred_then, then_builder));

  // Loop cond: loop_counter++ < num_iters;
  CommandBuffer::Builder cond_builder = [&](CommandBuffer* cond_cmd) {
    return cond_cmd->Launch(inc_and_cmp, ThreadDim(), BlockDim(length),
                            loop_counter, pred, num_iters);
  };

  CommandBuffer::Builder body_builder =
      [&](CommandBuffer* body_cmd) -> absl::Status {
    CHECK_OK(body_cmd->AddNestedCommandBuffer(*nested_cmd));
    return absl::OkStatus();
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = executor->CreateCommandBuffer(primary).value();
  TF_ASSERT_OK(cmd_buffer->While(pred, cond_builder, body_builder));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
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
        CHECK_OK(add.Launch(ThreadDim(), BlockDim(4), stream, b, b, b));
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

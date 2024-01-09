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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace stream_executor::gpu {

static Platform* GpuPlatform() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  return MultiPlatformManager::PlatformWithName(name).value();
}

static MultiKernelLoaderSpec GetAddI32KernelSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/3);
#if defined(GOOGLE_CUDA)
  spec.AddCudaPtxInMemory(internal::kAddI32Kernel, "add");
#elif defined(TENSORFLOW_USE_ROCM)
  spec.AddCudaCubinInMemory(
      reinterpret_cast<const char*>(&internal::kAddI32KernelModule[0]), "add");
#endif
  return spec;
}

using AddI32Kernel = TypedKernel<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                                 DeviceMemory<int32_t>>;
using MulI32Kernel = TypedKernel<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                                 DeviceMemory<int32_t>>;
using IncAndCmpKernel =
    TypedKernel<DeviceMemory<int32_t>, DeviceMemory<bool>, int32_t>;

using AddI32Ptrs3 = TypedKernel<internal::Ptrs3<int32_t>>;

static constexpr auto nested = CommandBuffer::Mode::kNested;    // NOLINT
static constexpr auto primary = CommandBuffer::Mode::kPrimary;  // NOLINT

TEST(GpuCommandBufferTest, LaunchSingleKernel) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");

  AddI32Kernel add(executor);
  TF_ASSERT_OK(executor->GetKernel(spec, &add));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemset32(&b, 2, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // Create a command buffer with a single kernel launch.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), a, b, c));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  stream.ThenMemZero(&d, byte_length);

  // Update command buffer to write into `d` buffer.
  TF_ASSERT_OK(cmd_buffer.Update());
  TF_ASSERT_OK(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), a, b, d));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  stream.ThenMemcpy(dst.data(), d, byte_length);
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

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Ptrs3 add(executor);

  // Register a kernel with a custom arguments packing function that packs
  // device memory arguments into a struct with pointers.
  MultiKernelLoaderSpec spec(/*arity=*/1, [&](const Kernel& kernel,
                                              const KernelArgs& args) {
    auto bufs = Cast<KernelArgsDeviceMemoryArray>(&args)->device_memory_args();
    auto cast = [](auto m) { return reinterpret_cast<int32_t*>(m.opaque()); };
    return PackKernelArgs(add, internal::Ptrs3<int32_t>{
                                   cast(bufs[0]),
                                   cast(bufs[1]),
                                   cast(bufs[2]),
                               });
  });
  spec.AddInProcessSymbol(internal::GetAddI32Ptrs3Kernel(), "add");

  TF_ASSERT_OK(executor->GetKernel(spec, &add));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemset32(&b, 2, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // Use an array of device memory base pointers as argument to test packing.
  KernelArgsDeviceMemoryArray args({a, b, c}, 0);

  // Create a command buffer by tracing kernel launch operations.
  auto cmd_buffer = CommandBuffer::Trace(
      executor,
      [&](Stream* stream) {
        return executor->Launch(stream, ThreadDim(), BlockDim(4), add, args);
      },
      primary);

  TF_ASSERT_OK(cmd_buffer.status());
  TF_ASSERT_OK(executor->Submit(&stream, *cmd_buffer));

  // Copy data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, LaunchNestedCommandBuffer) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  MultiKernelLoaderSpec spec = GetAddI32KernelSpec();

  AddI32Kernel add(executor);
  TF_ASSERT_OK(executor->GetKernel(spec, &add));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemset32(&b, 2, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // Create a command buffer with a single kernel launch.
  auto primary_cmd = CommandBuffer::Create(executor).value();
  auto nested_cmd = CommandBuffer::Create(executor, nested).value();
  TF_ASSERT_OK(nested_cmd.Launch(add, ThreadDim(), BlockDim(4), a, b, c));
  TF_ASSERT_OK(primary_cmd.AddNestedCommandBuffer(nested_cmd));
  TF_ASSERT_OK(primary_cmd.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, primary_cmd));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  stream.ThenMemZero(&d, byte_length);

  // Update command buffer to write into `d` buffer by creating a new nested
  // command buffer.
  nested_cmd = CommandBuffer::Create(executor, nested).value();
  TF_ASSERT_OK(nested_cmd.Launch(add, ThreadDim(), BlockDim(4), a, b, d));
  TF_ASSERT_OK(primary_cmd.Update());
  TF_ASSERT_OK(primary_cmd.AddNestedCommandBuffer(nested_cmd));
  TF_ASSERT_OK(primary_cmd.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, primary_cmd));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  stream.ThenMemcpy(dst.data(), d, byte_length);
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, MemcpyDeviceToDevice) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=uninitialized
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 42, byte_length);

  // Create a command buffer with a single a to b memcpy command.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.MemcpyDeviceToDevice(&b, a, byte_length));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  stream.ThenMemcpy(dst.data(), a, byte_length);

  std::vector<int32_t> expected = {42, 42, 42, 42};
  ASSERT_EQ(dst, expected);

  // Update command buffer to swap the memcpy direction.
  TF_ASSERT_OK(cmd_buffer.Update());
  TF_ASSERT_OK(cmd_buffer.MemcpyDeviceToDevice(&a, b, byte_length));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  // Clear destination to test that command buffer actually copied memory.
  stream.ThenMemset32(&a, 0, byte_length);

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `a` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  stream.ThenMemcpy(dst.data(), a, byte_length);
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, Memset) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  // Create a command buffer with a single memset command.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.Memset(&a, uint32_t{42}, length));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  stream.ThenMemcpy(dst.data(), a, byte_length);

  std::vector<int32_t> expected = {42, 42, 42, 42};
  ASSERT_EQ(dst, expected);

  // Update command buffer to use a new bit pattern.
  TF_ASSERT_OK(cmd_buffer.Update());
  TF_ASSERT_OK(cmd_buffer.Memset(&a, uint32_t{43}, length));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  stream.ThenMemcpy(dst.data(), a, byte_length);

  expected = {43, 43, 43, 43};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIf) {
  Platform* platform = GpuPlatform();
  if (!CommandBuffer::SupportsConditionalCommands(platform)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Kernel add(executor);

  {  // Load addition kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
    TF_ASSERT_OK(executor->GetKernel(spec, &add));
  }

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0, pred=true
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  stream.ThenMemcpy(&pred, &kTrue, 1);
  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemset32(&b, 2, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // if (pred == true) c = a + b
  CommandBuffer::Builder then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.If(executor, pred, then_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);

  // Reset predicate to false and clear output buffer.
  constexpr bool kFalse = false;
  stream.ThenMemcpy(&pred, &kFalse, 1);
  stream.ThenMemZero(&c, byte_length);

  // Submit the same command buffer, but this time it should not execute
  // conditional branch as conditional handle should be updated to false.
  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  stream.ThenMemcpy(dst.data(), c, byte_length);
  std::vector<int32_t> zeroes = {0, 0, 0, 0};
  ASSERT_EQ(dst, zeroes);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  stream.ThenMemZero(&d, byte_length);

  // Set predicate buffer to true to run conditional command buffer.
  stream.ThenMemcpy(&pred, &kTrue, 1);

  // if (pred == true) d = a + b (write to a new location).
  then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, d);
  };

  // Update command buffer with a conditional to use new builder.
  TF_ASSERT_OK(cmd_buffer.Update());
  TF_ASSERT_OK(cmd_buffer.If(executor, pred, then_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  stream.ThenMemcpy(dst.data(), d, byte_length);
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalIfElse) {
  Platform* platform = GpuPlatform();
  if (!CommandBuffer::SupportsConditionalCommands(platform)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Kernel add(executor);
  MulI32Kernel mul(executor);

  {  // Load addition kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
    TF_ASSERT_OK(executor->GetKernel(spec, &add));
  }

  {  // Load multiplication kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "mul");
    TF_ASSERT_OK(executor->GetKernel(spec, &mul));
  }

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=2, b=3, c=0, pred=true
  DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  stream.ThenMemcpy(&pred, &kTrue, 1);
  stream.ThenMemset32(&a, 2, byte_length);
  stream.ThenMemset32(&b, 3, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // if (pred == true) c = a + b
  CommandBuffer::Builder then_builder = [&](CommandBuffer* then_cmd) {
    return then_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // if (pred == false) c = a * b
  CommandBuffer::Builder else_builder = [&](CommandBuffer* else_cmd) {
    return else_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.IfElse(executor, pred, then_builder, else_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Reset predicate to false.
  constexpr bool kFalse = false;
  stream.ThenMemcpy(&pred, &kFalse, 1);

  // Submit the same command buffer, but this time it should execute `else`
  // branch and multiply inputs.
  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  stream.ThenMemcpy(dst.data(), c, byte_length);
  std::vector<int32_t> expected_mul = {6, 6, 6, 6};
  ASSERT_EQ(dst, expected_mul);

  // Prepare argument for graph update: d = 0
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);
  stream.ThenMemZero(&d, byte_length);

  // if (pred == false) d = a * b (write to a new location).
  else_builder = [&](CommandBuffer* else_cmd) {
    return else_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, d);
  };

  // Update command buffer with a conditional to use new `else` builder.
  TF_ASSERT_OK(cmd_buffer.Update());
  TF_ASSERT_OK(cmd_buffer.IfElse(executor, pred, then_builder, else_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 42);
  stream.ThenMemcpy(dst.data(), d, byte_length);
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalCase) {
  Platform* platform = GpuPlatform();
  if (!CommandBuffer::SupportsConditionalCommands(platform)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Kernel add(executor);
  MulI32Kernel mul(executor);

  {  // Load addition kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
    TF_ASSERT_OK(executor->GetKernel(spec, &add));
  }

  {  // Load multiplication kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetMulI32Kernel(), "mul");
    TF_ASSERT_OK(executor->GetKernel(spec, &mul));
  }

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=2, b=3, c=0, index=0
  DeviceMemory<int32_t> index = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&index, 0, sizeof(int32_t));
  stream.ThenMemset32(&a, 2, byte_length);
  stream.ThenMemset32(&b, 3, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // if (index == 0) c = a + b
  CommandBuffer::Builder branch0 = [&](CommandBuffer* branch0_cmd) {
    return branch0_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, c);
  };

  // if (index == 1) c = a * b
  CommandBuffer::Builder branch1 = [&](CommandBuffer* branch1_cmd) {
    return branch1_cmd->Launch(mul, ThreadDim(), BlockDim(4), a, b, c);
  };

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.Case(executor, index, {branch0, branch1}));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  // Copy `c` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected_add = {5, 5, 5, 5};
  ASSERT_EQ(dst, expected_add);

  // Set index to `1`
  stream.ThenMemset32(&index, 1, sizeof(int32_t));

  // Submit the same command buffer, but this time it should multiply inputs.
  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  stream.ThenMemcpy(dst.data(), c, byte_length);
  std::vector<int32_t> expected_mul = {6, 6, 6, 6};
  ASSERT_EQ(dst, expected_mul);

  // Set index to `-1` (out of bound index value).
  stream.ThenMemset32(&index, -1, sizeof(int32_t));

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  stream.ThenMemcpy(dst.data(), c, byte_length);
  ASSERT_EQ(dst, expected_mul);

  // Set index to `2` (out of bound index value).
  stream.ThenMemset32(&index, 2, sizeof(int32_t));

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  stream.ThenMemcpy(dst.data(), c, byte_length);
  ASSERT_EQ(dst, expected_mul);
}

TEST(GpuCommandBufferTest, ConditionalFor) {
  Platform* platform = GpuPlatform();
  if (!CommandBuffer::SupportsConditionalCommands(platform)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Kernel add(executor);

  {  // Load addition kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
    TF_ASSERT_OK(executor->GetKernel(spec, &add));
  }

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=0, loop_counter=100
  DeviceMemory<int32_t> loop_counter = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  // Set loop counter to 100 to check that command buffer resets it.
  stream.ThenMemset32(&loop_counter, 100, sizeof(int32_t));
  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemZero(&b, byte_length);

  // Loop body: b = a + b
  CommandBuffer::Builder body_builder = [&](CommandBuffer* body_cmd) {
    return body_cmd->Launch(add, ThreadDim(), BlockDim(4), a, b, b);
  };

  int32_t num_iters = 10;

  // Create a command buffer with a single conditional operation.
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.For(executor, num_iters, loop_counter, body_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), b, byte_length);

  std::vector<int32_t> expected = {10, 10, 10, 10};
  ASSERT_EQ(dst, expected);
}

TEST(GpuCommandBufferTest, ConditionalWhile) {
  Platform* platform = GpuPlatform();
  if (!CommandBuffer::SupportsConditionalCommands(platform)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  AddI32Kernel add(executor);
  IncAndCmpKernel inc_and_cmp(executor);

  {  // Load addition kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetAddI32Kernel(), "add");
    TF_ASSERT_OK(executor->GetKernel(spec, &add));
  }

  {  // Load inc_and_cmp kernel.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(internal::GetIncAndCmpKernel(), "inc_and_cmp");
    TF_ASSERT_OK(executor->GetKernel(spec, &inc_and_cmp));
  }

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
  stream.ThenMemcpy(&pred, &kFalse, 1);
  stream.ThenMemset32(&loop_counter, 0, sizeof(int32_t));
  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemZero(&b, byte_length);

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
  auto cmd_buffer = CommandBuffer::Create(executor).value();
  TF_ASSERT_OK(cmd_buffer.While(executor, pred, cond_builder, body_builder));
  TF_ASSERT_OK(cmd_buffer.Finalize());

  TF_ASSERT_OK(executor->Submit(&stream, cmd_buffer));

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), b, byte_length);

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

  MultiKernelLoaderSpec spec = GetAddI32KernelSpec();

  AddI32Kernel add(executor);
  CHECK_OK(executor->GetKernel(spec, &add));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  for (auto s : state) {
    auto cmd_buffer = CommandBuffer::Create(executor, nested).value();
    for (int i = 1; i < state.range(0); ++i) {
      CHECK_OK(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), b, b, b));
    }
    CHECK_OK(cmd_buffer.Finalize());
  }
}

BENCHMARK_SIZES(BM_CreateCommandBuffer);

static void BM_TraceCommandBuffer(benchmark::State& state) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  CHECK(stream.ok());

  MultiKernelLoaderSpec spec = GetAddI32KernelSpec();

  AddI32Kernel add(executor);
  CHECK_OK(executor->GetKernel(spec, &add));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  for (auto s : state) {
    auto launch_kernels = [&](Stream* stream) {
      for (int i = 1; i < state.range(0); ++i) {
        CHECK_OK(stream->ThenLaunch(ThreadDim(), BlockDim(4), add, b, b, b));
      }
      return absl::OkStatus();
    };

    CHECK_OK(CommandBuffer::Trace(executor, launch_kernels, nested));
  }
}

BENCHMARK_SIZES(BM_TraceCommandBuffer);

static void BM_UpdateCommandBuffer(benchmark::State& state) {
  Platform* platform = GpuPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  MultiKernelLoaderSpec spec = GetAddI32KernelSpec();

  AddI32Kernel add(executor);
  CHECK_OK(executor->GetKernel(spec, &add));

  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(1, 0);

  auto cmd_buffer = CommandBuffer::Create(executor, primary).value();
  for (int i = 1; i < state.range(0); ++i) {
    CHECK_OK(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), b, b, b));
  }
  CHECK_OK(cmd_buffer.Finalize());

  for (auto s : state) {
    CHECK_OK(cmd_buffer.Update());
    for (int i = 1; i < state.range(0); ++i) {
      CHECK_OK(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), b, b, b));
    }
    CHECK_OK(cmd_buffer.Finalize());
  }
}

BENCHMARK_SIZES(BM_UpdateCommandBuffer);

}  // namespace stream_executor::gpu

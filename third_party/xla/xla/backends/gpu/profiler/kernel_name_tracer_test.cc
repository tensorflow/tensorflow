/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/profiler/kernel_name_tracer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

absl::StatusOr<stream_executor::Platform*> GetPlatform() {
  TF_ASSIGN_OR_RETURN(std::string name,
                      PlatformUtil::CanonicalPlatformName("gpu"));
  return stream_executor::PlatformManager::PlatformWithName(
      absl::AsciiStrToUpper(name));
}

class KernelNameTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_, GetPlatform());
    TF_ASSERT_OK_AND_ASSIGN(stream_executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_,
                            stream_executor_->CreateStream(std::nullopt));
  }

  stream_executor::Platform* platform_;
  stream_executor::StreamExecutor* stream_executor_;
  std::unique_ptr<stream_executor::Stream> stream_;
};

void LaunchAddI32Kernels(stream_executor::StreamExecutor* executor,
                         stream_executor::Stream* stream) {
  using AddI32Kernel =
      stream_executor::TypedKernel<stream_executor::DeviceAddress<int>,
                                   stream_executor::DeviceAddress<int>,
                                   stream_executor::DeviceAddress<int>>;
  TF_ASSERT_OK_AND_ASSIGN(AddI32Kernel add,
                          stream_executor::gpu::LoadAddI32TestKernel(executor));

  constexpr int64_t kLength = 4;
  constexpr int64_t kLengthInBytes = sizeof(int32_t) * kLength;

  stream_executor::DeviceAddress<int32_t> a =
      executor->AllocateArray<int32_t>(kLength, 0);
  stream_executor::DeviceAddress<int32_t> b =
      executor->AllocateArray<int32_t>(kLength, 0);
  stream_executor::DeviceAddress<int32_t> c =
      executor->AllocateArray<int32_t>(kLength, 0);

  ASSERT_THAT(stream->Memset32(&a, 1, kLengthInBytes), IsOk());
  ASSERT_THAT(stream->Memset32(&b, 2, kLengthInBytes), IsOk());

  ASSERT_THAT(add.Launch(stream_executor::ThreadDim(),
                         stream_executor::BlockDim(kLength), stream, a, b, c),
              IsOk());

  // We launch this kernel twice because in the past we had issues with the
  // tracer only capturing the first kernel launch.
  ASSERT_THAT(add.Launch(stream_executor::ThreadDim(),
                         stream_executor::BlockDim(kLength), stream, a, b, c),
              IsOk());

  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  executor->Deallocate(&a);
  executor->Deallocate(&b);
  executor->Deallocate(&c);
}

void LaunchCommandBufferThunk(stream_executor::StreamExecutor* executor,
                              stream_executor::Stream* stream) {
  using AddI32Kernel =
      stream_executor::TypedKernel<stream_executor::DeviceAddress<int>,
                                   stream_executor::DeviceAddress<int>,
                                   stream_executor::DeviceAddress<int>>;
  TF_ASSERT_OK_AND_ASSIGN(AddI32Kernel add,
                          stream_executor::gpu::LoadAddI32TestKernel(executor));

  constexpr int64_t kLength = 4;
  constexpr int64_t kLengthInBytes = sizeof(int32_t) * kLength;

  stream_executor::DeviceAddress<int32_t> a =
      executor->AllocateArray<int32_t>(kLength, 0);
  stream_executor::DeviceAddress<int32_t> b =
      executor->AllocateArray<int32_t>(kLength, 0);
  stream_executor::DeviceAddress<int32_t> c =
      executor->AllocateArray<int32_t>(kLength, 0);

  ASSERT_THAT(stream->Memset32(&a, 1, kLengthInBytes), IsOk());
  ASSERT_THAT(stream->Memset32(&b, 2, kLengthInBytes), IsOk());

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, kLengthInBytes, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, kLengthInBytes, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, kLengthInBytes, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, kLengthInBytes);
  BufferAllocation::Slice slice_b(&alloc_b, 0, kLengthInBytes);
  BufferAllocation::Slice slice_c(&alloc_c, 0, kLengthInBytes);

  auto args = {slice_a, slice_b, slice_c};  // c = a + b
  auto args_access = {BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kRead,
                      BufferUse::MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, kLength),
                              /*shmem_bytes=*/0);
  commands.Emplace<LaunchCmd>("AddI32", args, args_access,
                              LaunchDimensions(1, kLength),
                              /*shmem_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor cmd_buffer_executor,
      CommandBufferCmdExecutor::Create(
          std::move(commands),
          CommandBufferCmdExecutor::SynchronizationMode::kConcurrent));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(cmd_buffer_executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({a, b, c}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream, stream, nullptr, nullptr);

  // This is where we're getting the 'AddI32' kernel from.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> fatbin,
                          stream_executor::gpu::GetGpuTestKernelsFatbin(
                              executor->GetPlatform()->Name()));
  ASSERT_THAT(
      thunk.Initialize({executor,
                        Thunk::ExecutableSource{/*text=*/"", fatbin,
                                                /*dnn_compiled_graphs=*/{}},
                        &allocations, stream}),
      IsOk());

  // Execute command buffer thunk and verify that it added the value.
  ASSERT_THAT(thunk.ExecuteOnStream(params), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  executor->Deallocate(&a);
  executor->Deallocate(&b);
  executor->Deallocate(&c);
}

TEST_F(KernelNameTracerTest, Create) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KernelNameTracer> tracer,
      KernelNameTracer::Create(stream_executor::cuda::kCudaPlatformId));
  tracer->start();
  std::vector<std::string> kernel_names = tracer->stop();
  EXPECT_THAT(kernel_names, IsEmpty());
}

TEST_F(KernelNameTracerTest, CreateUnsupportedPlatform) {
  EXPECT_THAT(KernelNameTracer::Create(stream_executor::rocm::kROCmPlatformId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(KernelNameTracerTest, CaptureKernelNames) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelNameTracer> tracer,
                          KernelNameTracer::Create(platform_->id()));
  tracer->start();

  LaunchAddI32Kernels(stream_executor_, stream_.get());

  std::vector<std::string> kernel_names = tracer->stop();
  EXPECT_THAT(kernel_names, ElementsAre("AddI32", "AddI32"));
}

TEST_F(KernelNameTracerTest, CaptureKernelNamesFromCommandBufferThunk) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<KernelNameTracer> tracer,
                          KernelNameTracer::Create(platform_->id()));
  tracer->start();

  LaunchCommandBufferThunk(stream_executor_, stream_.get());

  std::vector<std::string> kernel_names = tracer->stop();
  EXPECT_THAT(kernel_names, ElementsAre("AddI32", "AddI32"));
}

}  // namespace
}  // namespace xla::gpu

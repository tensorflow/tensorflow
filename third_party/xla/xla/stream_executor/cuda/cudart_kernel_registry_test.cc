/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cudart_kernel_registry.h"

#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/simple_kernel_cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::cuda {
namespace {

using Write42Kernel = TypedKernelFactory<DeviceAddress<int32_t>, int32_t>;

TEST(CudaRuntimeKernelRegistryTest, ResolveAndRunKernel) {
  // 1. Get the host function pointer for Write42 kernel.
  Write42KernelFn host_fn = GetWrite42Kernel();

  // 2. Resolve host function pointer to CUBIN.
  ASSERT_OK_AND_ASSIGN(auto cubin_spec, FindCudaRuntimeKernel(host_fn));
  EXPECT_THAT(cubin_spec.kernel_name(), ::testing::HasSubstr("Write42Kernel"));
  EXPECT_EQ(cubin_spec.arity(), 2);

  // 3. Use StreamExecutor APIs to load the CUBIN and execute the kernel.
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto write42_kernel,
                       Write42Kernel::Create(executor, cubin_spec));

  int32_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  DeviceAddress<int32_t> out = executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->MemZero(&out, byte_length));

  ASSERT_TRUE(
      write42_kernel.Launch(ThreadDim(), BlockDim(4), stream.get(), out, length)
          .ok());

  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), out, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  EXPECT_EQ(dst, expected);
}

// This test ensures that we don't accidentally break CUDA runtime
// functionality by testing whether the chevron syntax still works.
TEST(CudaRuntimeKernelRegistryTest, LaunchCudaKernelWithTripleAngleBrackets) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  DeviceAddress<int32_t> out = executor->AllocateArray<int32_t>(length, 0);
  ASSERT_OK(stream->MemZero(&out, byte_length));

  ASSERT_OK(ChevronLaunchWrite42Kernel(stream.get(), out, length));

  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), out, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  EXPECT_EQ(dst, expected);
}

}  // namespace
}  // namespace stream_executor::cuda

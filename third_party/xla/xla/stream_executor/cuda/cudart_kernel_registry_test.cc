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
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/simple_kernel_cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
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

using AddI32Kernel =
    TypedKernelFactory<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                       DeviceAddress<int32_t>>;

TEST(CudaRuntimeKernelRegistryTest, ResolveAndRunKernel) {
  // 1. Get the host function pointer for AddI32 kernel.
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                       gpu::GetAddI32TestKernelSpec(cuda::kCudaPlatformId));
  ASSERT_TRUE(spec.has_in_process_symbol());
  void* host_fn = spec.in_process_symbol()->symbol;

  // 2. Resolve host function pointer to CUBIN.
  auto cubin_span = FindCudaRuntimeKernel(host_fn);
  ASSERT_TRUE(cubin_span.has_value());

  // 3. Use StreamExecutor APIs to load the CUBIN and execute the kernel.
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  KernelLoaderSpec cubin_spec =
      KernelLoaderSpec::CreateCudaCubinInMemorySpec(*cubin_span, "AddI32", 3);

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto add_kernel,
                       AddI32Kernel::Create(executor, cubin_spec));

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  DeviceAddress<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceAddress<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceAddress<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  ASSERT_OK(stream->Memset32(&b, 2, byte_length));
  ASSERT_OK(stream->MemZero(&c, byte_length));

  ASSERT_TRUE(
      add_kernel.Launch(ThreadDim(), BlockDim(4), stream.get(), a, b, c).ok());

  std::vector<int32_t> dst(4, 42);
  ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  EXPECT_EQ(dst, expected);
}

// This test ensures that we don't accidentally break CUDA runtime
// functionality.
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

  ASSERT_OK(LaunchWrite42Kernel(stream.get(), out, length));

  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), out, byte_length));

  std::vector<int32_t> expected = {42, 42, 42, 42};
  EXPECT_EQ(dst, expected);
}

}  // namespace
}  // namespace stream_executor::cuda

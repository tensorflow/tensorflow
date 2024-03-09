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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {

TEST(GpuKernelTest, Add) {
  using AddI32Kernel = TypedKernel<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                                   DeviceMemory<int32_t>>;
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  Platform* platform = PlatformManager::PlatformWithName(name).value();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  MultiKernelLoaderSpec spec(/*arity=*/3);
#if defined(GOOGLE_CUDA)
  spec.AddCudaPtxInMemory(internal::kAddI32Kernel, "add");
#elif defined(TENSORFLOW_USE_ROCM)
  spec.AddCudaCubinInMemory(internal::kAddI32KernelModule, "add");
#endif

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

  // Launch kernel.
  ASSERT_TRUE(stream->ThenLaunch(ThreadDim(), BlockDim(4), add, a, b, c).ok());

  // Copy data back to host.
  std::vector<int32_t> dst(4, 42);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

}  // namespace stream_executor::gpu

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

#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu::kernel::gemm_universal {

TEST(CutlassGemmKernelTest, SimpleGemm) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("CUDA").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  // Load [4, 4] x [4, 4] gemm kernel written in CUDA C++ with CUTLASS.
  auto custom_kernel = GetCutlassGemmKernel(
      "cutlass_gemm", PrimitiveType::F32, 4, 4, 4,
      /*indices=*/{0, 1, 2}, /*slices=*/{}, executor->GetDeviceDescription());

  TF_ASSERT_OK_AND_ASSIGN(
      auto gemm, se::Kernel::Create(executor, custom_kernel->kernel_spec()));

  int64_t length = 4 * 4;
  int64_t byte_length = sizeof(float) * length;

  // Prepare arguments: a=2, b=2, c=0
  se::DeviceMemory<float> a = executor->AllocateArray<float>(length, 0);
  se::DeviceMemory<float> b = executor->AllocateArray<float>(length, 0);
  se::DeviceMemory<float> c = executor->AllocateArray<float>(length, 0);

  float value = 2.0;
  uint32_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));

  TF_ASSERT_OK(stream->Memset32(&a, pattern, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, pattern, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Launch gemm kernel with device memory arguments.
  se::KernelArgsDeviceMemoryArray arr(
      std::vector<se::DeviceMemoryBase>({a, b, c}),
      custom_kernel->shared_memory_bytes());
  TF_ASSERT_OK(executor->Launch(stream.get(), custom_kernel->thread_dims(),
                                custom_kernel->block_dims(), *gemm, arr));

  // Copy `c` data back to host.
  std::vector<float> dst(length, -1.0f);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<float> expected(length, 16.0);
  ASSERT_EQ(dst, expected);
}

TEST(CutlassGemmKernelTest, LoadFromSharedLibrary) {
  std::string kernel_lib_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu", "kernels",
                        "cutlass_gemm_kernel_f32xf32_to_f32.so");

  se::Platform* platform =
      se::PlatformManager::PlatformWithName("CUDA").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  // Load [4, 4] x [4, 4] gemm kernel written in CUDA C++ with CUTLASS.
  auto custom_kernel = LoadCutlassGemmKernel(
      "cutlass_gemm", kernel_lib_path, PrimitiveType::F32, 4, 4, 4,
      /*indices=*/{0, 1, 2}, /*slices=*/{}, executor->GetDeviceDescription());

  TF_ASSERT_OK_AND_ASSIGN(
      auto gemm, se::Kernel::Create(executor, custom_kernel->kernel_spec()));

  int64_t length = 4 * 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceMemory<float> a = executor->AllocateArray<float>(length, 0);
  se::DeviceMemory<float> b = executor->AllocateArray<float>(length, 0);
  se::DeviceMemory<float> c = executor->AllocateArray<float>(length, 0);

  float value = 2.0;
  uint32_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));

  TF_ASSERT_OK(stream->Memset32(&a, pattern, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, pattern, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Launch gemm kernel with device memory arguments.
  se::KernelArgsDeviceMemoryArray arr(
      std::vector<se::DeviceMemoryBase>({a, b, c}),
      custom_kernel->shared_memory_bytes());
  TF_ASSERT_OK(executor->Launch(stream.get(), custom_kernel->thread_dims(),
                                custom_kernel->block_dims(), *gemm, arr));

  // Copy `c` data back to host.
  std::vector<float> dst(length, -1.0f);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<float> expected(length, 16.0);
  ASSERT_EQ(dst, expected);
}

}  // namespace xla::gpu::kernel::gemm_universal

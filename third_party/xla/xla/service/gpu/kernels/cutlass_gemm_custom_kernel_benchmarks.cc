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
#include <cstring>
#include <vector>

#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::gpu::kernel::gemm_universal {

static uint32_t BitPattern(float value) {
  uint32_t pattern;
  std::memcpy(&pattern, &value, sizeof(float));
  return pattern;
}

static void BM_RowMajorGemm(benchmark::State& state) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("CUDA").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  const se::DeviceDescription& device = executor->GetDeviceDescription();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // GEMM: 8192x4096 * 4096x16384 -> 8192x16384
  int32_t m = 8192;
  int32_t n = 16384;
  int32_t k = 4096;

  auto custom_kernel =
      GetCutlassGemmKernel("cutlass_gemm", PrimitiveType::BF16, m, n, k,
                           /*indices=*/{0, 1, 2}, /*slices=*/{}, device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto gemm, se::Kernel::Create(executor, custom_kernel->kernel_spec()));

  // Prepare arguments: a=1.1, b=1.2, c=0.0
  se::DeviceMemory<float> a = executor->AllocateArray<float>(m * k, 0);
  se::DeviceMemory<float> b = executor->AllocateArray<float>(k * n, 0);
  se::DeviceMemory<float> c = executor->AllocateArray<float>(m * n, 0);

  TF_CHECK_OK(stream->Memset32(&a, BitPattern(1.1f), a.size()));
  TF_CHECK_OK(stream->Memset32(&b, BitPattern(1.2f), b.size()));
  TF_CHECK_OK(stream->MemZero(&c, c.size()));

  se::KernelArgsDeviceMemoryArray args(
      std::vector<se::DeviceMemoryBase>({a, b, c}),
      custom_kernel->shared_memory_bytes());

  for (auto s : state) {
    TF_CHECK_OK(executor->Launch(stream.get(), custom_kernel->thread_dims(),
                                 custom_kernel->block_dims(), *gemm, args));
    TF_CHECK_OK(stream->BlockHostUntilDone());
  }
}

BENCHMARK(BM_RowMajorGemm);

}  // namespace xla::gpu::kernel::gemm_universal

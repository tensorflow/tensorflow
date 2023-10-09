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
#include "xla/service/gpu/runtime/sleep_kernel.h"

namespace xla::gpu {
namespace {

#if GOOGLE_CUDA
__global__ void mock_nccl_call(unsigned sleep_ns) {
#if __CUDA_ARCH__ >= 700  // __nanosleep requires compute capability 7.0
  // Passing too high a number to __nanosleep makes it sleep for much less time
  // than the passed-in number. So only pass 1,000,000 and keep calling
  // __nanosleep in a loop.
  int n = sleep_ns / 1000000;
  unsigned rem = sleep_ns % 1000000;
  for (int i = 0; i < n; i++) __nanosleep(1000000U);
  __nanosleep(rem);
#endif
}
#elif TENSORFLOW_USE_ROCM
__global__ void mock_nccl_call(unsigned sleep_ns, unsigned clock_rate_khz) {
  if (threadIdx.x < warpSize) {
    // s_sleep causes a wave to sleep for (64 * SIMM16[6:0] + 1..64) clocks.
    uint32_t nclocks = (uint32_t)((float)clock_rate_khz / 64e6 * sleep_ns);
    for (uint32_t i = 0; i < nclocks / 64; i++) {
      __builtin_amdgcn_s_sleep(64);
    }
  }
  __syncthreads();
}
#endif  // TENSORFLOW_USE_ROCM
}  // namespace

void* GetSleepKernel() { return reinterpret_cast<void*>(&mock_nccl_call); }

}  // namespace xla::gpu

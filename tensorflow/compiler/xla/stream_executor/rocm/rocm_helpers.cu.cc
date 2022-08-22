/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <hip/hip_runtime.h>

#include <limits>
namespace stream_executor {
namespace gpu {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//

__global__ void __xla_MakeBatchPointers(char* base, int stride, int n,
                                        void** ptrs_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}

void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                            void** ptrs_out) {
  const int threads_per_block = 256;
  hipLaunchKernelGGL(
      __xla_MakeBatchPointers,
      dim3((n + threads_per_block - 1) / threads_per_block, 1, 1),
      dim3(threads_per_block, 1, 1), 0, (hipStream_t)stream, base, stride, n,
      ptrs_out);
}

};  // namespace gpu
};  // namespace stream_executor

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

#include "xla/stream_executor/cuda/cuda_executor_multigpu_test_kernels.h"

#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::gpu {
namespace {

__global__ void MulticastReduceKernel(int* input, int* output, size_t size) {
#if __CUDA_ARCH__ >= 900
  for (int i = 0; i < size; i++) {
    int* multimem_element_ptr = input + i;
    int result = 0;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
                 : "=r"(result)
                 : "l"(multimem_element_ptr)
                 : "memory");

    output[i] = result;
  }
#endif
}
}  // namespace

__host__ absl::Status MulticastReduce(int* input, int* output, size_t size) {
  TF_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cudaSetDevice(0)));
  TF_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cudaDeviceSynchronize()));
  MulticastReduceKernel<<<1, 1, 0>>>(input, output, size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("CUDA Kernel launch failed: ", cudaGetErrorString(err)));
  }
  return stream_executor::cuda::ToStatus(cudaDeviceSynchronize());
}
}  // namespace stream_executor::gpu

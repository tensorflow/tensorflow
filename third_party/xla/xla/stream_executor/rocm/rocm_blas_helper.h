/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_HELPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_HELPER_H_

namespace stream_executor::gpu {

// Copies one or more source matrices into strided destinations. Used to pack
// pointer-array batched GEMM operands into the strided layout rocBLAS expects.
void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_HELPER_H_

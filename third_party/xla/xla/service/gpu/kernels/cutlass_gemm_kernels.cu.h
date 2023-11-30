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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNELS_CU_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNELS_CU_H_

#include "third_party/gpus/cutlass/include/cutlass/gemm/device/gemm_universal.h"

namespace xla::gpu::kernel {

struct CutlassGemmKernels {
  using F32xF32toF32 =
      cutlass::gemm::device::GemmUniversal<float, cutlass::layout::RowMajor,
                                           float, cutlass::layout::RowMajor,
                                           float, cutlass::layout::RowMajor>;
};

namespace internal {

template <typename Gemm>
void* GetCutlassGemmKernel() {
  return reinterpret_cast<void*>(cutlass::Kernel2<typename Gemm::GemmKernel>);
}

// Extern templates for all supported CUTLASS Gemm kernels.
extern template void* GetCutlassGemmKernel<CutlassGemmKernels::F32xF32toF32>();

}  // namespace internal
}  // namespace xla::gpu::kernel

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNELS_CU_H_

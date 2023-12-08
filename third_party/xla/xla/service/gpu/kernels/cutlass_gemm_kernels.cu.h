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

#include <cstddef>
#include <cstdint>

#include "cutlass/gemm/device/gemm_universal.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"

namespace xla::gpu::kernel::gemm_universal {

//===----------------------------------------------------------------------===//
// CUTLASS kernels with default template parameters.
//===----------------------------------------------------------------------===//

struct Default {
  using F32xF32toF32 = cutlass::gemm::device::GemmUniversal<
      float, cutlass::layout::RowMajor,   // A
      float, cutlass::layout::RowMajor,   // B
      float, cutlass::layout::RowMajor>;  // C

  using BF16xBF16toBF16 = cutlass::gemm::device::GemmUniversal<
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // A
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // B
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // C
      float>;                                          // Accumulator
};

//===----------------------------------------------------------------------===//
// CUTLASS kernels optimized for Sm80 architecture.
//===----------------------------------------------------------------------===//

struct Sm80 {
  using BF16xBF16toBF16 = cutlass::gemm::device::GemmUniversal<
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // A
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // B
      cutlass::bfloat16_t, cutlass::layout::RowMajor,  // C
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,  // ThreadblockShape
      cutlass::gemm::GemmShape<64, 64, 64>,    // WarpShape
      cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape
      cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8,
                                                   float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      /*Stages=*/3, /*AlignmentA=*/8, /*AlignmentB=*/8>;
};

//===----------------------------------------------------------------------===//

// This entry point is based on `cutlass::Kernel2` template with an extra
// parameter to pass dynamic slices.
template <typename Gemm>
__global__ void Kernel(typename Gemm::Params params,
                       gemm_universal::DynamicSliceParams slices) {
  extern __shared__ int SharedStorageBase[];
  typename Gemm::SharedStorage* shared_storage =
      reinterpret_cast<typename Gemm::SharedStorage*>(SharedStorageBase);

  // Update output pointers to account for dynamic offsets.
  if (slices.out.has_value()) {
    auto m = params.problem_size.m();
    auto n = params.problem_size.n();

    int32_t out_offset = **slices.out;

    char* ptr_c = reinterpret_cast<char*>(params.ptr_C);
    char* ptr_d = reinterpret_cast<char*>(params.ptr_D);

    using ElementC = typename Gemm::ElementC;
    params.ptr_C = ptr_c + sizeof(ElementC) * out_offset * (m * n);
    params.ptr_D = ptr_d + sizeof(ElementC) * out_offset * (m * n);
  }

  Gemm::invoke(params, *shared_storage);
}

template <typename Gemm>
void* GetKernelSymbol() {
  return reinterpret_cast<void*>(Kernel<typename Gemm::GemmKernel>);
}

// Extern templates for all supported CUTLASS Gemm kernels.
extern template void* GetKernelSymbol<Default::F32xF32toF32>();
extern template void* GetKernelSymbol<Default::BF16xBF16toBF16>();
extern template void* GetKernelSymbol<Sm80::BF16xBF16toBF16>();

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNELS_CU_H_

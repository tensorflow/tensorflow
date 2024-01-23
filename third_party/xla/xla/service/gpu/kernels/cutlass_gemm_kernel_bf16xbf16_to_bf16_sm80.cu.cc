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

#include "cutlass/gemm/device/gemm_universal.h"
#include "xla/service/gpu/kernels/cutlass_gemm_adaptor.cu.h"

namespace xla::gpu::kernel::gemm_universal {

using GemmOperation = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,  // A
    cutlass::bfloat16_t, cutlass::layout::RowMajor,  // B
    cutlass::bfloat16_t, cutlass::layout::RowMajor,  // C
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,  // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,    // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float,
                                                 float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    /*Stages=*/3, /*AlignmentA=*/8, /*AlignmentB=*/8>;

XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(Bf16xBf16ToBf16<Arch::kSm80>, GemmOperation);

template struct Adaptor<Bf16xBf16ToBf16<Arch::kSm80>>;
template struct DeviceKernel<Bf16xBf16ToBf16<Arch::kSm80>>;

}  // namespace xla::gpu::kernel::gemm_universal

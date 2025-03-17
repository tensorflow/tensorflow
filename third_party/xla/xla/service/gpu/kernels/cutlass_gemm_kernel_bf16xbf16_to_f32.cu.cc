/* Copyright 2024 The OpenXLA Authors.

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

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "xla/service/gpu/kernels/cutlass_gemm_adaptor.cu.h"

namespace xla::gpu::kernel::gemm_universal {

namespace {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = float;
using ElementAccumulator = float;

}  // namespace

using GemmOperation = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor, ElementB, cutlass::layout::RowMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 32, 8>, cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,  // stages
    1,  // A alignment
    1,  // B alignment
    cutlass::arch::OpMultiplyAdd>;

XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(Bf16xBf16ToF32<Arch::kDefault>,
                                   GemmOperation);
template class Adaptor<Bf16xBf16ToF32<Arch::kDefault>>;
template class DeviceKernel<Bf16xBf16ToF32<Arch::kDefault>>;

}  // namespace xla::gpu::kernel::gemm_universal

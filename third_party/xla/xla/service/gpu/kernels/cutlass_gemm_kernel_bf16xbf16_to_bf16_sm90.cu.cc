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

#include "xla/service/gpu/kernels/cutlass_gemm_adaptor.cu.h"

// CUTLASS headers must be included after adaptor
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

// Custom epilogue must be included after CUTLASS headers
#include "xla/service/gpu/kernels/cutlass_gemm_epilogue.cu.h"

namespace xla::gpu::kernel::gemm_universal {

using EpilogueLoop = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8, cutlass::bfloat16_t,
    cutlass::layout::RowMajor, 8, cutlass::epilogue::NoSmemWarpSpecialized,
    LinearCombinationWithDynamicSlice<cutlass::bfloat16_t, float,
                                      /*dynamic_offset=*/1>>::CollectiveOp;

using MainLoop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::bfloat16_t,
    cutlass::layout::RowMajor, 8, cutlass::bfloat16_t,
    cutlass::layout::RowMajor, 8, float,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
        typename EpilogueLoop::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                         MainLoop, EpilogueLoop,
                                         cutlass::gemm::StreamKScheduler>;

using GemmOperation = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(Bf16xBf16ToBf16<Arch::kSm90>, GemmOperation);

template struct Adaptor<Bf16xBf16ToBf16<Arch::kSm90>>;
template struct DeviceKernel<Bf16xBf16ToBf16<Arch::kSm90>>;

}  // namespace xla::gpu::kernel::gemm_universal

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

#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm_kernel.cu.h"
#include "xla/service/gpu/kernels/cutlass_gemm_kernels.cu.h"
#include "xla/statusor.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::gemm_universal {

template <typename Gemm>
static StatusOr<CustomKernel> LoadCutlassGemmUniversal(
    std::string name, int32_t m, int32_t n, int32_t k,
    const ArgsIndices& indices, const DynamicSliceIndices& slices,
    const se::DeviceDescription& device) {
  using Kernel = typename Gemm::GemmKernel;

  cutlass::gemm::GemmCoord problem_size = {m, n, k};

  auto packing =
      ArgsPacking<Gemm>(problem_size, indices, slices, device.core_count());

  se::MultiKernelLoaderSpec spec(/*arity=*/2, std::move(packing));
  spec.AddInProcessSymbol(GetKernelSymbol<Gemm>(), name);

  return CustomKernel(std::move(name), std::move(spec),
                      BlockDim<Gemm>(problem_size), ThreadDim<Gemm>(),
                      sizeof(typename Kernel::SharedStorage));
}

StatusOr<CustomKernel> GetCutlassGemmKernel(
    std::string name, PrimitiveType dtype, int32_t m, int32_t n, int32_t k,
    const ArgsIndices& indices, const DynamicSliceIndices& slices,
    const se::DeviceDescription& device) {
  switch (dtype) {
    case PrimitiveType::F32:
      return LoadCutlassGemmUniversal<CutlassGemmKernels::F32xF32toF32>(
          std::move(name), m, n, k, indices, slices, device);
    case PrimitiveType::BF16:
      return LoadCutlassGemmUniversal<CutlassGemmKernels::BF16xBF16toBF16>(
          std::move(name), m, n, k, indices, slices, device);
    default:
      return absl::InvalidArgumentError("Unsupported CUTLASS gemm data type");
  }
}

}  // namespace xla::gpu::kernel::gemm_universal

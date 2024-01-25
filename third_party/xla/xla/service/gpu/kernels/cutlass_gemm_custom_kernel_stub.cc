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

#include <cstdint>

#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::gemm_universal {

absl::StatusOr<CustomKernel> GetCutlassGemmKernel(
    std::string name, PrimitiveType dtype, int32_t m, int32_t n, int32_t k,
    const ArgsIndices& indices, const DynamicSliceIndices& slices,
    const se::DeviceDescription& device) {
  return absl::InternalError("XLA compiled without CUDA support");
}

absl::StatusOr<CustomKernel> LoadCutlassGemmKernel(
    std::string name, const std::string& library_path, PrimitiveType dtype,
    int32_t m, int32_t n, int32_t k, const ArgsIndices& indices,
    const DynamicSliceIndices& slices, const se::DeviceDescription& device) {
  return absl::InternalError("XLA compiled without CUDA support");
}

}  // namespace xla::gpu::kernel::gemm_universal

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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_CUSTOM_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_CUSTOM_KERNEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::gemm_universal {

// Returns pre-compiled custom kernels for a given data type and problem size.
absl::StatusOr<std::vector<CustomKernel>> GetCutlassGemmKernels(
    std::string name, PrimitiveType dtype, int32_t m, int32_t n, int32_t k,
    const ArgsIndices& indices, const DynamicSliceIndices& slices,
    const se::DeviceDescription& device);

// Loads custom kernel for a given data type and problem size from a shared
// library. It's up to the caller to guarantee that CUTLASS kernel in the shared
// library is compatible with the data type and problem size.
absl::StatusOr<CustomKernel> LoadCutlassGemmKernel(
    std::string name, const std::string& library_path, PrimitiveType dtype,
    int32_t m, int32_t n, int32_t k, const ArgsIndices& indices,
    const DynamicSliceIndices& slices, const se::DeviceDescription& device);

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_CUSTOM_KERNEL_H_

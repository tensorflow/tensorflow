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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_HELPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_HELPER_H_

#include <cstdint>

#include "rocm/include/hip/hip_runtime_api.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"

namespace stream_executor::rocm {

// Fills device-side hipBLASLt grouped-GEMM user arguments for a ragged dot.
void GroupGemmUpdateArgs(
    hipStream_t stream, DeviceAddressBase args, DeviceAddressBase a,
    DeviceAddressBase b, DeviceAddressBase c, DeviceAddressBase d,
    DeviceAddressBase bias, DeviceAddressBase group_sizes,
    uint8_t group_size_bytewidth, uint8_t log2_byte_width_elem_a,
    uint8_t log2_byte_width_elem_b, uint8_t log2_byte_width_elem_c,
    uint8_t log2_byte_width_elem_d, uint32_t stride_ragged_dim,
    uint32_t stride_group_dim, uint32_t c_stride_ragged_dim,
    uint32_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    gpu::RaggedDotMode ragged_mode, uint32_t num_gemms, int32_t activation_type,
    int8_t bias_type, bool has_matrix_bias);

}  // namespace stream_executor::rocm

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_HELPER_H_

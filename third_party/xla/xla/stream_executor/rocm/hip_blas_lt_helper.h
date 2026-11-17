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

// Bundles all host-side parameters for a grouped/ragged-dot UserArguments
// update launch. This replaces a ~35-argument free-function signature.
struct GroupGemmUpdateArgsParams {
  DeviceAddressBase args, a, b, c, d, bias, group_sizes;
  uint8_t group_size_bytewidth;
  // Byte strides: the element stride premultiplied by the operand's element
  // width, with stride_a/stride_b already swapped when must_swap_operands.
  uint64_t stride_a_bytes, stride_b_bytes, c_stride_ragged_dim_bytes,
      output_stride_ragged_dim_bytes;
  bool must_swap_operands;
  uint32_t m, n, k, batch;
  uint32_t strideA1, strideA2, strideB1, strideB2, strideC1, strideC2, strideD1,
      strideD2;
  gpu::RaggedDotMode ragged_mode;
  uint32_t num_gemms;
  int32_t activation_type;
  int8_t bias_type;
  bool has_matrix_bias;
};

// Launches the kernel that populates the device-side hipBLASLt UserArguments
// array for a grouped/ragged-dot GEMM. Defined in hip_blas_lt_helper.cu.cc.
void GroupGemmUpdateArgs(hipStream_t stream,
                         const GroupGemmUpdateArgsParams& params);

}  // namespace stream_executor::rocm

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_HELPER_H_

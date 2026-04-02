/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_XNNPACK_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_XNNPACK_H_

#include <cstdint>

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// Executes GEMM using XNNPACK when supported. Returns true if XNNPACK handled
// the operation, or false if the caller must fall back to the reference path
// (e.g. XNNPACK disabled at build time, unsupported layout combination,
// asymmetric int8 filter zero point, or out-of-range requantization scale).
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingXnnpack {
  static bool Run(
      const MatrixParams<LhsScalar>& /*lhs_params*/,
      const LhsScalar* /*lhs_data*/,
      const MatrixParams<RhsScalar>& /*rhs_params*/,
      const RhsScalar* /*rhs_data*/,
      const MatrixParams<DstScalar>& /*dst_params*/, DstScalar* /*dst_data*/,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& /*params*/,
      CpuBackendContext* /*context*/) {
    return false;
  }
};

template <>
struct GemmImplUsingXnnpack<float, float, float, float,
                            QuantizationFlavor::kFloatingPoint> {
  static bool Run(const MatrixParams<float>& lhs_params, const float* lhs_data,
                  const MatrixParams<float>& rhs_params, const float* rhs_data,
                  const MatrixParams<float>& dst_params, float* dst_data,
                  const GemmParams<float, float,
                                   QuantizationFlavor::kFloatingPoint>& params,
                  CpuBackendContext* context);
};

template <>
struct GemmImplUsingXnnpack<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                            QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static bool Run(
      const MatrixParams<std::int8_t>& lhs_params, const std::int8_t* lhs_data,
      const MatrixParams<std::int8_t>& rhs_params, const std::int8_t* rhs_data,
      const MatrixParams<std::int8_t>& dst_params, std::int8_t* dst_data,
      const GemmParams<std::int32_t, std::int8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context);
};

template <>
struct GemmImplUsingXnnpack<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                            QuantizationFlavor::kIntegerWithPerRowMultiplier> {
  static bool Run(
      const MatrixParams<std::int8_t>& lhs_params, const std::int8_t* lhs_data,
      const MatrixParams<std::int8_t>& rhs_params, const std::int8_t* rhs_data,
      const MatrixParams<std::int8_t>& dst_params, std::int8_t* dst_data,
      const GemmParams<std::int32_t, std::int8_t,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>&
          params,
      CpuBackendContext* context);
};

template <>
struct GemmImplUsingXnnpack<std::uint8_t, std::uint8_t, std::int32_t,
                            std::uint8_t,
                            QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static bool Run(
      const MatrixParams<std::uint8_t>& lhs_params,
      const std::uint8_t* lhs_data,
      const MatrixParams<std::uint8_t>& rhs_params,
      const std::uint8_t* rhs_data,
      const MatrixParams<std::uint8_t>& dst_params, std::uint8_t* dst_data,
      const GemmParams<std::int32_t, std::uint8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context);
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_XNNPACK_H_

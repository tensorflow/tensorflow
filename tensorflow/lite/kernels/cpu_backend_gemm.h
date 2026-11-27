/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_

#include <cstdint>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_reference.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_xnnpack.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

namespace cpu_backend_gemm {

// Main entry point for CpuBackendGemm::Gemm.
//
// Dispatches supported float, int8, and uint8 GEMMs to XNNPACK when
// TFLITE_KERNEL_USE_XNNPACK is enabled, falling back to a portable C++
// reference implementation for unsupported layouts, asymmetric int8 weights,
// 16x8, or raw int32 accumulators.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
          const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    // For now, assert in debug mode, return in opt.
    // TODO(b/183099395) Eliminate debug/release discrepancy by plumbing in
    // TFLiteStatus so we can return an error here.
    TFLITE_DCHECK(false);
    return;
  }

  if (detail::GemmImplUsingXnnpack<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                   quantization_flavor>::Run(lhs_params,
                                                             lhs_data,
                                                             rhs_params,
                                                             rhs_data,
                                                             dst_params,
                                                             dst_data, params,
                                                             context)) {
    return;
  }

  detail::GemmImplUsingReference<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                 quantization_flavor>::Run(lhs_params, lhs_data,
                                                           rhs_params, rhs_data,
                                                           dst_params, dst_data,
                                                           params, context);
}

// Special path for 16x8 quant gemm.
template <QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<int8_t>& lhs_params, const int8_t* lhs_data,
          const MatrixParams<int16_t>& rhs_params, const int16_t* rhs_data,
          const MatrixParams<int16_t>& dst_params, int16_t* dst_data,
          const GemmParams<int32_t, int16_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    TFLITE_DCHECK(false);
    return;
  }

  detail::GemmImplUsingReference<int8_t, int16_t, int32_t, int16_t,
                                 quantization_flavor>::Run(lhs_params, lhs_data,
                                                           rhs_params, rhs_data,
                                                           dst_params, dst_data,
                                                           params, context);
}

// Special path for gemm with raw accumulator case. i.e. AccumScalar ==
// DstScalar == int32 case.
template <typename LhsScalar, typename RhsScalar,
          QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<int32_t>& dst_params, int32_t* dst_data,
          const GemmParams<int32_t, int32_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    TFLITE_DCHECK(false);
    return;
  }

  ruy::profiler::ScopeLabel label2("cpu_backend_gemm::Gemm: general GEMM");
  detail::GemmImplUsingReference<LhsScalar, RhsScalar, int32_t, int32_t,
                                 quantization_flavor>::Run(lhs_params, lhs_data,
                                                           rhs_params, rhs_data,
                                                           dst_params, dst_data,
                                                           params, context);
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_

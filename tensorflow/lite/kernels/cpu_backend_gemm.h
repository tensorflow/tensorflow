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
#include "tensorflow/lite/kernels/cpu_backend_gemm_custom_gemv.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"

#ifndef TFLITE_WITH_RUY
#include "tensorflow/lite/kernels/cpu_backend_gemm_eigen.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_gemmlowp.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_x86.h"
#endif

namespace tflite {

namespace cpu_backend_gemm {

// The main entry point for CpuBackendGemm::Gemm.
//
// If TFLITE_WITH_RUY is set, CpuBackendGemm::Gemm will always go to Ruy aka
// GemmImplUsingRuy. Other cases are as follows:
//
//                    |Quantized (uint8)|Quantized (int8)| Float |
// TFLITE_WITH_RUY    |      Ruy        |      Ruy       | Ruy   |
// !TFLITE_WITH_RUY   |      gemmlowp   |  Ruy/gemmlowp* | eigen |
// * - Ruy if NEON is not available.

//  On x86 platforms:
//  (default)         |      gemmlowp   |     Ruy        | eigen |
//  TFLITE_X86_RUY_\  |      Ruy        |     Ruy        | Ruy   |
//  ENABLED && (AVX
//  or above available)

#if !defined(TFLITE_WITH_RUY) && defined(TFLITE_X86_PLATFORM)
/* GEMM dispatch implementation for x86.
 */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplX86<LhsScalar, RhsScalar, AccumScalar,
                                      DstScalar, quantization_flavor> {};
#else
/* Generic implementation using ruy.
 * Non-ruy implementation will be partial specializations of this template.
 */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                           DstScalar, quantization_flavor> {};

#if !defined(TFLITE_WITH_RUY)

/* Specializations using gemmlowp */
template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingGemmlowp<SrcScalar, SrcScalar, std::int32_t,
                                    DstScalar, quantization_flavor> {};

// When SrcScalar=int8 or DstScalar=int8, gemmlowp fails to compile
// outside of NEON. We avoid the compilation failure by subspecializing these
// cases, rerouting it back to ruy.
#if !defined(GEMMLOWP_NEON)
template <typename SrcScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                               quantization_flavor> {};

template <typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               DstScalar, quantization_flavor> {};

template <QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               std::int8_t, quantization_flavor> {};
#endif  // not GEMMLOWP_NEON

/* Specializations using Eigen */

template <>
struct GemmImpl<float, float, float, float, QuantizationFlavor::kFloatingPoint>
    : detail::GemmImplUsingEigen {};

#endif  // not TFLITE_WITH_RUY

#endif  // not TFLITE_WITH_RUY and TFLITE_X86_PLATFORM

/* Public entry point */

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
  // In some cases we want to unconditionally use ruy as the backend, overriding
  // the `tflite_with_ruy` setting and the platform default.
  bool must_use_ruy = false;
  if (context->use_caching()) {
    // Only ruy supports caching of pre-packed matrices. Due to the large
    // performance impact in the cases where it's typically used, this overrides
    // the default.
    must_use_ruy = true;
  }
  if (lhs_params.order != Order::kRowMajor ||
      rhs_params.order != Order::kColMajor ||
      dst_params.order != Order::kColMajor) {
    // ruy supports all 2^3=8 combinations of storage orders with comparable
    // performance. In ruy, it's only a runtime switch. In other backends
    // (gemmlowp, Eigen), storage orders are template parameters, supporting
    // all 8 combinations would be up to a 8-fold code size increase, so we
    // prefer to force usage of ruy in these cases.
    must_use_ruy = true;
  }
  if (must_use_ruy) {
    detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             quantization_flavor>::Run(lhs_params, lhs_data,
                                                       rhs_params, rhs_data,
                                                       dst_params, dst_data,
                                                       params, context);
    return;
  }
  // If we did not choose to force usage of ruy above, then we may now consider
  // using custom GEMV code for the matrix*vector cases.
  const bool try_custom_gemv = (dst_params.cols == 1);
  if (try_custom_gemv) {
    // GEMV case: try a custom fast GEMV path. It will return true if it
    // actually handled it.
    if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
                           dst_params, dst_data, params, context)) {
      return;
    }
  }
  // Generic case: dispatch to any backend as a general GEMM.
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
           quantization_flavor>::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                                     dst_params, dst_data, params, context);
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

  // Currently, only Ruy backend supports 16x8 quant gemm so we use ruy
  // only.
  detail::GemmImplUsingRuy<int8_t, int16_t, int32_t, int16_t,
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

  // Currently, only Ruy backend supports get raw accumulator, so we use ruy
  // only.
  ruy::profiler::ScopeLabel label2("cpu_backend_gemm::Gemm: general GEMM");
  detail::GemmImplUsingRuy<LhsScalar, RhsScalar, int32_t, int32_t,
                           quantization_flavor>::Run(lhs_params, lhs_data,
                                                     rhs_params, rhs_data,
                                                     dst_params, dst_data,
                                                     params, context);
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_

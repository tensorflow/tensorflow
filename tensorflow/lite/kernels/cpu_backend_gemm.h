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

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_custom_gemv.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"

#ifndef TFLITE_WITH_RUY
#include "tensorflow/lite/kernels/cpu_backend_gemm_eigen.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_gemmlowp.h"
#endif

namespace tflite {

namespace cpu_backend_gemm {

/* Generic implementation using ruy.
 * Non-ruy implementation will be partial specializations of this template.
 */

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                           DstScalar, quantization_flavor> {};

#ifndef TFLITE_WITH_RUY

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
#ifndef GEMMLOWP_NEON
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
  bool do_custom_gemv = dst_params.cols == 1;
#ifdef TFLITE_WITH_RUY_GEMV
  // Prefer a Ruy GEMM to Custom GEMV unless we are doing float math.
  // TODO(b/148692500): Add float GEMV kernels to Ruy.
  do_custom_gemv = do_custom_gemv && std::is_floating_point<DstScalar>::value;
#endif
  if (do_custom_gemv) {
    // GEMV case: try a custom fast GEMV path.
    if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
                           dst_params, dst_data, params, context)) {
      return;
    }
  }
  ruy::profiler::ScopeLabel label2("cpu_backend_gemm::Gemm: general GEMM");
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
           quantization_flavor>::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                                     dst_params, dst_data, params, context);
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

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_

#include "ruy/matrix.h"  // from @ruy
#include "ruy/path.h"  // from @ruy
#include "ruy/ruy.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

inline ruy::CachePolicy ToRuyCachePolicy(CachePolicy cache_policy) {
  switch (cache_policy) {
    case CachePolicy::kNeverCache:
      return ruy::CachePolicy::kNeverCache;
    case CachePolicy::kCacheIfLargeSpeedup:
      return ruy::CachePolicy::kCacheIfLargeSpeedup;
    case CachePolicy::kAlwaysCache:
      return ruy::CachePolicy::kAlwaysCache;
    default:
      TFLITE_DCHECK(false);
      return ruy::CachePolicy::kNeverCache;
  }
}

template <typename Scalar, typename DataPointer>
void MakeRuyMatrix(const MatrixParams<Scalar>& params, DataPointer data_ptr,
                   ruy::Matrix<Scalar>* dst, bool use_caching = false) {
  ruy::Order ruy_order = params.order == Order::kColMajor
                             ? ruy::Order::kColMajor
                             : ruy::Order::kRowMajor;
  ruy::MakeSimpleLayout(params.rows, params.cols, ruy_order,
                        dst->mutable_layout());
  // Note that ruy::Matrix::data is a ConstCheckingPtr, not a plain pointer.
  // It does care whether we assign to it a Scalar* or a const Scalar*.
  dst->set_data(data_ptr);
  dst->set_zero_point(params.zero_point);
  if (use_caching) {
    dst->set_cache_policy(ToRuyCachePolicy(params.cache_policy));
  }
}

template <typename GemmParamsType, typename RuySpecType>
void MakeRuyMulParams(const GemmParamsType& params,
                      RuySpecType* ruy_mul_params) {
  // This validation has already been performed by the Gemm API entry point,
  // but it doesn't hurt to test specifically this again here, where it's
  // being used.
  ValidateGemmParams(params);

  ruy_mul_params->set_multiplier_fixedpoint(params.multiplier_fixedpoint);
  ruy_mul_params->set_multiplier_exponent(params.multiplier_exponent);
  ruy_mul_params->set_multiplier_fixedpoint_perchannel(
      params.multiplier_fixedpoint_perchannel);
  ruy_mul_params->set_multiplier_exponent_perchannel(
      params.multiplier_exponent_perchannel);
  ruy_mul_params->set_bias(params.bias);
  ruy_mul_params->set_clamp_min(params.clamp_min);
  ruy_mul_params->set_clamp_max(params.clamp_max);
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingRuy {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    ruy::Matrix<LhsScalar> ruy_lhs;
    ruy::Matrix<RhsScalar> ruy_rhs;
    ruy::Matrix<DstScalar> ruy_dst;
    MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs, context->use_caching());
    MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs, context->use_caching());
    MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::MulParams<AccumScalar, DstScalar> ruy_mul_params;
    MakeRuyMulParams(params, &ruy_mul_params);

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, context->ruy_context(),
             &ruy_dst);
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_

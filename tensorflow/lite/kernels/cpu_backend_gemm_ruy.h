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

#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename Scalar, typename DataPointer>
void MakeRuyMatrix(const MatrixParams<Scalar>& params, DataPointer data_ptr,
                   ruy::Matrix<Scalar>* dst) {
  dst->layout.rows = params.rows;
  dst->layout.cols = params.cols;
  if (params.order == Order::kColMajor) {
    dst->layout.order = ruy::Order::kColMajor;
    dst->layout.stride = params.rows;
  } else {
    dst->layout.order = ruy::Order::kRowMajor;
    dst->layout.stride = params.cols;
  }
  // Note that ruy::Matrix::data is a ConstCheckingPtr, not a plain pointer.
  // It does care whether we assign to it a Scalar* or a const Scalar*.
  dst->data = data_ptr;
  dst->zero_point = params.zero_point;
}

template <typename GemmParamsType, typename RuySpecType>
void MakeRuySpec(const GemmParamsType& params, RuySpecType* ruy_spec) {
  // This validation has already been performed by the Gemm API entry point,
  // but it doesn't hurt to test specifically this again here, where it's
  // being used.
  ValidateGemmParams(params);

  ruy_spec->multiplier_fixedpoint = params.multiplier_fixedpoint;
  ruy_spec->multiplier_exponent = params.multiplier_exponent;
  ruy_spec->multiplier_fixedpoint_perchannel =
      params.multiplier_fixedpoint_perchannel;
  ruy_spec->multiplier_exponent_perchannel =
      params.multiplier_exponent_perchannel;
  ruy_spec->bias = params.bias;
  ruy_spec->clamp_min = params.clamp_min;
  ruy_spec->clamp_max = params.clamp_max;
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
    MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs);
    MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs);
    MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::BasicSpec<AccumScalar, DstScalar> ruy_spec;
    MakeRuySpec(params, &ruy_spec);

    ruy::Mul<ruy::kAllPaths>(ruy_lhs, ruy_rhs, ruy_spec, context->ruy_context(),
                             &ruy_dst);
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_

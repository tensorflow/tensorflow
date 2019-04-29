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
#include <limits>
#include <type_traits>

#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"

namespace tflite {

namespace cpu_backend_gemm {

// Matrix storage order: column-major or row-major.
enum class Order { kColMajor, kRowMajor };

// MatrixParams encapsulates the parameters that Gemm needs about each
// matrix, besides the buffer data pointer.
// Compare to ruy::Matrix, which also encapsulates the data pointer.
// Rationale for leaving the data pointer out of here: doing so
// requires complicated const-correctness mechanics. See
// ruy::ConstCheckingPtr.
template <typename Scalar>
struct MatrixParams {
  // Storage layout order. For now we only do plain linear non-strided
  // layout. It would be easy to support a stride if needed.
  Order order = Order::kColMajor;
  // Number of rows of the matrix.
  int rows = 0;
  // Number of columns of the matrix.
  int cols = 0;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
};

// Additional parameters that Gemm needs, beyond what falls into
// the MatrixParams that it takes. Compare to ruy::Spec.
//
// Decoupling AccumScalar from DstScalar (rather than deducing it from that)
// is useful future-proofing. Think of a float16 path using float32 accum.
template <typename AccumScalar, typename DstScalar>
struct GemmParams {
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint = 0;
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent = 0;
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar* multiplier_fixedpoint_perchannel = nullptr;
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int* multiplier_exponent_perchannel = nullptr;
  // The bias vector data, if not null.
  const AccumScalar* bias = nullptr;
  // min clamp bound of destination values.
  DstScalar clamp_min = std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max = std::numeric_limits<DstScalar>::max();
};

/* Convenience typedefs */

template <typename DstScalar>
using QuantizedGemmParams = GemmParams<std::int32_t, DstScalar>;

using FloatGemmParams = GemmParams<float, float>;

/* Generic implementation using ruy */

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
  if (std::is_floating_point<typename RuySpecType::AccumScalar>::value) {
    TF_LITE_ASSERT(!params.multiplier_fixedpoint);
    TF_LITE_ASSERT(!params.multiplier_exponent);
    TF_LITE_ASSERT(!params.multiplier_fixedpoint_perchannel);
    TF_LITE_ASSERT(!params.multiplier_exponent_perchannel);
  } else {
    TF_LITE_ASSERT((params.multiplier_fixedpoint == 0) !=
                   (params.multiplier_fixedpoint_perchannel == nullptr));
  }
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

}  // namespace detail

// Non-ruy implementation will be partial specializations of this template.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct GemmImpl {
  static void Run(const MatrixParams<LhsScalar>& lhs_params,
                  const LhsScalar* lhs_data,
                  const MatrixParams<RhsScalar>& rhs_params,
                  const RhsScalar* rhs_data,
                  const MatrixParams<DstScalar>& dst_params,
                  DstScalar* dst_data,
                  const GemmParams<AccumScalar, DstScalar>& params,
                  CpuBackendContext* context) {
    ruy::Matrix<LhsScalar> ruy_lhs;
    ruy::Matrix<RhsScalar> ruy_rhs;
    ruy::Matrix<DstScalar> ruy_dst;
    detail::MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs);
    detail::MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs);
    detail::MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::BasicSpec<AccumScalar, DstScalar> ruy_spec;
    detail::MakeRuySpec(params, &ruy_spec);

    ruy::Mul<ruy::kAllPaths>(ruy_lhs, ruy_rhs, ruy_spec, context->ruy_context(),
                             &ruy_dst);
  }
};

/* Public entry point */

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
          const GemmParams<AccumScalar, DstScalar>& params,
          CpuBackendContext* context) {
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar>::Run(
      lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data, params,
      context);
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_

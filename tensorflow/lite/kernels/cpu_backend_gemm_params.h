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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"

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
  DstScalar clamp_min = std::is_floating_point<DstScalar>::value
                            ? -std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max = std::is_floating_point<DstScalar>::value
                            ? std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::max();
};

/* Convenience typedefs */

template <typename DstScalar>
using QuantizedGemmParams = GemmParams<std::int32_t, DstScalar>;

using FloatGemmParams = GemmParams<float, float>;

/* Validation functions */

// Note that this uses TFLITE_DCHECK from kernels/internal/compatibility.h
// and not TF_LITE_ASSERT from op_macros.h. We want this to be explicitly
// debug-build-only assertions so that there's not reason not to
// generously validate, and TF_LITE_ASSERT is actually at the moment
// a release-build assertion. See b/131587258.

// Validates self-consistency of GemmParams.
template <typename AccumScalar, typename DstScalar>
void ValidateGemmParams(const GemmParams<AccumScalar, DstScalar>& params) {
  // For now require a bias vector. Again, ruy does not rely on that requirement
  // but the gemmlowp and Eigen path would require more code to handle it,
  // and currently TFLite only uses the case where there is a bias vector.
  TFLITE_DCHECK(params.bias);
  // Guard consistency of the quantized multiplier fields.
  if (std::is_floating_point<AccumScalar>::value) {
    // Floating point case: must not have any quantized multipliers
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  } else {
    // Quantized case. Must have either uniform or perchannel multiplier,
    // not both.
    TFLITE_DCHECK((params.multiplier_fixedpoint == 0) !=
                  (params.multiplier_fixedpoint_perchannel == nullptr));
    // Consistency of the two _perchannel fields.
    TFLITE_DCHECK((params.multiplier_exponent_perchannel == nullptr) ==
                  (params.multiplier_fixedpoint_perchannel == nullptr));
  }
}

// Validates overall consistency of all the parameters taken by a Gemm call:
// the 3 MatrixParams and the GemmParams. Even if currently these are
// checked only separately, it's good to have this validation done in one
// function taking all of these parameters at once, as in the future there
// may be mutual consistency requirements.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
void ValidateParams(const MatrixParams<LhsScalar>& lhs_params,
                    const MatrixParams<RhsScalar>& rhs_params,
                    const MatrixParams<DstScalar>& dst_params,
                    const GemmParams<AccumScalar, DstScalar>& params) {
  ValidateGemmParams(params);
  // For now, Gemm only supports this particular combination of storage orders.
  // Actually the generic ruy path already supports all combinations (with
  // various performance penalties). On the other hand, gemmlowp and Eigen
  // paths would require more source code and larger binary code to handle
  // other combinations (because orders are template parameters in gemmlowp
  // and Eigen). Since this is TFLite's own internal Gemm library, there is
  // no point in supporting more than what TFlite currently uses, and that
  // is for now this single combination.
  TFLITE_DCHECK(lhs_params.order == Order::kRowMajor);
  TFLITE_DCHECK(rhs_params.order == Order::kColMajor);
  TFLITE_DCHECK(dst_params.order == Order::kColMajor);
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_

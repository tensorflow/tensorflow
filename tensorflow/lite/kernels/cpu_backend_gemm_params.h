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

// Enumeration of broad categories of Gemm.
//
// The primary reason for this to exist is to allow Gemm to compile
// only uniform-quantized or only per-channel-quantized code paths.
// This is unneeded with ruy as the back-end, as this is only a runtime
// difference in ruy, but with gemmlowp these really are separate code
// paths and templatizing in a QuantizationFlavor is necessary to avoid
// compiling unused gemmlowp code. Indeed, TFLite currently uses
// uint8 with uniform quantization and int8 with per-channel quantization,
// and does not use uint8 with per-channel. We want to avoid compiling
// the gemmlowp uint8 per-channel path when gemmlowp is the back-end.
//
// It's possible to drop this in the future if gemmlowp goes away and no
// other then-relevant backend library handles quantized paths in a way that
// requires knowing this at compile-time.
enum class QuantizationFlavor {
  // Floating-point Gemm: the accumulators are not multiplied by any
  // 'multiplier'.
  kFloatingPoint,
  // Quantized Gemm using a single multiplier for all accumulators.
  kIntegerWithUniformMultiplier,
  // Quantized Gemm using a separate multipliers for accumulators of each
  // row of the destination matrix. This is what is called 'per-channel'
  // in GemmParams. Here we use the more specific 'per-row' terminology
  // to allow for the possibility of 'per-column' in the future, and to
  // allow for that to be a separate code path in some back-end such as
  // gemmlowp.
  kIntegerWithPerRowMultiplier
};

// Additional parameters that Gemm needs, beyond what falls into
// the MatrixParams that it takes. Compare to ruy::Spec.
//
// Decoupling AccumScalar from DstScalar (rather than deducing it from that)
// is useful future-proofing. Think of a float16 path using float32 accum.
//
// QuantizationFlavor is passed here even though it's technically not used
// in this class. This is so that we retain the ability in the future to
// specialize this class for quantization flavor, and this allows for
// Gemm to be templatized in quantization_flavor via the GemmParams that it
// takes, allowing for automatic template parameter deduction to take place,
// so that most call sites don't need to specify a QuantizationFlavor
// (only those that need perchannel quantization do).
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor =
              std::is_floating_point<AccumScalar>::value
                  ? QuantizationFlavor::kFloatingPoint
                  : QuantizationFlavor::kIntegerWithUniformMultiplier>
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
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
void ValidateGemmParams(
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
  // Guard consistency of the quantized multiplier fields.
  if (quantization_flavor == QuantizationFlavor::kFloatingPoint) {
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  } else if (quantization_flavor ==
                 QuantizationFlavor::kIntegerWithUniformMultiplier &&
             !std::is_same<DstScalar, int32_t>::value) {
    TFLITE_DCHECK(params.multiplier_fixedpoint);
    // Nothing to check about multiplier_exponent
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  } else if (quantization_flavor ==
                 QuantizationFlavor::kIntegerWithPerRowMultiplier &&
             !std::is_same<DstScalar, int32_t>::value) {
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(params.multiplier_exponent_perchannel);
  } else {
    // For the get raw accumulator case, we should make sure none of the
    // quantization params are set.
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  }
}

namespace detail {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct ValidateTypes {
  // This generic implementation is for quantized flavors.
  // kFloatingPoint will be a specialization below.
  static_assert(!std::is_floating_point<LhsScalar>::value, "");
  static_assert(!std::is_floating_point<RhsScalar>::value, "");
  static_assert(!std::is_floating_point<AccumScalar>::value, "");
  // No requirement on DstScalar --- we might in the future allow it
  // to be floating point even in a quantized Gemm.
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct ValidateTypes<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                     QuantizationFlavor::kFloatingPoint> {
  static_assert(std::is_floating_point<LhsScalar>::value, "");
  static_assert(std::is_floating_point<RhsScalar>::value, "");
  static_assert(std::is_floating_point<AccumScalar>::value, "");
  static_assert(std::is_floating_point<DstScalar>::value, "");
};

}  // namespace detail

// Validates overall consistency of all the parameters taken by a Gemm call:
// the 3 MatrixParams and the GemmParams.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void ValidateParams(
    const MatrixParams<LhsScalar>& lhs_params,
    const MatrixParams<RhsScalar>& rhs_params,
    const MatrixParams<DstScalar>& dst_params,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
  (void)detail::ValidateTypes<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                              quantization_flavor>();
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

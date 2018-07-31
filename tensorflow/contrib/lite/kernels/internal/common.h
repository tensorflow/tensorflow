/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMMON_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMMON_H_

#ifndef ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#ifdef GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#endif
#endif

#ifndef USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>
#endif

#if defined __GNUC__ && defined __SSE4_1__
#define USE_NEON

#define OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wattributes"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wsequence-point"

#include "NEON_2_SSE.h"

#pragma GCC diagnostic pop
#endif
#endif

#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

inline void GetActivationMinMax(FusedActivationFunctionType ac,
                                float* output_activation_min,
                                float* output_activation_max) {
  switch (ac) {
    case FusedActivationFunctionType::kNone:
      *output_activation_min = std::numeric_limits<float>::lowest();
      *output_activation_max = std::numeric_limits<float>::max();
      break;
    case FusedActivationFunctionType::kRelu:
      *output_activation_min = 0.f;
      *output_activation_max = std::numeric_limits<float>::max();
      break;
    case FusedActivationFunctionType::kRelu1:
      *output_activation_min = -1.f;
      *output_activation_max = 1.f;
      break;
    case FusedActivationFunctionType::kRelu6:
      *output_activation_min = 0.f;
      *output_activation_max = 6.f;
      break;
  }
}

inline float ActivationFunctionWithMinMax(float x, float output_activation_min,
                                          float output_activation_max) {
  return std::min(std::max(x, output_activation_min), output_activation_max);
}

// Legacy function, left for compatibility only.
template <FusedActivationFunctionType Ac>
float ActivationFunction(float x) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  return ActivationFunctionWithMinMax(x, output_activation_min,
                                      output_activation_max);
}

inline int32 MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32 x, int32 quantized_multiplier, int left_shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

inline int32 MultiplyByQuantizedMultiplierGreaterThanOne(
    int32 x, int32 quantized_multiplier, int left_shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                           quantized_multiplier);
}

inline int32 MultiplyByQuantizedMultiplier(int32 x, int32 quantized_multiplier,
                                           int shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                 x * (1 << left_shift), quantized_multiplier),
                             right_shift);
}

template <typename T>
int CountLeadingZeros(T integer_input) {
  static_assert(std::is_unsigned<T>::value,
                "Only unsigned integer types handled.");
  const T one_in_leading_positive = static_cast<T>(1)
                                    << (std::numeric_limits<T>::digits - 1);
  int leading_zeros = 0;
  while (integer_input < one_in_leading_positive) {
    integer_input <<= 1;
    ++leading_zeros;
  }
  return leading_zeros;
}

// DO NOT USE THIS STRUCT FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING
// BROADCASTING.
//
// NdArrayDesc<N> describes the shape and memory layout of an N-dimensional
// rectangular array of numbers.
//
// NdArrayDesc<N> is basically identical to Dims<N> defined in types.h.
// However, as Dims<N> is to be deprecated, this class exists as an adaptor
// to enable simple unoptimized implementations of element-wise broadcasting
// operations.
template <int N>
struct NdArrayDesc {
  // The "extent" of each dimension. Indices along dimension d must be in the
  // half-open interval [0, extents[d]).
  int extents[N];

  // The number of *elements* (not bytes) between consecutive indices of each
  // dimension.
  int strides[N];
};

// DO NOT USE THIS FUNCTION FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING
// BROADCASTING.
//
// Same as Offset(), except takes as NdArrayDesc<N> instead of Dims<N>.
inline int SubscriptToIndex(const NdArrayDesc<4>& desc, int i0, int i1, int i2,
                            int i3) {
  TFLITE_DCHECK(i0 >= 0 && i0 < desc.extents[0]);
  TFLITE_DCHECK(i1 >= 0 && i1 < desc.extents[1]);
  TFLITE_DCHECK(i2 >= 0 && i2 < desc.extents[2]);
  TFLITE_DCHECK(i3 >= 0 && i3 < desc.extents[3]);
  return i0 * desc.strides[0] + i1 * desc.strides[1] + i2 * desc.strides[2] +
         i3 * desc.strides[3];
}

// Given the dimensions of the operands for an element-wise binary broadcast,
// adjusts them so that they can be directly iterated over with simple loops.
// Returns the adjusted dims as instances of NdArrayDesc in 'desc0_out' and
// 'desc1_out'. 'desc0_out' and 'desc1_out' cannot be nullptr.
//
// This function assumes that the two input shapes are compatible up to
// broadcasting and the shorter one has already been prepended with 1s to be the
// same length. E.g., if shape0 is (1, 16, 16, 64) and shape1 is (1, 64),
// shape1 must already have been prepended to be (1, 1, 1, 64). Recall that
// Dims<N> refer to shapes in reverse order. In this case, input0_dims will be
// (64, 16, 16, 1) and input1_dims will be (64, 1, 1, 1).
//
// When two shapes are compatible up to broadcasting, for each dimension d,
// the input extents are either equal, or one of them is 1.
//
// This function performs the following for each dimension d:
// - If the extents are equal, then do nothing since the loop that walks over
//   both of the input arrays is correct.
// - Otherwise, one (and only one) of the extents must be 1. Say extent0 is 1
//   and extent1 is e1. Then set extent0 to e1 and stride0 *to 0*. This allows
//   array0 to be referenced *at any index* in dimension d and still access the
//   same slice.
template <int N>
inline void NdArrayDescsForElementwiseBroadcast(const Dims<N>& input0_dims,
                                                const Dims<N>& input1_dims,
                                                NdArrayDesc<N>* desc0_out,
                                                NdArrayDesc<N>* desc1_out) {
  TFLITE_DCHECK(desc0_out != nullptr);
  TFLITE_DCHECK(desc1_out != nullptr);

  // Copy dims to desc.
  for (int i = 0; i < N; ++i) {
    desc0_out->extents[i] = input0_dims.sizes[i];
    desc0_out->strides[i] = input0_dims.strides[i];
    desc1_out->extents[i] = input1_dims.sizes[i];
    desc1_out->strides[i] = input1_dims.strides[i];
  }

  // Walk over each dimension. If the extents are equal do nothing.
  // Otherwise, set the desc with extent 1 to have extent equal to the other and
  // stride 0.
  for (int i = 0; i < N; ++i) {
    const int extent0 = ArraySize(input0_dims, i);
    const int extent1 = ArraySize(input1_dims, i);
    if (extent0 != extent1) {
      if (extent0 == 1) {
        desc0_out->strides[i] = 0;
        desc0_out->extents[i] = extent1;
      } else {
        TFLITE_DCHECK_EQ(extent1, 1);
        desc1_out->strides[i] = 0;
        desc1_out->extents[i] = extent0;
      }
    }
  }
}

template <int N>
inline void NdArrayDescsForElementwiseBroadcast(
    const RuntimeShape& input0_shape, const RuntimeShape& input1_shape,
    NdArrayDesc<N>* desc0_out, NdArrayDesc<N>* desc1_out) {
  TFLITE_DCHECK(desc0_out != nullptr);
  TFLITE_DCHECK(desc1_out != nullptr);

  auto extended_input0_shape = RuntimeShape::ExtendedShape(N, input0_shape);
  auto extended_input1_shape = RuntimeShape::ExtendedShape(N, input1_shape);

  // Copy dims to desc, calculating strides.
  int desc0_stride = 1;
  int desc1_stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    desc0_out->extents[i] = extended_input0_shape.Dims(i);
    desc0_out->strides[i] = desc0_stride;
    desc0_stride *= extended_input0_shape.Dims(i);
    desc1_out->extents[i] = extended_input1_shape.Dims(i);
    desc1_out->strides[i] = desc1_stride;
    desc1_stride *= extended_input1_shape.Dims(i);
  }

  // Walk over each dimension. If the extents are equal do nothing.
  // Otherwise, set the desc with extent 1 to have extent equal to the other and
  // stride 0.
  for (int i = 0; i < N; ++i) {
    const int extent0 = extended_input0_shape.Dims(i);
    const int extent1 = extended_input1_shape.Dims(i);
    if (extent0 != extent1) {
      if (extent0 == 1) {
        desc0_out->strides[i] = 0;
        desc0_out->extents[i] = extent1;
      } else {
        TFLITE_DCHECK_EQ(extent1, 1);
        desc1_out->strides[i] = 0;
        desc1_out->extents[i] = extent0;
      }
    }
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMMON_H_

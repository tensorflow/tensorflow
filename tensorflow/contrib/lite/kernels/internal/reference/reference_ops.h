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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_

#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

// TODO(b/77858996): Add these to gemmlowp.
template <typename IntegerType>
IntegerType SaturatingAddNonGemmlowp(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

template <>
inline std::int32_t SaturatingAddNonGemmlowp(std::int32_t a, std::int32_t b) {
  std::int64_t a64 = a;
  std::int64_t b64 = b;
  std::int64_t sum = a64 + b64;
  return static_cast<std::int32_t>(std::min(
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()),
      std::max(
          static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()),
          sum)));
}

template <typename tRawType, int tIntegerBits>
gemmlowp::FixedPoint<tRawType, tIntegerBits> SaturatingAddNonGemmlowp(
    gemmlowp::FixedPoint<tRawType, tIntegerBits> a,
    gemmlowp::FixedPoint<tRawType, tIntegerBits> b) {
  return gemmlowp::FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SaturatingAddNonGemmlowp(a.raw(), b.raw()));
}

template <typename IntegerType>
IntegerType SaturatingSub(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

template <>
inline std::int16_t SaturatingSub(std::int16_t a, std::int16_t b) {
  std::int32_t a32 = a;
  std::int32_t b32 = b;
  std::int32_t diff = a32 - b32;
  return static_cast<std::int16_t>(std::min(32767, std::max(-32768, diff)));
}

template <>
inline std::int32_t SaturatingSub(std::int32_t a, std::int32_t b) {
  std::int64_t a64 = a;
  std::int64_t b64 = b;
  std::int64_t diff = a64 - b64;
  return static_cast<std::int32_t>(std::min(
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()),
      std::max(
          static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()),
          diff)));
}

template <typename tRawType, int tIntegerBits>
gemmlowp::FixedPoint<tRawType, tIntegerBits> SaturatingSub(
    gemmlowp::FixedPoint<tRawType, tIntegerBits> a,
    gemmlowp::FixedPoint<tRawType, tIntegerBits> b) {
  return gemmlowp::FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SaturatingSub(a.raw(), b.raw()));
}
// End section to be moved to gemmlowp.

namespace reference_ops {

// TODO(b/80247582) Remove this constant.
// This will be phased out as the shifts are revised with more thought. Use of a
// constant enables us to track progress on this work.
//
// Used mainly to convert from old-style shifts (right) to new-style (left).
static constexpr int kReverseShift = -1;

template <typename T>
int CountLeadingZeros(T integer_input) {
  static_assert(std::is_unsigned<T>::value,
                "Only unsigned integer types handled.");
  if (integer_input == 0) {
    return std::numeric_limits<T>::digits;
  }
  const T one_in_leading_positive = static_cast<T>(1)
                                    << (std::numeric_limits<T>::digits - 1);
  int leading_zeros = 0;
  while (integer_input < one_in_leading_positive) {
    integer_input <<= 1;
    ++leading_zeros;
  }
  return leading_zeros;
}

template <typename IntegerType>
IntegerType SaturatingRoundingMultiplyByPOTParam(IntegerType x, int exponent) {
  if (exponent == 0) {
    return x;
  }
  using ScalarIntegerType =
      typename gemmlowp::FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
  const IntegerType min =
      gemmlowp::Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::min());
  const IntegerType max =
      gemmlowp::Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::max());
  const int ScalarIntegerTypeBits = 8 * sizeof(ScalarIntegerType);

  const std::int32_t threshold =
      ((1 << (ScalarIntegerTypeBits - 1 - exponent)) - 1);
  const IntegerType positive_mask =
      gemmlowp::MaskIfGreaterThan(x, gemmlowp::Dup<IntegerType>(threshold));
  const IntegerType negative_mask =
      gemmlowp::MaskIfLessThan(x, gemmlowp::Dup<IntegerType>(-threshold));

  IntegerType result = gemmlowp::ShiftLeft(x, exponent);
  result = gemmlowp::SelectUsingMask(positive_mask, max, result);
  result = gemmlowp::SelectUsingMask(negative_mask, min, result);
  return result;
}

// If we want to leave IntegerBits fixed, then multiplication
// by a power of two has to be saturating/rounding, not exact anymore.
template <typename tRawType, int tIntegerBits>
gemmlowp::FixedPoint<tRawType, tIntegerBits>
SaturatingRoundingMultiplyByPOTParam(
    gemmlowp::FixedPoint<tRawType, tIntegerBits> a, int exponent) {
  return gemmlowp::FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SaturatingRoundingMultiplyByPOTParam(a.raw(), exponent));
}

// DO NOT USE THIS STRUCT FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING ELEMENT-WISE
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
// ELEMENT-WISE BROADCASTING.
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

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  (void)im2col_data;  // only used in optimized code.
  (void)im2col_dims;  // only used in optimized code.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  if (bias_data) {
    TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  }
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int dilation_width_factor,
          int dilation_height_factor, int pad_width, int pad_height,
          float* output_data, const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, dilation_width_factor,
       dilation_height_factor, pad_width, pad_height, output_activation_min,
       output_activation_max, output_data, output_dims, im2col_data,
       im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, 1, 1, pad_width, pad_height,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  Conv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
           bias_dims, stride, stride, 1, 1, pad_width, pad_height, output_data,
           output_dims, im2col_data, im2col_dims);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_dims;   // only used in optimized code.
  (void)gemm_context;  // only used in optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth =
      MatchingArraySize(filter_dims, 3, bias_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32 acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32 input_val = input_data[Offset(input_dims, in_channel,
                                                      in_x, in_y, batch)];
                  int32 filter_val =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
          }
          if (bias_data) {
            acc += bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          acc = MultiplyByQuantizedMultiplierSmallerThanOneExp(
              acc, output_multiplier, kReverseShift * output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              static_cast<uint8>(acc);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const uint8* input_data, const Dims<4>& input_dims,
          int32 input_offset, const uint8* filter_data,
          const Dims<4>& filter_dims, int32 filter_offset,
          const int32* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, int32 output_offset,
          int32 output_multiplier, int output_shift,
          int32 output_activation_min, int32 output_activation_max,
          uint8* output_data, const Dims<4>& output_dims, uint8* im2col_data,
          const Dims<4>& im2col_dims, gemmlowp::GemmContext* gemm_context) {
  Conv<Ac>(input_data, input_dims, input_offset, filter_data, filter_dims,
           filter_offset, bias_data, bias_dims, stride, stride, pad_width,
           pad_height, output_offset, output_multiplier, output_shift,
           output_activation_min, output_activation_max, output_data,
           output_dims, im2col_data, im2col_dims, gemm_context);
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int input_batch = ArraySize(input_dims, 3);

  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_batch = ArraySize(output_dims, 3);

  TFLITE_DCHECK_EQ(input_width * block_size, output_width);
  TFLITE_DCHECK_EQ(input_height * block_size, output_height);
  TFLITE_DCHECK_EQ(input_depth, output_depth * block_size * block_size);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          const int in_d =
              out_d + ((out_h % block_size) * block_size + out_w % block_size) *
                          output_depth;

          const int in_w = out_w / block_size;
          const int in_h = out_h / block_size;
          const int in_b = out_b;

          const int output_index =
              Offset(output_dims, out_d, out_w, out_h, out_b);
          const int input_index = Offset(input_dims, in_d, in_w, in_h, in_b);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int input_batch = ArraySize(input_dims, 3);

  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_batch = ArraySize(output_dims, 3);

  TFLITE_DCHECK_EQ(input_width, output_width * block_size);
  TFLITE_DCHECK_EQ(input_height, output_height * block_size);
  TFLITE_DCHECK_EQ(input_depth * block_size * block_size, output_depth);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int in_b = 0; in_b < input_batch; ++in_b) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      for (int in_w = 0; in_w < input_width; ++in_w) {
        for (int in_d = 0; in_d < input_depth; ++in_d) {
          const int out_d =
              in_d + ((in_h % block_size) * block_size + in_w % block_size) *
                         input_depth;
          const int out_w = in_w / block_size;
          const int out_h = in_h / block_size;
          const int out_b = in_b;

          const int output_index =
              Offset(output_dims, out_d, out_w, out_h, out_b);
          const int input_index = Offset(input_dims, in_d, in_w, in_h, in_b);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(weights_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(weights_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(weights_dims));
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      float total = 0.f;
      for (int d = 0; d < accum_depth; ++d) {
        total += input_data[b * accum_depth + d] *
                 weights_data[out_c * accum_depth + d];
      }
      float bias_value = 0.0f;
      if (bias_data) {
        bias_value = bias_data[Offset(bias_dims, out_c, 0, 0, 0)];
      }
      output_data[out_c + output_depth * b] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemm_context) {
  (void)gemm_context;  // only used in optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(filter_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(filter_dims));
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32 acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32 input_val = input_data[b * accum_depth + d];
        int32 filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[Offset(bias_dims, out_c, 0, 0, 0)];
      }
      acc = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          acc, output_multiplier, kReverseShift * output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8>(acc);
    }
  }
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, int16* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemm_context) {
  (void)gemm_context;  // only used in optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(filter_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(filter_dims));
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum = bias_data[out_c];
      // Accumulation loop.
      for (int d = 0; d < accum_depth; ++d) {
        int16 input_val = input_data[b * accum_depth + d] + input_offset;
        int16 filter_val = filter_data[out_c * accum_depth + d] + filter_offset;
        accum += filter_val * input_val;
      }
      // Down-scale the final int32 accumulator to the scale used by our
      // (16-bit, typically 3 integer bits) fixed-point format. The quantized
      // multiplier and shift here have been pre-computed offline
      // (e.g. by toco).
      accum = MultiplyByQuantizedMultiplier(accum, output_multiplier,
                                            -output_shift);
      // Saturate, cast to int16, and store to output array.
      accum = std::max(accum, output_activation_min - output_offset);
      accum = std::min(accum, output_activation_max - output_offset);
      accum += output_offset;
      output_data[out_c + output_depth * b] = accum;
    }
  }
}

inline void ShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data, gemmlowp::GemmContext* gemm_context) {
  (void)gemm_context;  // only used in optimized code.

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(weights_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(weights_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(weights_dims));
  TFLITE_DCHECK((accum_depth % 16) == 0);
  TFLITE_DCHECK((output_depth % 4) == 0);

  // Shuffling and xoring of input activations into the workspace buffer
  uint8* shuffled_input_workspace_ptr = shuffled_input_workspace_data;
  if (batches == 1) {
    for (int i = 0; i < accum_depth; i++) {
      shuffled_input_workspace_data[i] = input_data[i] ^ 0x80;
    }
  } else if (batches == 4) {
    for (int c = 0; c < accum_depth; c += 16) {
      for (int b = 0; b < 4; b++) {
        const uint8* src_data_ptr = input_data + b * accum_depth + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the kernel will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_input_workspace_ptr++ = dst_val;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }

  // Actual computation
  if (batches == 1) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4] = {0};
      // Accumulation loop.
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 16; j++) {
            int8 input_val = shuffled_input_data[d + j];
            int8 weights_val = *shuffled_weights_ptr++;
            accum[i] += weights_val * input_val;
          }
        }
      }
      for (int i = 0; i < 4; i++) {
        // Add bias value
        int acc = accum[i] + bias_data[c + i];
        // Down-scale the final int32 accumulator to the scale used by our
        // (16-bit, typically 3 integer bits) fixed-point format. The quantized
        // multiplier and shift here have been pre-computed offline
        // (e.g. by toco).
        acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                            -output_shift);
        // Saturate, cast to int16, and store to output array.
        acc = std::max(acc, output_activation_min);
        acc = std::min(acc, output_activation_max);
        output_ptr[c + i] = acc;
      }
    }
  } else if (batches == 4) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr = shuffled_input_data;
      // Accumulation loop.
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4][4];
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          accum[i][b] = 0;
        }
      }
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 16; j++) {
              int8 input_val = shuffled_input_ptr[16 * b + j];
              int8 weights_val = shuffled_weights_ptr[16 * i + j];
              accum[i][b] += weights_val * input_val;
            }
          }
        }
        shuffled_input_ptr += 64;
        shuffled_weights_ptr += 64;
      }
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          // Add bias value
          int acc = accum[i][b] + bias_data[c + i];
          // Down-scale the final int32 accumulator to the scale used by our
          // (16-bit, typically 3 integer bits) fixed-point format. The
          // quantized multiplier and shift here have been pre-computed offline
          // (e.g. by toco).
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              -output_shift);
          // Saturate, cast to int16, and store to output array.
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_ptr[b * output_depth + c + i] = acc;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims,
                    gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims, gemm_context);
}

template <FusedActivationFunctionType Ac>
void NonGlobalBatchNormalization(
    const float* input_data, const Dims<4>& input_dims, const float* mean_data,
    const Dims<4>& mean_dims, const float* multiplier_data,
    const Dims<4>& multiplier_dims, const float* offset_data,
    const Dims<4>& offset_dims, float* output_data,
    const Dims<4>& output_dims) {
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int inner_size = MatchingFlatSizeSkipDim(
      input_dims, 3, mean_dims, multiplier_dims, offset_dims, output_dims);

  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < inner_size; ++i) {
      output_data[b * inner_size + i] = ActivationFunction<Ac>(
          (input_data[b * inner_size + i] - mean_data[i]) * multiplier_data[i] +
          offset_data[i]);
    }
  }
}

template <FusedActivationFunctionType Ac>
void GlobalBatchNormalization(const float* input_data,
                              const Dims<4>& input_dims, const float* mean_data,
                              const Dims<4>& mean_dims,
                              const float* multiplier_data,
                              const Dims<4>& multiplier_dims,
                              const float* offset_data,
                              const Dims<4>& offset_dims, float* output_data,
                              const Dims<4>& output_dims) {
  const int outer_size = MatchingFlatSizeSkipDim(input_dims, 0, output_dims);
  const int depth =
      MatchingArraySize(input_dims, 0, mean_dims, 0, multiplier_dims, 0,
                        offset_dims, 0, output_dims, 0);

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      output_data[depth * i + c] = ActivationFunction<Ac>(
          (input_data[depth * i + c] - mean_data[c]) * multiplier_data[c] +
          offset_data[c]);
    }
  }
}

inline void Relu(const float* input_data, const RuntimeShape& input_shape,
                 float* output_data, const RuntimeShape& output_shape) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float lower = 0;
    const float clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu1(const float* input_data, const RuntimeShape& input_shape,
                  float* output_data, const RuntimeShape& output_shape) {
  gemmlowp::ScopedProfilingLabel label("Relu1 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 1;
    const float lower = -1;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu6(const float* input_data, const RuntimeShape& input_shape,
                  float* output_data, const RuntimeShape& output_shape) {
  gemmlowp::ScopedProfilingLabel label("Relu6 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6;
    const float lower = 0;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void ReluX(uint8 min_value, uint8 max_value, const uint8* input_data,
                  const RuntimeShape& input_shape, uint8* output_data,
                  const RuntimeShape& output_shape) {
  gemmlowp::ScopedProfilingLabel label("Quantized ReluX (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const uint8 val = input_data[i];
    const uint8 clamped =
        val > max_value ? max_value : val < min_value ? min_value : val;
    output_data[i] = clamped;
  }
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[depth * i + c];
      squared_l2_norm += val * val;
    }
    const float l2_norm = std::sqrt(squared_l2_norm);
    for (int c = 0; c < depth; ++c) {
      output_data[depth * i + c] = input_data[depth * i + c] / l2_norm;
    }
  }
}

inline void GetInvSqrtQuantizedMultiplierExp(int32 input,
                                             int32* output_inv_sqrt,
                                             int* output_shift) {
  *output_shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*output_shift;
  }
  TFLITE_DCHECK_GT(input, 0);
  const unsigned max_left_shift_bits = __builtin_clz(input) - 1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  TFLITE_DCHECK_GE(input, (1 << 27));
  TFLITE_DCHECK_LT(input, (1 << 29));
  using gemmlowp::FixedPoint;
  using gemmlowp::Rescale;
  using gemmlowp::SaturatingRoundingMultiplyByPOT;
  // Using 3 integer bits gives us enough room for the internal arithmetic in
  // this Newton-Raphson iteration.
  using F3 = FixedPoint<int32, 3>;
  using F0 = FixedPoint<int32, 0>;
  const F3 fixedpoint_input = F3::FromRaw(input >> 1);
  const F3 fixedpoint_half_input =
      SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
  const F3 fixedpoint_half_three =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
  // Newton-Raphson iteration
  // Naive unoptimized starting guess: x = 1
  F3 x = F3::One();
  // Naive unoptimized number of iterations: 5
  for (int i = 0; i < 5; i++) {
    const F3 x3 = Rescale<3>(x * x * x);
    x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
  }
  const F0 fixedpoint_half_sqrt_2 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
  x = x * fixedpoint_half_sqrt_2;
  *output_inv_sqrt = x.raw();
  if (*output_shift < 0) {
    *output_inv_sqrt <<= -*output_shift;
    *output_shift = 0;
  }
  *output_shift *= kReverseShift;
}

inline void L2Normalization(const uint8* input_data,
                            const RuntimeShape& input_shape,
                            int32 input_zero_point, uint8* output_data,
                            const RuntimeShape& output_shape) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  for (int i = 0; i < outer_size; ++i) {
    int32 square_l2_norm = 0;
    for (int c = 0; c < depth; c++) {
      int32 diff = input_data[depth * i + c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    int32 inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, &inv_l2norm_multiplier,
                                     &inv_l2norm_shift);

    for (int c = 0; c < depth; c++) {
      int32 diff = input_data[depth * i + c] - input_zero_point;
      int32 rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32 unclamped_output_val = 128 + rescaled_diff;
      int32 output_val = std::min(255, std::max(0, unclamped_output_val));
      output_data[depth * i + c] = static_cast<uint8>(output_val);
    }
  }
}

template <typename T>
inline void Add(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] + input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Add(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          const int32 input1_val =
              input1_offset + input1_data[Offset(input1_dims, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[Offset(input2_dims, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier,
                  kReverseShift * input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier,
                  kReverseShift * input2_shift);
          const int32 raw_sum = scaled_input1_val + scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sum, output_multiplier, kReverseShift * output_shift) +
              output_offset;
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, raw_output));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size = MatchingFlatSize(output_dims, input1_dims, input2_dims);

  TFLITE_DCHECK(input1_shift == 0 || input2_shift == 0);
  TFLITE_DCHECK_GE(input1_shift, 0);
  TFLITE_DCHECK_GE(input2_shift, 0);
  const int16* not_shift_input = input1_shift == 0 ? input1_data : input2_data;
  const int16* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_shift = input1_shift == 0 ? input2_shift : input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input =
        F0::FromRaw(gemmlowp::RoundingDivideByPOT(shift_input[i], input_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16 raw_output = result.raw();
    const int16 clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  Add(input1_data, input1_dims, input1_shift, input2_data, input2_dims,
      input2_shift, output_activation_min, output_activation_max, output_data,
      output_dims);
}

// TODO(jiawen): We can implement BroadcastAdd on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAdd");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] +
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAdd/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier,
                  kReverseShift * input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier,
                  kReverseShift * input2_shift);
          const int32 raw_sum = scaled_input1_val + scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sum, output_multiplier, kReverseShift * output_shift) +
              output_offset;
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, raw_output));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAddFivefold/8bit");

  int sb1 = y0;
  int sa2 = y0;
  int sb2 = y0 * y1;
  int sa3 = y0 * y2;
  int sa4 = y0 * y2 * y3;
  int sb4 = y0 * y1 * y2;

  uint8* output_data_ptr = output_data;
  for (int i4 = 0; i4 < y4; ++i4) {
    for (int i3 = 0; i3 < y3; ++i3) {
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i1 = 0; i1 < y1; ++i1) {
          for (int i0 = 0; i0 < y0; ++i0) {
            const int32 input1_val =
                input1_offset +
                input1_data[i4 * sa4 + i3 * sa3 + i2 * sa2 + i0];
            const int32 input2_val =
                input2_offset +
                input2_data[i4 * sb4 + i2 * sb2 + i1 * sb1 + i0];
            const int32 shifted_input1_val = input1_val * (1 << left_shift);
            const int32 shifted_input2_val = input2_val * (1 << left_shift);
            const int32 scaled_input1_val =
                MultiplyByQuantizedMultiplierSmallerThanOneExp(
                    shifted_input1_val, input1_multiplier,
                    kReverseShift * input1_shift);
            const int32 scaled_input2_val =
                MultiplyByQuantizedMultiplierSmallerThanOneExp(
                    shifted_input2_val, input2_multiplier,
                    kReverseShift * input2_shift);
            const int32 raw_sum = scaled_input1_val + scaled_input2_val;
            const int32 raw_output =
                MultiplyByQuantizedMultiplierSmallerThanOneExp(
                    raw_sum, output_multiplier, kReverseShift * output_shift) +
                output_offset;
            const int32 clamped_output =
                std::min(output_activation_max,
                         std::max(output_activation_min, raw_output));
            *output_data_ptr = static_cast<uint8>(clamped_output);
            ++output_data_ptr;
          }
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  BroadcastAdd(left_shift, input1_data, input1_dims, input1_offset,
               input1_multiplier, input1_shift, input2_data, input2_dims,
               input2_offset, input2_multiplier, input2_shift, output_offset,
               output_multiplier, output_shift, output_activation_min,
               output_activation_max, output_data, output_dims);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  BroadcastAddFivefold(y0, y1, y2, y3, y4, left_shift, input1_data, input1_dims,
                       input1_offset, input1_multiplier, input1_shift,
                       input2_data, input2_dims, input2_offset,
                       input2_multiplier, input2_shift, output_offset,
                       output_multiplier, output_shift, output_activation_min,
                       output_activation_max, output_data, output_dims);
}

template <typename T>
inline void Mul(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] * input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Mul(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

// TODO(jiawen): We can implement BroadcastMul on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMul");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] *
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastMul(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMul/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 unclamped_result =
              output_offset + MultiplyByQuantizedMultiplierSmallerThanOneExp(
                                  input1_val * input2_val, output_multiplier,
                                  kReverseShift * output_shift);
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, unclamped_result));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16");

  const int flat_size = MatchingFlatSize(output_dims, input1_dims, input2_dims);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    output_data[i] = unclamped_result.raw();
  }
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16Uint8");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size = MatchingFlatSize(output_dims, input1_dims, input2_dims);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16 rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16 clamped_result =
        std::min<int16>(output_activation_max - output_offset, rescaled_result);
    clamped_result =
        std::max<int16>(output_activation_min - output_offset, clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastDiv(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDiv");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] /
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

template <typename T>
inline void Div(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] / input2_data[i], output_activation_min,
        output_activation_max);
  }
}

template <typename T>
inline void Sub(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// TODO(jiawen): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastSub(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastSub");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] -
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

inline void BroadcastSub(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastSub/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier,
                  kReverseShift * input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier,
                  kReverseShift * input2_shift);
          const int32 raw_sub = scaled_input1_val - scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sub, output_multiplier, kReverseShift * output_shift) +
              output_offset;
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, raw_output));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac, typename Scalar>
void Concatenation(int concat_dim, const Scalar* const* input_data,
                   const Dims<4>* const* input_dims, int inputs_count,
                   Scalar* output_data, const Dims<4>& output_dims) {
  int concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    for (int j = 0; j < 4; j++) {
      if (j != concat_dim) {
        MatchingArraySize(*input_dims[i], j, output_dims, j);
      }
    }
    concat_size += ArraySize(*input_dims[i], concat_dim);
  }
  TFLITE_DCHECK_EQ(concat_size, ArraySize(output_dims, concat_dim));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  // For now we don't have a model with a Concatenation with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);
  int outer_size = 1;
  for (int i = concat_dim + 1; i < 4; i++) {
    outer_size *= output_dims.sizes[i];
  }
  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size =
          input_dims[i]->sizes[concat_dim] * input_dims[i]->strides[concat_dim];
      memcpy(output_ptr, input_data[i] + k * copy_size,
             copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

// TODO(prabhumk): This is the same as the optimized implementation.
// TODO(prabhumk): The quantized implementation of concatentation isn't fully
// quantized as it takes scale as a floating point value. This should be fixed
// when optimizng this routine further.
inline void Concatenation(int concat_dim, const uint8* const* input_data,
                          const Dims<4>* const* input_dims,
                          const int32* input_zeropoint,
                          const float* input_scale, int inputs_count,
                          uint8* output_data, const Dims<4>& output_dims,
                          const int32 output_zeropoint,
                          const float output_scale) {
  // The arguments input_zeropoint and input_scale are expected to be an array
  // that have the quantization parameters for all the inputs to the concat
  // operator.
  TFLITE_DCHECK_GT(inputs_count, 1);
  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    for (int j = 0; j < 4; j++) {
      if (j != concat_dim) {
        MatchingArraySize(*input_dims[i], j, output_dims, j);
      }
    }
    concat_size += ArraySize(*input_dims[i], concat_dim);
  }
  TFLITE_DCHECK_EQ(concat_size, ArraySize(output_dims, concat_dim));
  int64_t outer_size = 1;
  for (int i = concat_dim + 1; i < 4; i++) {
    outer_size *= output_dims.sizes[i];
  }
  const float inverse_output_scale = 1.f / output_scale;
  uint8* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size =
          input_dims[i]->sizes[concat_dim] * input_dims[i]->strides[concat_dim];
      const uint8* input_ptr = input_data[i] + k * copy_size;
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        memcpy(output_ptr, input_ptr, copy_size);
      } else {
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value =
              static_cast<int32_t>(round(input_ptr[j] * scale + bias)) +
              output_zeropoint;
          output_ptr[j] =
              static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

template <FusedActivationFunctionType Ac, typename Scalar>
void DepthConcatenation(const Scalar* const* input_data,
                        const Dims<4>* const* input_dims, int inputs_count,
                        Scalar* output_data, const Dims<4>& output_dims) {
  Concatenation<Ac, Scalar>(0, input_data, input_dims, inputs_count,
                            output_data, output_dims);
}

inline void LstmCell(const float* input_data, const Dims<4>& input_dims,
                     const float* prev_activ_data,
                     const Dims<4>& prev_activ_dims, const float* weights_data,
                     const Dims<4>& weights_dims, const float* bias_data,
                     const Dims<4>& bias_dims, const float* prev_state_data,
                     const Dims<4>& prev_state_dims, float* output_state_data,
                     const Dims<4>& output_state_dims, float* output_activ_data,
                     const Dims<4>& output_activ_dims, float* concat_temp_data,
                     const Dims<4>& concat_temp_dims, float* activ_temp_data,
                     const Dims<4>& activ_temp_dims) {
  const int batches =
      MatchingArraySize(input_dims, 3, prev_activ_dims, 3, prev_state_dims, 3,
                        output_state_dims, 3, output_activ_dims, 3);
  const int height =
      MatchingArraySize(input_dims, 2, prev_activ_dims, 2, prev_state_dims, 2,
                        output_state_dims, 2, output_activ_dims, 2);
  const int width =
      MatchingArraySize(input_dims, 1, prev_activ_dims, 1, prev_state_dims, 1,
                        output_state_dims, 1, output_activ_dims, 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 2), 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 3), 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int prev_activ_depth = ArraySize(prev_activ_dims, 0);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 0), total_input_depth);
  TFLITE_CHECK_EQ(MatchingArraySize(bias_dims, 1, bias_dims, 2, bias_dims, 3),
                  1);
  const int intern_activ_depth =
      MatchingArraySize(weights_dims, 1, bias_dims, 0);
  TFLITE_CHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingArraySize(prev_state_dims, 0, prev_activ_dims, 0,
                        output_state_dims, 0, output_activ_dims, 0);
  TFLITE_CHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<Dims<4> const*> concat_input_arrays_dims;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_dims.push_back(&input_dims);
  concat_input_arrays_dims.push_back(&prev_activ_dims);
  Concatenation<FusedActivationFunctionType::kNone, float>(
      0, &(concat_input_arrays_data[0]), &(concat_input_arrays_dims[0]),
      concat_input_arrays_data.size(), concat_temp_data, concat_temp_dims);

  // Fully connected
  FullyConnected<FusedActivationFunctionType::kNone>(
      concat_temp_data, concat_temp_dims, weights_data, weights_dims, bias_data,
      bias_dims, activ_temp_data, activ_temp_dims);

  // Memory state update (the LSTM "guts")
  for (int b = 0; b < batches; ++b) {
    for (int w = 0; w < width; ++w) {
      for (int h = 0; h < height; ++h) {
        for (int c = 0; c < output_depth; ++c) {
          const float input_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(
                         activ_temp_dims, 0 * output_depth + c, w, h, b)]));
          const float new_input = std::tanh(activ_temp_data[Offset(
              activ_temp_dims, 1 * output_depth + c, w, h, b)]);
          const float forget_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(
                         activ_temp_dims, 2 * output_depth + c, w, h, b)]));
          const float output_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(
                         activ_temp_dims, 3 * output_depth + c, w, h, b)]));
          const float new_state =
              input_gate * new_input +
              forget_gate *
                  prev_state_data[Offset(prev_state_dims, c, w, h, b)];
          output_state_data[Offset(output_state_dims, c, w, h, b)] = new_state;
          output_activ_data[Offset(output_activ_dims, c, w, h, b)] =
              output_gate * std::tanh(new_state);
        }
      }
    }
  }
}

// Quantized LSTM cell implementation.
// The quantization of the input, output arrays is as follows:
//  - The input activations are quantized as uint8 on the interval
//    [-1, 127/128].
//    The rationale for that is that is the natural interval for output
//    activations (see next point) and these need to be concatenated together.
//    We could accommodate different ranges by re-scaling, but we empirically
//    found that setting the input activations range to be [-1, 127/128] in the
//    first place, removing the need for re-scaling, greatly improves accuracy.
//  - The output activations are quantized as uint8 on the interval
//    [-1, 127/128].
//    The rationale for that is that the definition of a LSTM cell makes them
//    intrinsically constrained in [-1, 1]; tweaking that to [-1, 127/128]
//    makes for simpler, more accurate fixed-point arithmetic.
//  - The output-at-previous-timestep state array is obviously quantized as
//    the output activations.
//  - The internal LSTM memory (not the output-at-previous-timestep, the other
//    internal state array) is int16-quantized and may use any power-of-two,
//    symmetric range i.e. [-2^N, 2^N * 32767/32768] for any N, which we call
//    StateIntegerBits below, see the below discussion of that template
//    parameter ("The StateIntegerBits template parameter").
//  - The output of the internal fully-connected node is int16-quantized
//    on the interval [-8, 8 * 32767/32768], the rationale for which is
//    explained just below ("Why [-8, 8] for fully-connected output?").
//
//
// === The StateIntegerBits template parameter ===
//
// The StateIntegerBits template parameter controls the fixed-point format used
// to represent the internal memory of the LSTM cell (not the
// output-at-previous-timestep, the other internal state array). It's currently
// a template parameter so that the model can control that. The most typical
// value for StateIntegerBits is 4. Other plausible values are anywhere between
// 3 and 5. We might eventually standardize on a single supported value, e.g. 4,
// and drop that template parameter. The reason why it can't be a runtime
// parameter is that this controls the fixed-point format used, i.e. we need to
// generate actually different code based on it. In particular, we generate code
// for a fixed-point tanh() implementation for that format, which internally
// uses a fixed-point exp() implementation, which internally uses a
// barrel-shifter with a number of steps that depends on StateIntegerBits.
// Another consequence of that is that a higher value of StateIntegerBits
// results in a more expensive implementation (more barrel shifter steps
// needed).
//
//
// === Why [-8, 8] for fully-connected output? ===
//
// This array is only fed to Logistic and Tanh functions, for which
// the quantized implementation will want to use fixed-point arithmetic,
// requiring a power-of-two representation interval. Thus, we should right
// away quantize this array to a power-of-two interval; otherwise,
// implementation will need to rescale that, losing any benefit that a tighter
// representation interval might otherwise yield, while introducing some
// numerical error and computational overhead.
//
// Now, Logistic and Tanh
// are nearly constant (nearly equal to their horizontal asymptotes)
// outside of a small bounded interval around 0:
//
//   Logistic(4) = 1 - 1.8e-2     Tanh(4) = 1 - 6.7e-4
//   Logistic(8) = 1 - 3.4e-4     Tanh(8) = 1 - 2.3e-7
//   Logistic(16) = 1 - 1.1e-7    Tanh(16) = 1 - 2.5e-14
//
// From this, we see that clamping to [-4, 4] would be too inaccurate
// (the error of 1.8e-2 on Logistic would be felt even in 8bit precision)
// while clamping to [-16, 16] would make no difference even in float32.
// However, for a fixed-point implementation in 16-bit integers, using 5
// integer bits to represent the [-16, 16] range would leave only 11
// fractional bits, giving an increment of 2^-11 = 4.9e-4 between consecutive
// representable values. Notice that is higher than the
// worst-case clamping error with clamping to [-8, 8]: 3.4e-4 for Logistic.
// Using [-8, 8] thus seems like the better compromise overall, enjoying
// an increment of 2.4e-4 between representable values and a worst-case
// clamping error of 3.4e-4, both better than the increment of 4.9e-4 with
// [-16, 16].
//
// Moreover, all other things being equal, it is nice to choose the narrower
// representation range, as that makes the implementation of fixed-point
// math functions a little cheaper (each integer bit requires an additional
// barrel-shifter atep in the implementation of exp(-x)). That is further
// reason to prefer [-8, 8] over [-16, 16]. The choice of [-16, 16] would make
// sense for 32-bit float or 32-bit fixed-point quantization, but we are
// aiming for 16-bit fixed-point quantization of these internal nodes here.
//
template <int StateIntegerBits>
void LstmCell(const uint8* input_data_uint8, const Dims<4>& input_dims,
              const uint8* prev_activ_data_uint8,
              const Dims<4>& prev_activ_dims, const uint8* weights_data_uint8,
              const Dims<4>& weights_dims, const int32* bias_data_int32,
              const Dims<4>& bias_dims, const int16* prev_state_data_int16,
              const Dims<4>& prev_state_dims, int16* output_state_data_int16,
              const Dims<4>& output_state_dims, uint8* output_activ_data_uint8,
              const Dims<4>& output_activ_dims, uint8* concat_temp_data_uint8,
              const Dims<4>& concat_temp_dims, int16* activ_temp_data_int16,
              const Dims<4>& activ_temp_dims, int32 weights_zero_point,
              int32 accum_multiplier, int accum_shift,
              gemmlowp::GemmContext* gemm_context) {
  (void)gemm_context;  // only used in optimized code.

  // Gather dimensions information, and perform consistency checks.
  const int outer_size =
      MatchingFlatSizeSkipDim(input_dims, 0, prev_activ_dims, prev_state_dims,
                              output_state_dims, output_activ_dims);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 2), 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 3), 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int prev_activ_depth = ArraySize(prev_activ_dims, 0);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 0), total_input_depth);
  TFLITE_CHECK_EQ(MatchingArraySize(bias_dims, 1, bias_dims, 2, bias_dims, 3),
                  1);
  const int intern_activ_depth =
      MatchingArraySize(weights_dims, 1, bias_dims, 0);
  TFLITE_CHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingArraySize(prev_state_dims, 0, prev_activ_dims, 0,
                        output_state_dims, 0, output_activ_dims, 0);
  TFLITE_CHECK_EQ(output_depth, intern_activ_depth / 4);
  const int fc_batches = FlatSizeSkipDim(activ_temp_dims, 0);
  const int fc_output_depth =
      MatchingArraySize(weights_dims, 1, activ_temp_dims, 0);
  const int fc_accum_depth = ArraySize(weights_dims, 0);
  TFLITE_CHECK_EQ(fc_output_depth, 4 * output_depth);

  // Depth-concatenate prev_activ and input data together.
  uint8 const* concat_input_arrays_data[2] = {input_data_uint8,
                                              prev_activ_data_uint8};
  Dims<4> const* concat_input_arrays_dims[2] = {&input_dims, &prev_activ_dims};
  Concatenation<FusedActivationFunctionType::kNone, uint8>(
      0, concat_input_arrays_data, concat_input_arrays_dims, 2,
      concat_temp_data_uint8, concat_temp_dims);

  // Implementation of the fully connected node inside the LSTM cell.
  // The operands are 8-bit integers, the accumulators are internally 32bit
  // integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
  for (int b = 0; b < fc_batches; ++b) {
    for (int out_c = 0; out_c < fc_output_depth; ++out_c) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum = bias_data_int32[out_c];
      // Accumulation loop.
      for (int d = 0; d < fc_accum_depth; ++d) {
        int16 input_val = concat_temp_data_uint8[b * fc_accum_depth + d] - 128;
        int16 weights_val =
            weights_data_uint8[out_c * fc_accum_depth + d] - weights_zero_point;
        accum += input_val * weights_val;
      }
      // Down-scale the final int32 accumulator to the scale used by our
      // (16-bit, using 3 integer bits) fixed-point format. The quantized
      // multiplier and shift here have been pre-computed offline
      // (e.g. by toco).
      accum =
          MultiplyByQuantizedMultiplier(accum, accum_multiplier, accum_shift);
      // Saturate, cast to int16, and store to the temporary activations array.
      accum = std::max(-32768, std::min(32767, accum));
      activ_temp_data_int16[out_c + fc_output_depth * b] = accum;
    }
  }

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  for (int b = 0; b < outer_size; ++b) {
    for (int c = 0; c < output_depth; ++c) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<std::int16_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 0 * output_depth + c]);
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 1 * output_depth + c]);
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 2 * output_depth + c]);
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 3 * output_depth + c]);
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(prev_state_data_int16[b * output_depth + c]);
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      output_state_data_int16[b * output_depth + c] = new_state.raw();
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16 rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int16 clamped_output_activ =
          std::max<int16>(-128, std::min<int16>(127, rescaled_output_activ));
      output_activ_data_uint8[b * output_depth + c] =
          128 + clamped_output_activ;
    }
  }
}

template <typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int axis, int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  const int batches = ArraySize(*output_dims[0], 3);
  const int height = ArraySize(*output_dims[0], 2);
  const int width = ArraySize(*output_dims[0], 1);
  const int depth = ArraySize(*output_dims[0], 0);

  const int slice_size = ArraySize(*output_dims[0], axis);

  for (int i = 0; i < outputs_count; ++i) {
    int offset = i * slice_size * input_dims.strides[axis];
    for (int b = 0; b < batches; ++b) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          for (int c = 0; c < depth; ++c) {
            auto out = Offset(*output_dims[i], c, x, y, b);
            auto in = Offset(input_dims, c, x, y, b);
            output_data[i][out] = input_data[offset + in];
          }
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac, typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  TFLITE_DCHECK_GE(outputs_count, 1);
  for (int i = 0; i < outputs_count; i++) {
    /* batches = */ MatchingArraySize(*output_dims[i], 3, input_dims, 3);
    /* height = */ MatchingArraySize(*output_dims[i], 2, input_dims, 2);
    /* width = */ MatchingArraySize(*output_dims[i], 1, input_dims, 1);
  }
  // for now we dont have a model with a TensorFlowSplit
  // with fused activation function.
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  TensorFlowSplit(input_data, input_dims, /*axis=*/0, outputs_count,
                  output_data, output_dims);
}

// TODO(benoitjacob) make this a proper reference impl without Eigen!
template <typename Scalar>
using MatrixMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar* data,
                                                const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar* data,
                                               const Dims<N>& dims) {
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++) {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float total = 0.f;
          float filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              total +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          const float average = total / filter_count;
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(average, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8* input_data,
                        const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int32 acc = 0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          acc = (acc + filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<uint8>(acc);
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float sum_squares = 0.f;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              const float val =
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              sum_squares += val * val;
              filter_count++;
            }
          }
          const float l2pool_result = std::sqrt(sum_squares / filter_count);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(l2pool_result,
                                           params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(max, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8* input_data, const RuntimeShape& output_shape,
                    uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, 0);
  TFLITE_DCHECK_LE(params.quantized_activation_max, 255);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          uint8 max = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          max = std::max<uint8>(max, params.quantized_activation_min);
          max = std::min<uint8>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<uint8>(max);
        }
      }
    }
  }
}

inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
  const int outer_size = MatchingFlatSizeSkipDim(input_dims, 0, output_dims);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      const int begin_input_c = std::max(0, c - range);
      const int end_input_c = std::min(depth, c + range);
      float accum = 0.f;
      for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
        const float input_val = input_data[i * depth + input_c];
        accum += input_val * input_val;
      }
      const float multiplier = std::pow(bias + alpha * accum, -beta);
      output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
    }
  }
}

inline void Softmax(const float* input_data, const RuntimeShape& input_shape,
                    float beta, float* output_data,
                    const RuntimeShape& output_shape) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp((input_data[i * depth + c] - max) * beta);
    }

    // Compute result.
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] =
          std::exp((input_data[i * depth + c] - max) * beta) / sum;
    }
  }
}

inline void Softmax(const uint8* input_data, const RuntimeShape& input_shape,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const RuntimeShape& output_shape) {
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    uint8 max_in_row = 0;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int32 fixed_sum_of_exps = sum_of_exps.raw();
    int headroom_plus_one =
        CountLeadingZeros(static_cast<uint32>(fixed_sum_of_exps));
    // This is the number of bits to the left of the binary point above 1.0.
    // Consider fixed_sum_of_exps=1.25.  In that case shifted_scale=0.8 and
    // no later adjustment will be needed.
    int num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one;
    int32 shifted_sum_minus_one = static_cast<int32>(
        (static_cast<uint32>(fixed_sum_of_exps) << headroom_plus_one) -
        (static_cast<uint32>(1) << 31));

    FixedPoint0 shifted_scale = gemmlowp::one_over_one_plus_x_for_x_in_0_1(
        FixedPoint0::FromRaw(shifted_sum_minus_one));

    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        int32 unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

        output_data[i * depth + c] = static_cast<uint8>(
            std::max(std::min(unsat_output, static_cast<int32>(255)), 0));

      } else {
        output_data[i * depth + c] = 0;
      }
    }
  }
}

inline void LogSoftmax(const float* input_data, const RuntimeShape& input_shape,
                       float* output_data, const RuntimeShape& output_shape) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp(input_data[i * depth + c] - max);
    }

    // Compute result.
    const float log_sum = std::log(sum);
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = input_data[i * depth + c] - max - log_sum;
    }
  }
}

// Although currently the name of this function says that it cannot handle
// values less than 1, in practice it can handle as low as 1/x_max, where
// x_max is the largest representable input.  In other words, the output range
// is symmetric.
template <int OutputIntegerBits, int InputIntegerBits>
inline gemmlowp::FixedPoint<int32, OutputIntegerBits>
log_x_for_x_greater_than_or_equal_to_1_impl(
    gemmlowp::FixedPoint<int32, InputIntegerBits> input_val) {
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
  // The reason for accumulating the result with an extra bit of headroom is
  // that z_pow_2_adj * log_2 might be saturated, and adding num_scaled *
  // recip_denom will otherwise introduce an error.
  static constexpr int kAccumIntegerBits = OutputIntegerBits + 1;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumIntegerBits>;

  const FixedPoint0 log_2 = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 1488522236, std::log(2.0));
  const FixedPoint0 sqrt_sqrt_half = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 1805811301, std::sqrt(std::sqrt(0.5)));
  const FixedPoint0 sqrt_half = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 1518500250, std::sqrt(0.5));
  const FixedPoint0 one_quarter =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPoint0, 536870912, 1.0 / 4.0);

  const FixedPoint0 alpha_n = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 117049297, 11.0 / 240.0 * std::sqrt(std::sqrt(2.0)));
  const FixedPoint0 alpha_d = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 127690142, 1.0 / 20.0 * std::sqrt(std::sqrt(2.0)));
  const FixedPoint0 alpha_i = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 1057819769,
      2.0 / std::sqrt(std::sqrt(2.0)) - std::sqrt(std::sqrt(2.0)));
  const FixedPoint0 alpha_f = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(
      FixedPoint0, 638450708, 1.0 / 4.0 * std::sqrt(std::sqrt(2.0)));

  const FixedPointAccum shifted_quarter =
      gemmlowp::Rescale<kAccumIntegerBits>(one_quarter);

  // Reinterpret the input value as Q0.31, because we will figure out the
  // required shift "ourselves" instead of using, say, Rescale.
  FixedPoint0 z_a = FixedPoint0::FromRaw(input_val.raw());
  // z_a_pow_2 = input_integer_bits - z_a_headroom;
  int z_a_headroom_plus_1 = CountLeadingZeros(static_cast<uint32>(z_a.raw()));
  FixedPoint0 r_a_tmp =
      SaturatingRoundingMultiplyByPOTParam(z_a, (z_a_headroom_plus_1 - 1));
  const int32 r_a_raw =
      SaturatingRoundingMultiplyByPOTParam((r_a_tmp * sqrt_half).raw(), 1);
  // z_pow_2_adj = max(z_pow_2_a - 0.75, z_pow_2_b - 0.25);
  // z_pow_2_adj = max(InputIntegerBits - z_a_headroom_plus_1 + 0.25,
  //                   InputIntegerBits - z_b_headroom - 0.25);
  const FixedPointAccum z_a_pow_2_adj = SaturatingAddNonGemmlowp(
      FixedPointAccum::FromRaw(SaturatingRoundingMultiplyByPOTParam(
          InputIntegerBits - z_a_headroom_plus_1, 31 - kAccumIntegerBits)),
      shifted_quarter);

  // z_b is treated like z_a, but premultiplying by sqrt(0.5).
  FixedPoint0 z_b = z_a * sqrt_half;
  int z_b_headroom = CountLeadingZeros(static_cast<uint32>(z_b.raw())) - 1;
  const int32 r_b_raw =
      SaturatingRoundingMultiplyByPOTParam(z_a.raw(), z_b_headroom);
  const FixedPointAccum z_b_pow_2_adj = SaturatingSub(
      FixedPointAccum::FromRaw(SaturatingRoundingMultiplyByPOTParam(
          InputIntegerBits - z_b_headroom, 31 - kAccumIntegerBits)),
      shifted_quarter);

  const FixedPoint0 r = FixedPoint0::FromRaw(std::min(r_a_raw, r_b_raw));
  const FixedPointAccum z_pow_2_adj = FixedPointAccum::FromRaw(
      std::max(z_a_pow_2_adj.raw(), z_b_pow_2_adj.raw()));

  const FixedPoint0 p = gemmlowp::RoundingHalfSum(r, sqrt_sqrt_half);
  FixedPoint0 q = r - sqrt_sqrt_half;
  q = q + q;

  const FixedPoint0 common_sq = q * q;
  const FixedPoint0 num = q * r + q * common_sq * alpha_n;
  const FixedPoint0 denom_minus_one_0 =
      p * (alpha_i + q + alpha_d * common_sq) + alpha_f * q;
  const FixedPoint0 recip_denom =
      one_over_one_plus_x_for_x_in_0_1(denom_minus_one_0);

  const FixedPointAccum num_scaled = gemmlowp::Rescale<kAccumIntegerBits>(num);
  return gemmlowp::Rescale<OutputIntegerBits>(z_pow_2_adj * log_2 +
                                              num_scaled * recip_denom);
}

// Minimum output bits to accommodate log of maximum input range.  It actually
// does not matter if one considers, say, [-64,64] or [-64,64).
//
// For example, run this through Octave:
// [0:127; ...
//  ceil(log(abs( log(2.^(0:127))+1 ))/log(2)); ...
//  ceil(log(abs( log(2.^(0:127))+1 ))/log(2))]
constexpr int min_log_x_output_bits(int input_bits) {
  return input_bits > 90
             ? 7
             : input_bits > 44
                   ? 6
                   : input_bits > 21
                         ? 5
                         : input_bits > 10
                               ? 4
                               : input_bits > 4 ? 3 : input_bits > 1 ? 2 : 1;
}

template <int OutputIntegerBits, int InputIntegerBits>
inline gemmlowp::FixedPoint<int32, OutputIntegerBits>
log_x_for_x_greater_than_or_equal_to_1(
    gemmlowp::FixedPoint<int32, InputIntegerBits> input_val) {
  static_assert(
      OutputIntegerBits >= min_log_x_output_bits(InputIntegerBits),
      "Output integer bits must be sufficent to accommodate logs of inputs.");
  return log_x_for_x_greater_than_or_equal_to_1_impl<OutputIntegerBits,
                                                     InputIntegerBits>(
      input_val);
}

inline void LogSoftmax(const uint8* input_data, const RuntimeShape& input_shape,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const RuntimeShape& output_shape) {
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    uint8 max_in_row = 0;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    const int32 fixed_log_sum_of_exps =
        log_x_for_x_greater_than_or_equal_to_1<kScaledDiffIntegerBits>(
            sum_of_exps)
            .raw();

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32>::lowest();
    const int adjusted_diff_min =
        std::max(diff_min - 1,  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOneExp(
                     rescaled_diff_min, reverse_scaling_divisor,
                     kReverseShift * reverse_scaling_right_shift));

    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff > adjusted_diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        int32 unsat_output =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_rescaled - fixed_log_sum_of_exps),
                31 - kScaledDiffIntegerBits - kOutputIntegerBits) +
            255;

        output_data[i * depth + c] = static_cast<uint8>(
            std::max(std::min(unsat_output, static_cast<int32>(255)), 0));
      } else {
        // Set output to smallest value.
        output_data[i * depth + c] = 0;
      }
    }
  }
}

inline void Logistic(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    float val = input_data[i];
    float result = 1.f / (1.f + std::exp(-val));
    output_data[i] = result;
  }
}

inline void Logistic(const uint8* input_data, const RuntimeShape& input_shape,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const RuntimeShape& output_shape) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const uint8 input_val_u8 = input_data[i];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      // Convert from Q0.31 to Q23.8.
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 23);
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      // Reinterpret as U0.8.
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[i] = output_val;
  }
}

inline void Logistic(const int16* input_data, const RuntimeShape& input_shape,
                     int16* output_data, const RuntimeShape& output_shape) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    const F3 input = F3::FromRaw(input_data[i]);
    F0 output = gemmlowp::logistic(input);
    output_data[i] = output.raw();
  }
}

inline void Tanh(const float* input_data, const RuntimeShape& input_shape,
                 float* output_data, const RuntimeShape& output_shape) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    float val = input_data[i];
    float result = std::tanh(val);
    output_data[i] = result;
  }
}

inline void Tanh(const uint8* input_data, const RuntimeShape& input_shape,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const RuntimeShape& output_shape) {
  const int32 output_zero_point = 128;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const uint8 input_val_u8 = input_data[i];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      // Convert from Q0.31 to Q24.7.
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 24);
      output_val_s32 += output_zero_point;
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      // Reinterpret as Q0.7, encoded in uint8.
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[i] = output_val;
  }
}

inline void Tanh(const int16* input_data, const RuntimeShape& input_shape,
                 int input_left_shift, int16* output_data,
                 const RuntimeShape& output_shape) {
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // F0 uses 0 integer bits, range [-1, 1].
  // This is the return type of math functions such as tanh, logistic,
  // whose range is in [-1, 1].
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
  using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

  if (input_left_shift == 0) {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(input_data[i]);
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  } else {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(
          gemmlowp::SaturatingRoundingMultiplyByPOT<1>(input_data[i]));
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  }
}

inline void Dequantize(const uint8* input_data, const Dims<4>& input_dims,
                       int32 zero_point, double scale, float* output_data,
                       const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(output_dims, input_dims);

  for (int i = 0; i < flat_size; i++) {
    int32 val = input_data[i];
    float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void FakeQuant(const float* input_data, const Dims<4>& input_dims,
                      float rmin, float rmax, int num_bits, float* output_data,
                      const Dims<4>& output_dims) {
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(rmin, 0.0f);
  TFLITE_DCHECK_GE(rmax, 0.0f);
  TFLITE_DCHECK_LT(rmin, rmax);

  // Code matches tensorflow's FakeQuantWithMinMaxArgsFunctor.
  int quant_min = 0;
  int quant_max = (1 << num_bits) - 1;
  float nudged_min, nudged_max, nudged_scale;
  NudgeQuantizationRange(rmin, rmax, quant_min, quant_max, &nudged_min,
                         &nudged_max, &nudged_scale);
  const float inv_nudged_scale = 1.0f / nudged_scale;

  const int flat_size = MatchingFlatSize(output_dims, input_dims);
  for (int i = 0; i < flat_size; i++) {
    const float src_val = input_data[i];
    const float clamped = std::min(nudged_max, std::max(nudged_min, src_val));
    const float clamped_shifted = clamped - nudged_min;
    const float dst_val =
        TfLiteRound(clamped_shifted * inv_nudged_scale) * nudged_scale +
        nudged_min;
    output_data[i] = dst_val;
  }
}

template <typename SrcT, typename DstT>
inline void Cast(const SrcT* input_data, const Dims<4>& input_dims,
                 DstT* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(output_dims, input_dims);

  for (int i = 0; i < flat_size; i++) {
    int offset = i;
    output_data[offset] = static_cast<DstT>(input_data[offset]);
  }
}

inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(output_dims, input_dims);

  for (int i = 0; i < flat_size; i++) {
    int offset = i;
    output_data[offset] = std::floor(input_data[offset]);
  }
}

template <typename T>
inline void Gather(const T* input_data, const Dims<4>& input_dims,
                   int input_rank, const int32* coords_data,
                   const Dims<4>& coords_dims, T* output_data,
                   const Dims<4>& output_dims) {
  TFLITE_DCHECK(coords_dims.sizes[0] == output_dims.sizes[input_rank - 1]);
  int stride = input_dims.strides[input_rank - 1];
  T* out = output_data;

  for (int i = 0; i < coords_dims.sizes[0]; i++) {
    TFLITE_DCHECK_GE(coords_data[i], 0);
    TFLITE_DCHECK_LT(coords_data[i], input_dims.sizes[input_rank - 1]);
    const T* in = input_data + coords_data[i] * stride;
    memcpy(out, in, sizeof(T) * stride);
    out += stride;
  }
}

template <typename T>
inline void ResizeBilinear(const T* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, T* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  int32 batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  int32 input_height = ArraySize(input_dims, 2);
  int32 input_width = ArraySize(input_dims, 1);
  int32 depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 3), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 2), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 1), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 0), 2);
  int32 output_height = output_size_data[Offset(output_size_dims, 0, 0, 0, 0)];
  int32 output_width = output_size_data[Offset(output_size_dims, 1, 0, 0, 0)];
  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(std::floor(input_x));
        int32 x1 = std::min(x0 + 1, input_width - 1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_dims, c, x0, y0, b)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_dims, c, x0, y1, b)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_dims, c, x1, y0, b)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_dims, c, x1, y1, b)] *
                                 (input_y - y0) * (input_x - x0));
          output_data[Offset(output_dims, c, x, y, b)] = interpolation;
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear<float>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear<uint8>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims) {
  const int output_batch_size = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int input_batch_size = ArraySize(input_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  const int block_shape_height = block_shape_data[0];
  const int block_shape_width = block_shape_data[1];
  const int padding_top = paddings_data[0];
  const int padding_left = paddings_data[2];

  for (int out_b = 0; out_b < output_batch_size; ++out_b) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        T* out = output_data + Offset(output_dims, 0, out_w, out_h, out_b);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >=
                padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          memset(out, 0, depth * sizeof(T));
        } else {
          const T* in =
              input_data +
              Offset(input_dims, 0,
                     (out_w * block_shape_width + shift_w) - padding_left,
                     (out_h * block_shape_height + shift_h) - padding_top,
                     input_batch);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
}

template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
  const int output_batch_size = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int input_batch_size = ArraySize(input_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];
  const int crops_top = crops_data[0];
  const int crops_left = crops_data[2];

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      if (out_h < 0 || out_h >= output_height) {
        continue;
      }
      for (int in_w = 0; in_w < input_width; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;

        if (out_w < 0 || out_w >= output_width) {
          continue;
        }
        T* out = output_data + Offset(output_dims, 0, out_w, out_h, out_batch);
        const T* in = input_data + Offset(input_dims, 0, in_w, in_h, in_batch);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);

  const int output_batch = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int output_depth = ArraySize(output_dims, 0);

  const int left_b_padding = left_paddings[3];
  const int left_h_padding = left_paddings[2];
  const int left_w_padding = left_paddings[1];
  const int left_d_padding = left_paddings[0];

  const int right_b_padding = right_paddings[3];
  const int right_h_padding = right_paddings[2];
  const int right_w_padding = right_paddings[1];
  const int right_d_padding = right_paddings[0];

  const T* in_ptr = input_data;
  T* out_ptr = output_data;
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          if (out_b < left_b_padding ||
              out_b >= output_batch - right_b_padding ||
              out_h < left_h_padding ||
              out_h >= output_height - right_h_padding ||
              out_w < left_w_padding ||
              out_w >= output_width - right_w_padding ||
              out_d < left_d_padding ||
              out_d >= output_depth - right_d_padding) {
            *out_ptr++ = pad_value;
          } else {
            *out_ptr++ = *in_ptr++;
          }
        }
      }
    }
  }
}

// Legacy Pad() method that casts an int32_t to T before padding.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  Pad(input_data, input_dims, left_paddings, right_paddings, output_data,
      output_dims, 0);
}

template <typename T>
inline void StridedSlice(const T* input_data, const Dims<4>& input_dims,
                         int begin_mask, int end_mask, int shrink_axis_mask,
                         const std::vector<int>& start_indices,
                         const std::vector<int>& stop_indices,
                         const std::vector<int>& strides, T* output_data,
                         const Dims<4>& output_dims) {
  // Note that the axis orders are reversed for runtime ops, so the indices,
  // strides and masks must be as well too.
  TFLITE_DCHECK_EQ(start_indices.size(), 4);
  TFLITE_DCHECK_EQ(stop_indices.size(), 4);
  TFLITE_DCHECK_EQ(strides.size(), 4);
  const int start_b = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 3);
  const int stop_b =
      strided_slice::StopForAxis(end_mask, shrink_axis_mask, stop_indices,
                                 strides, input_dims.sizes, 3, start_b);
  const int start_h = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 2);
  const int stop_h =
      strided_slice::StopForAxis(end_mask, shrink_axis_mask, stop_indices,
                                 strides, input_dims.sizes, 2, start_h);
  const int start_w = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 1);
  const int stop_w =
      strided_slice::StopForAxis(end_mask, shrink_axis_mask, stop_indices,
                                 strides, input_dims.sizes, 1, start_w);
  const int start_d = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 0);
  const int stop_d =
      strided_slice::StopForAxis(end_mask, shrink_axis_mask, stop_indices,
                                 strides, input_dims.sizes, 0, start_d);

  T* out_ptr = output_data;
  for (int in_b = start_b;
       !strided_slice::LoopCondition(in_b, stop_b, strides[3]);
       in_b += strides[3]) {
    for (int in_h = start_h;
         !strided_slice::LoopCondition(in_h, stop_h, strides[2]);
         in_h += strides[2]) {
      for (int in_w = start_w;
           !strided_slice::LoopCondition(in_w, stop_w, strides[1]);
           in_w += strides[1]) {
        for (int in_d = start_d;
             !strided_slice::LoopCondition(in_d, stop_d, strides[0]);
             in_d += strides[0]) {
          *out_ptr++ = input_data[Offset(input_dims, in_d, in_w, in_h, in_b)];
        }
      }
    }
  }
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
  // TODO(dkalenichenko): This op only supports 4D tensors.
  TFLITE_DCHECK_EQ(begin.size(), 4);
  TFLITE_DCHECK_EQ(size.size(), 4);
  const int start_b = begin[3];
  const int stop_b =
      size[3] == -1 ? input_dims.sizes[3] - start_b : start_b + size[3];
  const int start_h = begin[2];
  const int stop_h =
      size[2] == -1 ? input_dims.sizes[2] - start_h : start_h + size[2];
  const int start_w = begin[1];
  const int stop_w =
      size[1] == -1 ? input_dims.sizes[1] - start_w : start_w + size[1];
  const int start_d = begin[0];
  const int stop_d =
      size[0] == -1 ? input_dims.sizes[0] - start_d : start_d + size[0];

  T* out_ptr = output_data;
  for (int in_b = start_b; in_b < stop_b; ++in_b) {
    for (int in_h = start_h; in_h < stop_h; ++in_h) {
      for (int in_w = start_w; in_w < stop_w; ++in_w) {
        for (int in_d = start_d; in_d < stop_d; ++in_d) {
          *out_ptr++ = input_data[Offset(input_dims, in_d, in_w, in_h, in_b)];
        }
      }
    }
  }
}

template <typename T>
inline void Exp(const T* input_data, const size_t num_elements,
                T* output_data) {
  for (size_t idx = 0; idx < num_elements; ++idx) {
    output_data[idx] = exp(input_data[idx]);
  }
}

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   Out reducer(const Out current, const In in),
                   Out* output_data) {
  // Reset input iterator.
  TFLITE_DCHECK(input_num_dims > 0);
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                               input_iter, num_axis, axis);
    output_data[output_offset] =
        reducer(output_data[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index.
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
  auto reducer = [](const Out current, const In in) -> Out {
    const Out actual_in = static_cast<Out>(in);
    return current + actual_in;
  };
  return Reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims,
                         output_num_dims, axis, num_axis, input_iter, reducer,
                         output_data);
}

template <typename T>
inline bool InitTensorDataForReduce(const int* dims, const int num_dims,
                                    const T init_value, T* data) {
  size_t num_elements = 1;
  for (int idx = 0; idx < num_dims; ++idx) {
    size_t current = static_cast<size_t>(dims[idx]);
    // Overflow prevention.
    if (num_elements > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_elements *= current;
  }
  for (size_t idx = 0; idx < num_elements; ++idx) {
    data[idx] = init_value;
  }
  return true;
}

// Computes the sum of elements across dimensions given in axis.
template <typename T>
inline bool Sum(const T* input_data, const int* input_dims,
                const int input_num_dims, T* output_data,
                const int* output_dims, const int output_num_dims,
                const int* axis, const int num_axis_dimensions, bool keep_dims,
                int* temp_index, int* resolved_axis) {
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, static_cast<T>(0),
                               output_data)) {
    return false;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  return ReduceSumImpl<T, T>(input_data, input_dims, output_dims,
                             input_num_dims, output_num_dims, resolved_axis,
                             num_resolved_axis, temp_index, output_data);
}

// Computes the max of elements across dimensions given in axis.
template <typename T>
inline bool ReduceMax(const T* input_data, const int* input_dims,
                      const int input_num_dims, T* output_data,
                      const int* output_dims, const int output_num_dims,
                      const int* axis, const int64_t num_axis_dimensions,
                      bool keep_dims, int* temp_index, int* resolved_axis) {
  T init_value = std::numeric_limits<T>::lowest();
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, init_value,
                               output_data)) {
    return false;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  auto reducer = [](const T current, const T in) -> T {
    return (in > current) ? in : current;
  };
  return Reduce<T, T>(input_data, input_dims, output_dims, input_num_dims,
                      output_num_dims, resolved_axis, num_resolved_axis,
                      temp_index, reducer, output_data);
}

// Computes the prod of elements across dimensions given in axis.
template <typename T>
inline bool ReduceProd(const T* input_data, const int* input_dims,
                       const int input_num_dims, T* output_data,
                       const int* output_dims, const int output_num_dims,
                       const int* axis, const int64_t num_axis_dimensions,
                       bool keep_dims, int* temp_index, int* resolved_axis) {
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, static_cast<T>(1),
                               output_data)) {
    return false;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  auto reducer = [](const T current, const T in) -> T { return in * current; };
  return Reduce<T, T>(input_data, input_dims, output_dims, input_num_dims,
                      output_num_dims, resolved_axis, num_resolved_axis,
                      temp_index, reducer, output_data);
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis.
template <typename T, typename U>
inline bool Mean(const T* input_data, const int* input_dims,
                 const int input_num_dims, T* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, bool keep_dims,
                 int* temp_index, int* resolved_axis, U* temp_sum) {
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  U num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<U>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      output_data[idx] =
          static_cast<T>(temp_sum[idx] / static_cast<U>(num_elements_in_axis));
    }
  }
  return true;
}

template <typename T>
inline void Mean(const T* input_data, const Dims<4>& input_dims,
                 const std::vector<int>& reduction_indices, T* output_data,
                 const Dims<4>& output_dims) {
  const int output_batch = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int output_depth = ArraySize(output_dims, 0);

  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);

  // The current implementation only supports simultaneous reduction over
  // width and height.
  TFLITE_DCHECK_EQ(reduction_indices.size(), 2);
  TFLITE_DCHECK((reduction_indices[0] == 1 && reduction_indices[1] == 2) ||
                (reduction_indices[0] == 2 && reduction_indices[1] == 1));
  TFLITE_DCHECK_EQ(output_height, 1);
  TFLITE_DCHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_dims, out_d, in_w, in_h, out_b)];
        }
      }
      output_data[Offset(output_dims, out_d, 0, 0, out_b)] =
          value / (input_width * input_height);
    }
  }
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              input1_data[SubscriptToIndex(desc1, c, x, y, b)] -
              input2_data[SubscriptToIndex(desc2, c, x, y, b)];
        }
      }
    }
  }
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(output_dims, input1_dims);

  auto min_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] > min_value ? min_value : input1_data[i];
  }
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(output_dims, input1_dims);

  auto max_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] < max_value ? max_value : input1_data[i];
  }
}

template <typename T, typename Op>
void TensorFlowMaximumMinimum(const T* input1_data, const Dims<4>& input1_dims,
                              const T* input2_data, const Dims<4>& input2_dims,
                              T* output_data, const Dims<4>& output_dims,
                              Op op) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          auto out_idx = Offset(output_dims, c, x, y, b);
          auto in1_idx = SubscriptToIndex(desc1, c, x, y, b);
          auto in2_idx = SubscriptToIndex(desc2, c, x, y, b);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = op(in1_val, in2_val);
        }
      }
    }
  }
}

template <typename T1, typename T2, typename T3, typename Cmp>
void ArgMinMax(const T3* axis, const T1* input_data, const Dims<4>& input_dims,
               T2* output_data, const Dims<4>& output_dims, const Cmp& cmp) {
  // The current ArgMax implemention can only determine the index of the maximum
  // value in the last dimension. So the axis argument is ignored.

  // For ArgMax, the number of output dimensions = (number of input dimensions -
  // 1). For the sake of simplicity, the output dimensions are equal to the
  // input dimensions here. We enforce the constraint that the last dimension
  // must always be 1.
  TFLITE_DCHECK_EQ(ArraySize(output_dims, 0), 1);
  const int outer_size = MatchingFlatSizeSkipDim(input_dims, 0, output_dims);
  const int depth = ArraySize(input_dims, 0);

  for (int i = 0; i < outer_size; ++i) {
    auto min_max_value = input_data[i * depth];
    int min_max_index = 0;
    for (int d = 1; d < depth; ++d) {
      const auto& curr_value = input_data[i * depth + d];
      if (cmp(curr_value, min_max_value)) {
        min_max_value = curr_value;
        min_max_index = d;
      }
    }
    output_data[i] = min_max_index;
  }
}

// TODO(renjieliu): Remove this one.
template <typename T1, typename T2, typename T3>
void ArgMax(const T3* axis, const T1* input_data,
            const tflite::Dims<4>& input_dims, T2* output_data,
            const tflite::Dims<4>& output_dims) {
  ArgMinMax(axis, input_data, input_dims, output_data, output_dims,
            std::greater<T1>());
}

template <typename T>
void Transpose(const T* input, const Dims<4>& input_dims, T* output,
               const Dims<4>& output_dims, const int* permuted_axes) {
  int out_sizes[4];
  // Compute the inverse permutation array so we can do an output centered
  // transpose. Also, check to make sure output_dims is matching input_dims.
  for (int k = 0; k < 4; k++) {
    out_sizes[k] =
        MatchingArraySize(input_dims, permuted_axes[k], output_dims, k);
  }

  // Naive transpose loop (iterate on output index and compute input index).
  int o[4];  // loop index (on output).
  int i[4];
  for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
    i[permuted_axes[3]] = o[3];
    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
      i[permuted_axes[2]] = o[2];
      for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
        i[permuted_axes[1]] = o[1];
        for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
          i[permuted_axes[0]] = o[0];
          output[Offset(output_dims, o)] = input[Offset(input_dims, i)];
        }
      }
    }
  }
}

inline void TransposeConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, float* output_data,
                          const Dims<4>& output_dims, float* /*im2col_data*/,
                          const Dims<4>& /*im2col_dims*/) {
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  // Although transpose convolution simplifies to convolution with transposed
  // weights for strides of 1, non-unitary striding complicates matters. To
  // keep this reference implementation as clear as possible, we use a
  // "scatter" access pattern, where we loop through all the input elements,
  // computing their influence on the output, rather than looping through the
  // output elements in the typical "gather" access pattern of a conv. We
  // therefore must initialize the output array to zero.
  const int num_elements = FlatSize(output_dims);
  for (int i = 0; i < num_elements; i++) {
    output_data[i] = 0.0f;
  }

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  output_data[Offset(output_dims, out_channel, out_x, out_y,
                                     batch)] += input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
inline bool EqualFn(T lhs, T rhs) {
  return lhs == rhs;
}

template <typename T>
inline bool NotEqualFn(T lhs, T rhs) {
  return lhs != rhs;
}

template <typename T>
inline bool GreaterFn(T lhs, T rhs) {
  return lhs > rhs;
}
template <typename T>
inline bool GreaterEqualFn(T lhs, T rhs) {
  return lhs >= rhs;
}
template <typename T>
inline bool LessFn(T lhs, T rhs) {
  return lhs < rhs;
}
template <typename T>
inline bool LessEqualFn(T lhs, T rhs) {
  return lhs <= rhs;
}

template <typename T>
using ComparisonFn = bool (*)(T, T);

template <typename T, ComparisonFn<T> F>
inline void Comparison(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, const Dims<4>& input2_dims,
                       bool* output_data, const Dims<4>& output_dims) {
  const int64_t flatsize =
      MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] = F(input1_data[i], input2_data[i]);
  }
}

template <typename T, ComparisonFn<int32> F>
inline void Comparison(int left_shift, const T* input1_data,
                       const Dims<4>& input1_dims, int32 input1_offset,
                       int32 input1_multiplier, int input1_shift,
                       const T* input2_data, const Dims<4>& input2_dims,
                       int32 input2_offset, int32 input2_multiplier,
                       int input2_shift, bool* output_data,
                       const Dims<4>& output_dims) {
  const int64_t flatsize =
      MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int64_t i = 0; i < flatsize; ++i) {
    const int32 input1_val = input1_offset + input1_data[i];
    const int32 input2_val = input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << left_shift);
    const int32 shifted_input2_val = input2_val * (1 << left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, input1_multiplier,
            kReverseShift * input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, input2_multiplier,
            kReverseShift * input2_shift);
    output_data[i] = F(scaled_input1_val, scaled_input2_val);
  }
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison(const T* input1_data,
                                const Dims<4>& input1_dims,
                                const T* input2_data,
                                const Dims<4>& input2_dims, bool* output_data,
                                const Dims<4>& output_dims) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              F(input1_data[SubscriptToIndex(desc1, c, x, y, b)],
                input2_data[SubscriptToIndex(desc2, c, x, y, b)]);
        }
      }
    }
  }
}

template <typename T, ComparisonFn<int32> F>
inline void BroadcastComparison(int left_shift, const T* input1_data,
                                const Dims<4>& input1_dims, int32 input1_offset,
                                int32 input1_multiplier, int input1_shift,
                                const T* input2_data,
                                const Dims<4>& input2_dims, int32 input2_offset,
                                int32 input2_multiplier, int input2_shift,
                                bool* output_data, const Dims<4>& output_dims) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier,
                  kReverseShift * input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier,
                  kReverseShift * input2_shift);
          output_data[Offset(output_dims, c, x, y, b)] =
              F(scaled_input1_val, scaled_input2_val);
        }
      }
    }
  }
}

#define TFLITE_COMPARISON_OP(name)                                            \
  template <typename T>                                                       \
  inline void name(const T* input1_data, const Dims<4>& input1_dims,          \
                   const T* input2_data, const Dims<4>& input2_dims,          \
                   bool* output_data, const Dims<4>& output_dims) {           \
    gemmlowp::ScopedProfilingLabel label(#name);                              \
    Comparison<T, name##Fn>(input1_data, input1_dims, input2_data,            \
                            input2_dims, output_data, output_dims);           \
  }                                                                           \
  template <typename T>                                                       \
  inline void name(                                                           \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label(#name "/8bit");                      \
    Comparison<T, name##Fn>(left_shift, input1_data, input1_dims,             \
                            input1_offset, input1_multiplier, input1_shift,   \
                            input2_data, input2_dims, input2_offset,          \
                            input2_multiplier, input2_shift, output_data,     \
                            output_dims);                                     \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      const T* input1_data, const Dims<4>& input1_dims, const T* input2_data, \
      const Dims<4>& input2_dims, bool* output_data,                          \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label("Broadcast" #name);                  \
    BroadcastComparison<T, name##Fn>(input1_data, input1_dims, input2_data,   \
                                     input2_dims, output_data, output_dims);  \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label("Broadcast" #name "/8bit");          \
    BroadcastComparison<T, name##Fn>(left_shift, input1_data, input1_dims,    \
                                     input1_offset, input1_multiplier,        \
                                     input1_shift, input2_data, input2_dims,  \
                                     input2_offset, input2_multiplier,        \
                                     input2_shift, output_data, output_dims); \
  }
TFLITE_COMPARISON_OP(Equal);
TFLITE_COMPARISON_OP(NotEqual);
TFLITE_COMPARISON_OP(Greater);
TFLITE_COMPARISON_OP(GreaterEqual);
TFLITE_COMPARISON_OP(Less);
TFLITE_COMPARISON_OP(LessEqual);
#undef TFLITE_COMPARISON_OP

template <typename D, typename T>
inline void Select(const D* input_condition_data,
                   const Dims<4>& input_condition_dims, const T* input_x_data,
                   const Dims<4>& input_x_dims, const T* input_y_data,
                   const Dims<4>& input_y_dims, T* output_data,
                   const Dims<4>& output_dims) {
  const int64_t flatsize =
      MatchingFlatSize(input_x_dims, input_y_dims, output_dims);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] =
        input_condition_data[i] ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
inline void RankOneSelect(const D* input_condition_data,
                          const Dims<4>& input_condition_dims,
                          const T* input_x_data, const Dims<4>& input_x_dims,
                          const T* input_y_data, const Dims<4>& input_y_dims,
                          T* output_data, const Dims<4>& output_dims) {
  const int64_t rank = MatchingArraySize(input_condition_dims, 0, input_x_dims,
                                         3, input_y_dims, 3, output_dims, 3);
  const int64_t inner_size =
      MatchingFlatSizeSkipDim(input_x_dims, 3, input_y_dims, output_dims);

  int64_t offset = 0;
  for (int64_t i = 0; i < rank; i++) {
    const T* input_data = input_condition_data[i] ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

// For easy implementation, the indices is always a vector of size-4 vectors.
template <typename T, typename I>
inline void SparseToDense(const std::vector<std::vector<I>>& indices,
                          const T* values, T default_value, T* output_data,
                          const Dims<4>& output_dims, bool value_is_scalar) {
  const int value_count = indices.size();

  // First fill the output_data with default value.
  const int num_elements = FlatSize(output_dims);
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = default_value;
  }

  // Special handle for value is scalar case to avoid checking the boolean
  // condition within the loop every time.
  if (value_is_scalar) {
    for (int i = 0; i < value_count; ++i) {
      const std::vector<I>& index = indices[i];
      TFLITE_DCHECK_EQ(index.size(), 4);
      const T value = *values;  // just use the first value.
      output_data[Offset(output_dims, index[3], index[2], index[1], index[0])] =
          value;
    }
    return;
  }

  // Go through the values and indices to fill the sparse values.
  for (int i = 0; i < value_count; ++i) {
    const std::vector<I>& index = indices[i];
    TFLITE_DCHECK_EQ(index.size(), 4);
    const T value = values[i];
    output_data[Offset(output_dims, index[3], index[2], index[1], index[0])] =
        value;
  }
}

template <typename T>
inline void Pow(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T* output_data, const Dims<4>& output_dims) {
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline void BroadcastPow(const T* input1_data, const Dims<4>& input1_dims,
                         const T* input2_data, const Dims<4>& input2_dims,
                         T* output_data, const Dims<4>& output_dims) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              std::pow(input1_data[SubscriptToIndex(desc1, c, x, y, b)],
                       input2_data[SubscriptToIndex(desc2, c, x, y, b)]);
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_

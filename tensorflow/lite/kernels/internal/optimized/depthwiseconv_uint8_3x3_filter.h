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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

#include <memory>

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace depthwise_conv {

constexpr int kDepthwiseConvScratchWorkspaceSize = 10 * 10 * 64;
constexpr int kDepthwiseConvAdjustedBiasLimit = 64;
// In cases such as depth multiplication, we want to be able to load data from
// the workspace that is beyond the valid range. Macro-block sizes are adjusted
// to allow for this.
constexpr int kWorkspaceExtension = 16;

#ifdef USE_NEON
// Lane operations are for clarity and convenience. We want to load and store
// 4 8-bit lanes together. So these are treated much like 32-bit loads and
// 32-bit stores. Stores require 32-bit alignment.

#define vst1_lane_8x4(dst, reg, lane_num)                         \
  TFLITE_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % 4, 0); \
  vst1_lane_u32(reinterpret_cast<uint32_t*>(dst), reg, lane_num)
#define vst1q_lane_8x4(dst, reg, lane_num)                        \
  TFLITE_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % 4, 0); \
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(dst), reg, lane_num)

#define vld1q_lane_s8x8(src, reg, lane_num) \
  vld1q_lane_u64(reinterpret_cast<const uint64_t*>(src), reg, lane_num)
#define vld1_lane_8x4(src, reg, lane_num) \
  vld1_lane_s32(reinterpret_cast<const int32*>(src), reg, lane_num)
#define vld1q_lane_8x4(src, reg, lane_num) \
  vld1q_lane_s32(reinterpret_cast<const int32*>(src), reg, lane_num)
#define vld1q_dup_s8x4(src) vld1q_dup_s32(reinterpret_cast<const int32*>(src))

#ifndef __aarch64__
inline int8x16_t vqtbl4q_s8(int8x16x4_t a, int8x16_t b) {
  const uint8x16_t mask = vtstq_s8(b, vdupq_n_s8(8));

  // Delete bit 3 from the indices.
  const int8x16_t high_bits = vshrq_n_s8(b, 4);
  int8x16_t deleted_bit_3 = b;
  deleted_bit_3 = vsliq_n_s8(deleted_bit_3, high_bits, 3);

  int8x8x4_t repacked_data;

  // Calculate for lower indices.
  repacked_data.val[0] = vget_low_s8(a.val[0]);
  repacked_data.val[1] = vget_low_s8(a.val[1]);
  repacked_data.val[2] = vget_low_s8(a.val[2]);
  repacked_data.val[3] = vget_low_s8(a.val[3]);
  const int8x16_t output_for_lower =
      vcombine_s8(vtbl4_s8(repacked_data, vget_low_s8(deleted_bit_3)),
                  vtbl4_s8(repacked_data, vget_high_s8(deleted_bit_3)));

  // Calculate for high indices.
  repacked_data.val[0] = vget_high_s8(a.val[0]);
  repacked_data.val[1] = vget_high_s8(a.val[1]);
  repacked_data.val[2] = vget_high_s8(a.val[2]);
  repacked_data.val[3] = vget_high_s8(a.val[3]);
  const int8x16_t output_for_higher =
      vcombine_s8(vtbl4_s8(repacked_data, vget_low_s8(deleted_bit_3)),
                  vtbl4_s8(repacked_data, vget_high_s8(deleted_bit_3)));

  // Merge.
  int8x16_t output = vbslq_s8(mask, output_for_higher, output_for_lower);
  return output;
}
#endif  // !__aarch64__

// Convenience-compatibility functions.
// Compatibility: Intrinsics reflect a mixture of older and newer ARM
//     instructions. This actually results in ZIP1 / ZIP2 asm instructions, but
//     one intrinsic is provided. Also older instructions operated in place,
//     and it seems more defensive to assume that some versions of intrinsics
//     might reflect this
// Convenience: Callers in these kernels want both ZIP1 and ZIP2, and we do not
//     want the calling code to get cluttered with unpacking int8x16x2_t.
inline void vzipq_s8_in_place(int8x16_t* a, int8x16_t* b) {
  int8x16x2_t r8x16;
  r8x16 = vzipq_s8(*a, *b);
  *a = r8x16.val[0];
  *b = r8x16.val[1];
}

inline void vzipq_s8x2_in_place(int8x16_t* a, int8x16_t* b) {
  int16x8x2_t r16x8;
  r16x8 = vzipq_s16(vreinterpretq_s16_s8(*a), vreinterpretq_s16_s8(*b));
  *a = vreinterpretq_s8_s16(r16x8.val[0]);
  *b = vreinterpretq_s8_s16(r16x8.val[1]);
}

// Similar rationale to the zip-in_place functions, but callers only actually
// need the TRN1 asm instruction result.
inline void vtrn1_s8x2_in_place(int8x16_t* a, int8x16_t* b) {
  int16x8x2_t r16x8;
  r16x8 = vtrnq_s16(vreinterpretq_s16_s8(*a), vreinterpretq_s16_s8(*b));
  *a = vreinterpretq_s8_s16(r16x8.val[0]);
}

// Similar rationale to the zip-in_place functions, but callers only actually
// need the ZIP1 or ZIP2 asm instruction results.
inline int8x16_t vzip1q_s8(int8x16_t a, int8x16_t b) {
  return vzipq_s8(a, b).val[0];
}
inline int8x16_t vzip2q_s8(int8x16_t a, int8x16_t b) {
  return vzipq_s8(a, b).val[1];
}

inline void biregister_rotate_8(int8x16_t* left, int8x16_t* right) {
  *left = vreinterpretq_s8_u32(vshrq_n_u32(vreinterpretq_u32_s8(*left), 8));
  *left = vreinterpretq_s8_u32(vsliq_n_u32(vreinterpretq_u32_s8(*left),
                                           vreinterpretq_u32_s8(*right), 24));
  *right = vreinterpretq_s8_u32(vshrq_n_u32(vreinterpretq_u32_s8(*right), 8));
}

#ifndef __aarch64__
inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x4x2_t deinterleaved = vuzpq_s32(a, b);
  return vqaddq_s32(deinterleaved.val[0], deinterleaved.val[1]);
}
#endif  // !__aarch64__

#ifdef __ARM_FEATURE_DOTPROD
// The vdotq_lane_s32 takes int8x8t for the rhs parameter, whereas the actual
// instruction selects from between 4 32-bit (4x8-bit packed) sub-registers, an
// unusual interpretation of "lane".
inline int32x4_t vdotq_four_lane_s32(int32x4_t acc, int8x16_t lhs,
                                     int8x16_t rhs, const int lane) {
  switch (lane) {
    case 0:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_low_s8(rhs)), 0);
    case 1:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_low_s8(rhs)), 1);
    case 2:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_high_s8(rhs)),
                            0);
    case 3:
    default:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_high_s8(rhs)),
                            1);
  }
}

#else

inline int32x4_t vdotq_s32(int32x4_t acc, int8x16_t lhs, int8x16_t rhs) {
  int32x4_t sum0 = vpaddlq_s16(vmull_s8(vget_low_s8(lhs), vget_low_s8(rhs)));
  int32x4_t sum1 = vpaddlq_s16(vmull_s8(vget_high_s8(lhs), vget_high_s8(rhs)));
  int32x4_t sum = vpaddq_s32(sum0, sum1);
  return vaddq_s32(acc, sum);
}

inline int32x4_t vdotq_four_lane_s32(int32x4_t acc, int8x16_t lhs,
                                     int8x16_t rhs, int lane) {
  int8x8_t lane_rhs;
  if (lane == 0) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(rhs)), 0));
  } else if (lane == 1) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(rhs)), 1));
  } else if (lane == 2) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(rhs)), 0));
  } else {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(rhs)), 1));
  }
  int32x4_t sum0 = vpaddlq_s16(vmull_s8(vget_low_s8(lhs), lane_rhs));
  int32x4_t sum1 = vpaddlq_s16(vmull_s8(vget_high_s8(lhs), lane_rhs));
  int32x4_t sum = vpaddq_s32(sum0, sum1);
  return vaddq_s32(acc, sum);
}

#endif  // !__ARM_FEATURE_DOTPROD
#endif  // ARM NEON

template <DepthwiseConvOutputRounding output_rounding>
struct DivideByPOT {};

template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kAwayFromZero> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
    return RoundingDivideByPOT(x, exponent);
  }
};

#ifdef USE_NEON
template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kUpward> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
    return vqrshlq_s32(x, vdupq_n_s32(static_cast<int32>(-exponent)));
  }
};
#endif  // ARM NEON

// See CategorizeDotProductKernel for definitive taxonomy.
enum class DotProduct3x3KernelType {
  kNone = 0,  // Parameter combination is not supported for dot product kernels.
  kPlain,
  kWithDepthMultiplicationStride1,
  kWithDepthMultiplicationStride2,
  kStride2,
};

inline DotProduct3x3KernelType CategorizeDotProductKernel(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    const DepthwiseParams& params) {
  constexpr int kSymmetricZeroPoint = 128;
  const int padding =
      std::max(params.padding_values.width, params.padding_values.height);
  const int stride = params.stride_width;
  const int32 input_depth = input_shape.Dims(3);
  const int32 depth_multiplier = params.depth_multiplier;
  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);

  bool supported =
      params.weights_offset == -kSymmetricZeroPoint &&
      stride == params.stride_height && stride <= 2 && padding <= 1 &&
      filter_width == 3 && filter_height == 3 && params.output_shift <= 0 &&
      params.dilation_width_factor == 1 && params.dilation_height_factor == 1 &&
      (((input_depth % 8) == 0 && depth_multiplier == 1) ||
       (input_depth == 1 && depth_multiplier > 1));

  if (!supported) {
    return DotProduct3x3KernelType::kNone;
  }

  if (params.depth_multiplier == 1) {
    if (stride == 1) {
      return DotProduct3x3KernelType::kPlain;
    } else if (stride == 2) {
      return DotProduct3x3KernelType::kStride2;
    } else {
      return DotProduct3x3KernelType::kNone;
    }
  } else {
    if (stride == 1) {
      return DotProduct3x3KernelType::kWithDepthMultiplicationStride1;
    } else if (stride == 2) {
      return DotProduct3x3KernelType::kWithDepthMultiplicationStride2;
    } else {
      return DotProduct3x3KernelType::kNone;
    }
  }
}

#ifdef USE_NEON

#define STR(s) STR_UNEXPANDED(s)
#define STR_UNEXPANDED(s) #s

// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__) && !defined(GOOGLE_L4T)
#include <stddef.h>

// Encapsulates constant parameters used in DepthwiseConv.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
struct DepthwiseConvParams {
  int64_t input_depth;
  int64_t input_row_size;
  int64_t output_depth;
  int64_t output_row_size;
  int64_t filter_row_size;
  int32 input_offset;
  int32 output_offset;
  int32 filter_offset;
  int32 output_multiplier;
  int32 output_activation_min;
  int32 output_activation_max;
  int32 output_right_shift;
  int32 input_width;
  int32 input_height;
  int32 stride_width;
  int32 stride_height;
  int32 output_width;
  int32 output_height;
};

// Represents the number of bytes offset from the start of the
// DepthwiseConvParams struct. This is used in the asm to load parameters.
// Keep these values in sync with the static_asserts below.
#define OFFSET_INPUT_DEPTH 0
#define OFFSET_INPUT_ROW_SIZE 8
#define OFFSET_OUTPUT_DEPTH 16
#define OFFSET_OUTPUT_ROW_SIZE 24
#define OFFSET_FILTER_ROW_SIZE 32
#define OFFSET_INPUT_OFFSET 40
#define OFFSET_OUTPUT_OFFSET 44
#define OFFSET_FILTER_OFFSET 48
#define OFFSET_OUTPUT_MULTIPLIER 52
#define OFFSET_OUTPUT_ACTIVATION_MIN 56
#define OFFSET_OUTPUT_ACTIVATION_MAX 60
#define OFFSET_OUTPUT_RIGHT_SHIFT 64
#define OFFSET_INPUT_WIDTH 68
#define OFFSET_INPUT_HEIGHT 72
#define OFFSET_STRIDE_WIDTH 76
#define OFFSET_STRIDE_HEIGHT 80
#define OFFSET_OUTPUT_WIDTH 84
#define OFFSET_OUTPUT_HEIGHT 88

static_assert(offsetof(DepthwiseConvParams, input_depth) == OFFSET_INPUT_DEPTH,
              "");
static_assert(offsetof(DepthwiseConvParams, input_row_size) ==
                  OFFSET_INPUT_ROW_SIZE,
              "");
static_assert(offsetof(DepthwiseConvParams, output_depth) ==
                  OFFSET_OUTPUT_DEPTH,
              "");
static_assert(offsetof(DepthwiseConvParams, output_row_size) ==
                  OFFSET_OUTPUT_ROW_SIZE,
              "");
static_assert(offsetof(DepthwiseConvParams, filter_row_size) ==
                  OFFSET_FILTER_ROW_SIZE,
              "");
static_assert(offsetof(DepthwiseConvParams, input_offset) ==
                  OFFSET_INPUT_OFFSET,
              "");
static_assert(offsetof(DepthwiseConvParams, output_offset) ==
                  OFFSET_OUTPUT_OFFSET,
              "");
static_assert(offsetof(DepthwiseConvParams, filter_offset) ==
                  OFFSET_FILTER_OFFSET,
              "");
static_assert(offsetof(DepthwiseConvParams, output_multiplier) ==
                  OFFSET_OUTPUT_MULTIPLIER,
              "");
static_assert(offsetof(DepthwiseConvParams, output_activation_min) ==
                  OFFSET_OUTPUT_ACTIVATION_MIN,
              "");
static_assert(offsetof(DepthwiseConvParams, output_activation_max) ==
                  OFFSET_OUTPUT_ACTIVATION_MAX,
              "");
static_assert(offsetof(DepthwiseConvParams, output_right_shift) ==
                  OFFSET_OUTPUT_RIGHT_SHIFT,
              "");
static_assert(offsetof(DepthwiseConvParams, input_width) == OFFSET_INPUT_WIDTH,
              "");
static_assert(offsetof(DepthwiseConvParams, input_height) ==
                  OFFSET_INPUT_HEIGHT,
              "");
static_assert(offsetof(DepthwiseConvParams, stride_width) ==
                  OFFSET_STRIDE_WIDTH,
              "");
static_assert(offsetof(DepthwiseConvParams, stride_height) ==
                  OFFSET_STRIDE_HEIGHT,
              "");
static_assert(offsetof(DepthwiseConvParams, output_width) ==
                  OFFSET_OUTPUT_WIDTH,
              "");
static_assert(offsetof(DepthwiseConvParams, output_height) ==
                  OFFSET_OUTPUT_HEIGHT,
              "");
#endif  // __aarch64__
#endif  // ARM NEON

// Encapsulates constant parameters used in DepthwiseConv using dot-product ops.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
//
// This structure is specifically designed for use in asm.
struct DepthwiseConvDotProdParams {
  int64_t input_depth;
  int64_t output_depth;
  int32 stride;
  int32 bias_increment;
  //
  int32 input_offset;
  int32 output_offset;
  int32 output_multiplier;
  int32 output_shift;
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  //
  int32 padding_left;
  int32 padding_right;
  int32 padding_top;
  int32 padding_bottom;
  //
  int32 depth_micro_repeats;
  //
  int32 width_macro_count;
  int32 input_width_overall_micro_repeats;
  int32 input_width_micro_repeats;
  int32 residual_width;
  int32 output_width_overall_micro_repeats;
  int32 output_width_micro_repeats;
  int32 output_residual_width;
  int32 workspace_width_micro_repeats;
  //
  int32 height_macro_count;
  int32 inbound_block_height;
  int32 outbound_block_height;
  int32 input_height_stride;
  int32 output_height_stride;
  int32 workspace_height_stride;
  //
  int32 four_over_stride;
};

#ifdef USE_NEON
#if defined(__ARM_FEATURE_DOTPROD) && !defined(GOOGLE_L4T)
// Represents the number of bytes offset from the start of the
// DepthwiseConvDotProdParams struct. This is used in the asm to load
// parameters. Keep these values in sync with the static_asserts below.

#define DP_OFFSET_INPUT_DEPTH 0
#define DP_OFFSET_OUTPUT_DEPTH DP_OFFSET_INPUT_DEPTH + 8
#define DP_OFFSET_STRIDE DP_OFFSET_OUTPUT_DEPTH + 8
#define DP_OFFSET_BIAS_INCREMENT DP_OFFSET_STRIDE + 4
//
#define DP_OFFSET_INPUT_OFFSET 24
#define DP_OFFSET_OUTPUT_OFFSET DP_OFFSET_INPUT_OFFSET + 4
#define DP_OFFSET_OUTPUT_MULTIPLIER DP_OFFSET_OUTPUT_OFFSET + 4
#define DP_OFFSET_OUTPUT_SHIFT DP_OFFSET_OUTPUT_MULTIPLIER + 4
#define DP_OFFSET_QUANTIZED_ACTIVATION_MIN DP_OFFSET_OUTPUT_SHIFT + 4
#define DP_OFFSET_QUANTIZED_ACTIVATION_MAX \
  DP_OFFSET_QUANTIZED_ACTIVATION_MIN + 4
//
#define DP_OFFSET_PADDING_LEFT 48
#define DP_OFFSET_PADDING_RIGHT DP_OFFSET_PADDING_LEFT + 4
#define DP_OFFSET_PADDING_TOP DP_OFFSET_PADDING_RIGHT + 4
#define DP_OFFSET_PADDING_BOTTOM DP_OFFSET_PADDING_TOP + 4
//
#define DP_OFFSET_DEPTH_MICRO_REPEATS DP_OFFSET_PADDING_BOTTOM + 4
//
#define DP_OFFSET_WIDTH_MACRO_COUNT 68
#define DP_OFFSET_INPUT_WIDTH_OVERALL_MICRO_REPEATS \
  DP_OFFSET_WIDTH_MACRO_COUNT + 4
#define DP_OFFSET_INPUT_WIDTH_MICRO_REPEATS \
  DP_OFFSET_INPUT_WIDTH_OVERALL_MICRO_REPEATS + 4
#define DP_OFFSET_RESIDUAL_WIDTH DP_OFFSET_INPUT_WIDTH_MICRO_REPEATS + 4
#define DP_OFFSET_OUTPUT_WIDTH_OVERALL_MICRO_REPEATS \
  DP_OFFSET_RESIDUAL_WIDTH + 4
#define DP_OFFSET_OUTPUT_WIDTH_MICRO_REPEATS \
  DP_OFFSET_OUTPUT_WIDTH_OVERALL_MICRO_REPEATS + 4
#define DP_OFFSET_OUTPUT_RESIDUAL_WIDTH DP_OFFSET_OUTPUT_WIDTH_MICRO_REPEATS + 4
#define DP_OFFSET_WORKSPACE_WIDTH_MICRO_REPEATS \
  DP_OFFSET_OUTPUT_RESIDUAL_WIDTH + 4
//
#define DP_OFFSET_HEIGHT_MACRO_COUNT 100
#define DP_OFFSET_INBOUND_BLOCK_HEIGHT DP_OFFSET_HEIGHT_MACRO_COUNT + 4
#define DP_OFFSET_OUTBOUND_BLOCK_HEIGHT DP_OFFSET_INBOUND_BLOCK_HEIGHT + 4
#define DP_OFFSET_INPUT_HEIGHT_STRIDE DP_OFFSET_OUTBOUND_BLOCK_HEIGHT + 4
#define DP_OFFSET_OUTPUT_HEIGHT_STRIDE DP_OFFSET_INPUT_HEIGHT_STRIDE + 4
#define DP_OFFSET_WORKSPACE_HEIGHT_STRIDE DP_OFFSET_OUTPUT_HEIGHT_STRIDE + 4
//
#define DP_OFFSET_FOUR_OVER_STRIDE DP_OFFSET_WORKSPACE_HEIGHT_STRIDE + 4

static_assert(offsetof(DepthwiseConvDotProdParams, input_depth) ==
                  DP_OFFSET_INPUT_DEPTH,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_depth) ==
                  DP_OFFSET_OUTPUT_DEPTH,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, stride) == DP_OFFSET_STRIDE,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, bias_increment) ==
                  DP_OFFSET_BIAS_INCREMENT,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, input_offset) ==
                  DP_OFFSET_INPUT_OFFSET,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_offset) ==
                  DP_OFFSET_OUTPUT_OFFSET,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_multiplier) ==
                  DP_OFFSET_OUTPUT_MULTIPLIER,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_shift) ==
                  DP_OFFSET_OUTPUT_SHIFT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, quantized_activation_min) ==
                  DP_OFFSET_QUANTIZED_ACTIVATION_MIN,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, quantized_activation_max) ==
                  DP_OFFSET_QUANTIZED_ACTIVATION_MAX,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, padding_left) ==
                  DP_OFFSET_PADDING_LEFT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, padding_right) ==
                  DP_OFFSET_PADDING_RIGHT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, padding_top) ==
                  DP_OFFSET_PADDING_TOP,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, padding_bottom) ==
                  DP_OFFSET_PADDING_BOTTOM,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, depth_micro_repeats) ==
                  DP_OFFSET_DEPTH_MICRO_REPEATS,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, width_macro_count) ==
                  DP_OFFSET_WIDTH_MACRO_COUNT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams,
                       input_width_overall_micro_repeats) ==
                  DP_OFFSET_INPUT_WIDTH_OVERALL_MICRO_REPEATS,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, input_width_micro_repeats) ==
                  DP_OFFSET_INPUT_WIDTH_MICRO_REPEATS,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, residual_width) ==
                  DP_OFFSET_RESIDUAL_WIDTH,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams,
                       output_width_overall_micro_repeats) ==
                  DP_OFFSET_OUTPUT_WIDTH_OVERALL_MICRO_REPEATS,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams,
                       output_width_micro_repeats) ==
                  DP_OFFSET_OUTPUT_WIDTH_MICRO_REPEATS,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_residual_width) ==
                  DP_OFFSET_OUTPUT_RESIDUAL_WIDTH,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams,
                       workspace_width_micro_repeats) ==
                  DP_OFFSET_WORKSPACE_WIDTH_MICRO_REPEATS,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, height_macro_count) ==
                  DP_OFFSET_HEIGHT_MACRO_COUNT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, inbound_block_height) ==
                  DP_OFFSET_INBOUND_BLOCK_HEIGHT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, outbound_block_height) ==
                  DP_OFFSET_OUTBOUND_BLOCK_HEIGHT,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, input_height_stride) ==
                  DP_OFFSET_INPUT_HEIGHT_STRIDE,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, output_height_stride) ==
                  DP_OFFSET_OUTPUT_HEIGHT_STRIDE,
              "");
static_assert(offsetof(DepthwiseConvDotProdParams, workspace_height_stride) ==
                  DP_OFFSET_WORKSPACE_HEIGHT_STRIDE,
              "");
//
static_assert(offsetof(DepthwiseConvDotProdParams, four_over_stride) ==
                  DP_OFFSET_FOUR_OVER_STRIDE,
              "");
#endif  // __ARM_FEATURE_DOTPROD && !GOOGLE_L4T

#if defined(__aarch64__) && !defined(GOOGLE_L4T)
template <int32 kDepth, int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvWindow {};

template <>
struct DepthwiseConvWindow<8, 1, 1> {
 public:
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         int64_t input_depth, int64_t input_row_size,
                         int32 output_window_height, int32 output_window_width,
                         const DepthwiseConvParams* params_ptr) {
    const int64_t input_width_increment = 2 * input_depth;
    const int64_t input_height_increment = 2 * input_row_size;
    const int64_t output_height_increment = 2 * params_ptr->output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "9"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "10"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "11"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time, load inputs for a 2x1 (2
        //            height, 1 width) output window (4x3 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 2x1 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time, load inputs for a 1x2 (1
        //            height, 2 width) output window (3x4 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 1x2 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        // We use x9--x15 general purpose registers as they are caller-saved
        // temporary registers (see
        // http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf).  // NOLINT
        "ldr w9, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr x3, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "cmp %w[output_window_height], #2\n"
        "dup v26.8h, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "ldr w2, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v29.4s, w2\n"
        "ldr w4, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "dup v30.4s, w4\n"
        "ldr w0, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v31.4s, w0\n"
        "neg w9, w9\n"
        "dup v28.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "add x10, %[bias_ptr], #16\n"
        "ldr x1, [%[params_ptr], #" STR(OFFSET_OUTPUT_ROW_SIZE) "]\n"
        "dup v9.8h, w9\n"

        // Load filters and add offsets.
        "ld1 {v0.8b}, [%[filter_ptr]], x3\n"
        "ld1 {v1.8b}, [%[filter_ptr]], x3\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], x3\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], x3\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], x3\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], x3\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], x3\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], x3\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]], x3\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // This loop processes 2x2 outputs. To avoid register exhaustion,
          // inputs for the left 2 outputs are loaded first, then the right
          // two outputs.
          "mov x11, %[input_ptr]\n"
          "mov x12, x11\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "add x13, x11, %[input_row_size]\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "add x14, x13, %[input_row_size]\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x15, x14, %[input_row_size]\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "mov w5, %w[output_window_width]\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "mov x6, %[output_ptr]\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "add x7, %[output_ptr], x1\n"
          "ld1 {v15.8b}, [x14], %[input_depth]\n"
          // The height 2 / width 2 loop loads an extra 2x1 outputs (2 height,
          // 1 width) in anticipation for the next iteration. Make sure
          // |output_window_width| is large enough to handle the additional
          // loads, otherwise jump to specific the appropriate label to handle
          // smaller widths.
          "cmp w5, #2\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "ld1 {v16.8b}, [x14], %[input_depth]\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "ld1 {v17.8b}, [x14], %[input_depth]\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "ld1 {v18.8b}, [x15], %[input_depth]\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "ld1 {v19.8b}, [x15], %[input_depth]\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "ld1 {v20.8b}, [x15], %[input_depth]\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v22.4s}, [x10]\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "ld1 {v24.4s}, [x10]\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "f\n"
          "cmp w5, #1\n"
          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            // Mul-add left outputs.
            "smlal v21.4s, v0.4h, v9.4h\n"
            "subs w5, w5, #2\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "cmp w5, #3\n"
            "smlal v23.4s, v0.4h, v12.4h\n"
            "ld1 {v9.8b}, [x12]\n"
            "smlal2 v24.4s, v0.8h, v12.8h\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "smlal v23.4s, v1.4h, v13.4h\n"
            "smlal2 v24.4s, v1.8h, v13.8h\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "smlal v23.4s, v2.4h, v14.4h\n"
            "smlal2 v24.4s, v2.8h, v14.8h\n"
            "smlal v21.4s, v3.4h, v12.4h\n"
            "smlal2 v22.4s, v3.8h, v12.8h\n"
            "ld1 {v12.8b}, [x13]\n"
            "smlal v23.4s, v3.4h, v15.4h\n"
            "smlal2 v24.4s, v3.8h, v15.8h\n"
            "smlal v21.4s, v4.4h, v13.4h\n"
            "smlal2 v22.4s, v4.8h, v13.8h\n"
            "smlal v23.4s, v4.4h, v16.4h\n"
            "smlal2 v24.4s, v4.8h, v16.8h\n"
            "smlal v21.4s, v5.4h, v14.4h\n"
            "smlal2 v22.4s, v5.8h, v14.8h\n"
            "smlal v23.4s, v5.4h, v17.4h\n"
            "smlal2 v24.4s, v5.8h, v17.8h\n"
            "smlal v21.4s, v6.4h, v15.4h\n"
            "smlal2 v22.4s, v6.8h, v15.8h\n"
            "ld1 {v15.8b}, [x14]\n"
            "smlal v23.4s, v6.4h, v18.4h\n"
            "smlal2 v24.4s, v6.8h, v18.8h\n"
            "ld1 {v18.8b}, [x15]\n"
            "smlal v21.4s, v7.4h, v16.4h\n"
            "smlal2 v22.4s, v7.8h, v16.8h\n"
            "smlal v23.4s, v7.4h, v19.4h\n"
            "smlal2 v24.4s, v7.8h, v19.8h\n"
            "smlal v21.4s, v8.4h, v17.4h\n"
            "smlal2 v22.4s, v8.8h, v17.8h\n"
            "smlal v23.4s, v8.4h, v20.4h\n"
            "smlal2 v24.4s, v8.8h, v20.8h\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w4\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w0\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x3\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "st1 {v23.8b}, [x7], x3\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"

            // Mul-add right outputs.
            "smlal v21.4s, v0.4h, v10.4h\n"
            "add x11, x11, %[input_width_increment]\n"
            "smlal2 v22.4s, v0.8h, v10.8h\n"
            "mov x12, x11\n"
            "smlal v23.4s, v0.4h, v13.4h\n"
            "add x13, x11, %[input_row_size]\n"
            "smlal2 v24.4s, v0.8h, v13.8h\n"
            "add x14, x13, %[input_row_size]\n"
            "smlal v21.4s, v1.4h, v11.4h\n"
            "add x15, x14, %[input_row_size]\n"
            "smlal2 v22.4s, v1.8h, v11.8h\n"
            "smlal v23.4s, v1.4h, v14.4h\n"
            "smlal2 v24.4s, v1.8h, v14.8h\n"
            "smlal v21.4s, v2.4h, v9.4h\n"
            "smlal2 v22.4s, v2.8h, v9.8h\n"
            "ld1 {v9.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v12.4h\n"
            "ld1 {v10.8b}, [x12], %[input_depth]\n"
            "smlal2 v24.4s, v2.8h, v12.8h\n"
            "ld1 {v11.8b}, [x12], %[input_depth]\n"
            "smlal v21.4s, v3.4h, v13.4h\n"
            "smlal2 v22.4s, v3.8h, v13.8h\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "smlal v21.4s, v4.4h, v14.4h\n"
            "smlal2 v22.4s, v4.8h, v14.8h\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "smlal v21.4s, v5.4h, v12.4h\n"
            "smlal2 v22.4s, v5.8h, v12.8h\n"
            "ld1 {v12.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v15.4h\n"
            "ld1 {v13.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v5.8h, v15.8h\n"
            "ld1 {v14.8b}, [x13], %[input_depth]\n"
            "smlal v21.4s, v6.4h, v16.4h\n"
            "smlal2 v22.4s, v6.8h, v16.8h\n"
            "smlal v23.4s, v6.4h, v19.4h\n"
            "smlal2 v24.4s, v6.8h, v19.8h\n"
            "smlal v21.4s, v7.4h, v17.4h\n"
            "smlal2 v22.4s, v7.8h, v17.8h\n"
            "smlal v23.4s, v7.4h, v20.4h\n"
            "smlal2 v24.4s, v7.8h, v20.8h\n"
            "smlal v21.4s, v8.4h, v15.4h\n"
            "smlal2 v22.4s, v8.8h, v15.8h\n"
            "ld1 {v15.8b}, [x14], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v18.4h\n"
            "ld1 {v16.8b}, [x14], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v18.8h\n"
            "ld1 {v17.8b}, [x14], %[input_depth]\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "ld1 {v18.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "ld1 {v19.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "ld1 {v20.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w4\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w0\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x3\n"
            "uaddw v10.8h, v26.8h, v10.8b\n"
            "st1 {v23.8b}, [x7], x3\n"
            "uaddw v11.8h, v26.8h, v11.8b\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "uaddw v13.8h, v26.8h, v13.8b\n"
            "uaddw v14.8h, v26.8h, v14.8b\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v16.8h, v26.8h, v16.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "uaddw v17.8h, v26.8h, v17.8b\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "uaddw v19.8h, v26.8h, v19.8b\n"
            "uaddw v20.8h, v26.8h, v20.8b\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          // At this point, there will be one of 2 width or 1 width leftover,
          // not both.
          "cmp w5, #2\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          // Handle last 2 columns if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER ":\n"
          // Mul-add left outputs.
          "smlal v21.4s, v0.4h, v9.4h\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "ld1 {v9.8b}, [x12]\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "smlal v23.4s, v2.4h, v14.4h\n"
          "smlal2 v24.4s, v2.8h, v14.8h\n"
          "smlal v21.4s, v3.4h, v12.4h\n"
          "smlal2 v22.4s, v3.8h, v12.8h\n"
          "ld1 {v12.8b}, [x13]\n"
          "smlal v23.4s, v3.4h, v15.4h\n"
          "smlal2 v24.4s, v3.8h, v15.8h\n"
          "smlal v21.4s, v4.4h, v13.4h\n"
          "smlal2 v22.4s, v4.8h, v13.8h\n"
          "smlal v23.4s, v4.4h, v16.4h\n"
          "smlal2 v24.4s, v4.8h, v16.8h\n"
          "smlal v21.4s, v5.4h, v14.4h\n"
          "smlal2 v22.4s, v5.8h, v14.8h\n"
          "smlal v23.4s, v5.4h, v17.4h\n"
          "smlal2 v24.4s, v5.8h, v17.8h\n"
          "smlal v21.4s, v6.4h, v15.4h\n"
          "smlal2 v22.4s, v6.8h, v15.8h\n"
          "ld1 {v15.8b}, [x14]\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "ld1 {v18.8b}, [x15]\n"
          "smlal v21.4s, v7.4h, v16.4h\n"
          "smlal2 v22.4s, v7.8h, v16.8h\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "st1 {v21.8b}, [x6], x3\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "st1 {v23.8b}, [x7], x3\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"

          // Mul-add right outputs.
          "smlal v21.4s, v0.4h, v10.4h\n"
          "smlal2 v22.4s, v0.8h, v10.8h\n"
          "smlal v23.4s, v0.4h, v13.4h\n"
          "smlal2 v24.4s, v0.8h, v13.8h\n"
          "smlal v21.4s, v1.4h, v11.4h\n"
          "smlal2 v22.4s, v1.8h, v11.8h\n"
          "smlal v23.4s, v1.4h, v14.4h\n"
          "smlal2 v24.4s, v1.8h, v14.8h\n"
          "smlal v21.4s, v2.4h, v9.4h\n"
          "smlal2 v22.4s, v2.8h, v9.8h\n"
          "smlal v23.4s, v2.4h, v12.4h\n"
          "smlal2 v24.4s, v2.8h, v12.8h\n"
          "smlal v21.4s, v3.4h, v13.4h\n"
          "smlal2 v22.4s, v3.8h, v13.8h\n"
          "smlal v23.4s, v3.4h, v16.4h\n"
          "smlal2 v24.4s, v3.8h, v16.8h\n"
          "smlal v21.4s, v4.4h, v14.4h\n"
          "smlal2 v22.4s, v4.8h, v14.8h\n"
          "smlal v23.4s, v4.4h, v17.4h\n"
          "smlal2 v24.4s, v4.8h, v17.8h\n"
          "smlal v21.4s, v5.4h, v12.4h\n"
          "smlal2 v22.4s, v5.8h, v12.8h\n"
          "smlal v23.4s, v5.4h, v15.4h\n"
          "smlal2 v24.4s, v5.8h, v15.8h\n"
          "smlal v21.4s, v6.4h, v16.4h\n"
          "smlal2 v22.4s, v6.8h, v16.8h\n"
          "smlal v23.4s, v6.4h, v19.4h\n"
          "smlal2 v24.4s, v6.8h, v19.8h\n"
          "smlal v21.4s, v7.4h, v17.4h\n"
          "smlal2 v22.4s, v7.8h, v17.8h\n"
          "smlal v23.4s, v7.4h, v20.4h\n"
          "smlal2 v24.4s, v7.8h, v20.8h\n"
          "smlal v21.4s, v8.4h, v15.4h\n"
          "smlal2 v22.4s, v8.8h, v15.8h\n"
          "smlal v23.4s, v8.4h, v18.4h\n"
          "smlal2 v24.4s, v8.8h, v18.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v21.8b}, [x6], x3\n"
          "st1 {v23.8b}, [x7], x3\n"
          "b " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "smlal v23.4s, v2.4h, v14.4h\n"
          "smlal2 v24.4s, v2.8h, v14.8h\n"
          "smlal v21.4s, v3.4h, v12.4h\n"
          "smlal2 v22.4s, v3.8h, v12.8h\n"
          "smlal v23.4s, v3.4h, v15.4h\n"
          "smlal2 v24.4s, v3.8h, v15.8h\n"
          "smlal v21.4s, v4.4h, v13.4h\n"
          "smlal2 v22.4s, v4.8h, v13.8h\n"
          "smlal v23.4s, v4.4h, v16.4h\n"
          "smlal2 v24.4s, v4.8h, v16.8h\n"
          "smlal v21.4s, v5.4h, v14.4h\n"
          "smlal2 v22.4s, v5.8h, v14.8h\n"
          "smlal v23.4s, v5.4h, v17.4h\n"
          "smlal2 v24.4s, v5.8h, v17.8h\n"
          "smlal v21.4s, v6.4h, v15.4h\n"
          "smlal2 v22.4s, v6.8h, v15.8h\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v16.4h\n"
          "smlal2 v22.4s, v7.8h, v16.8h\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v9.16b, v21.16b, v28.16b\n"
          "and v12.16b, v22.16b, v28.16b\n"
          "and v15.16b, v23.16b, v28.16b\n"
          "and v18.16b, v24.16b, v28.16b\n"
          "sshr v9.4s, v9.4s, #31\n"
          "sshr v12.4s, v12.4s, #31\n"
          "sshr v15.4s, v15.4s, #31\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sqadd v21.4s, v21.4s, v9.4s\n"
          "sqadd v22.4s, v22.4s, v12.4s\n"
          "sqadd v23.4s, v23.4s, v15.4s\n"
          "sqadd v24.4s, v24.4s, v18.4s\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v21.8b}, [x6], x3\n"
          "st1 {v23.8b}, [x7], x3\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        "mov x12, %[input_ptr]\n"
        "ld1 {v9.8b}, [x12], %[input_depth]\n"
        "add x13, %[input_ptr], %[input_row_size]\n"
        "ld1 {v10.8b}, [x12], %[input_depth]\n"
        "add x14, x13, %[input_row_size]\n"
        "ld1 {v11.8b}, [x12], %[input_depth]\n"
        "add x15, x14, %[input_row_size]\n"
        "mov w5, %w[output_window_width]\n"
        "ld1 {v13.8b}, [x13], %[input_depth]\n"
        "mov x6, %[output_ptr]\n"
        "ld1 {v14.8b}, [x13], %[input_depth]\n"
        "add x7, %[output_ptr], x1\n"
        "ld1 {v15.8b}, [x13], %[input_depth]\n"
        // The height 1 / width 2 loop loads an extra 1x1 output in anticipation
        // for the next iteration. Make sure |output_window_width| is large
        // enough to handle the additional load, otherwise jump to the
        // appropriate label to handle smaller widths.
        "cmp w5, #2\n"
        "ld1 {v17.8b}, [x14], %[input_depth]\n"
        "ld1 {v18.8b}, [x14], %[input_depth]\n"
        "ld1 {v19.8b}, [x14], %[input_depth]\n"
        "ld1 {v21.4s}, [%[bias_ptr]]\n"
        "ld1 {v22.4s}, [x10]\n"
        "ld1 {v23.4s}, [%[bias_ptr]]\n"
        "ld1 {v24.4s}, [x10]\n"

        "uaddw v9.8h, v26.8h, v9.8b\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"
        "uaddw v14.8h, v26.8h, v14.8b\n"
        "uaddw v15.8h, v26.8h, v15.8b\n"
        "uaddw v17.8h, v26.8h, v17.8b\n"
        "uaddw v18.8h, v26.8h, v18.8b\n"
        "uaddw v19.8h, v26.8h, v19.8b\n"

        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "f\n"
        "cmp w5, #1\n"
        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          // Load inputs for 3x4 input window which corresponds to a 1x2 output
          // window.
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v12.8b}, [x12]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v16.8b}, [x13]\n"
          "smlal v23.4s, v0.4h, v10.4h\n"
          "ld1 {v20.8b}, [x14]\n"
          "smlal2 v24.4s, v0.8h, v10.8h\n"
          "subs w5, w5, #2\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "cmp w5, #3\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "add %[input_ptr], %[input_ptr], %[input_width_increment]\n"
          "smlal v23.4s, v1.4h, v11.4h\n"
          "mov x12, %[input_ptr]\n"
          "smlal2 v24.4s, v1.8h, v11.8h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x13, %[input_ptr], %[input_row_size]\n"
          "smlal v23.4s, v2.4h, v12.4h\n"
          "add x14, x13, %[input_row_size]\n"
          "smlal2 v24.4s, v2.8h, v12.8h\n"
          "smlal v21.4s, v3.4h, v13.4h\n"
          "add x15, x14, %[input_row_size]\n"
          "smlal2 v22.4s, v3.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v3.4h, v14.4h\n"
          "smlal2 v24.4s, v3.8h, v14.8h\n"
          "smlal v21.4s, v4.4h, v14.4h\n"
          "smlal2 v22.4s, v4.8h, v14.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v4.4h, v15.4h\n"
          "smlal2 v24.4s, v4.8h, v15.8h\n"
          "smlal v21.4s, v5.4h, v15.4h\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "smlal2 v22.4s, v5.8h, v15.8h\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v5.4h, v16.4h\n"
          "smlal2 v24.4s, v5.8h, v16.8h\n"
          "smlal v21.4s, v6.4h, v17.4h\n"
          "smlal2 v22.4s, v6.8h, v17.8h\n"
          "ld1 {v17.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v18.4h\n"
          "smlal2 v22.4s, v7.8h, v18.8h\n"
          "ld1 {v18.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v19.4h\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"
          "smlal2 v22.4s, v8.8h, v19.8h\n"
          "ld1 {v19.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "st1 {v21.8b}, [%[output_ptr]], x3\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "st1 {v23.8b}, [%[output_ptr]], x3\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        // At this point, there will be one of 2 width or 1 width leftover,
        // not both.
        "cmp w5, #2\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        // Handle last two horizontal outputs if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER ":\n"
        "smlal v21.4s, v0.4h, v9.4h\n"
        "ld1 {v12.8b}, [x12], %[input_depth]\n"
        "smlal2 v22.4s, v0.8h, v9.8h\n"
        "ld1 {v16.8b}, [x13], %[input_depth]\n"
        "smlal v23.4s, v0.4h, v10.4h\n"
        "ld1 {v20.8b}, [x14], %[input_depth]\n"
        "smlal2 v24.4s, v0.8h, v10.8h\n"
        "smlal v21.4s, v1.4h, v10.4h\n"
        "smlal2 v22.4s, v1.8h, v10.8h\n"
        "smlal v23.4s, v1.4h, v11.4h\n"
        "smlal2 v24.4s, v1.8h, v11.8h\n"
        "smlal v21.4s, v2.4h, v11.4h\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "smlal2 v22.4s, v2.8h, v11.8h\n"
        "smlal v23.4s, v2.4h, v12.4h\n"
        "smlal2 v24.4s, v2.8h, v12.8h\n"
        "smlal v21.4s, v3.4h, v13.4h\n"
        "smlal2 v22.4s, v3.8h, v13.8h\n"
        "smlal v23.4s, v3.4h, v14.4h\n"
        "smlal2 v24.4s, v3.8h, v14.8h\n"
        "smlal v21.4s, v4.4h, v14.4h\n"
        "smlal2 v22.4s, v4.8h, v14.8h\n"
        "smlal v23.4s, v4.4h, v15.4h\n"
        "smlal2 v24.4s, v4.8h, v15.8h\n"
        "smlal v21.4s, v5.4h, v15.4h\n"
        "uaddw v16.8h, v26.8h, v16.8b\n"
        "smlal2 v22.4s, v5.8h, v15.8h\n"
        "smlal v23.4s, v5.4h, v16.4h\n"
        "smlal2 v24.4s, v5.8h, v16.8h\n"
        "smlal v21.4s, v6.4h, v17.4h\n"
        "smlal2 v22.4s, v6.8h, v17.8h\n"
        "smlal v23.4s, v6.4h, v18.4h\n"
        "smlal2 v24.4s, v6.8h, v18.8h\n"
        "smlal v21.4s, v7.4h, v18.4h\n"
        "smlal2 v22.4s, v7.8h, v18.8h\n"
        "smlal v23.4s, v7.4h, v19.4h\n"
        "smlal2 v24.4s, v7.8h, v19.8h\n"
        "smlal v21.4s, v8.4h, v19.4h\n"
        "uaddw v20.8h, v26.8h, v20.8b\n"
        "smlal2 v22.4s, v8.8h, v19.8h\n"
        "smlal v23.4s, v8.4h, v20.4h\n"
        "smlal2 v24.4s, v8.8h, v20.8h\n"

        "sqrdmulh v21.4s, v21.4s, v27.4s\n"
        "sqrdmulh v22.4s, v22.4s, v27.4s\n"
        "sqrdmulh v23.4s, v23.4s, v27.4s\n"
        "sqrdmulh v24.4s, v24.4s, v27.4s\n"
        "and v25.16b, v21.16b, v28.16b\n"
        "and v29.16b, v22.16b, v28.16b\n"
        "and v30.16b, v23.16b, v28.16b\n"
        "and v31.16b, v24.16b, v28.16b\n"
        "sshr v25.4s, v25.4s, #31\n"
        "sshr v29.4s, v29.4s, #31\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v21.4s, v21.4s, v25.4s\n"
        "sqadd v22.4s, v22.4s, v29.4s\n"
        "dup v29.4s, w2\n"
        "sqadd v23.4s, v23.4s, v30.4s\n"
        "dup v30.4s, w4\n"
        "sqadd v24.4s, v24.4s, v31.4s\n"
        "dup v31.4s, w0\n"
        "srshl v21.4s, v21.4s, v28.4s\n"
        "srshl v22.4s, v22.4s, v28.4s\n"
        "srshl v23.4s, v23.4s, v28.4s\n"
        "srshl v24.4s, v24.4s, v28.4s\n"
        "add v21.4s, v21.4s, v29.4s\n"
        "add v22.4s, v22.4s, v29.4s\n"
        "add v23.4s, v23.4s, v29.4s\n"
        "add v24.4s, v24.4s, v29.4s\n"
        "smax v21.4s, v21.4s, v30.4s\n"
        "smax v22.4s, v22.4s, v30.4s\n"
        "smax v23.4s, v23.4s, v30.4s\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smin v21.4s, v21.4s, v31.4s\n"
        "smin v22.4s, v22.4s, v31.4s\n"
        "smin v23.4s, v23.4s, v31.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "sqxtn v21.4h, v21.4s\n"
        "sqxtn v23.4h, v23.4s\n"
        "sqxtn2 v21.8h, v22.4s\n"
        "sqxtn2 v23.8h, v24.4s\n"
        "sqxtun v21.8b, v21.8h\n"
        "sqxtun v23.8b, v23.8h\n"
        "st1 {v21.8b}, [%[output_ptr]], x3\n"
        "st1 {v23.8b}, [%[output_ptr]], x3\n"
        "b " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        // Handle bottom right output if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER ":\n"
        "smlal v21.4s, v0.4h, v9.4h\n"
        "smlal2 v22.4s, v0.8h, v9.8h\n"
        "smlal v21.4s, v1.4h, v10.4h\n"
        "smlal2 v22.4s, v1.8h, v10.8h\n"
        "smlal v21.4s, v2.4h, v11.4h\n"
        "smlal2 v22.4s, v2.8h, v11.8h\n"
        "smlal v21.4s, v3.4h, v13.4h\n"
        "smlal2 v22.4s, v3.8h, v13.8h\n"
        "smlal v21.4s, v4.4h, v14.4h\n"
        "smlal2 v22.4s, v4.8h, v14.8h\n"
        "smlal v21.4s, v5.4h, v15.4h\n"
        "smlal2 v22.4s, v5.8h, v15.8h\n"
        "smlal v21.4s, v6.4h, v17.4h\n"
        "smlal2 v22.4s, v6.8h, v17.8h\n"
        "smlal v21.4s, v7.4h, v18.4h\n"
        "smlal2 v22.4s, v7.8h, v18.8h\n"
        "smlal v21.4s, v8.4h, v19.4h\n"
        "smlal2 v22.4s, v8.8h, v19.8h\n"

        "sqrdmulh v21.4s, v21.4s, v27.4s\n"
        "sqrdmulh v22.4s, v22.4s, v27.4s\n"
        "and v9.16b, v21.16b, v28.16b\n"
        "and v12.16b, v22.16b, v28.16b\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v12.4s, v12.4s, #31\n"
        "sqadd v21.4s, v21.4s, v9.4s\n"
        "sqadd v22.4s, v22.4s, v12.4s\n"
        "srshl v21.4s, v21.4s, v28.4s\n"
        "srshl v22.4s, v22.4s, v28.4s\n"
        "add v21.4s, v21.4s, v29.4s\n"
        "add v22.4s, v22.4s, v29.4s\n"
        "smax v21.4s, v21.4s, v30.4s\n"
        "smax v22.4s, v22.4s, v30.4s\n"
        "smin v21.4s, v21.4s, v31.4s\n"
        "smin v22.4s, v22.4s, v31.4s\n"
        "sqxtn v21.4h, v21.4s\n"
        "sqxtn2 v21.8h, v22.4s\n"
        "sqxtun v21.8b, v21.8h\n"
        "st1 {v21.8b}, [%[output_ptr]]\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"
    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment),
    [params_ptr] "r"(params_ptr)
    :
    // Clobbers.
    "cc", "memory",
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
    "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
  }
};

template <>
struct DepthwiseConvWindow<8, 2, 2> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         int64_t input_depth, int64_t input_row_size,
                         int32 output_window_height, int32 output_window_width,
                         const DepthwiseConvParams* params_ptr) {
    const int64_t input_width_increment = 4 * input_depth;
    const int64_t input_height_increment = 4 * input_row_size;
    const int64_t output_height_increment = 2 * params_ptr->output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "9"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "10"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "11"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time at stride 2, a 5x5 input
        //            window is required. To avoid register exhaustion, we load
        //            the first 2 rows of the 5x5 input window into registers
        //            v9--v18, and use the same registers to load the next 2
        //            rows, and finally v9--v13 to load the last row.
        //            Accumulators for all 2x2 outputs are reserved by registers
        //            v21-v22 (top left output), v23-v24 (top right output),
        //            v19-v20 (bottom left output), v25-v26 (bottom right
        //            output).
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time at stride 2, load inputs for
        //            a 1x2 (1 height, 2 width) output window (3x5 input
        //            window). Registers v9--v24 hold input values. Mul-add with
        //            accumulators v24--v27.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        // We use x9--x15 general purpose registers as they are caller-saved
        // temporary registers (see http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf).  // NOLINT
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "ldr w0, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "cmp %w[output_window_height], #2\n"
        "dup v28.8h, w0\n"
        "neg w9, w9\n"
        "ldr w1, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.4s, w9\n"
        "ldr w2, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w1\n"
        "ldr w3, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "dup v29.4s, w2\n"
        "ldr w4, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w3\n"
        "ldr x5, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "dup v31.4s, w4\n"
        "ldr x19, [%[params_ptr], #" STR(OFFSET_OUTPUT_ROW_SIZE) "]\n"
        "ldr w20, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"

        // Load filters and add offsets.
        "add x10, %[bias_ptr], #16\n"
        "ld1 {v0.8b}, [%[filter_ptr]], x5\n"
        "dup v9.8h, w20\n"
        "ld1 {v1.8b}, [%[filter_ptr]], x5\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], x5\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], x5\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], x5\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], x5\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], x5\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], x5\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]]\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // Load the first two rows of the 5x5 input window, then reuse the
          // same registers to load subsequent rows as they become available.
          "mov x11, %[input_ptr]\n"
          "mov x12, x11\n"
          "add x13, x12, %[input_row_size]\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "mov w14, %w[output_window_width]\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          // The height 2 / width 2 loop loads an extra 1 output horizontally in
          // anticipation for the next iteration. Make sure
          // |output_window_width| is large enough to handle the additional
          // load, otherwise jump to the appropriate label to handle smaller
          // widths.
          "cmp w14, #2\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x15, x13, %[input_row_size]\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "mov x6, %[output_ptr]\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "add x7, %[output_ptr], x19\n"
          "ld1 {v16.8b}, [x13], %[input_depth]\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "ld1 {v22.4s}, [x10]\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "ld1 {v24.4s}, [x10]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "ld1 {v19.4s}, [%[bias_ptr]]\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "ld1 {v20.4s}, [x10]\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "ld1 {v25.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v26.4s}, [x10]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"

          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "f\n"
          "cmp w14, #1\n"
          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            "smlal v21.4s, v0.4h, v9.4h\n"
            "ld1 {v12.8b}, [x12], %[input_depth]\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "ld1 {v13.8b}, [x12]\n"
            "add x12, x15, %[input_row_size]\n"
            "smlal v23.4s, v0.4h, v11.4h\n"
            "ld1 {v17.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v0.8h, v11.8h\n"
            "ld1 {v18.8b}, [x13]\n"
            "add x13, x12, %[input_row_size]\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "ld1 {v9.8b}, [x15], %[input_depth]\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x15], %[input_depth]\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "ld1 {v11.8b}, [x15], %[input_depth]\n"
            "smlal v21.4s, v3.4h, v14.4h\n"
            "smlal2 v22.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "subs w14, w14, #2\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "cmp w14, #3\n"
            "smlal v21.4s, v4.4h, v15.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v22.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x12], %[input_depth]\n"
            "smlal v21.4s, v5.4h, v16.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v22.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v1.4h, v12.4h\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "smlal2 v24.4s, v1.8h, v12.8h\n"
            "ld1 {v12.8b}, [x15], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v13.4h\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "smlal2 v24.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x15]\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "ld1 {v17.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v18.4h\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "smlal2 v24.4s, v5.8h, v18.8h\n"
            "ld1 {v18.8b}, [x12]\n"

            "smlal v21.4s, v6.4h, v9.4h\n"
            "smlal2 v22.4s, v6.8h, v9.8h\n"
            "smlal v19.4s, v0.4h, v9.4h\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "smlal2 v20.4s, v0.8h, v9.8h\n"
            "ld1 {v9.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v6.4h, v11.4h\n"
            "smlal2 v24.4s, v6.8h, v11.8h\n"
            "smlal v21.4s, v7.4h, v10.4h\n"
            "smlal2 v22.4s, v7.8h, v10.8h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal v19.4s, v1.4h, v10.4h\n"
            "smlal2 v20.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v7.4h, v12.4h\n"
            "smlal2 v24.4s, v7.8h, v12.8h\n"
            "smlal v25.4s, v1.4h, v12.4h\n"
            "smlal2 v26.4s, v1.8h, v12.8h\n"
            "smlal v21.4s, v8.4h, v11.4h\n"
            "smlal2 v22.4s, v8.8h, v11.8h\n"
            "add x11, x11, %[input_width_increment]\n"
            "smlal v19.4s, v2.4h, v11.4h\n"
            "mov x12, x11\n"
            "smlal2 v20.4s, v2.8h, v11.8h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal v25.4s, v0.4h, v11.4h\n"
            "smlal2 v26.4s, v0.8h, v11.8h\n"
            "ld1 {v11.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v13.4h\n"
            "ld1 {v12.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v13.8h\n"
            "smlal v25.4s, v2.4h, v13.4h\n"
            "smlal2 v26.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x13]\n"
            "add x13, x12, %[input_row_size]\n"
            "add x15, x13, %[input_row_size]\n"

            "dup v28.4s, w9\n"
            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v27.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v27.4s\n"
            "dup v27.4s, w1\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w3\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w4\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "dup v28.8h, w0\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x5\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "st1 {v23.8b}, [x6], x5\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"

            "smlal v19.4s, v6.4h, v9.4h\n"
            "smlal2 v20.4s, v6.8h, v9.8h\n"
            "ld1 {v9.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v6.4h, v11.4h\n"
            "smlal2 v26.4s, v6.8h, v11.8h\n"
            "smlal v19.4s, v7.4h, v10.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v20.4s, v7.8h, v10.8h\n"
            "ld1 {v10.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v7.4h, v12.4h\n"
            "smlal2 v26.4s, v7.8h, v12.8h\n"
            "smlal v19.4s, v8.4h, v11.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v20.4s, v8.8h, v11.8h\n"
            "ld1 {v11.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v8.4h, v13.4h\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "smlal2 v26.4s, v8.8h, v13.8h\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"
            "smlal v19.4s, v3.4h, v14.4h\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "smlal2 v20.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v3.4h, v16.4h\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "smlal2 v26.4s, v3.8h, v16.8h\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "smlal v19.4s, v4.4h, v15.4h\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "smlal2 v20.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v4.4h, v17.4h\n"
            "smlal2 v26.4s, v4.8h, v17.8h\n"
            "smlal v19.4s, v5.4h, v16.4h\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "smlal2 v20.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v5.4h, v18.4h\n"
            "smlal2 v26.4s, v5.8h, v18.8h\n"

            "dup v28.4s, w9\n"
            "sqrdmulh v19.4s, v19.4s, v27.4s\n"
            "sqrdmulh v20.4s, v20.4s, v27.4s\n"
            "sqrdmulh v25.4s, v25.4s, v27.4s\n"
            "sqrdmulh v26.4s, v26.4s, v27.4s\n"
            "and v27.16b, v19.16b, v28.16b\n"
            "and v29.16b, v20.16b, v28.16b\n"
            "and v30.16b, v25.16b, v28.16b\n"
            "and v31.16b, v26.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v19.4s, v19.4s, v27.4s\n"
            "dup v27.4s, w1\n"
            "sqadd v20.4s, v20.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v25.4s, v25.4s, v30.4s\n"
            "dup v30.4s, w3\n"
            "sqadd v26.4s, v26.4s, v31.4s\n"
            "dup v31.4s, w4\n"
            "srshl v19.4s, v19.4s, v28.4s\n"
            "srshl v20.4s, v20.4s, v28.4s\n"
            "srshl v25.4s, v25.4s, v28.4s\n"
            "srshl v26.4s, v26.4s, v28.4s\n"
            "dup v28.8h, w0\n"
            "add v19.4s, v19.4s, v29.4s\n"
            "add v20.4s, v20.4s, v29.4s\n"
            "add v25.4s, v25.4s, v29.4s\n"
            "add v26.4s, v26.4s, v29.4s\n"
            "smax v19.4s, v19.4s, v30.4s\n"
            "smax v20.4s, v20.4s, v30.4s\n"
            "smax v25.4s, v25.4s, v30.4s\n"
            "smax v26.4s, v26.4s, v30.4s\n"
            "smin v19.4s, v19.4s, v31.4s\n"
            "smin v20.4s, v20.4s, v31.4s\n"
            "smin v25.4s, v25.4s, v31.4s\n"
            "smin v26.4s, v26.4s, v31.4s\n"
            "sqxtn v19.4h, v19.4s\n"
            "sqxtn v25.4h, v25.4s\n"
            "sqxtn2 v19.8h, v20.4s\n"
            "ld1 {v20.4s}, [x10]\n"
            "sqxtn2 v25.8h, v26.4s\n"
            "ld1 {v26.4s}, [x10]\n"
            "sqxtun v19.8b, v19.8h\n"
            "sqxtun v25.8b, v25.8h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "st1 {v19.8b}, [x7], x5\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "st1 {v25.8b}, [x7], x5\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "ld1 {v19.4s}, [%[bias_ptr]]\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "ld1 {v25.4s}, [%[bias_ptr]]\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          // At this point, there will be one of 2 width or 1 width leftover,
          // not both.
          "cmp w14, #2\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          // Handle last 2 columns if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v12.8b}, [x12], %[input_depth]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v13.8b}, [x12]\n"
          "add x12, x15, %[input_row_size]\n"
          "smlal v23.4s, v0.4h, v11.4h\n"
          "ld1 {v17.8b}, [x13], %[input_depth]\n"
          "smlal2 v24.4s, v0.8h, v11.8h\n"
          "ld1 {v18.8b}, [x13]\n"
          "add x13, x12, %[input_row_size]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x15], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v3.4h, v14.4h\n"
          "smlal2 v22.4s, v3.8h, v14.8h\n"
          "ld1 {v14.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v3.4h, v16.4h\n"
          "smlal2 v24.4s, v3.8h, v16.8h\n"
          "smlal v21.4s, v4.4h, v15.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v22.4s, v4.8h, v15.8h\n"
          "ld1 {v15.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v5.4h, v16.4h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal2 v22.4s, v5.8h, v16.8h\n"
          "ld1 {v16.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v1.4h, v12.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v24.4s, v1.8h, v12.8h\n"
          "ld1 {v12.8b}, [x15], %[input_depth]\n"
          "smlal v23.4s, v2.4h, v13.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v24.4s, v2.8h, v13.8h\n"
          "ld1 {v13.8b}, [x15]\n"
          "smlal v23.4s, v4.4h, v17.4h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "smlal2 v24.4s, v4.8h, v17.8h\n"
          "ld1 {v17.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v5.4h, v18.4h\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "smlal2 v24.4s, v5.8h, v18.8h\n"
          "ld1 {v18.8b}, [x12]\n"

          "smlal v21.4s, v6.4h, v9.4h\n"
          "smlal2 v22.4s, v6.8h, v9.8h\n"
          "smlal v19.4s, v0.4h, v9.4h\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "smlal2 v20.4s, v0.8h, v9.8h\n"
          "ld1 {v9.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v6.4h, v11.4h\n"
          "smlal2 v24.4s, v6.8h, v11.8h\n"
          "smlal v21.4s, v7.4h, v10.4h\n"
          "smlal2 v22.4s, v7.8h, v10.8h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal v19.4s, v1.4h, v10.4h\n"
          "smlal2 v20.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v7.4h, v12.4h\n"
          "smlal2 v24.4s, v7.8h, v12.8h\n"
          "smlal v25.4s, v1.4h, v12.4h\n"
          "smlal2 v26.4s, v1.8h, v12.8h\n"
          "smlal v21.4s, v8.4h, v11.4h\n"
          "smlal2 v22.4s, v8.8h, v11.8h\n"
          "smlal v19.4s, v2.4h, v11.4h\n"
          "smlal2 v20.4s, v2.8h, v11.8h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal v25.4s, v0.4h, v11.4h\n"
          "smlal2 v26.4s, v0.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v8.4h, v13.4h\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "smlal2 v24.4s, v8.8h, v13.8h\n"
          "smlal v25.4s, v2.4h, v13.4h\n"
          "smlal2 v26.4s, v2.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13]\n"

          "dup v28.4s, w9\n"
          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v27.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v27.4s, v27.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v27.4s\n"
          "dup v27.4s, w1\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v21.8b}, [x6], x5\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "st1 {v23.8b}, [x6]\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"

          "smlal v19.4s, v6.4h, v9.4h\n"
          "smlal2 v20.4s, v6.8h, v9.8h\n"
          "smlal v25.4s, v6.4h, v11.4h\n"
          "smlal2 v26.4s, v6.8h, v11.8h\n"
          "smlal v19.4s, v7.4h, v10.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v20.4s, v7.8h, v10.8h\n"
          "smlal v25.4s, v7.4h, v12.4h\n"
          "smlal2 v26.4s, v7.8h, v12.8h\n"
          "smlal v19.4s, v8.4h, v11.4h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal2 v20.4s, v8.8h, v11.8h\n"
          "smlal v25.4s, v8.4h, v13.4h\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "smlal2 v26.4s, v8.8h, v13.8h\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "smlal v19.4s, v3.4h, v14.4h\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "smlal2 v20.4s, v3.8h, v14.8h\n"
          "smlal v25.4s, v3.4h, v16.4h\n"
          "smlal2 v26.4s, v3.8h, v16.8h\n"
          "smlal v19.4s, v4.4h, v15.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v20.4s, v4.8h, v15.8h\n"
          "smlal v25.4s, v4.4h, v17.4h\n"
          "smlal2 v26.4s, v4.8h, v17.8h\n"
          "smlal v19.4s, v5.4h, v16.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v20.4s, v5.8h, v16.8h\n"
          "smlal v25.4s, v5.4h, v18.4h\n"
          "smlal2 v26.4s, v5.8h, v18.8h\n"

          "dup v28.4s, w9\n"
          "sqrdmulh v19.4s, v19.4s, v27.4s\n"
          "sqrdmulh v20.4s, v20.4s, v27.4s\n"
          "sqrdmulh v25.4s, v25.4s, v27.4s\n"
          "sqrdmulh v26.4s, v26.4s, v27.4s\n"
          "and v27.16b, v19.16b, v28.16b\n"
          "and v29.16b, v20.16b, v28.16b\n"
          "and v30.16b, v25.16b, v28.16b\n"
          "and v31.16b, v26.16b, v28.16b\n"
          "sshr v27.4s, v27.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v19.4s, v19.4s, v27.4s\n"
          "dup v27.4s, w1\n"
          "sqadd v20.4s, v20.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v25.4s, v25.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v26.4s, v26.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v19.4s, v19.4s, v28.4s\n"
          "srshl v20.4s, v20.4s, v28.4s\n"
          "srshl v25.4s, v25.4s, v28.4s\n"
          "srshl v26.4s, v26.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "add v19.4s, v19.4s, v29.4s\n"
          "add v20.4s, v20.4s, v29.4s\n"
          "add v25.4s, v25.4s, v29.4s\n"
          "add v26.4s, v26.4s, v29.4s\n"
          "smax v19.4s, v19.4s, v30.4s\n"
          "smax v20.4s, v20.4s, v30.4s\n"
          "smax v25.4s, v25.4s, v30.4s\n"
          "smax v26.4s, v26.4s, v30.4s\n"
          "smin v19.4s, v19.4s, v31.4s\n"
          "smin v20.4s, v20.4s, v31.4s\n"
          "smin v25.4s, v25.4s, v31.4s\n"
          "smin v26.4s, v26.4s, v31.4s\n"
          "sqxtn v19.4h, v19.4s\n"
          "sqxtn v25.4h, v25.4s\n"
          "sqxtn2 v19.8h, v20.4s\n"
          "sqxtn2 v25.8h, v26.4s\n"
          "sqxtun v19.8b, v19.8h\n"
          "sqxtun v25.8b, v25.8h\n"
          "st1 {v19.8b}, [x7], x5\n"
          "st1 {v25.8b}, [x7]\n"
          "b " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          // Handle last column if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER ":\n"
          // Registers v9, v10, v11, v14, v15, and v16 have already been loaded
          // with the correct values at this point. This corresponds to the
          // first two input rows of the top left output. Now load the last
          // input row for this output. Once these inputs are no longer needed,
          // load the input rows for the bottom left output.
          "add x12, x15, %[input_row_size]\n"
          "add x13, x12, %[input_row_size]\n"

          "ld1 {v12.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v13.8b}, [x15], %[input_depth]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v17.8b}, [x15]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x12]\n"
          "smlal v21.4s, v3.4h, v14.4h\n"
          "smlal2 v22.4s, v3.8h, v14.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v21.4s, v4.4h, v15.4h\n"
          "smlal2 v22.4s, v4.8h, v15.8h\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "smlal v21.4s, v5.4h, v16.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v22.4s, v5.8h, v16.8h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "ld1 {v16.8b}, [x13]\n"

          "smlal v21.4s, v6.4h, v12.4h\n"
          "smlal2 v22.4s, v6.8h, v12.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v7.4h, v13.4h\n"
          "smlal2 v22.4s, v7.8h, v13.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v2.4h, v17.4h\n"
          "smlal2 v24.4s, v2.8h, v17.8h\n"

          "dup v26.4s, w9\n"
          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "and v18.16b, v21.16b, v26.16b\n"
          "and v19.16b, v22.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v21.4s, v21.4s, v18.4s\n"
          "sqadd v22.4s, v22.4s, v19.4s\n"
          "srshl v21.4s, v21.4s, v26.4s\n"
          "srshl v22.4s, v22.4s, v26.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v21.8b}, [x6]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"

          "smlal v23.4s, v3.4h, v9.4h\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "smlal2 v24.4s, v3.8h, v9.8h\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "smlal v23.4s, v4.4h, v10.4h\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "smlal2 v24.4s, v4.8h, v10.8h\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "smlal v23.4s, v5.4h, v11.4h\n"
          "smlal2 v24.4s, v5.8h, v11.8h\n"

          "smlal v23.4s, v6.4h, v14.4h\n"
          "smlal2 v24.4s, v6.8h, v14.8h\n"
          "smlal v23.4s, v7.4h, v15.4h\n"
          "smlal2 v24.4s, v7.8h, v15.8h\n"
          "smlal v23.4s, v8.4h, v16.4h\n"
          "smlal2 v24.4s, v8.8h, v16.8h\n"

          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v18.16b, v23.16b, v26.16b\n"
          "and v19.16b, v24.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v23.4s, v23.4s, v18.4s\n"
          "sqadd v24.4s, v24.4s, v19.4s\n"
          "srshl v23.4s, v23.4s, v26.4s\n"
          "srshl v24.4s, v24.4s, v26.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v23.8b}, [x7]\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        "mov x11, %[input_ptr]\n"
        "mov x12, x11\n"
        "add x13, x12, %[input_row_size]\n"
        "ld1 {v9.8b}, [x12], %[input_depth]\n"
        "add x15, x13, %[input_row_size]\n"
        "ld1 {v10.8b}, [x12], %[input_depth]\n"
        "mov x6, %[output_ptr]\n"
        "ld1 {v11.8b}, [x12], %[input_depth]\n"
        "mov w14, %w[output_window_width]\n"
        // The height 1 / width 2 loop loads an extra 1x1 output in anticipation
        // for the next iteration. Make sure |output_window_width| is large
        // enough to handle the additional load, otherwise jump to the
        // appropriate label to handle smaller widths.
        "cmp w14, #2\n"
        "ld1 {v12.8b}, [x13], %[input_depth]\n"
        "ld1 {v13.8b}, [x13], %[input_depth]\n"
        "ld1 {v14.8b}, [x13], %[input_depth]\n"
        "ld1 {v15.8b}, [x15], %[input_depth]\n"
        "ld1 {v16.8b}, [x15], %[input_depth]\n"
        "ld1 {v17.8b}, [x15], %[input_depth]\n"

        "uaddw v9.8h, v28.8h, v9.8b\n"
        "ld1 {v24.4s}, [%[bias_ptr]]\n"
        "uaddw v10.8h, v28.8h, v10.8b\n"
        "ld1 {v25.4s}, [x10]\n"
        "uaddw v11.8h, v28.8h, v11.8b\n"
        "ld1 {v26.4s}, [%[bias_ptr]]\n"
        "ld1 {v27.4s}, [x10]\n"
        "uaddw v12.8h, v28.8h, v12.8b\n"
        "uaddw v13.8h, v28.8h, v13.8b\n"
        "uaddw v14.8h, v28.8h, v14.8b\n"
        "uaddw v15.8h, v28.8h, v15.8b\n"
        "uaddw v16.8h, v28.8h, v16.8b\n"
        "uaddw v17.8h, v28.8h, v17.8b\n"

        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "f\n"
        "cmp w14, #1\n"
        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          "smlal v24.4s, v0.4h, v9.4h\n"
          "ld1 {v18.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v0.8h, v9.8h\n"
          "ld1 {v19.8b}, [x12]\n"
          "smlal v26.4s, v0.4h, v11.4h\n"
          "ld1 {v20.8b}, [x13], %[input_depth]\n"
          "smlal2 v27.4s, v0.8h, v11.8h\n"
          "ld1 {v21.8b}, [x13]\n"
          "smlal v24.4s, v1.4h, v10.4h\n"
          "ld1 {v22.8b}, [x15], %[input_depth]\n"
          "smlal2 v25.4s, v1.8h, v10.8h\n"
          "ld1 {v23.8b}, [x15]\n"
          "smlal v24.4s, v2.4h, v11.4h\n"
          "subs w14, w14, #2\n"
          "smlal2 v25.4s, v2.8h, v11.8h\n"
          "cmp w14, #3\n"
          "smlal v24.4s, v3.4h, v12.4h\n"
          "add x11, x11, %[input_width_increment]\n"
          "smlal2 v25.4s, v3.8h, v12.8h\n"
          "mov x12, x11\n"
          "smlal v26.4s, v3.4h, v14.4h\n"
          "add x13, x12, %[input_row_size]\n"
          "smlal2 v27.4s, v3.8h, v14.8h\n"
          "add x15, x13, %[input_row_size]\n"
          "smlal v24.4s, v4.4h, v13.4h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v4.8h, v13.8h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "smlal v24.4s, v5.4h, v14.4h\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v5.8h, v14.8h\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "smlal v24.4s, v6.4h, v15.4h\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "smlal2 v25.4s, v6.8h, v15.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v26.4s, v6.4h, v17.4h\n"
          "ld1 {v15.8b}, [x15], %[input_depth]\n"
          "smlal2 v27.4s, v6.8h, v17.8h\n"
          "smlal v24.4s, v7.4h, v16.4h\n"
          "smlal2 v25.4s, v7.8h, v16.8h\n"
          "ld1 {v16.8b}, [x15], %[input_depth]\n"
          "smlal v24.4s, v8.4h, v17.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v25.4s, v8.8h, v17.8h\n"
          "ld1 {v17.8b}, [x15], %[input_depth]\n"
          "uaddw v19.8h, v28.8h, v19.8b\n"

          "smlal v26.4s, v1.4h, v18.4h\n"
          "uaddw v20.8h, v28.8h, v20.8b\n"
          "smlal2 v27.4s, v1.8h, v18.8h\n"
          "smlal v26.4s, v2.4h, v19.4h\n"
          "uaddw v21.8h, v28.8h, v21.8b\n"
          "smlal2 v27.4s, v2.8h, v19.8h\n"
          "smlal v26.4s, v4.4h, v20.4h\n"
          "smlal v26.4s, v5.4h, v21.4h\n"
          "smlal2 v27.4s, v4.8h, v20.8h\n"
          "uaddw v22.8h, v28.8h, v22.8b\n"
          "smlal2 v27.4s, v5.8h, v21.8h\n"
          "uaddw v23.8h, v28.8h, v23.8b\n"
          "smlal v26.4s, v7.4h, v22.4h\n"
          "smlal2 v27.4s, v7.8h, v22.8h\n"
          "smlal v26.4s, v8.4h, v23.4h\n"
          "smlal2 v27.4s, v8.8h, v23.8h\n"

          "dup v28.4s, w1\n"
          "dup v29.4s, w9\n"
          "sqrdmulh v24.4s, v24.4s, v28.4s\n"
          "sqrdmulh v25.4s, v25.4s, v28.4s\n"
          "sqrdmulh v26.4s, v26.4s, v28.4s\n"
          "sqrdmulh v27.4s, v27.4s, v28.4s\n"
          "dup v28.4s, w2\n"
          "and v30.16b, v24.16b, v29.16b\n"
          "and v31.16b, v25.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v24.4s, v24.4s, v30.4s\n"
          "sqadd v25.4s, v25.4s, v31.4s\n"
          "and v30.16b, v26.16b, v29.16b\n"
          "and v31.16b, v27.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v26.4s, v26.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v27.4s, v27.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v24.4s, v24.4s, v29.4s\n"
          "srshl v25.4s, v25.4s, v29.4s\n"
          "srshl v26.4s, v26.4s, v29.4s\n"
          "srshl v27.4s, v27.4s, v29.4s\n"
          "add v24.4s, v24.4s, v28.4s\n"
          "add v25.4s, v25.4s, v28.4s\n"
          "add v26.4s, v26.4s, v28.4s\n"
          "add v27.4s, v27.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smax v25.4s, v25.4s, v30.4s\n"
          "smax v26.4s, v26.4s, v30.4s\n"
          "smax v27.4s, v27.4s, v30.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "smin v25.4s, v25.4s, v31.4s\n"
          "smin v26.4s, v26.4s, v31.4s\n"
          "smin v27.4s, v27.4s, v31.4s\n"
          "sqxtn v24.4h, v24.4s\n"
          "sqxtn v26.4h, v26.4s\n"
          "sqxtn2 v24.8h, v25.4s\n"
          "ld1 {v25.4s}, [x10]\n"
          "sqxtn2 v26.8h, v27.4s\n"
          "ld1 {v27.4s}, [x10]\n"
          "sqxtun v24.8b, v24.8h\n"
          "sqxtun v26.8b, v26.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v24.8b}, [x6], x5\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "st1 {v26.8b}, [x6], x5\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "ld1 {v24.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v26.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        // At this point, there will be one of 2 width or 1 width leftover,
        // not both.
        "cmp w14, #2\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        // Handle last two horizontal outputs if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER ":\n"
        "smlal v24.4s, v0.4h, v9.4h\n"
        "ld1 {v18.8b}, [x12], %[input_depth]\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "ld1 {v19.8b}, [x12]\n"
        "smlal v26.4s, v0.4h, v11.4h\n"
        "ld1 {v20.8b}, [x13], %[input_depth]\n"
        "smlal2 v27.4s, v0.8h, v11.8h\n"
        "ld1 {v21.8b}, [x13]\n"
        "smlal v24.4s, v1.4h, v10.4h\n"
        "ld1 {v22.8b}, [x15], %[input_depth]\n"
        "smlal2 v25.4s, v1.8h, v10.8h\n"
        "ld1 {v23.8b}, [x15]\n"
        "smlal v24.4s, v2.4h, v11.4h\n"
        "smlal2 v25.4s, v2.8h, v11.8h\n"
        "smlal v24.4s, v3.4h, v12.4h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "smlal v26.4s, v3.4h, v14.4h\n"
        "smlal2 v27.4s, v3.8h, v14.8h\n"
        "smlal v24.4s, v4.4h, v13.4h\n"
        "smlal2 v25.4s, v4.8h, v13.8h\n"
        "smlal v24.4s, v5.4h, v14.4h\n"
        "smlal2 v25.4s, v5.8h, v14.8h\n"
        "smlal v24.4s, v6.4h, v15.4h\n"
        "smlal2 v25.4s, v6.8h, v15.8h\n"
        "smlal v26.4s, v6.4h, v17.4h\n"
        "smlal2 v27.4s, v6.8h, v17.8h\n"
        "smlal v24.4s, v7.4h, v16.4h\n"
        "smlal2 v25.4s, v7.8h, v16.8h\n"
        "smlal v24.4s, v8.4h, v17.4h\n"
        "uaddw v18.8h, v28.8h, v18.8b\n"
        "smlal2 v25.4s, v8.8h, v17.8h\n"
        "uaddw v19.8h, v28.8h, v19.8b\n"

        "smlal v26.4s, v1.4h, v18.4h\n"
        "uaddw v20.8h, v28.8h, v20.8b\n"
        "smlal2 v27.4s, v1.8h, v18.8h\n"
        "smlal v26.4s, v2.4h, v19.4h\n"
        "uaddw v21.8h, v28.8h, v21.8b\n"
        "smlal2 v27.4s, v2.8h, v19.8h\n"
        "smlal v26.4s, v4.4h, v20.4h\n"
        "smlal v26.4s, v5.4h, v21.4h\n"
        "smlal2 v27.4s, v4.8h, v20.8h\n"
        "uaddw v22.8h, v28.8h, v22.8b\n"
        "smlal2 v27.4s, v5.8h, v21.8h\n"
        "uaddw v23.8h, v28.8h, v23.8b\n"
        "smlal v26.4s, v7.4h, v22.4h\n"
        "smlal2 v27.4s, v7.8h, v22.8h\n"
        "smlal v26.4s, v8.4h, v23.4h\n"
        "smlal2 v27.4s, v8.8h, v23.8h\n"

        "dup v28.4s, w1\n"
        "dup v29.4s, w9\n"
        "sqrdmulh v24.4s, v24.4s, v28.4s\n"
        "sqrdmulh v25.4s, v25.4s, v28.4s\n"
        "sqrdmulh v26.4s, v26.4s, v28.4s\n"
        "sqrdmulh v27.4s, v27.4s, v28.4s\n"
        "dup v28.4s, w2\n"
        "and v30.16b, v24.16b, v29.16b\n"
        "and v31.16b, v25.16b, v29.16b\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v24.4s, v24.4s, v30.4s\n"
        "sqadd v25.4s, v25.4s, v31.4s\n"
        "and v30.16b, v26.16b, v29.16b\n"
        "and v31.16b, v27.16b, v29.16b\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v26.4s, v26.4s, v30.4s\n"
        "dup v30.4s, w3\n"
        "sqadd v27.4s, v27.4s, v31.4s\n"
        "dup v31.4s, w4\n"
        "srshl v24.4s, v24.4s, v29.4s\n"
        "srshl v25.4s, v25.4s, v29.4s\n"
        "srshl v26.4s, v26.4s, v29.4s\n"
        "srshl v27.4s, v27.4s, v29.4s\n"
        "add v24.4s, v24.4s, v28.4s\n"
        "add v25.4s, v25.4s, v28.4s\n"
        "add v26.4s, v26.4s, v28.4s\n"
        "add v27.4s, v27.4s, v28.4s\n"
        "dup v28.8h, w0\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smax v25.4s, v25.4s, v30.4s\n"
        "smax v26.4s, v26.4s, v30.4s\n"
        "smax v27.4s, v27.4s, v30.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "smin v25.4s, v25.4s, v31.4s\n"
        "smin v26.4s, v26.4s, v31.4s\n"
        "smin v27.4s, v27.4s, v31.4s\n"
        "sqxtn v24.4h, v24.4s\n"
        "sqxtn v26.4h, v26.4s\n"
        "sqxtn2 v24.8h, v25.4s\n"
        "sqxtn2 v26.8h, v27.4s\n"
        "sqxtun v24.8b, v24.8h\n"
        "sqxtun v26.8b, v26.8h\n"
        "st1 {v24.8b}, [x6], x5\n"
        "st1 {v26.8b}, [x6]\n"
        "b " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        // Handle bottom right output if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER ":\n"
        "dup v26.4s, w9\n"
        "dup v27.4s, w1\n"
        "dup v29.4s, w2\n"

        "smlal v24.4s, v0.4h, v9.4h\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "smlal v24.4s, v1.4h, v10.4h\n"
        "smlal2 v25.4s, v1.8h, v10.8h\n"
        "smlal v24.4s, v2.4h, v11.4h\n"
        "smlal2 v25.4s, v2.8h, v11.8h\n"
        "smlal v24.4s, v3.4h, v12.4h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "smlal v24.4s, v4.4h, v13.4h\n"
        "smlal2 v25.4s, v4.8h, v13.8h\n"
        "smlal v24.4s, v5.4h, v14.4h\n"
        "smlal2 v25.4s, v5.8h, v14.8h\n"
        "smlal v24.4s, v6.4h, v15.4h\n"
        "smlal2 v25.4s, v6.8h, v15.8h\n"
        "smlal v24.4s, v7.4h, v16.4h\n"
        "smlal2 v25.4s, v7.8h, v16.8h\n"
        "smlal v24.4s, v8.4h, v17.4h\n"
        "smlal2 v25.4s, v8.8h, v17.8h\n"

        "sqrdmulh v24.4s, v24.4s, v27.4s\n"
        "sqrdmulh v25.4s, v25.4s, v27.4s\n"
        "and v18.16b, v24.16b, v26.16b\n"
        "and v19.16b, v25.16b, v26.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v24.4s, v24.4s, v18.4s\n"
        "sqadd v25.4s, v25.4s, v19.4s\n"
        "srshl v24.4s, v24.4s, v26.4s\n"
        "srshl v25.4s, v25.4s, v26.4s\n"
        "add v24.4s, v24.4s, v29.4s\n"
        "add v25.4s, v25.4s, v29.4s\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smax v25.4s, v25.4s, v30.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "smin v25.4s, v25.4s, v31.4s\n"
        "sqxtn v24.4h, v24.4s\n"
        "sqxtn2 v24.8h, v25.4s\n"
        "sqxtun v24.8b, v24.8h\n"
        "st1 {v24.8b}, [x6]\n"

        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"
    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment),
    [params_ptr] "r"(params_ptr)
    :
    // Clobbers.
    "cc", "memory",
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
    "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x19", "x20");
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
  }
};

enum class EdgeType { kCorner, kHorizontal, kVertical, kCenter };

template <EdgeType kEdgeType, int kPadWidth, int kPadHeight>
struct DepthwiseConvPartial {};

template <>
struct DepthwiseConvPartial<EdgeType::kCenter, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 1x1 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the 1x1 input and filter values.
        "ld1 {v8.8b}, [%[input_ptr]], #8\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w10\n"
        "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
        "cmp x11, #16\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w10, w10\n"
        "dup v29.4s, w10\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w10\n"
        "dup v25.8h, w9\n"

        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v0.8h, v25.8h, v0.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "smlal v16.4s, v0.4h, v8.4h\n"
          "subs x11, x11, #8\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [%[input_ptr]], #8\n"
          "cmp x11, #16\n"
          "ld1 {v0.8b}, [%[filter_ptr]], #8\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v0.8h, v25.8h, v0.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"

        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v8", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31",
        // We use these general-purpose registers.
        "x9", "x10", "x11");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kCorner, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 2x2 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 2x2 input and
        // filter values.

        // Load input and filter values.
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "ldr x9, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "cmp x15, #16\n"
        "add x12, %[input_ptr], x15\n"
        "add x13, %[input_ptr], x9\n"
        "ld1 {v8.8b}, [%[input_ptr]], #8\n"
        "add x14, x13, x15\n"
        "ld1 {v9.8b}, [x12], #8\n"
        "ldr x6, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"

        "add x9, %[filter_ptr], x15\n"
        "ld1 {v10.8b}, [x13], #8\n"
        "add x10, %[filter_ptr], x6\n"
        "ld1 {v11.8b}, [x14], #8\n"
        "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
        "add x11, x10, x15\n"
        "ld1 {v1.8b}, [x9], #8\n"
        "ld1 {v2.8b}, [x10], #8\n"
        "ld1 {v3.8b}, [x11], #8\n"

        // Load constants.
        "ldr w6, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w7\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w7, w7\n"
        "dup v29.4s, w7\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w7\n"
        "dup v25.8h, w6\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "smlal v16.4s, v0.4h, v8.4h\n"
          "subs x15, x15, #8\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [%[input_ptr]], #8\n"
          "cmp x15, #16\n"
          "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "ld1 {v9.8b}, [x12], #8\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "ld1 {v1.8b}, [x9], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], #8\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v2.8b}, [x10], #8\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x14], #8\n"
          "ld1 {v3.8b}, [x11], #8\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"

        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
        "v19", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "x6", "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 2x3 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 2x3 input and
        // filter values.

        // Load input and filter values.
        "ldr x7, [%[params_ptr], #" STR(OFFSET_INPUT_DEPTH) "]\n"
        "mov x12, %[input_ptr]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "mov x9, %[filter_ptr]\n"
        "ldr x14, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"
        "add x13, x12, x11\n"
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"

        "ld1 {v8.8b}, [x12], x7\n"
        "add x10, x9, x14\n"
        "ld1 {v9.8b}, [x12], x7\n"
        "cmp x15, #16\n"
        "ld1 {v10.8b}, [x12]\n"
        "add %[input_ptr], %[input_ptr], #8\n"
        "ld1 {v11.8b}, [x13], x7\n"
        "add %[filter_ptr], %[filter_ptr], #8\n"
        "ld1 {v12.8b}, [x13], x7\n"
        "ld1 {v13.8b}, [x13]\n"

        "ld1 {v0.8b}, [x9], x7\n"
        "ld1 {v1.8b}, [x9], x7\n"
        "ld1 {v2.8b}, [x9]\n"
        "ld1 {v3.8b}, [x10], x7\n"
        "ld1 {v4.8b}, [x10], x7\n"
        "ld1 {v5.8b}, [x10]\n"

        // Load constants.
        "ldr w12, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w13, w13\n"
        "dup v29.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w13\n"
        "dup v25.8h, w12\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"
        "uaddw v4.8h, v25.8h, v4.8b\n"
        "uaddw v5.8h, v25.8h, v5.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "mov x12, %[input_ptr]\n"
          "subs x15, x15, #8\n"
          "add x13, x12, x11\n"
          "cmp x15, #16\n"
          "add %[input_ptr], %[input_ptr], #8\n"

          "smlal v16.4s, v0.4h, v8.4h\n"
          "mov x9, %[filter_ptr]\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [x12], x7\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "add x10, x9, x14\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "ld1 {v9.8b}, [x12], x7\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "add %[filter_ptr], %[filter_ptr], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x12]\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v0.8b}, [x9], x7\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13], x7\n"
          "smlal v16.4s, v4.4h, v12.4h\n"
          "ld1 {v1.8b}, [x9], x7\n"
          "smlal2 v17.4s, v4.8h, v12.8h\n"
          "ld1 {v12.8b}, [x13], x7\n"
          "smlal v16.4s, v5.4h, v13.4h\n"
          "ld1 {v2.8b}, [x9]\n"
          "smlal2 v17.4s, v5.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13]\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "ld1 {v3.8b}, [x10], x7\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "ld1 {v4.8b}, [x10], x7\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "ld1 {v5.8b}, [x10]\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"

          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v4.8h, v25.8h, v4.8b\n"
          "uaddw v5.8h, v25.8h, v5.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"
        "smlal v16.4s, v4.4h, v12.4h\n"
        "smlal2 v17.4s, v4.8h, v12.8h\n"
        "smlal v16.4s, v5.4h, v13.4h\n"
        "smlal2 v17.4s, v5.8h, v13.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"
        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
        "v13", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28", "v29",
        "v30", "v31",
        // We use these general-purpose registers.
        "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kVertical, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 3x2 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 3x2 input and
        // filter values.

        // Load input and filter values.
        "ldr x6, [%[params_ptr], #" STR(OFFSET_INPUT_DEPTH) "]\n"
        "mov x12, %[input_ptr]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "mov x7, %[filter_ptr]\n"
        "ldr x5, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"
        "add x13, x12, x11\n"
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "add x14, x13, x11\n"

        "ld1 {v8.8b}, [x12], x6\n"
        "add x9, x7, x5\n"
        "ld1 {v9.8b}, [x12]\n"
        "cmp x15, #16\n"
        "add x10, x9, x5\n"
        "ld1 {v10.8b}, [x13], x6\n"
        "add %[input_ptr], %[input_ptr], #8\n"
        "ld1 {v11.8b}, [x13]\n"
        "add %[filter_ptr], %[filter_ptr], #8\n"
        "ld1 {v12.8b}, [x14], x6\n"
        "ld1 {v13.8b}, [x14]\n"

        "ld1 {v0.8b}, [x7], x6\n"
        "ld1 {v1.8b}, [x7]\n"
        "ld1 {v2.8b}, [x9], x6\n"
        "ld1 {v3.8b}, [x9]\n"
        "ld1 {v4.8b}, [x10], x6\n"
        "ld1 {v5.8b}, [x10]\n"

        // Load constants.
        "ldr w12, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w13, w13\n"
        "dup v29.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w13\n"
        "dup v25.8h, w12\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"
        "uaddw v4.8h, v25.8h, v4.8b\n"
        "uaddw v5.8h, v25.8h, v5.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "mov x12, %[input_ptr]\n"
          "subs x15, x15, #8\n"
          "add x13, x12, x11\n"
          "cmp x15, #16\n"
          "add x14, x13, x11\n"
          "add %[input_ptr], %[input_ptr], #8\n"

          "smlal v16.4s, v0.4h, v8.4h\n"
          "mov x7, %[filter_ptr]\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [x12], x6\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "add x9, x7, x5\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "add x10, x9, x5\n"
          "ld1 {v9.8b}, [x12]\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "add %[filter_ptr], %[filter_ptr], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], x6\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v0.8b}, [x7], x6\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13]\n"
          "smlal v16.4s, v4.4h, v12.4h\n"
          "ld1 {v1.8b}, [x7]\n"
          "smlal2 v17.4s, v4.8h, v12.8h\n"
          "ld1 {v12.8b}, [x14], x6\n"
          "smlal v16.4s, v5.4h, v13.4h\n"
          "ld1 {v2.8b}, [x9], x6\n"
          "smlal2 v17.4s, v5.8h, v13.8h\n"
          "ld1 {v13.8b}, [x14]\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "ld1 {v3.8b}, [x9]\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "ld1 {v4.8b}, [x10], x6\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "ld1 {v5.8b}, [x10]\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"

          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v4.8h, v25.8h, v4.8b\n"
          "uaddw v5.8h, v25.8h, v5.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"
        "smlal v16.4s, v4.4h, v12.4h\n"
        "smlal2 v17.4s, v4.8h, v12.8h\n"
        "smlal v16.4s, v5.4h, v13.4h\n"
        "smlal2 v17.4s, v5.8h, v13.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"
        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
        "v13", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28", "v29",
        "v30", "v31",
        // We use these general-purpose registers.
        "x5", "x6", "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

#undef OFFSET_INPUT_DEPTH
#undef OFFSET_INPUT_ROW_SIZE
#undef OFFSET_OUTPUT_DEPTH
#undef OFFSET_OUTPUT_ROW_SIZE
#undef OFFSET_INPUT_OFFSET
#undef OFFSET_OUTPUT_OFFSET
#undef OFFSET_FILTER_OFFSET
#undef OFFSET_OUTPUT_MULTIPLIER
#undef OFFSET_OUTPUT_ACTIVATION_MIN
#undef OFFSET_OUTPUT_ACTIVATION_MAX
#undef OFFSET_OUTPUT_RIGHT_SHIFT
#undef OFFSET_INPUT_WIDTH
#undef OFFSET_INPUT_HEIGHT
#undef OFFSET_OUTPUT_WIDTH
#undef OFFSET_OUTPUT_HEIGHT

// Copies a subset of the input designated by |input_ptr| into |output_ptr|
// with the specified output dimensions. Supports output depths of 64 only as
// this is the cache line size.
inline void ShuffleInput(const uint8* input_ptr, int64_t input_depth,
                         int32 input_width, int32 input_height,
                         int64_t output_depth, int32 output_width,
                         int32 output_height, uint8* output_ptr) {
  const int64_t input_row_size = input_depth * input_width;
  for (int32 y = 0; y < output_height; y++) {
    const uint8* ptr = input_ptr;
    for (int32 x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

// Calculates the input size depending on stride and output.
inline int32 get_shuffle_input_size(int32 stride, int32 output) {
  return stride * (output - 1) + 3;
}

// Indicates the input and output dimensions used when shuffling input
// activations.
struct ShuffleParams {
  int32 output_width;
  int32 output_height;
  int32 input_width;
  int32 input_height;

  ShuffleParams() = default;
  ShuffleParams(int32 output_width, int32 output_height, int32 stride_width,
                int32 stride_height)
      : output_width(output_width),
        output_height(output_height),
        input_width(get_shuffle_input_size(stride_width, output_width)),
        input_height(get_shuffle_input_size(stride_height, output_height)) {}
};

template <int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvThroughDepth {
  // Runs the DepthwiseConvWindow kernels through the depth dimension from
  // |start_depth| to |end_depth|. Keep this not inlined to maintain a small
  // binary size. We use a DepthwiseConvParams struct for read only params
  // to minimize call overhead.
  static __attribute__((noinline)) void Run(
      const uint8* input_ptr, const uint8* filter_ptr, const int32* bias_ptr,
      uint8* output_ptr, int64_t start_depth, int64_t end_depth,
      int64_t input_depth, int64_t input_row_size, int32 output_window_height,
      int32 output_window_width, const DepthwiseConvParams& params) {
    for (; start_depth <= end_depth - 8; start_depth += 8) {
      DepthwiseConvWindow<8, kStrideWidth, kStrideHeight>::Run(
          input_ptr, filter_ptr, bias_ptr, output_ptr, input_depth,
          input_row_size, output_window_height, output_window_width, &params);
      input_ptr += 8;
      output_ptr += 8;
      filter_ptr += 8;
      bias_ptr += 8;
    }
  }
};

template <int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvMultiRow {
  using ConvKernel = DepthwiseConvThroughDepth<kStrideWidth, kStrideHeight>;

  static inline void Run(const uint8* input_data, int32 start_x, int32 end_x,
                         const uint8* filter_data, const int32* bias_data,
                         uint8* output_data, const DepthwiseConvParams& params,
                         const ShuffleParams& shuffle_params,
                         uint8* shuffle_workspace) {
    TFLITE_DCHECK(
        shuffle_params.input_height ==
        get_shuffle_input_size(kStrideHeight, shuffle_params.output_height));
    TFLITE_DCHECK(
        shuffle_params.input_width ==
        get_shuffle_input_size(kStrideWidth, shuffle_params.output_width));
    TFLITE_DCHECK_LE(
        64 * shuffle_params.input_width * shuffle_params.input_height,
        kDepthwiseConvScratchWorkspaceSize);

    int32 out_x = start_x;

    // Run shuffling on inputs with sufficiently large depth and width. When
    // these parameters are large enough, more time is taken to load inputs
    // from memory. At this point, it becomes useful to prefetch and
    // preshuffle the input data to maximize locality.
    if (params.output_depth > 64 ||
        (params.output_depth <= 64 && params.input_width > 150)) {
      for (; out_x <= (end_x - shuffle_params.output_width);
           out_x += shuffle_params.output_width) {
        const uint8* input_ptr = input_data;
        const int32* bias_ptr = bias_data;
        const uint8* filter_ptr = filter_data;
        uint8* output_ptr = output_data;
        int64_t depth = 0;
        const int64_t shuffle_row_size = 64 * shuffle_params.input_width;

        for (; depth <= params.output_depth - 64; depth += 64) {
          // Preload.
          const uint8* h_ptr = input_ptr;
          for (int32 i = 0; i < shuffle_params.input_height; i++) {
            const uint8* ptr = h_ptr;
            for (int32 j = 0; j < shuffle_params.input_width; j++) {
              asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
              ptr += params.input_depth;
            }
            h_ptr += params.input_row_size;
          }

          // For a large enough input, shuffle into buckets.
          ShuffleInput(input_ptr, params.input_depth, params.input_width,
                       params.input_height, 64, shuffle_params.input_width,
                       shuffle_params.input_height, shuffle_workspace);
          ConvKernel::Run(shuffle_workspace, filter_ptr, bias_ptr, output_ptr,
                          0, 64, 64, shuffle_row_size,
                          shuffle_params.output_height,
                          shuffle_params.output_width, params);
          input_ptr += 64;
          output_ptr += 64;
          filter_ptr += 64;
          bias_ptr += 64;
        }

        // Preload.
        const uint8* h_ptr = input_ptr;
        for (int32 i = 0; i < shuffle_params.input_height; i++) {
          const uint8* ptr = h_ptr;
          for (int32 j = 0; j < shuffle_params.input_width; j++) {
            asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
            ptr += params.input_depth;
          }
          h_ptr += params.input_row_size;
        }

        // Handle leftover depth.
        ConvKernel::Run(input_ptr, filter_ptr, bias_ptr, output_ptr, depth,
                        params.output_depth, params.input_depth,
                        params.input_row_size, shuffle_params.output_height,
                        shuffle_params.output_width, params);

        input_data +=
            shuffle_params.output_width * kStrideWidth * params.input_depth;
        output_data += shuffle_params.output_width * params.output_depth;
      }
    }

    const int32 output_leftover_width = end_x - out_x;
    if (output_leftover_width > 0) {
      ConvKernel::Run(input_data, filter_data, bias_data, output_data, 0,
                      params.output_depth, params.input_depth,
                      params.input_row_size, shuffle_params.output_height,
                      output_leftover_width, params);
    }
  }
};

// Processes the borders of the input for pad_width and pad_height = 1.
// Calls 4 asm kernels:
//   * 1x1 input shape.
//   * Corner edges.
//   * Horizontal edges.
//   * Vertical edges.
inline void DepthwiseConvHandlePadding(const uint8* input_data,
                                       const uint8* filter_data,
                                       const int32* bias_data,
                                       uint8* output_data,
                                       const DepthwiseConvParams& params) {
  if (params.input_width == 1 && params.input_height == 1) {
    const uint8* filter_ptr =
        filter_data + params.filter_row_size + params.output_depth;
    DepthwiseConvPartial<EdgeType::kCenter, 1, 1>::Run(
        input_data, filter_ptr, bias_data, output_data, &params);
    return;
  }

  const int32 out_x_start_corner = 0;
  const int32 out_x_end_corner = params.output_width - 1;
  const int32 out_y_start_corner = 0;
  const int32 out_y_end_corner = params.output_height - 1;

  // Handle top row.
  const uint8* input_ptr = input_data;
  const uint8* filter_ptr =
      filter_data + params.filter_row_size + params.output_depth;
  uint8* output_ptr = output_data;

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(
      input_ptr, filter_ptr, bias_data, output_ptr, &params);

  input_ptr += (params.stride_width - 1) * params.input_depth;
  filter_ptr = filter_data + params.filter_row_size;
  output_ptr += params.output_depth;

  for (int32 out_x = out_x_start_corner + 1; out_x < out_x_end_corner;
       out_x++) {
    DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_depth;
    output_ptr += params.output_depth;
  }

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(
      input_ptr, filter_ptr, bias_data, output_ptr, &params);

  // Handle left side.
  input_ptr = input_data + (params.stride_width - 1) * params.input_row_size;
  filter_ptr = filter_data + params.input_depth;
  output_ptr = output_data + params.output_row_size;

  for (int32 out_y = out_y_start_corner + 1; out_y < out_y_end_corner;
       out_y++) {
    DepthwiseConvPartial<EdgeType::kVertical, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_row_size;
    output_ptr += params.output_row_size;
  }

  // Handle right side.
  input_ptr = input_data + (params.input_width - 2) * params.input_depth +
              (params.stride_width - 1) * params.input_row_size;
  filter_ptr = filter_data;
  output_ptr = output_data + params.output_row_size +
               (params.output_width - 1) * params.output_depth;

  for (int32 out_y = out_y_start_corner + 1; out_y < out_y_end_corner;
       out_y++) {
    DepthwiseConvPartial<EdgeType::kVertical, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_row_size;
    output_ptr += params.output_row_size;
  }

  // Handle bottom row.
  input_ptr = input_data + (params.input_height - 2) * params.input_row_size;
  filter_ptr = filter_data + params.output_depth;
  output_ptr =
      output_data + (params.output_height - 1) * params.output_row_size;

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(
      input_ptr, filter_ptr, bias_data, output_ptr, &params);

  input_ptr += (params.stride_width == 1) ? 0 : params.input_depth;
  filter_ptr = filter_data;
  output_ptr += params.output_depth;

  for (int32 out_x = out_x_start_corner + 1; out_x < out_x_end_corner;
       out_x++) {
    DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_depth;
    output_ptr += params.output_depth;
  }

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(
      input_ptr, filter_ptr, bias_data, output_ptr, &params);
}

inline bool Fast3x3FilterKernelSupported(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    int32 stride_width, int32 stride_height, int32 dilation_width_factor,
    int32 dilation_height_factor, int32 pad_width, int32 pad_height,
    int32 depth_multiplier, const RuntimeShape& output_shape,
    int32 output_shift) {
  const int32 input_height = input_shape.Dims(1);
  const int32 input_width = input_shape.Dims(2);
  const int32 input_depth = input_shape.Dims(3);
  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);
  const int32 output_height = output_shape.Dims(1);
  const int32 output_width = output_shape.Dims(2);

  bool supported =
      filter_width == 3 && filter_height == 3 && depth_multiplier == 1 &&
      (stride_width == 1 || stride_width == 2) &&
      (stride_height == 1 || stride_height == 2) &&
      (stride_width == stride_height) && (pad_width == 0 || pad_width == 1) &&
      (pad_height == 0 || pad_height == 1) && (pad_width == pad_height) &&
      (input_depth % 8) == 0 && (output_shift <= 0) &&
      dilation_width_factor == 1 && dilation_height_factor == 1;

  if (!supported) {
    return false;
  }

  // Handle case where padding is zero but padding type is not kValid.
  // This would require special boundary case handling that is not supported.

  const int32 out_x = output_width - 1;
  const int32 out_y = output_height - 1;

  const int32 in_x_origin = (out_x * stride_width) - pad_width;
  const int32 in_y_origin = (out_y * stride_height) - pad_height;

  const int32 in_x_end = in_x_origin + filter_width;
  const int32 in_y_end = in_y_origin + filter_height;

  // Supported only if filter on the right and bottom boundary lies completely
  // within the input if padding is zero.
  if (pad_width == 0 && pad_height == 0) {
    return in_x_end <= input_width && in_y_end <= input_height;
  }

  // Else if padding is 1, supported if bottom right filter lies +1 past input
  // width and height.
  supported = in_x_end <= (input_width + 1) && in_y_end <= (input_height + 1);

  if (!supported) {
    return false;
  }

  // Shapes with width 1 and height > 1, and vice versa are not supported yet.
  if (input_width == 1) {
    supported = (input_width == input_height);
  } else if (input_height == 1) {
    supported = (input_width == input_height);
  }
  return supported;
}

inline void DepthwiseConv3x3Filter(
    const DepthwiseParams& rt_params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, int thread_start, int thread_end, int thread_dim) {
  gemmlowp::ScopedProfilingLabel label(__PRETTY_FUNCTION__);
  DepthwiseConvParams params;

  const int32 stride_width = rt_params.stride_width;
  const int32 stride_height = rt_params.stride_height;
  const int32 pad_width = rt_params.padding_values.width;
  const int32 pad_height = rt_params.padding_values.height;
  const int32 depth_multiplier = rt_params.depth_multiplier;
  const int32 output_activation_min = rt_params.quantized_activation_min;
  const int32 output_activation_max = rt_params.quantized_activation_max;
  const int32 input_offset = rt_params.input_offset;
  const int32 filter_offset = rt_params.weights_offset;
  const int32 output_offset = rt_params.output_offset;
  const int32 output_multiplier = rt_params.output_multiplier;
  const int32 output_shift = rt_params.output_shift;

  params.input_depth = input_shape.Dims(3);
  params.input_width = input_shape.Dims(2);
  params.input_height = input_shape.Dims(1);
  params.input_row_size = params.input_depth * params.input_width;
  params.input_offset = input_offset;
  params.stride_width = stride_width;
  params.stride_height = stride_height;
  params.output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  params.output_width = output_shape.Dims(2);
  params.output_height = output_shape.Dims(1);
  params.output_row_size = params.output_depth * params.output_width;
  params.output_offset = output_offset;
  params.filter_offset = filter_offset;
  params.output_multiplier = output_multiplier;
  params.output_right_shift = -output_shift;
  params.output_activation_min = output_activation_min;
  params.output_activation_max = output_activation_max;

  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);
  params.filter_row_size = params.output_depth * filter_width;

  // Algorithm assumes below constraints. It is optimized for depth
  // multiplier of 1, 3x3 filter, no padding and strides 1 and 2.
  TFLITE_DCHECK(params.output_depth == params.input_depth * depth_multiplier);
  TFLITE_DCHECK(depth_multiplier == 1);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 3);
  TFLITE_DCHECK(stride_height == 1 || stride_height == 2);
  TFLITE_DCHECK(stride_width == 1 || stride_width == 2);
  TFLITE_DCHECK(stride_width == stride_height);
  TFLITE_DCHECK(pad_height == 0 || pad_height == 1);
  TFLITE_DCHECK(pad_width == 0 || pad_width == 1);
  TFLITE_DCHECK(pad_width == pad_height);
  TFLITE_DCHECK(thread_dim == 0 || thread_dim == 1);

  const int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int64_t input_batch_size = params.input_row_size * params.input_height;
  const int64_t output_batch_size =
      params.output_row_size * params.output_height;

  ShuffleParams one_row_shuffle_params, two_row_shuffle_params,
      four_row_shuffle_params, eight_row_shuffle_params;
  if (stride_width == 1) {
    one_row_shuffle_params = ShuffleParams(30, 1, 1, 1);
    two_row_shuffle_params = ShuffleParams(22, 2, 1, 1);
    four_row_shuffle_params = ShuffleParams(14, 4, 1, 1);
    eight_row_shuffle_params = ShuffleParams(8, 8, 1, 1);
  } else {
    one_row_shuffle_params = ShuffleParams(14, 1, 2, 2);
    two_row_shuffle_params = ShuffleParams(8, 2, 2, 2);
    four_row_shuffle_params = ShuffleParams(4, 4, 2, 2);
    eight_row_shuffle_params = ShuffleParams(2, 8, 2, 2);
  }

  using conv_multirow_func_t = decltype(&DepthwiseConvMultiRow<1, 1>::Run);
  conv_multirow_func_t conv_multirow_func = DepthwiseConvMultiRow<1, 1>::Run;
  if (stride_width == 2) {
    conv_multirow_func = DepthwiseConvMultiRow<2, 2>::Run;
  }

  // Allocate maximum memory needed for shuffled input.
  // TODO(mariewhite): The size of this workspace is small enough to be
  // allocated on the stack. Eventually we will want to move it to the heap
  // and have it allocated outside of this function, like the im2col_array
  // used in gemmlowp.
  uint8 shuffle_workspace[kDepthwiseConvScratchWorkspaceSize];

  int batch_start = 0;
  int batch_end = batches;
  int row_start = 0;
  int row_end = params.output_height;

  switch (thread_dim) {
    case 0:
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, batches);
      batch_start = thread_start;
      batch_end = thread_end;
      break;
    case 1:
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, params.output_height);
      row_start = thread_start;
      row_end = thread_end;
      break;
  }

  for (int32 b = batch_start; b < batch_end; ++b) {
    const uint8* input_ptr = input_data + b * input_batch_size;
    uint8* output_ptr = output_data + b * output_batch_size +
                        row_start * params.output_width * params.output_depth;

    int32 out_x = 0;
    int32 out_y = row_start;
    int32 end_x = params.output_width;
    int32 end_y = row_end;

    if (pad_width == 1 && pad_height == 1) {
      DepthwiseConvHandlePadding(input_ptr, filter_data, bias_data, output_ptr,
                                 params);

      // Update extents now that the edges have been handled.
      out_x = 1;
      end_x = params.output_width - 1;
      out_y = std::max(1, out_y);
      end_y = std::min(params.output_height - 1, end_y);
      const int in_x = (out_x * stride_width) - pad_width;
      const int in_y = (out_y * stride_height) - pad_height;
      input_ptr += in_y * params.input_row_size + in_x * params.input_depth;
      output_ptr +=
          out_y * params.output_row_size + out_x * params.output_depth;
    }

    // Shuffling shapes that maximize width over the shuffle workspace size
    // perform better since the inputs are closer together, minimizing
    // shuffling time.
    //
    // If the input shape has width large enough for the 2 row kernels,
    // we prefer to use this. The innermost loop of the kernels handle
    // 2 height x 2 width so this is the fastest path.
    //
    // If the input shape has smaller width but larger height, shuffling is
    // still useful and can benefit from kernels 4 row and 8 row kernels.

    // Handle 8 rows at a time.
    if (params.input_width < four_row_shuffle_params.input_width) {
      for (; out_y <= end_y - 8; out_y += 8) {
        conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                           output_ptr, params, eight_row_shuffle_params,
                           shuffle_workspace);
        input_ptr += 8 * stride_height * params.input_row_size;
        output_ptr += 8 * params.output_row_size;
      }
    }

    // Handle 4 rows at a time.
    if (params.input_width < two_row_shuffle_params.input_width) {
      for (; out_y <= end_y - 4; out_y += 4) {
        conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                           output_ptr, params, four_row_shuffle_params,
                           shuffle_workspace);
        input_ptr += 4 * stride_height * params.input_row_size;
        output_ptr += 4 * params.output_row_size;
      }
    }

    // Handle 2 rows at a time.
    for (; out_y <= end_y - 2; out_y += 2) {
      conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                         output_ptr, params, two_row_shuffle_params,
                         shuffle_workspace);
      input_ptr += 2 * stride_height * params.input_row_size;
      output_ptr += 2 * params.output_row_size;
    }

    // Handle one row at a time.
    for (; out_y < end_y; out_y++) {
      conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                         output_ptr, params, one_row_shuffle_params,
                         shuffle_workspace);
      input_ptr += stride_height * params.input_row_size;
      output_ptr += params.output_row_size;
    }
  }
}
#endif  // __aarch64__

#endif

// Permute filter data, and adjust bias data to account for symmetric input
// offset. Details are provided in the implementation of the
// kUseCModel3x3DotProduct version.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation>
struct ProcessPerDepth {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// Copy a macro block of data from the input buffer into the workspace,
// permuting data within each micro block.
//
// (a) Copy a macro block of data, padding as required along the width and
//     height.
// (b) Transpose the data within each micro block.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          DepthwiseConvDepthMultiplication depth_multiplication,
          int32 max_padding>
struct PackMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// Apply filter to macro block of input data and store results. Details are
// provided in the implementation of the kUseCModel3x3DotProduct version.
//
// Parameters for repeats and residual sizes are in terms of outputs.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          DepthwiseConvDepthMultiplication depth_multiplication, int32 stride>
struct KernelMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

#if defined(USE_NEON) && defined(__aarch64__)
// Experiments suggest that a modest performance improvement is seen, at least
// on 855 chipset big cores, with cache hints.
inline void PreloadInputBlock(
    const uint8* input_block_data,
    const DepthwiseConvDotProdParams* function_params) {
  // Preload.
  const int input_width_micro_repeats =
      function_params->input_width_micro_repeats;
  const int block_height = function_params->inbound_block_height;
  const int residual_width = function_params->residual_width;
  const int input_height_stride = function_params->input_height_stride;
  const int input_depth = function_params->input_depth;

  const int total_width = 4 * input_width_micro_repeats + residual_width;
  const uint8* row_ptr = input_block_data;
  for (int k_height = 0; k_height < block_height; ++k_height) {
    const uint8* ptr = row_ptr;
    for (int j = 0; j < total_width; ++j) {
      // Input data is loaded once.
      asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
      ptr += input_depth;
    }
    row_ptr += input_height_stride;
  }
}
#endif  // USE_NEON &&__aarch64__

#if defined(__ARM_FEATURE_DOTPROD) && !defined(GOOGLE_L4T)

template <>
struct ProcessPerDepth<DepthwiseConvImplementation::kUseNeon3x3DotProduct> {
  static void ProcessPerDepthNeon(
      const uint8* filter_data, const int32* bias_data,
      int8* shuffled_filter_data, int32* adjusted_bias_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int depth = function_params->output_depth;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int bias_increment = function_params->bias_increment;

    constexpr int kSymmetricZeroPoint = 128;
    constexpr uint8 kSignBit = 0x80;
    const int32 input_offset = function_params->input_offset;
    TFLITE_DCHECK_GE(input_offset, -255);
    TFLITE_DCHECK_LE(input_offset, 0);
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;
    const int8x16_t ones_vector = vdupq_n_s8(1);

    // Simulate NEON-register transposition of subset of filter.
    int8x16_t filter_reg_0_a;
    int8x16_t filter_reg_0_b;
    int8x16_t filter_reg_1_a;
    int8x16_t filter_reg_1_b;
    int8x16_t filter_reg_2_a;
    int8x16_t filter_reg_2_b;

    // Register pairs for each height.
    // Effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);

    const uint8* filter_block = filter_data;
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      // Filter data is provided as filter_block[3][3][depth/8][2][4].
      // height 3, width 3, micro-blocks, sub-block 0 or 1, depth 4.
      // filter_bank[3][2][4][4]; Sub-block, height 3, depth 4, width 4.

      // Load zero-point into effective position of zero-padding of filter
      // (register B, upper part).
      filter_reg_0_b = vdupq_n_u8(kSignBit);
      filter_reg_1_b = vdupq_n_u8(kSignBit);
      filter_reg_2_b = vdupq_n_u8(kSignBit);

      const uint8* filter_block_ptr = filter_block;
      filter_reg_0_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_0_a, 0);
      filter_block_ptr += depth;
      filter_reg_0_b = vld1q_lane_s8x8(filter_block_ptr, filter_reg_0_b, 0);
      filter_block_ptr += depth;
      filter_reg_0_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_0_a, 1);
      filter_block_ptr += depth;
      filter_reg_1_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_1_a, 0);
      filter_block_ptr += depth;
      filter_reg_1_b = vld1q_lane_s8x8(filter_block_ptr, filter_reg_1_b, 0);
      filter_block_ptr += depth;
      filter_reg_1_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_1_a, 1);
      filter_block_ptr += depth;
      filter_reg_2_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_2_a, 0);
      filter_block_ptr += depth;
      filter_reg_2_b = vld1q_lane_s8x8(filter_block_ptr, filter_reg_2_b, 0);
      filter_block_ptr += depth;
      filter_reg_2_a = vld1q_lane_s8x8(filter_block_ptr, filter_reg_2_a, 1);

      filter_reg_0_a = veorq_s8(filter_reg_0_a, sign_bit);
      filter_reg_0_b = veorq_s8(filter_reg_0_b, sign_bit);
      filter_reg_1_a = veorq_s8(filter_reg_1_a, sign_bit);
      filter_reg_1_b = veorq_s8(filter_reg_1_b, sign_bit);
      filter_reg_2_a = veorq_s8(filter_reg_2_a, sign_bit);
      filter_reg_2_b = veorq_s8(filter_reg_2_b, sign_bit);

      vzipq_s8_in_place(&filter_reg_0_a, &filter_reg_0_b);
      vzipq_s8_in_place(&filter_reg_1_a, &filter_reg_1_b);
      vzipq_s8_in_place(&filter_reg_2_a, &filter_reg_2_b);
      vzipq_s8x2_in_place(&filter_reg_0_a, &filter_reg_0_b);
      vzipq_s8x2_in_place(&filter_reg_1_a, &filter_reg_1_b);
      vzipq_s8x2_in_place(&filter_reg_2_a, &filter_reg_2_b);

      vst1q_s8(shuffled_filter_data, filter_reg_0_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_reg_0_b);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_reg_1_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_reg_1_b);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_reg_2_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_reg_2_b);
      shuffled_filter_data += 16;

      int32x4_t adjusted_bias_data_a = vld1q_s32(bias_data);
      bias_data += bias_increment;
      int32x4_t adjusted_bias_data_b = vld1q_s32(bias_data);
      bias_data += bias_increment;
      // For instance, if input_offset == 128, no adjustment is needed.

      int32x4_t filter_sum_a = vdupq_n_s32(0);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_reg_0_a, ones_vector);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_reg_1_a, ones_vector);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_reg_2_a, ones_vector);
      int32x4_t filter_sum_b = vdupq_n_s32(0);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_reg_0_b, ones_vector);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_reg_1_b, ones_vector);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_reg_2_b, ones_vector);

      adjusted_bias_data_a = vmlaq_n_s32(adjusted_bias_data_a, filter_sum_a,
                                         input_offset_difference);
      adjusted_bias_data_b = vmlaq_n_s32(adjusted_bias_data_b, filter_sum_b,
                                         input_offset_difference);

      vst1q_s32(adjusted_bias_data, adjusted_bias_data_a);
      adjusted_bias_data += 4;
      vst1q_s32(adjusted_bias_data, adjusted_bias_data_b);
      adjusted_bias_data += 4;

      filter_block += 8;
    }
  }

  static inline void Run(const uint8* filter_data, const int32* bias_data,
                         int8* shuffled_filter_data, int32* adjusted_bias_data,
                         const DepthwiseConvDotProdParams* function_params) {
    ProcessPerDepthNeon(filter_data, bias_data, shuffled_filter_data,
                        adjusted_bias_data, function_params);
  }
};

template <>
struct PackMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/0> {
  static inline void PackMacroBlockNeon(
      const uint8* input_block_data, int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK_EQ(function_params->padding_bottom, 0);
    TFLITE_DCHECK_EQ(function_params->padding_top, 0);
    TFLITE_DCHECK_EQ(function_params->padding_left, 0);
    TFLITE_DCHECK_EQ(function_params->padding_right, 0);
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int input_width_micro_repeats =
        function_params->input_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int block_height = function_params->inbound_block_height;
    const int residual_width = function_params->residual_width;
    const int input_height_stride = function_params->input_height_stride;
    const int input_depth = function_params->input_depth;

    TFLITE_DCHECK_GE(depth_micro_repeats, 0);
    constexpr uint8 kSignBit = 0x80;
    const int micro_block_size = 4 * 8;
    const int depth_advance = width_overall_micro_repeats * micro_block_size;
    const int width_advance =
        micro_block_size *
        (1 - depth_micro_repeats * width_overall_micro_repeats);
    const int height_advance = workspace_height_stride -
                               width_overall_micro_repeats * micro_block_size;
    const int input_depth_skip = 4 * input_depth - 8 * depth_micro_repeats;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in NEON
    // code. Note the blocks of 4x4 are still interleaved down the depth.
    int8x16_t work_reg_a;
    int8x16_t work_reg_b;

    // Effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);

    // Work through one slice, by row, at a time.
    int8* scratch_data_0 = scratch_block_data;

    for (int k_height = 0; k_height < block_height; ++k_height) {
      const uint8* input_data_0 = input_block_data;
      int8x16_t input_data_a;
      int8x16_t input_data_b;
      int8x16_t input_data_c;
      int8x16_t input_data_d;

      // Traverse the width one point at a time, but the depth in (micro) blocks
      // of size 8.
      //
      // The depth and width margins, which are filled with "zeros", may be
      // larger than is strictly needed to calculate output. This is because the
      // conv calculation is performed across complete micro blocks.
      for (int j_width = 0; j_width < input_width_micro_repeats; ++j_width) {
        int8x16_t work_reg_a_sp;
        int8x16_t work_reg_b_sp;

        int i_depth = 0;

        if (depth_micro_repeats >= 2) {
          i_depth += 2;

          //

          input_data_a = vld1q_u8(input_data_0);
          input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
          input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
          input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
          input_data_0 += 16;

          //

          for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
            work_reg_a = vzip1q_s8(input_data_a, input_data_b);
            work_reg_b = vzip1q_s8(input_data_c, input_data_d);
            vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
            work_reg_a = veorq_s8(work_reg_a, sign_bit);
            work_reg_b = veorq_s8(work_reg_b, sign_bit);

            work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
            work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
            vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

            input_data_a = vld1q_u8(input_data_0);
            input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
            vst1q_s8(scratch_data_0, work_reg_a);
            vst1q_s8(scratch_data_0 + 16, work_reg_b);

            scratch_data_0 += depth_advance;

            work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
            work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

            input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
            input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
            vst1q_s8(scratch_data_0, work_reg_a_sp);
            vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

            scratch_data_0 += depth_advance;

            //

            input_data_0 += 16;
          }

          work_reg_a = vzip1q_s8(input_data_a, input_data_b);
          work_reg_b = vzip1q_s8(input_data_c, input_data_d);
          vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
          work_reg_a = veorq_s8(work_reg_a, sign_bit);
          work_reg_b = veorq_s8(work_reg_b, sign_bit);
          vst1q_s8(scratch_data_0, work_reg_a);
          vst1q_s8(scratch_data_0 + 16, work_reg_b);

          scratch_data_0 += depth_advance;
          //

          work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
          work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
          vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
          work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
          work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

          vst1q_s8(scratch_data_0, work_reg_a_sp);
          vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

          scratch_data_0 += depth_advance;
        }
        for (; i_depth < depth_micro_repeats; ++i_depth) {
          input_data_a = vld1q_lane_s8x8(input_data_0, input_data_a, 0);
          input_data_b =
              vld1q_lane_s8x8(input_data_0 + 1 * input_depth, input_data_b, 0);
          input_data_c =
              vld1q_lane_s8x8(input_data_0 + 2 * input_depth, input_data_c, 0);
          input_data_d =
              vld1q_lane_s8x8(input_data_0 + 3 * input_depth, input_data_d, 0);
          work_reg_a = vzip1q_s8(input_data_a, input_data_b);
          work_reg_b = vzip1q_s8(input_data_c, input_data_d);

          input_data_0 += 8;

          vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
          work_reg_a = veorq_s8(work_reg_a, sign_bit);
          work_reg_b = veorq_s8(work_reg_b, sign_bit);

          vst1q_s8(scratch_data_0, work_reg_a);
          vst1q_s8(scratch_data_0 + 16, work_reg_b);

          scratch_data_0 += depth_advance;
        }
        scratch_data_0 += width_advance;
        input_data_0 += input_depth_skip;
      }
      if (width_overall_micro_repeats > input_width_micro_repeats) {
        TFLITE_DCHECK_EQ(width_overall_micro_repeats,
                         input_width_micro_repeats + 1);
        TFLITE_DCHECK_GT(residual_width, 0);
        TFLITE_DCHECK_LT(residual_width, 4);
        for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
          input_data_c = vdupq_n_u8(kSignBit);
          input_data_a = vld1q_lane_s8x8(input_data_0, input_data_a, 0);
          input_data_d = vdupq_n_u8(kSignBit);
          if (residual_width > 1) {
            input_data_b =
                vld1q_lane_s8x8(input_data_0 + input_depth, input_data_b, 0);
            if (residual_width == 3) {
              input_data_c = vld1q_lane_s8x8(input_data_0 + 2 * input_depth,
                                             input_data_c, 0);
            }
          }
          work_reg_a = vzip1q_s8(input_data_a, input_data_b);
          work_reg_b = vzip1q_s8(input_data_c, input_data_d);

          work_reg_a = veorq_s8(work_reg_a, sign_bit);
          work_reg_b = veorq_s8(work_reg_b, sign_bit);
          vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);

          vst1q_s8(scratch_data_0, work_reg_a);
          vst1q_s8(scratch_data_0 + 16, work_reg_b);

          scratch_data_0 += depth_advance;
          input_data_0 += 8;
        }
        scratch_data_0 += width_advance;
        input_data_0 += input_depth_skip;
      }
      scratch_data_0 += height_advance;
      input_block_data += input_height_stride;
    }
    TFLITE_DCHECK_EQ(
        scratch_data_0,
        scratch_block_data + block_height * workspace_height_stride);
  }

  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    PreloadInputBlock(input_block_data, function_params);
    PackMacroBlockNeon(input_block_data, scratch_block_data, function_params);
  }
};

template <>
struct PackMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/1> {
  static inline void PackMacroBlockNeon(
      int32 height_block_number, int32 width_block_number,
      const uint8* input_block_data, int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    constexpr uint8 kSignBit = 0x80;

    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int input_width_micro_repeats =
        function_params->input_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int block_height = function_params->inbound_block_height;
    const int residual_width = function_params->residual_width;
    const int input_height_stride = function_params->input_height_stride;
    const int input_depth = function_params->input_depth;

    const int padding_left = function_params->padding_left;
    const int padding_right = function_params->padding_right;
    const int padding_top = function_params->padding_top;
    const int padding_bottom = function_params->padding_bottom;

    TFLITE_DCHECK_GT(depth_micro_repeats, 0);
    constexpr int kSymmetricZeroPoint = 128;

    const int micro_block_size = 4 * 8;
    const int depth_advance = width_overall_micro_repeats * micro_block_size;
    const int width_advance =
        micro_block_size *
        (1 - depth_micro_repeats * width_overall_micro_repeats);
    const int height_advance = workspace_height_stride -
                               width_overall_micro_repeats * micro_block_size;
    const int input_depth_skip = 4 * input_depth - 8 * depth_micro_repeats;

    const bool leading_width_padding =
        padding_left > 0 && width_block_number == 0;
    const bool trailing_width_padding =
        padding_right > 0 &&
        width_block_number == (function_params->width_macro_count - 1);
    const bool leading_height_padding =
        padding_top > 0 && height_block_number < 0;
    const bool trailing_height_padding =
        padding_bottom > 0 &&
        height_block_number == (function_params->height_macro_count - 1);

    const int32 input_offset = function_params->input_offset;
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in NEON
    // code. Note the blocks of 4x4 are still interleaved down the depth.
    int8x16_t work_reg_a;
    int8x16_t work_reg_b;

    // Effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);

    // Work through one slice, by row, at a time.
    int8* scratch_data_0 = scratch_block_data;

    int copy_block_height = block_height;
    if (leading_height_padding) {
      copy_block_height -= 1;
      memset(scratch_data_0, -input_offset_difference, workspace_height_stride);
      scratch_data_0 += workspace_height_stride;
      input_block_data += input_height_stride;
    }
    if (trailing_height_padding) {
      copy_block_height -= 1;
    }

    for (int k_height = 0; k_height < copy_block_height; ++k_height) {
      const uint8* input_data_0 = input_block_data;
      int8x16_t input_data_a;
      int8x16_t input_data_b;
      int8x16_t input_data_c;
      int8x16_t input_data_d;

      // Traverse the width one point at a time, but the depth in (micro) blocks
      // of size 8.
      //
      // The depth and width margins, which are filled with "zeros", may be
      // larger than is strictly needed to calculate output. This is because the
      // conv calculation is performed across complete micro blocks.
      for (int j_width = 0; j_width < width_overall_micro_repeats; ++j_width) {
        // Figure out division of work (available input vs zero-ed).
        int adjusted_residual_width =
            j_width == (input_width_micro_repeats) ? residual_width : 4;

        if (trailing_width_padding &&
            j_width == (width_overall_micro_repeats - 1)) {
          adjusted_residual_width -= 1;
        }
        int start_width = 0;
        if (leading_width_padding && j_width == 0) {
          start_width = 1;
        }
        if (start_width == 0) {
          if (adjusted_residual_width == 4) {
            int8x16_t work_reg_a_sp;
            int8x16_t work_reg_b_sp;

            int i_depth = 0;

            if (depth_micro_repeats >= 2) {
              i_depth += 2;

              //

              input_data_a = vld1q_u8(input_data_0);
              input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
              input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
              input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
              input_data_0 += 16;

              //

              for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
                work_reg_a = vzip1q_s8(input_data_a, input_data_b);
                work_reg_b = vzip1q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);

                work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
                work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

                input_data_a = vld1q_u8(input_data_0);
                input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a);
                vst1q_s8(scratch_data_0 + 16, work_reg_b);

                scratch_data_0 += depth_advance;

                work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

                input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
                input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a_sp);
                vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

                scratch_data_0 += depth_advance;

                //

                input_data_0 += 16;
              }

              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);
              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
              //

              work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
              work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
              work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
              work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

              vst1q_s8(scratch_data_0, work_reg_a_sp);
              vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

              scratch_data_0 += depth_advance;
            }
            for (; i_depth < depth_micro_repeats; ++i_depth) {
              input_data_a = vld1q_lane_s8x8(input_data_0, input_data_a, 0);
              input_data_b = vld1q_lane_s8x8(input_data_0 + 1 * input_depth,
                                             input_data_b, 0);
              input_data_c = vld1q_lane_s8x8(input_data_0 + 2 * input_depth,
                                             input_data_c, 0);
              input_data_d = vld1q_lane_s8x8(input_data_0 + 3 * input_depth,
                                             input_data_d, 0);
              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);

              input_data_0 += 8;

              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);

              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
            }
            scratch_data_0 += width_advance;
            input_data_0 += input_depth_skip;
          } else {
            TFLITE_DCHECK_LT(adjusted_residual_width, 4);
            for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
              input_data_a = vdupq_n_u8(-input_offset);
              input_data_b = vdupq_n_u8(-input_offset);
              input_data_c = vdupq_n_u8(-input_offset);
              input_data_d = vdupq_n_u8(-input_offset);
              if (adjusted_residual_width > 0) {
                input_data_a = vld1q_lane_s8x8(input_data_0, input_data_a, 0);
                if (adjusted_residual_width > 1) {
                  input_data_b = vld1q_lane_s8x8(input_data_0 + input_depth,
                                                 input_data_b, 0);
                  if (adjusted_residual_width == 3) {
                    input_data_c = vld1q_lane_s8x8(
                        input_data_0 + 2 * input_depth, input_data_c, 0);
                  }
                }
              }
              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);

              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);

              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
              input_data_0 += 8;
            }
            scratch_data_0 += width_advance;
            input_data_0 += input_depth_skip;
          }
        } else {
          if (adjusted_residual_width == 4) {
            int8x16_t work_reg_a_sp;
            int8x16_t work_reg_b_sp;

            int i_depth = 0;

            if (depth_micro_repeats >= 2) {
              i_depth += 2;

              //

              input_data_a = vdupq_n_u8(-input_offset);
              input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
              input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
              input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
              input_data_0 += 16;

              //

              for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
                work_reg_a = vzip1q_s8(input_data_a, input_data_b);
                work_reg_b = vzip1q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);

                work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
                work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

                input_data_a = vdupq_n_u8(-input_offset);
                input_data_b = vld1q_u8(input_data_0 + 1 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a);
                vst1q_s8(scratch_data_0 + 16, work_reg_b);

                scratch_data_0 += depth_advance;

                work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

                input_data_c = vld1q_u8(input_data_0 + 2 * input_depth);
                input_data_d = vld1q_u8(input_data_0 + 3 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a_sp);
                vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

                scratch_data_0 += depth_advance;

                //

                input_data_0 += 16;
              }

              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);
              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
              //

              work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
              work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
              work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
              work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);

              vst1q_s8(scratch_data_0, work_reg_a_sp);
              vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

              scratch_data_0 += depth_advance;
            }
            for (; i_depth < depth_micro_repeats; ++i_depth) {
              input_data_a = vdupq_n_u8(-input_offset);
              input_data_b = vld1q_lane_s8x8(input_data_0 + 1 * input_depth,
                                             input_data_b, 0);
              input_data_c = vld1q_lane_s8x8(input_data_0 + 2 * input_depth,
                                             input_data_c, 0);
              input_data_d = vld1q_lane_s8x8(input_data_0 + 3 * input_depth,
                                             input_data_d, 0);
              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);

              input_data_0 += 8;

              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);

              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
            }
            scratch_data_0 += width_advance;
            input_data_0 += input_depth_skip;
          } else {
            TFLITE_DCHECK_LT(adjusted_residual_width, 4);

            for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
              input_data_a = vdupq_n_u8(-input_offset);
              input_data_b = vdupq_n_u8(-input_offset);
              input_data_c = vdupq_n_u8(-input_offset);
              input_data_d = vdupq_n_u8(-input_offset);
              // Skip loading first column.
              if (adjusted_residual_width > 1) {
                input_data_b = vld1q_lane_s8x8(input_data_0 + input_depth,
                                               input_data_b, 0);
                if (adjusted_residual_width == 3) {
                  input_data_c = vld1q_lane_s8x8(input_data_0 + 2 * input_depth,
                                                 input_data_c, 0);
                }
              }
              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);

              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);

              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;
              input_data_0 += 8;
            }
            scratch_data_0 += width_advance;
            input_data_0 += input_depth_skip;
          }
        }
      }
      scratch_data_0 += height_advance;
      input_block_data += input_height_stride;
    }

    if (trailing_height_padding) {
      memset(scratch_data_0, -input_offset_difference, workspace_height_stride);
      scratch_data_0 += workspace_height_stride;
    }

    TFLITE_DCHECK_EQ(
        scratch_data_0,
        scratch_block_data + block_height * workspace_height_stride);
  }

  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    PreloadInputBlock(input_block_data, function_params);
    PackMacroBlockNeon(height_block_number, width_block_number,
                       input_block_data, scratch_block_data, function_params);
  }
};

template <>
struct PackMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kUnitInputDepth,
                      /*max_padding=*/1> {
  static inline void PackMacroBlockNeon(
      int32 height_block_number, int32 width_block_number,
      const uint8* input_block_data, int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int input_width_micro_repeats =
        function_params->input_width_micro_repeats;
    const int block_height = function_params->inbound_block_height;
    const int residual_width = function_params->residual_width;
    const int input_height_stride = function_params->input_height_stride;

    const int padding_left = function_params->padding_left;
    const int padding_right = function_params->padding_right;
    const int padding_top = function_params->padding_top;
    const int padding_bottom = function_params->padding_bottom;

    constexpr int kSymmetricZeroPoint = 128;

    TFLITE_DCHECK_GE(workspace_height_stride, 4 * width_overall_micro_repeats);

    const bool leading_width_padding =
        padding_left > 0 && width_block_number == 0;
    const bool trailing_width_padding =
        padding_right > 0 &&
        width_block_number == (function_params->width_macro_count - 1);
    const bool leading_height_padding =
        padding_top > 0 && height_block_number < 0;
    const bool trailing_height_padding =
        padding_bottom > 0 &&
        height_block_number == (function_params->height_macro_count - 1);

    const int32 input_offset = function_params->input_offset;
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;

    // Work through one slice, by row, at a time.
    int8* scratch_data_base = scratch_block_data;

    int copy_block_height = block_height;
    if (leading_height_padding) {
      copy_block_height -= 1;
      memset(scratch_data_base, -input_offset_difference,
             workspace_height_stride + kWorkspaceExtension);
      scratch_data_base += workspace_height_stride;
      input_block_data += input_height_stride;
    }
    if (trailing_height_padding) {
      copy_block_height -= 1;
    }

    int adjusted_residual_width =
        input_width_micro_repeats < width_overall_micro_repeats ? residual_width
                                                                : 4;

    if (trailing_width_padding) {
      adjusted_residual_width -= 1;
    }
    int start_width = 0;
    if (leading_width_padding) {
      start_width = 1;
      input_block_data += 1;
    }

    const int copy_size = (width_overall_micro_repeats - 1) * 4 +
                          adjusted_residual_width - start_width;
    // Adjusted so that later conditionals are simplified.
    const int copy_size_adjusted =
        trailing_width_padding ? copy_size + 1 : copy_size;

    TFLITE_DCHECK_LE(
        copy_size,
        input_height_stride - width_block_number * input_width_micro_repeats);
    // We may drop up to stride-1 of trailing input.
    TFLITE_DCHECK_GE(copy_size, input_height_stride - 1);

    int scratch_data_offset = 0;
    int input_block_offset = 0;

    constexpr uint8 kSignBit = 0x80;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in NEON
    // code. Note the blocks of 4x4 are still interleaved down the depth.
    int8x16_t work_reg;
    int8x8_t half_work_reg;
    int8x8_t padding_mask;

    // Effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);
    const uint8x16_t padding_reg = vdupq_n_u8(-input_offset);
    padding_mask = vdup_n_s8(-1);
    half_work_reg = vdup_n_s8(0);

    if (copy_size >= 16) {
      const int copy_remaining = (copy_size + start_width) & 0x7;
      padding_mask = vshl_u64(padding_mask, vdup_n_s64(8 * copy_remaining));

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // The surrounding condition ensures that we always need at least one
        // iteration of the main copy loop. In the case of leading width
        // padding, we unroll this specially.
        if (leading_width_padding) {
          work_reg = vld1q_u8(input_block_data + input_block_offset);
          work_reg = vextq_s8(padding_reg, work_reg, 15);
          work_reg = veorq_s8(work_reg, sign_bit);
          vst1q_s8(scratch_data, work_reg);
          copy_done += 15;
        }

        // Main copy loop.
        for (; (copy_done + 16) <= copy_size; copy_done += 16) {
          work_reg =
              vld1q_u8(input_block_data + input_block_offset + copy_done);
          work_reg = veorq_s8(work_reg, sign_bit);
          TFLITE_DCHECK_EQ((start_width + copy_done) % 16, 0);
          vst1q_s8(scratch_data + start_width + copy_done, work_reg);
        }

        if (copy_done + 8 <= copy_size) {
          half_work_reg =
              vld1_u8(input_block_data + input_block_offset + copy_done);
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ((start_width + copy_done) % 8, 0);
          vst1_s8(scratch_data + start_width + copy_done, half_work_reg);
          copy_done += 8;
        }

        TFLITE_DCHECK_EQ(copy_remaining, copy_size - copy_done);
        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.

        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size_adjusted) {
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          // This has the advantage of resulting in zeros after shifting.
          half_work_reg =
              vld1_u8(input_block_data + input_block_offset + copy_size - 8);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (8 - copy_remaining)));
          half_work_reg =
              vbsl_s8(padding_mask, vget_low_s8(padding_reg), half_work_reg);

          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ((start_width + copy_done) % 8, 0);
          vst1_s8(scratch_data + start_width + copy_done, half_work_reg);
        }

        // Trailing guard.
        vst1_s8(scratch_data + start_width + copy_done, half_work_reg);
        vst1_s8(scratch_data + start_width + copy_done + 8, half_work_reg);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else if (copy_size >= 4) {
      const int copy_remaining = (copy_size + start_width) & 0x3;
      padding_mask = vshl_u64(padding_mask, vdup_n_s64(8 * copy_remaining));

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // The surrounding condition ensures that we always need at least one
        // iteration of the main copy loop. In the case of leading width
        // padding, we unroll this specially.
        if (leading_width_padding) {
          half_work_reg = vld1_lane_8x4(input_block_data + input_block_offset,
                                        half_work_reg, 0);
          half_work_reg = vext_s8(vget_low_s8(padding_reg), half_work_reg, 7);
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          vst1_lane_8x4(scratch_data, half_work_reg, 0);
          copy_done += 3;
        }

        // Main copy loop.
        for (; (copy_done + 4) <= copy_size; copy_done += 4) {
          // Important! Most compilation configurations will compile and run
          // without the reinterpret_cast. Sanitizers may fail silently on
          // lane-loading, with a obscure bug or mis-feature probably in
          // unhygienic macro expansion.
          half_work_reg =
              vld1_lane_8x4(input_block_data + input_block_offset + copy_done,
                            half_work_reg, 0);
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ((start_width + copy_done) % 4, 0);
          vst1_lane_8x4(scratch_data + start_width + copy_done, half_work_reg,
                        0);
        }

        TFLITE_DCHECK_EQ(copy_remaining, copy_size - copy_done);
        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.

        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size_adjusted) {
          TFLITE_DCHECK_LT(copy_remaining, 4);
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          // This has the advantage of resulting in zeros after shifting.
          half_work_reg = vld1_lane_8x4(
              input_block_data + input_block_offset + copy_size - 4,
              half_work_reg, 0);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (4 - copy_remaining)));
          half_work_reg =
              vbsl_s8(padding_mask, vget_low_s8(padding_reg), half_work_reg);

          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ((start_width + copy_done) % 4, 0);
          vst1_lane_8x4(scratch_data + start_width + copy_done, half_work_reg,
                        0);
          copy_done += 4;
        }
        // Trailing guard.
        vst1_lane_8x4(scratch_data + start_width + copy_done, half_work_reg, 0);
        vst1_lane_8x4(scratch_data + start_width + copy_done + 4, half_work_reg,
                      0);
        vst1_lane_8x4(scratch_data + start_width + copy_done + 8, half_work_reg,
                      0);
        vst1_lane_8x4(scratch_data + start_width + copy_done + 12,
                      half_work_reg, 0);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else if (width_overall_micro_repeats == 2) {
      // Special case of 1 + 3 + 1, padding + copy + padding.
      // This is rarely executed in practice.
      TFLITE_DCHECK_EQ(copy_size, 3);
      TFLITE_DCHECK_EQ(start_width, 1);
      TFLITE_DCHECK(leading_width_padding);
      TFLITE_DCHECK(trailing_width_padding);
      // ASM should use MOVI 64-bit set.
      padding_mask = vcreate_u64(~0xffffff00L);

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        half_work_reg = vld1_lane_s8(reinterpret_cast<const int8*>(
                                         input_block_data + input_block_offset),
                                     half_work_reg, 1);
        half_work_reg =
            vld1_lane_s8(reinterpret_cast<const int8*>(input_block_data +
                                                       input_block_offset + 1),
                         half_work_reg, 2);
        half_work_reg =
            vld1_lane_s8(reinterpret_cast<const int8*>(input_block_data +
                                                       input_block_offset + 2),
                         half_work_reg, 3);
        half_work_reg =
            vbsl_s8(padding_mask, vget_low_s8(padding_reg), half_work_reg);

        half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
        TFLITE_DCHECK_EQ(scratch_data_offset % 8, 0);
        vst1_s8(scratch_data_base + scratch_data_offset, half_work_reg);

        // Trailing guard.
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 4,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 8,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 12,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 16,
                      half_work_reg, 0);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else {
      TFLITE_DCHECK_EQ(width_overall_micro_repeats, 1);
      const int copy_remaining = (copy_size + start_width) & 0x3;
      padding_mask = vshl_u64(padding_mask, vdup_n_s64(8 * copy_remaining));
      if (leading_width_padding) {
        padding_mask = vset_lane_u8(255, padding_mask, 0);
      }

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        for (int i = 0; i < copy_size; ++i) {
          half_work_reg = vshl_n_u64(half_work_reg, 8);
          half_work_reg = vld1_lane_s8(
              reinterpret_cast<const int8*>(
                  input_block_data + input_block_offset + copy_size - 1 - i),
              half_work_reg, 0);
        }
        if (leading_width_padding) {
          half_work_reg = vshl_n_s64(half_work_reg, 8);
        }
        half_work_reg =
            vbsl_s8(padding_mask, vget_low_s8(padding_reg), half_work_reg);

        half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
        TFLITE_DCHECK_EQ(scratch_data_offset % 4, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset, half_work_reg,
                      0);

        // Trailing guard.
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 4,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 8,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 12,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 16,
                      half_work_reg, 0);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    }

    scratch_data_base += copy_block_height * workspace_height_stride;

    if (trailing_height_padding) {
      memset(scratch_data_base, -input_offset_difference,
             workspace_height_stride + kWorkspaceExtension);
      scratch_data_base += workspace_height_stride;
    }

    TFLITE_DCHECK_EQ(
        scratch_data_base,
        scratch_block_data + block_height * workspace_height_stride);
  }

  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    PreloadInputBlock(input_block_data, function_params);
    PackMacroBlockNeon(height_block_number, width_block_number,
                       input_block_data, scratch_block_data, function_params);
  }
};

template <>
struct PackMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kUnitInputDepth,
                      /*max_padding=*/0> {
  static inline void PackMacroBlockNeon(
      int32 height_block_number, int32 width_block_number,
      const uint8* input_block_data, int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int input_width_micro_repeats =
        function_params->input_width_micro_repeats;
    const int block_height = function_params->inbound_block_height;
    const int residual_width = function_params->residual_width;
    const int input_height_stride = function_params->input_height_stride;

    TFLITE_DCHECK_EQ(function_params->padding_left, 0);
    TFLITE_DCHECK_EQ(function_params->padding_right, 0);
    TFLITE_DCHECK_EQ(function_params->padding_top, 0);
    TFLITE_DCHECK_EQ(function_params->padding_bottom, 0);

    TFLITE_DCHECK_GE(workspace_height_stride, 4 * width_overall_micro_repeats);

    // Work through one slice, by row, at a time.
    int8* scratch_data_base = scratch_block_data;

    const int copy_block_height = block_height;

    int adjusted_residual_width =
        input_width_micro_repeats < width_overall_micro_repeats ? residual_width
                                                                : 4;

    const int copy_size =
        (width_overall_micro_repeats - 1) * 4 + adjusted_residual_width;

    TFLITE_DCHECK_LE(
        copy_size,
        input_height_stride - width_block_number * input_width_micro_repeats);
    // We may drop up to stride-1 of trailing input.
    TFLITE_DCHECK_GE(copy_size, input_height_stride - 1);

    int scratch_data_offset = 0;
    int input_block_offset = 0;

    constexpr uint8 kSignBit = 0x80;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in NEON
    // code. Note the blocks of 4x4 are still interleaved down the depth.
    int8x16_t work_reg;
    int8x8_t half_work_reg;

    // Effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);
    half_work_reg = vdup_n_s8(0);

    if (copy_size >= 16) {
      const int copy_remaining = copy_size & 0x7;

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // Main copy loop.
        for (; (copy_done + 16) <= copy_size; copy_done += 16) {
          work_reg =
              vld1q_u8(input_block_data + input_block_offset + copy_done);
          work_reg = veorq_s8(work_reg, sign_bit);
          TFLITE_DCHECK_EQ(copy_done % 16, 0);
          vst1q_s8(scratch_data + copy_done, work_reg);
        }

        if (copy_done + 8 <= copy_size) {
          half_work_reg =
              vld1_u8(input_block_data + input_block_offset + copy_done);
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ(copy_done % 8, 0);
          vst1_s8(scratch_data + copy_done, half_work_reg);
          copy_done += 8;
        }

        TFLITE_DCHECK_EQ(copy_remaining, copy_size - copy_done);
        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.

        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size) {
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          // This has the advantage of resulting in zeros after shifting.
          half_work_reg =
              vld1_u8(input_block_data + input_block_offset + copy_size - 8);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (8 - copy_remaining)));

          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ(copy_done % 8, 0);
          vst1_s8(scratch_data + copy_done, half_work_reg);
          copy_done += 8;
        }

        // Trailing guard.
        vst1_s8(scratch_data + copy_done, half_work_reg);
        vst1_s8(scratch_data + copy_done + 8, half_work_reg);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else if (copy_size >= 4) {
      const int copy_remaining = copy_size & 0x3;

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // Main copy loop.
        for (; (copy_done + 4) <= copy_size; copy_done += 4) {
          // Important! Most compilation configurations will compile and run
          // without the reinterpret_cast. Sanitizers may fail silently on
          // lane-loading, with a obscure bug or mis-feature probably in
          // unhygienic macro expansion.
          half_work_reg =
              vld1_lane_8x4(input_block_data + input_block_offset + copy_done,
                            half_work_reg, 0);
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ(copy_done % 4, 0);
          vst1_lane_8x4(scratch_data + copy_done, half_work_reg, 0);
        }

        TFLITE_DCHECK_EQ(copy_remaining, copy_size - copy_done);
        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.

        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size) {
          TFLITE_DCHECK_LT(copy_remaining, 4);
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          // This has the advantage of resulting in zeros after shifting.
          half_work_reg = vld1_lane_8x4(
              input_block_data + input_block_offset + copy_size - 4,
              half_work_reg, 0);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (4 - copy_remaining)));

          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          TFLITE_DCHECK_EQ(copy_done % 4, 0);
          vst1_lane_8x4(scratch_data + copy_done, half_work_reg, 0);
          copy_done += 4;
        }
        // Trailing guard.
        vst1_lane_8x4(scratch_data + copy_done, half_work_reg, 0);
        vst1_lane_8x4(scratch_data + copy_done + 4, half_work_reg, 0);
        vst1_lane_8x4(scratch_data + copy_done + 8, half_work_reg, 0);
        vst1_lane_8x4(scratch_data + copy_done + 12, half_work_reg, 0);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else {
      TFLITE_DCHECK_EQ(width_overall_micro_repeats, 1);

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        for (int i = 0; i < copy_size; ++i) {
          half_work_reg = vshl_n_u64(half_work_reg, 8);
          half_work_reg = vld1_lane_s8(
              reinterpret_cast<const int8*>(
                  input_block_data + input_block_offset + copy_size - 1 - i),
              half_work_reg, 0);
        }

        half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
        TFLITE_DCHECK_EQ(scratch_data_offset % 4, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset, half_work_reg,
                      0);

        // Trailing guard.
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 4,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 8,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 12,
                      half_work_reg, 0);
        vst1_lane_8x4(scratch_data_base + scratch_data_offset + 16,
                      half_work_reg, 0);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    }

    scratch_data_base += copy_block_height * workspace_height_stride;

    TFLITE_DCHECK_EQ(
        scratch_data_base,
        scratch_block_data + block_height * workspace_height_stride);
  }

  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    PreloadInputBlock(input_block_data, function_params);
    PackMacroBlockNeon(height_block_number, width_block_number,
                       input_block_data, scratch_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kNoMultiplication,
                        /*stride=*/1> {
  static inline void KernelMacroBlockNeon(
      const int8* scratch_block_data, const int8* filter_workspace,
      const int32* bias_data, uint8* output_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int input_width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int depth = function_params->input_depth;

    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    const int bias_increment = function_params->bias_increment;

    TFLITE_DCHECK(depth_micro_repeats > 0);
    const int width_micro_stride = 4 * 8;
    const int depth_micro_stride =
        width_micro_stride * input_width_overall_micro_repeats;

    const int32 output_activation_min =
        function_params->quantized_activation_min;
    const int32 output_activation_max =
        function_params->quantized_activation_max;
    const int32 output_multiplier = function_params->output_multiplier;
    const int32 output_shift = function_params->output_shift;
    const int32 output_offset = function_params->output_offset;
    TFLITE_DCHECK_GE(output_activation_min, 0);
    TFLITE_DCHECK_LT(output_activation_min, 256);
    TFLITE_DCHECK_GE(output_activation_max, 0);
    TFLITE_DCHECK_LT(output_activation_max, 256);
    TFLITE_DCHECK_GE(output_offset, -32878);
    TFLITE_DCHECK_LT(output_offset, 32768);

    const int16x8_t output_offset_vec =
        vdupq_n_s16(static_cast<int16>(output_offset));
    const uint8x16_t output_activation_min_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_min));
    const uint8x16_t output_activation_max_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_max));

    const int8* input_data_depthwise = scratch_block_data;
    uint8* output_data_depthwise = output_block_data;
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      // Simulate NEON-register transposition of subset of filter.
      int8x16_t filter_reg_0_a;
      int8x16_t filter_reg_0_b;
      int8x16_t filter_reg_1_a;
      int8x16_t filter_reg_1_b;
      int8x16_t filter_reg_2_a;
      int8x16_t filter_reg_2_b;
      int8x16_t filter_reg_0_a_shifted;
      int8x16_t filter_reg_1_a_shifted;
      int8x16_t filter_reg_2_a_shifted;

      filter_reg_0_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_0_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;

      filter_reg_0_a_shifted = vshlq_n_u32(filter_reg_0_a, 8);
      filter_reg_1_a_shifted = vshlq_n_u32(filter_reg_1_a, 8);
      filter_reg_2_a_shifted = vshlq_n_u32(filter_reg_2_a, 8);

      if (block_height == 4) {
        for (int s = 0; s < 2; ++s) {
          // Work through one slice, by row, at a time.
          const int8* input_data_base = input_data_depthwise + 2 * 8 * s;
          uint8* output_data_base = output_data_depthwise + 4 * s;

          const int8* next_input_data = input_data_base;
          uint8* output_data = output_data_base;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);
          bias_data += bias_increment;

          // Load first sub-micro block of data into operational banks.
          int8x16_t left_bank_0_reg = vld1q_s8(next_input_data);
          int8x16_t left_bank_1_reg =
              vld1q_s8(next_input_data + workspace_height_stride);
          int8x16_t left_bank_2_reg =
              vld1q_s8(next_input_data + 2 * workspace_height_stride);
          int8x16_t left_bank_3_reg =
              vld1q_s8(next_input_data + 3 * workspace_height_stride);
          int8x16_t left_bank_4_reg =
              vld1q_s8(next_input_data + 4 * workspace_height_stride);
          int8x16_t left_bank_5_reg =
              vld1q_s8(next_input_data + 5 * workspace_height_stride);

          int32x4_t acc0;
          int32x4_t acc1;
          int32x4_t acc2;
          int32x4_t acc3;

          acc0 = adjusted_bias_data;
          acc1 = adjusted_bias_data;
          acc2 = adjusted_bias_data;
          acc3 = adjusted_bias_data;

          acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
          acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_2_reg);
          acc2 = vdotq_s32(acc2, filter_reg_0_a, left_bank_2_reg);
          acc3 = vdotq_s32(acc3, filter_reg_0_a, left_bank_3_reg);

          for (int i_width = 0; i_width < output_width_micro_repeats;
               ++i_width) {
            next_input_data += width_micro_stride;

            // Iterate over input width shifts within 4x4 blocks.
            {
              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_1_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_2_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_1_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_2_a, left_bank_5_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              output_data += depth;
            }

            // Load next sub-micro block of data.
            int8x16_t right_bank_0_reg;
            int8x16_t right_bank_1_reg;
            int8x16_t right_bank_2_reg;
            int8x16_t right_bank_3_reg;
            int8x16_t right_bank_4_reg;
            int8x16_t right_bank_5_reg;

            // Loading of next block always valid.
            right_bank_0_reg = vld1q_s8(next_input_data);
            right_bank_1_reg =
                vld1q_s8(next_input_data + workspace_height_stride);
            right_bank_2_reg =
                vld1q_s8(next_input_data + 2 * workspace_height_stride);
            right_bank_3_reg =
                vld1q_s8(next_input_data + 3 * workspace_height_stride);
            right_bank_4_reg =
                vld1q_s8(next_input_data + 4 * workspace_height_stride);
            right_bank_5_reg =
                vld1q_s8(next_input_data + 5 * workspace_height_stride);

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a_shifted, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a_shifted, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a_shifted, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a_shifted, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a_shifted, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a_shifted, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_0_a_shifted, left_bank_2_reg);
              acc2 = vdotq_s32(acc2, filter_reg_1_a_shifted, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_2_a_shifted, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_0_a_shifted, left_bank_3_reg);
              acc3 = vdotq_s32(acc3, filter_reg_1_a_shifted, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_2_a_shifted, left_bank_5_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              left_bank_0_reg = vrev32q_u16(left_bank_0_reg);
              left_bank_1_reg = vrev32q_u16(left_bank_1_reg);
              left_bank_2_reg = vrev32q_u16(left_bank_2_reg);
              left_bank_3_reg = vrev32q_u16(left_bank_3_reg);
              left_bank_4_reg = vrev32q_u16(left_bank_4_reg);
              left_bank_5_reg = vrev32q_u16(left_bank_5_reg);
              vtrn1_s8x2_in_place(&left_bank_0_reg, &right_bank_0_reg);
              vtrn1_s8x2_in_place(&left_bank_1_reg, &right_bank_1_reg);
              vtrn1_s8x2_in_place(&left_bank_2_reg, &right_bank_2_reg);
              vtrn1_s8x2_in_place(&left_bank_3_reg, &right_bank_3_reg);
              vtrn1_s8x2_in_place(&left_bank_4_reg, &right_bank_4_reg);
              vtrn1_s8x2_in_place(&left_bank_5_reg, &right_bank_5_reg);

              output_data += depth;
            }

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_0_a, left_bank_2_reg);
              acc2 = vdotq_s32(acc2, filter_reg_1_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_2_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_0_a, left_bank_3_reg);
              acc3 = vdotq_s32(acc3, filter_reg_1_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_2_a, left_bank_5_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              output_data += depth;
            }

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a_shifted, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a_shifted, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a_shifted, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a_shifted, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a_shifted, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a_shifted, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_0_a_shifted, left_bank_2_reg);
              acc2 = vdotq_s32(acc2, filter_reg_1_a_shifted, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_2_a_shifted, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_0_a_shifted, left_bank_3_reg);
              acc3 = vdotq_s32(acc3, filter_reg_1_a_shifted, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_2_a_shifted, left_bank_5_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              left_bank_0_reg = right_bank_0_reg;
              left_bank_1_reg = right_bank_1_reg;
              left_bank_2_reg = right_bank_2_reg;
              left_bank_3_reg = right_bank_3_reg;
              left_bank_4_reg = right_bank_4_reg;
              left_bank_5_reg = right_bank_5_reg;

              output_data += depth;
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_2_reg);
              acc2 = vdotq_s32(acc2, filter_reg_0_a, left_bank_2_reg);
              acc3 = vdotq_s32(acc3, filter_reg_0_a, left_bank_3_reg);
            }
          }

          if (residual_width > 0) {
            next_input_data += width_micro_stride;
            const int output_width = residual_width;

            // Load next sub-micro block of data.
            int8x16_t right_bank_0_reg;
            int8x16_t right_bank_1_reg;
            int8x16_t right_bank_2_reg;
            int8x16_t right_bank_3_reg;
            int8x16_t right_bank_4_reg;
            int8x16_t right_bank_5_reg;
            // Logic: (output_width - 1) * stride_val < 2.
            const bool no_right_block = output_width < 3;

            if (no_right_block) {
              // Only needed for santizer checks.
              right_bank_0_reg = vdupq_n_s8(0);
              right_bank_1_reg = vdupq_n_s8(0);
              right_bank_2_reg = vdupq_n_s8(0);
              right_bank_3_reg = vdupq_n_s8(0);
              right_bank_4_reg = vdupq_n_s8(0);
              right_bank_5_reg = vdupq_n_s8(0);
            } else {
              right_bank_0_reg = vld1q_s8(next_input_data);
              right_bank_1_reg =
                  vld1q_s8(next_input_data + workspace_height_stride);
              right_bank_2_reg =
                  vld1q_s8(next_input_data + 2 * workspace_height_stride);
              right_bank_3_reg =
                  vld1q_s8(next_input_data + 3 * workspace_height_stride);
              right_bank_4_reg =
                  vld1q_s8(next_input_data + 4 * workspace_height_stride);
              right_bank_5_reg =
                  vld1q_s8(next_input_data + 5 * workspace_height_stride);
            }

            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_1_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_1_a, left_bank_3_reg);
              acc2 = vdotq_s32(acc2, filter_reg_2_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_1_a, left_bank_4_reg);
              acc3 = vdotq_s32(acc3, filter_reg_2_a, left_bank_5_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              biregister_rotate_8(&left_bank_0_reg, &right_bank_0_reg);
              biregister_rotate_8(&left_bank_1_reg, &right_bank_1_reg);
              biregister_rotate_8(&left_bank_2_reg, &right_bank_2_reg);
              biregister_rotate_8(&left_bank_3_reg, &right_bank_3_reg);
              biregister_rotate_8(&left_bank_4_reg, &right_bank_4_reg);
              biregister_rotate_8(&left_bank_5_reg, &right_bank_5_reg);

              output_data += depth;

              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_2_reg);
              acc2 = vdotq_s32(acc2, filter_reg_0_a, left_bank_2_reg);
              acc3 = vdotq_s32(acc3, filter_reg_0_a, left_bank_3_reg);
            }
          }
          input_data_base += 4 * workspace_height_stride;
          output_data_base += 4 * output_height_stride;

          // Move to next sub-block: advance to second set of filters, to new
          // bias.
          filter_reg_0_a = filter_reg_0_b;
          filter_reg_1_a = filter_reg_1_b;
          filter_reg_2_a = filter_reg_2_b;
          filter_reg_0_a_shifted = vshlq_n_u32(filter_reg_0_a, 8);
          filter_reg_1_a_shifted = vshlq_n_u32(filter_reg_1_a, 8);
          filter_reg_2_a_shifted = vshlq_n_u32(filter_reg_2_a, 8);
        }
      } else {
        for (int s = 0; s < 2; ++s) {
          // Work through one slice, by row, at a time.
          const int8* input_data_base = input_data_depthwise + 2 * 8 * s;
          uint8* output_data_base = output_data_depthwise + 4 * s;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);
          bias_data += bias_increment;

          for (int k_height = 0; k_height < block_height; ++k_height) {
            const int8* next_input_data = input_data_base;
            uint8* output_data = output_data_base;

            // Load first sub-micro block of data into operational banks.
            int8x16_t left_bank_0_reg = vld1q_s8(next_input_data);
            int8x16_t left_bank_1_reg =
                vld1q_s8(next_input_data + workspace_height_stride);
            int8x16_t left_bank_2_reg =
                vld1q_s8(next_input_data + 2 * workspace_height_stride);

            for (int i_width = 0; i_width < output_width_overall_micro_repeats;
                 ++i_width) {
              next_input_data += width_micro_stride;
              const int output_width =
                  i_width == output_width_micro_repeats ? residual_width : 4;

              // Load next sub-micro block of data.
              int8x16_t right_bank_0_reg;
              int8x16_t right_bank_1_reg;
              int8x16_t right_bank_2_reg;
              // Logic: (output_width - 1) * stride_val < 2.
              const bool no_right_block = output_width < 3;

              if (no_right_block) {
                // Only needed for santizer checks.
                right_bank_0_reg = vdupq_n_s8(0);
                right_bank_1_reg = vdupq_n_s8(0);
                right_bank_2_reg = vdupq_n_s8(0);
              } else {
                right_bank_0_reg = vld1q_s8(next_input_data);
                right_bank_1_reg =
                    vld1q_s8(next_input_data + workspace_height_stride);
                right_bank_2_reg =
                    vld1q_s8(next_input_data + 2 * workspace_height_stride);
              }
              // Load next sub-micro block of data.

              // Iterate over input width shifts within 4x4 blocks.
              for (int x = 0; x < output_width; ++x) {
                int32x4_t acc = adjusted_bias_data;
                acc = vdotq_s32(acc, filter_reg_0_a, left_bank_0_reg);
                acc = vdotq_s32(acc, filter_reg_1_a, left_bank_1_reg);
                acc = vdotq_s32(acc, filter_reg_2_a, left_bank_2_reg);

                // Fixed-point multiplication.
                acc = vqrdmulhq_n_s32(acc, output_multiplier);
                acc = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                    acc, -output_shift);
                // Add the output offset.
                // Note that we need to fill the top half with vcombine, but can
                // drop the instruction in ASM code.
                int16x8_t acc_s16_0_0 =
                    vcombine_s16(vqmovn_s32(acc), vqmovn_s32(acc));
                acc_s16_0_0 = vqaddq_s16(acc_s16_0_0, output_offset_vec);
                // Apply the activation function.
                uint8x8_t acc_u8_0_0 = vqmovun_s16(acc_s16_0_0);
                acc_u8_0_0 =
                    vmax_u8(acc_u8_0_0, vget_low_u8(output_activation_min_vec));
                acc_u8_0_0 =
                    vmin_u8(acc_u8_0_0, vget_low_u8(output_activation_max_vec));

                vst1_lane_8x4(output_data, acc_u8_0_0, 0);

                biregister_rotate_8(&left_bank_0_reg, &right_bank_0_reg);
                biregister_rotate_8(&left_bank_1_reg, &right_bank_1_reg);
                biregister_rotate_8(&left_bank_2_reg, &right_bank_2_reg);

                output_data += depth;
              }
            }
            input_data_base += workspace_height_stride;
            output_data_base += output_height_stride;
          }

          // Move to next sub-block: advance to second set of filters.
          filter_reg_0_a = filter_reg_0_b;
          filter_reg_1_a = filter_reg_1_b;
          filter_reg_2_a = filter_reg_2_b;
        }
      }
      input_data_depthwise += depth_micro_stride;
      output_data_depthwise += 8;
    }
  }  // NOLINT(readability/fn_size) Manually unrolled.

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockNeon(scratch_block_data, filter_workspace, bias_data,
                         output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kNoMultiplication,
                        /*stride=*/2> {
  static inline void KernelMacroBlockNeon(
      const int8* scratch_block_data, const int8* filter_workspace,
      const int32* bias_data, uint8* output_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int input_width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int depth = function_params->input_depth;
    const int stride_val = function_params->stride;
    const int four_over_stride = function_params->four_over_stride;

    const int workspace_width_micro_repeats =
        function_params->workspace_width_micro_repeats;
    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    const int bias_increment = function_params->bias_increment;

    TFLITE_DCHECK(depth_micro_repeats > 0);
    const int width_micro_stride = 4 * 8;
    const int depth_micro_stride =
        width_micro_stride * input_width_overall_micro_repeats;

    const int32 output_activation_min =
        function_params->quantized_activation_min;
    const int32 output_activation_max =
        function_params->quantized_activation_max;
    const int32 output_multiplier = function_params->output_multiplier;
    const int32 output_shift = function_params->output_shift;
    const int32 output_offset = function_params->output_offset;
    TFLITE_DCHECK_GE(output_activation_min, 0);
    TFLITE_DCHECK_LT(output_activation_min, 256);
    TFLITE_DCHECK_GE(output_activation_max, 0);
    TFLITE_DCHECK_LT(output_activation_max, 256);
    TFLITE_DCHECK_GE(output_offset, -32878);
    TFLITE_DCHECK_LT(output_offset, 32768);

    // This version only does min/max on 64 bits.
    const int16x8_t output_offset_vec =
        vdupq_n_s16(static_cast<int16>(output_offset));
    const uint8x8_t output_activation_min_vec =
        vdup_n_u8(static_cast<uint8>(output_activation_min));
    const uint8x8_t output_activation_max_vec =
        vdup_n_u8(static_cast<uint8>(output_activation_max));

    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;

    TFLITE_DCHECK_EQ(stride_val, 2);
    TFLITE_DCHECK_LE(block_height, 2);

    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      const int8* filter_block =
          filter_workspace + shuffled_filter_increment * j_depth;

      if (block_height == 2) {
        for (int s = 0; s < 2; ++s) {
          // Simulate NEON-register transposition of subset of filter.
          int8x16_t filter_reg_0_a;
          int8x16_t filter_reg_1_a;
          int8x16_t filter_reg_2_a;

          filter_reg_0_a = vld1q_s8(filter_block + s * 16);
          filter_reg_1_a = vld1q_s8(filter_block + s * 16 + 32);
          filter_reg_2_a = vld1q_s8(filter_block + s * 16 + 64);

          const int8* scratch_data =
              scratch_block_data + depth_micro_stride * j_depth;
          uint8* output_data = output_block_data + 8 * j_depth;
          const int8* input_data_0 = scratch_data + s * 2 * 8;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);

          // Load first sub-micro block of data into operational banks.
          int8x16_t left_bank_0_reg = vld1q_s8(input_data_0);
          int8x16_t left_bank_1_reg =
              vld1q_s8(input_data_0 + workspace_height_stride);
          int8x16_t left_bank_2_reg =
              vld1q_s8(input_data_0 + 2 * workspace_height_stride);
          int8x16_t left_bank_3_reg =
              vld1q_s8(input_data_0 + 3 * workspace_height_stride);
          int8x16_t left_bank_4_reg =
              vld1q_s8(input_data_0 + 4 * workspace_height_stride);

          int8x16_t right_bank_0_reg;
          int8x16_t right_bank_1_reg;
          int8x16_t right_bank_2_reg;
          int8x16_t right_bank_3_reg;
          int8x16_t right_bank_4_reg;

          int32x4_t acc0;
          int32x4_t acc1;
          int16x8_t acc_s16_0_1;
          uint8x8_t acc_u8;

          int i_width = 0;

          // When output_width_micro_repeats <
          // output_width_overall_micro_repeats, 0 < residual_width <= 2, and so
          // residual_width == 1 is then true iff residual_width < 2.
          const int adjusted_width_micro_repeats =
              (output_width_micro_repeats <
               output_width_overall_micro_repeats) &&
                      (residual_width == 1)
                  ? output_width_micro_repeats
                  : output_width_overall_micro_repeats;

          for (; i_width < adjusted_width_micro_repeats; ++i_width) {
            const int output_width = four_over_stride;
            TFLITE_DCHECK_LE(output_width * stride_val, 4);
            const int8* input_data =
                input_data_0 + width_micro_stride * i_width;
            acc0 = adjusted_bias_data;
            acc1 = adjusted_bias_data;
            right_bank_0_reg = vld1q_s8(input_data + width_micro_stride);
            right_bank_1_reg = vld1q_s8(input_data + width_micro_stride +
                                        workspace_height_stride);

            acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
            acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_2_reg);
            uint8* output_data_base = output_data + depth * 2 * i_width + 4 * s;

            right_bank_2_reg = vld1q_s8(input_data + width_micro_stride +
                                        2 * workspace_height_stride);
            right_bank_3_reg = vld1q_s8(input_data + width_micro_stride +
                                        3 * workspace_height_stride);
            acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
            acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
            acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_3_reg);
            acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_4_reg);
            right_bank_4_reg = vld1q_s8(input_data + width_micro_stride +
                                        4 * workspace_height_stride);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8 = vqmovun_s16(acc_s16_0_1);
            acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
            acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

            left_bank_0_reg = vrev32q_u16(left_bank_0_reg);
            left_bank_1_reg = vrev32q_u16(left_bank_1_reg);
            left_bank_2_reg = vrev32q_u16(left_bank_2_reg);
            left_bank_3_reg = vrev32q_u16(left_bank_3_reg);
            left_bank_4_reg = vrev32q_u16(left_bank_4_reg);
            acc0 = adjusted_bias_data;
            acc1 = adjusted_bias_data;
            vtrn1_s8x2_in_place(&left_bank_0_reg, &right_bank_0_reg);
            vtrn1_s8x2_in_place(&left_bank_1_reg, &right_bank_1_reg);
            vtrn1_s8x2_in_place(&left_bank_2_reg, &right_bank_2_reg);
            vst1_lane_8x4(output_data_base, acc_u8, 0);
            vst1_lane_8x4(output_data_base + output_height_stride, acc_u8, 1);

            vtrn1_s8x2_in_place(&left_bank_3_reg, &right_bank_3_reg);
            vtrn1_s8x2_in_place(&left_bank_4_reg, &right_bank_4_reg);

            acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
            acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_2_reg);
            acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
            acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_3_reg);
            acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
            acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_4_reg);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8 = vqmovun_s16(acc_s16_0_1);
            acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
            acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

            vst1_lane_8x4(output_data_base + depth, acc_u8, 0);
            vst1_lane_8x4(output_data_base + depth + output_height_stride,
                          acc_u8, 1);

            left_bank_0_reg = right_bank_0_reg;
            left_bank_1_reg = right_bank_1_reg;
            left_bank_2_reg = right_bank_2_reg;
            left_bank_3_reg = right_bank_3_reg;
            left_bank_4_reg = right_bank_4_reg;
          }
          for (; i_width < output_width_overall_micro_repeats; ++i_width) {
            TFLITE_DCHECK_NE(residual_width, four_over_stride);

            // No need to load next ("right") block of data.

            uint8* output_data_base = output_data + depth * 2 * i_width + 4 * s;

            // Iterate over input width shifts within 4x4 blocks.
            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_0_a, left_bank_2_reg);
              acc1 = vdotq_s32(acc1, filter_reg_1_a, left_bank_3_reg);
              acc1 = vdotq_s32(acc1, filter_reg_2_a, left_bank_4_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              // Apply the activation function.
              uint8x8_t acc_u8 = vqmovun_s16(acc_s16_0_1);
              acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
              acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

              vst1_lane_8x4(output_data_base, acc_u8, 0);
              vst1_lane_8x4(output_data_base + output_height_stride, acc_u8, 1);

              left_bank_0_reg = vrev32q_u16(left_bank_0_reg);
              left_bank_1_reg = vrev32q_u16(left_bank_1_reg);
              left_bank_2_reg = vrev32q_u16(left_bank_2_reg);
              left_bank_3_reg = vrev32q_u16(left_bank_3_reg);
              left_bank_4_reg = vrev32q_u16(left_bank_4_reg);
              vtrn1_s8x2_in_place(&left_bank_0_reg, &right_bank_0_reg);
              vtrn1_s8x2_in_place(&left_bank_1_reg, &right_bank_1_reg);
              vtrn1_s8x2_in_place(&left_bank_2_reg, &right_bank_2_reg);
              vtrn1_s8x2_in_place(&left_bank_3_reg, &right_bank_3_reg);
              vtrn1_s8x2_in_place(&left_bank_4_reg, &right_bank_4_reg);
            }
          }
          bias_data += bias_increment;
        }
      } else {
        for (int s = 0; s < 2; ++s) {
          // Simulate NEON-register transposition of subset of filter.
          int8x16_t filter_reg_0_a;
          int8x16_t filter_reg_1_a;
          int8x16_t filter_reg_2_a;

          filter_reg_0_a = vld1q_s8(filter_block + s * 16);
          filter_reg_1_a = vld1q_s8(filter_block + s * 16 + 32);
          filter_reg_2_a = vld1q_s8(filter_block + s * 16 + 64);

          const int8* scratch_data =
              scratch_block_data + depth_micro_stride * j_depth;
          uint8* output_data = output_block_data + 8 * j_depth;
          const int8* input_data_0 = scratch_data + s * 2 * 8;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);

          // Load first sub-micro block of data into operational banks.
          int8x16_t left_bank_0_reg = vld1q_s8(input_data_0);
          int8x16_t left_bank_1_reg =
              vld1q_s8(input_data_0 + workspace_height_stride);
          int8x16_t left_bank_2_reg =
              vld1q_s8(input_data_0 + 2 * workspace_height_stride);

          int8x16_t right_bank_0_reg;
          int8x16_t right_bank_1_reg;
          int8x16_t right_bank_2_reg;

          int32x4_t acc0;

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            const int output_width = i_width == output_width_micro_repeats
                                         ? residual_width
                                         : four_over_stride;
            TFLITE_DCHECK_LE(output_width * stride_val, 4);
            const int8* input_data =
                input_data_0 + width_micro_stride * i_width;
            const bool no_right_block = i_width == output_width_micro_repeats &&
                                        output_width_overall_micro_repeats ==
                                            workspace_width_micro_repeats;

            if (!no_right_block) {
              // Load next sub-micro block of data.
              right_bank_0_reg = vld1q_s8(input_data + width_micro_stride);
              right_bank_1_reg = vld1q_s8(input_data + width_micro_stride +
                                          workspace_height_stride);
              right_bank_2_reg = vld1q_s8(input_data + width_micro_stride +
                                          2 * workspace_height_stride);
            }

            uint8* output_data_base = output_data + depth * 2 * i_width + 4 * s;

            // Iterate over input width shifts within 4x4 blocks.
            {
              acc0 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc0));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              // Apply the activation function.
              uint8x8_t acc_u8 = vqmovun_s16(acc_s16_0_1);
              acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
              acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

              vst1_lane_8x4(output_data_base, acc_u8, 0);

              left_bank_0_reg = vrev32q_u16(left_bank_0_reg);
              left_bank_1_reg = vrev32q_u16(left_bank_1_reg);
              left_bank_2_reg = vrev32q_u16(left_bank_2_reg);
              vtrn1_s8x2_in_place(&left_bank_0_reg, &right_bank_0_reg);
              vtrn1_s8x2_in_place(&left_bank_1_reg, &right_bank_1_reg);
              vtrn1_s8x2_in_place(&left_bank_2_reg, &right_bank_2_reg);
            }

            if (output_width > 1) {
              acc0 = adjusted_bias_data;

              acc0 = vdotq_s32(acc0, filter_reg_0_a, left_bank_0_reg);
              acc0 = vdotq_s32(acc0, filter_reg_1_a, left_bank_1_reg);
              acc0 = vdotq_s32(acc0, filter_reg_2_a, left_bank_2_reg);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc0));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              // Apply the activation function.
              uint8x8_t acc_u8 = vqmovun_s16(acc_s16_0_1);
              acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
              acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

              vst1_lane_8x4(output_data_base + depth, acc_u8, 0);

              left_bank_0_reg = right_bank_0_reg;
              left_bank_1_reg = right_bank_1_reg;
              left_bank_2_reg = right_bank_2_reg;
            }
          }
          bias_data += bias_increment;
        }
      }
    }
  }  // NOLINT(readability/fn_size) Manually unrolled.

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockNeon(scratch_block_data, filter_workspace, bias_data,
                         output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kUnitInputDepth,
                        /*stride=*/1> {
  static inline void KernelMacroBlockNeon(
      const int8* scratch_block_data, const int8* filter_workspace,
      const int32* bias_data, uint8* output_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK_EQ(function_params->stride, 1);
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int output_depth = function_params->output_depth;

    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    const int bias_increment = function_params->bias_increment;

    TFLITE_DCHECK(depth_micro_repeats > 0);

    TFLITE_DCHECK_EQ(bias_increment, 4);

    const int32 output_activation_min =
        function_params->quantized_activation_min;
    const int32 output_activation_max =
        function_params->quantized_activation_max;
    const int32 output_multiplier = function_params->output_multiplier;
    const int32 output_shift = function_params->output_shift;
    const int32 output_offset = function_params->output_offset;
    TFLITE_DCHECK_GE(output_activation_min, 0);
    TFLITE_DCHECK_LT(output_activation_min, 256);
    TFLITE_DCHECK_GE(output_activation_max, 0);
    TFLITE_DCHECK_LT(output_activation_max, 256);
    TFLITE_DCHECK_GE(output_offset, -32878);
    TFLITE_DCHECK_LT(output_offset, 32768);

    const int16x8_t output_offset_vec =
        vdupq_n_s16(static_cast<int16>(output_offset));
    const uint8x16_t output_activation_min_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_min));
    const uint8x16_t output_activation_max_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_max));

    uint8* output_data_depthwise = output_block_data;
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      // Simulate NEON-register transposition of subset of filter.
      int8x16_t filter_reg_0_a;
      int8x16_t filter_reg_0_b;
      int8x16_t filter_reg_1_a;
      int8x16_t filter_reg_1_b;
      int8x16_t filter_reg_2_a;
      int8x16_t filter_reg_2_b;
      int8x16_t filter_reg_0_a_shifted;
      int8x16_t filter_reg_1_a_shifted;
      int8x16_t filter_reg_2_a_shifted;

      filter_reg_0_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_0_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;

      filter_reg_0_a_shifted = vshlq_n_u32(filter_reg_0_a, 8);
      filter_reg_1_a_shifted = vshlq_n_u32(filter_reg_1_a, 8);
      filter_reg_2_a_shifted = vshlq_n_u32(filter_reg_2_a, 8);

      // When output_width_micro_repeats < output_width_overall_micro_repeats,
      // 0 < residual_width <= 2, and so residual_width == 1 is then true iff
      // residual_width < 2.
      const int adjusted_width_micro_repeats =
          (output_width_micro_repeats < output_width_overall_micro_repeats) &&
                  (residual_width < 4)
              ? output_width_micro_repeats
              : output_width_overall_micro_repeats;

      if (block_height == 4) {
        for (int s = 0; s < 2; ++s) {
          // Work through one slice, by row, at a time.
          uint8* output_data_base = output_data_depthwise + 4 * s;

          const int8* next_input_data = scratch_block_data;
          uint8* output_data = output_data_base;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);
          bias_data += bias_increment;

          int8x16_t input_bank_a_reg;  //  left 0, right 0, left 1, right 1.
          int8x16_t input_bank_b_reg;  //  left 2, right 2, left 3, right 3.
          int8x16_t input_bank_c_reg;  //  left 4, right 4, left 5, right 5.

          // Load first sub-micro block of data into operational banks.
          input_bank_a_reg =
              vld1q_dup_s8x4(next_input_data);  // Load lane 0, avoiding
                                                // uninitialized variable.
          input_bank_a_reg = vld1q_lane_8x4(
              next_input_data + workspace_height_stride, input_bank_a_reg, 2);
          input_bank_b_reg = vld1q_dup_s8x4(
              next_input_data +
              2 * workspace_height_stride);  // Load lane 0, avoiding
                                             // uninitialized variable.
          input_bank_b_reg =
              vld1q_lane_8x4(next_input_data + 3 * workspace_height_stride,
                             input_bank_b_reg, 2);
          input_bank_c_reg = vld1q_dup_s8x4(
              next_input_data +
              4 * workspace_height_stride);  // Load lane 0, avoiding
                                             // uninitialized variable.
          input_bank_c_reg =
              vld1q_lane_8x4(next_input_data + 5 * workspace_height_stride,
                             input_bank_c_reg, 2);

          int32x4_t acc0;
          int32x4_t acc1;
          int32x4_t acc2;
          int32x4_t acc3;

          acc0 = adjusted_bias_data;
          acc1 = adjusted_bias_data;
          acc2 = adjusted_bias_data;
          acc3 = adjusted_bias_data;

          acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg, 0);
          acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a, input_bank_b_reg, 0);
          acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a, input_bank_b_reg, 2);

          int i_width = 0;
          for (; i_width < adjusted_width_micro_repeats; ++i_width) {
            next_input_data += 4;

            // Iterate over input width shifts within 4x4 blocks.
            {
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg,
                                         0);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg,
                                         2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_a_reg,
                                         2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_1_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_2_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_1_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_2_a, input_bank_c_reg,
                                         2);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              output_data += output_depth;
            }
            // Load next sub-micro block of data.
            input_bank_a_reg =
                vld1q_lane_8x4(next_input_data, input_bank_a_reg, 1);
            input_bank_a_reg = vld1q_lane_8x4(
                next_input_data + workspace_height_stride, input_bank_a_reg, 3);
            input_bank_b_reg =
                vld1q_lane_8x4(next_input_data + 2 * workspace_height_stride,
                               input_bank_b_reg, 1);
            input_bank_b_reg =
                vld1q_lane_8x4(next_input_data + 3 * workspace_height_stride,
                               input_bank_b_reg, 3);
            input_bank_c_reg =
                vld1q_lane_8x4(next_input_data + 4 * workspace_height_stride,
                               input_bank_c_reg, 1);
            input_bank_c_reg =
                vld1q_lane_8x4(next_input_data + 5 * workspace_height_stride,
                               input_bank_c_reg, 3);

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a_shifted,
                                         input_bank_a_reg, 0);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a_shifted,
                                         input_bank_a_reg, 2);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a_shifted,
                                         input_bank_b_reg, 0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a_shifted,
                                         input_bank_a_reg, 2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a_shifted,
                                         input_bank_b_reg, 0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a_shifted,
                                         input_bank_b_reg, 2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a_shifted,
                                         input_bank_b_reg, 0);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_1_a_shifted,
                                         input_bank_b_reg, 2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_2_a_shifted,
                                         input_bank_c_reg, 0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a_shifted,
                                         input_bank_b_reg, 2);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_1_a_shifted,
                                         input_bank_c_reg, 0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_2_a_shifted,
                                         input_bank_c_reg, 2);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
              input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);
              input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 16);

              output_data += output_depth;
            }

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg,
                                         0);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg,
                                         2);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg,
                                         0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_a_reg,
                                         2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg,
                                         0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a, input_bank_b_reg,
                                         0);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_1_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_2_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a, input_bank_b_reg,
                                         2);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_1_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_2_a, input_bank_c_reg,
                                         2);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              output_data += output_depth;
            }

            {
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a_shifted,
                                         input_bank_a_reg, 0);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a_shifted,
                                         input_bank_a_reg, 2);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a_shifted,
                                         input_bank_b_reg, 0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a_shifted,
                                         input_bank_a_reg, 2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a_shifted,
                                         input_bank_b_reg, 0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a_shifted,
                                         input_bank_b_reg, 2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a_shifted,
                                         input_bank_b_reg, 0);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_1_a_shifted,
                                         input_bank_b_reg, 2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_2_a_shifted,
                                         input_bank_c_reg, 0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a_shifted,
                                         input_bank_b_reg, 2);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_1_a_shifted,
                                         input_bank_c_reg, 0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_2_a_shifted,
                                         input_bank_c_reg, 2);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
              input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);
              input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 16);

              output_data += output_depth;
              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg,
                                         0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg,
                                         0);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a, input_bank_b_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a, input_bank_b_reg,
                                         2);
            }
          }

          if (i_width < output_width_overall_micro_repeats) {
            next_input_data += 4;
            const int output_width = residual_width;

            // Load next sub-micro block of data.
            input_bank_a_reg =
                vld1q_lane_8x4(next_input_data, input_bank_a_reg, 1);
            input_bank_a_reg = vld1q_lane_8x4(
                next_input_data + workspace_height_stride, input_bank_a_reg, 3);
            input_bank_b_reg =
                vld1q_lane_8x4(next_input_data + 2 * workspace_height_stride,
                               input_bank_b_reg, 1);
            input_bank_b_reg =
                vld1q_lane_8x4(next_input_data + 3 * workspace_height_stride,
                               input_bank_b_reg, 3);
            input_bank_c_reg =
                vld1q_lane_8x4(next_input_data + 4 * workspace_height_stride,
                               input_bank_c_reg, 1);
            input_bank_c_reg =
                vld1q_lane_8x4(next_input_data + 5 * workspace_height_stride,
                               input_bank_c_reg, 3);

            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg,
                                         0);
              acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg,
                                         2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_a_reg,
                                         2);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_1_a, input_bank_b_reg,
                                         2);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_2_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_1_a, input_bank_c_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_2_a, input_bank_c_reg,
                                         2);

              // Fixed-point multiplication.
              acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
              acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc0, -output_shift);
              acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
              acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc1, -output_shift);
              acc2 = vqrdmulhq_n_s32(acc2, output_multiplier);
              acc2 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc2, -output_shift);
              acc3 = vqrdmulhq_n_s32(acc3, output_multiplier);
              acc3 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc3, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_1 =
                  vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
              int16x8_t acc_s16_2_3 =
                  vcombine_s16(vqmovn_s32(acc2), vqmovn_s32(acc3));
              acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
              acc_s16_2_3 = vqaddq_s16(acc_s16_2_3, output_offset_vec);
              // Apply the activation function.
              uint8x16_t acc_u8_all = vcombine_u8(vqmovun_s16(acc_s16_0_1),
                                                  vqmovun_s16(acc_s16_2_3));
              acc_u8_all = vmaxq_u8(acc_u8_all, output_activation_min_vec);
              acc_u8_all = vminq_u8(acc_u8_all, output_activation_max_vec);

              vst1q_lane_8x4(output_data, acc_u8_all, 0);
              vst1q_lane_8x4(output_data + output_height_stride, acc_u8_all, 1);
              vst1q_lane_8x4(output_data + 2 * output_height_stride, acc_u8_all,
                             2);
              vst1q_lane_8x4(output_data + 3 * output_height_stride, acc_u8_all,
                             3);

              input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 8);
              input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 8);
              input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 8);

              output_data += output_depth;

              acc0 = adjusted_bias_data;
              acc1 = adjusted_bias_data;
              acc2 = adjusted_bias_data;
              acc3 = adjusted_bias_data;

              acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg,
                                         0);
              acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg,
                                         0);
              acc2 = vdotq_four_lane_s32(acc2, filter_reg_0_a, input_bank_b_reg,
                                         0);
              acc3 = vdotq_four_lane_s32(acc3, filter_reg_0_a, input_bank_b_reg,
                                         2);
            }
          }
          // scratch_block_data += 4 * workspace_height_stride;
          output_data_base += 4 * output_height_stride;

          // Move to next sub-block: advance to second set of filters, to new
          // bias.
          filter_reg_0_a = filter_reg_0_b;
          filter_reg_1_a = filter_reg_1_b;
          filter_reg_2_a = filter_reg_2_b;
          filter_reg_0_a_shifted = vshlq_n_u32(filter_reg_0_a, 8);
          filter_reg_1_a_shifted = vshlq_n_u32(filter_reg_1_a, 8);
          filter_reg_2_a_shifted = vshlq_n_u32(filter_reg_2_a, 8);
        }
      } else {
        // Block height < 4.
        for (int s = 0; s < 2; ++s) {
          // Work through one slice, by row, at a time.
          uint8* output_data_base = output_data_depthwise + 4 * s;

          const int32x4_t adjusted_bias_data = vld1q_s32(bias_data);
          TFLITE_DCHECK_EQ(bias_increment, 4);
          bias_data += bias_increment;

          for (int k_height = 0; k_height < block_height; ++k_height) {
            const int8* next_input_data =
                scratch_block_data + k_height * workspace_height_stride;
            uint8* output_data = output_data_base;

            int8x16_t input_bank_a_reg;  //  left 0, right 0, left 1, right 1.
            int8x16_t input_bank_b_reg;  //  left 2, right 2, left 3, right 3.

            // Load first sub-micro block of data into operational banks.
            input_bank_a_reg =
                vld1q_dup_s8x4(next_input_data);  // Load lane 0, avoiding
                                                  // uninitialized variable.
            input_bank_a_reg = vld1q_lane_8x4(
                next_input_data + workspace_height_stride, input_bank_a_reg, 2);
            input_bank_b_reg = vld1q_dup_s8x4(
                next_input_data +
                2 * workspace_height_stride);  // Load lane 0, avoiding
                                               // uninitialized variable.

            for (int i_width = 0; i_width < output_width_overall_micro_repeats;
                 ++i_width) {
              next_input_data += 4;
              const int output_width =
                  i_width == output_width_micro_repeats ? residual_width : 4;

              // Load next sub-micro block of data.
              input_bank_a_reg =
                  vld1q_lane_8x4(next_input_data, input_bank_a_reg, 1);
              input_bank_a_reg =
                  vld1q_lane_8x4(next_input_data + workspace_height_stride,
                                 input_bank_a_reg, 3);
              input_bank_b_reg =
                  vld1q_lane_8x4(next_input_data + 2 * workspace_height_stride,
                                 input_bank_b_reg, 1);
              // Iterate over input width shifts within 4x4 blocks.
              for (int x = 0; x < output_width; ++x) {
                int32x4_t acc = adjusted_bias_data;
                acc = vdotq_four_lane_s32(acc, filter_reg_0_a, input_bank_a_reg,
                                          0);
                acc = vdotq_four_lane_s32(acc, filter_reg_1_a, input_bank_a_reg,
                                          2);
                acc = vdotq_four_lane_s32(acc, filter_reg_2_a, input_bank_b_reg,
                                          0);

                // Fixed-point multiplication.
                acc = vqrdmulhq_n_s32(acc, output_multiplier);
                acc = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                    acc, -output_shift);
                // Add the output offset.
                // Note that we need to fill the top half with vcombine, but can
                // drop the instruction in ASM code.
                int16x8_t acc_s16_0_0 =
                    vcombine_s16(vqmovn_s32(acc), vqmovn_s32(acc));
                acc_s16_0_0 = vqaddq_s16(acc_s16_0_0, output_offset_vec);
                // Apply the activation function.
                uint8x8_t acc_u8_0_0 = vqmovun_s16(acc_s16_0_0);
                acc_u8_0_0 =
                    vmax_u8(acc_u8_0_0, vget_low_u8(output_activation_min_vec));
                acc_u8_0_0 =
                    vmin_u8(acc_u8_0_0, vget_low_u8(output_activation_max_vec));

                vst1_lane_8x4(output_data, acc_u8_0_0, 0);

                input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 8);
                input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 8);

                output_data += output_depth;
              }
            }
            output_data_base += output_height_stride;
          }

          // Move to next sub-block: advance to second set of filters.
          filter_reg_0_a = filter_reg_0_b;
          filter_reg_1_a = filter_reg_1_b;
          filter_reg_2_a = filter_reg_2_b;
        }
      }
      output_data_depthwise += 8;
    }
  }  // NOLINT(readability/fn_size) Manually unrolled.

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockNeon(scratch_block_data, filter_workspace, bias_data,
                         output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseNeon3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kUnitInputDepth,
                        /*stride=*/2> {
  static inline void KernelMacroBlockNeon(
      const int8* scratch_block_data, const int8* filter_workspace,
      const int32* bias_data, uint8* output_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int output_depth = function_params->output_depth;
    const int stride_val = function_params->stride;

    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    const int bias_increment = function_params->bias_increment;

    const int32 output_activation_min =
        function_params->quantized_activation_min;
    const int32 output_activation_max =
        function_params->quantized_activation_max;
    const int32 output_multiplier = function_params->output_multiplier;
    const int32 output_shift = function_params->output_shift;
    const int32 output_offset = function_params->output_offset;
    TFLITE_DCHECK_GE(output_activation_min, 0);
    TFLITE_DCHECK_LT(output_activation_min, 256);
    TFLITE_DCHECK_GE(output_activation_max, 0);
    TFLITE_DCHECK_LT(output_activation_max, 256);
    TFLITE_DCHECK_GE(output_offset, -32878);
    TFLITE_DCHECK_LT(output_offset, 32768);

    TFLITE_DCHECK_GE(depth_micro_repeats, 1);
    TFLITE_DCHECK_EQ(bias_increment, 4);

    const int16x8_t output_offset_vec =
        vdupq_n_s16(static_cast<int16>(output_offset));
    const uint8x16_t output_activation_min_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_min));
    const uint8x16_t output_activation_max_vec =
        vdupq_n_u8(static_cast<uint8>(output_activation_max));

    for (int j_depth = 0; j_depth < (depth_micro_repeats * 1 + 0); ++j_depth) {
      int8x16_t filter_reg_0_a;
      int8x16_t filter_reg_0_b;
      int8x16_t filter_reg_1_a;
      int8x16_t filter_reg_1_b;
      int8x16_t filter_reg_2_a;
      int8x16_t filter_reg_2_b;

      filter_reg_0_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_0_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_1_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_a = vld1q_s8(filter_workspace);
      filter_workspace += 16;
      filter_reg_2_b = vld1q_s8(filter_workspace);
      filter_workspace += 16;

      TFLITE_DCHECK_EQ(bias_increment, 4);
      const int32x4_t adjusted_bias_data_s_0 = vld1q_s32(bias_data);
      bias_data += bias_increment;
      const int32x4_t adjusted_bias_data_s_1 = vld1q_s32(bias_data);
      bias_data += bias_increment;

      if (block_height == 2) {
        const int8* scratch_data = scratch_block_data;
        uint8* output_data = output_block_data + 8 * j_depth;

        int8x16_t input_bank_a_reg;  //  left 0, right 0, left 1, right 1.
        int8x16_t input_bank_b_reg;  //  left 2, right 2, left 3, right 3.
        int8x16_t input_bank_c_reg;  //  left 4, right 4, xxx, xxx.

        // Load first sub-micro block of data into operational banks.
        input_bank_a_reg =
            vld1q_dup_s8x4(scratch_data);  // Load lane 0, avoiding
                                           // uninitialized variable.
        input_bank_a_reg = vld1q_lane_8x4(
            scratch_data + workspace_height_stride, input_bank_a_reg, 2);
        input_bank_b_reg = vld1q_dup_s8x4(
            scratch_data +
            2 * workspace_height_stride);  // Load lane 0, avoiding
                                           // uninitialized variable.
        input_bank_b_reg = vld1q_lane_8x4(
            scratch_data + 3 * workspace_height_stride, input_bank_b_reg, 2);
        input_bank_c_reg = vld1q_dup_s8x4(
            scratch_data +
            4 * workspace_height_stride);  // Load lane 0, avoiding
                                           // uninitialized variable.

        int32x4_t acc0;
        int32x4_t acc1;

        // When output_width_micro_repeats < output_width_overall_micro_repeats,
        // 0 < residual_width <= 2, and so residual_width == 1 is then true iff
        // residual_width < 2.
        const int adjusted_width_micro_repeats =
            (output_width_micro_repeats < output_width_overall_micro_repeats) &&
                    (residual_width < 2)
                ? output_width_micro_repeats
                : output_width_overall_micro_repeats;

        int i_width = 0;
        for (; i_width < adjusted_width_micro_repeats; ++i_width) {
          const int8* input_data = scratch_data + 4 + 4 * i_width;

          // Load next sub-micro block of data.
          input_bank_a_reg = vld1q_lane_8x4(input_data, input_bank_a_reg, 1);
          input_bank_a_reg = vld1q_lane_8x4(
              input_data + workspace_height_stride, input_bank_a_reg, 3);
          input_bank_b_reg = vld1q_lane_8x4(
              input_data + 2 * workspace_height_stride, input_bank_b_reg, 1);
          input_bank_b_reg = vld1q_lane_8x4(
              input_data + 3 * workspace_height_stride, input_bank_b_reg, 3);
          input_bank_c_reg = vld1q_lane_8x4(
              input_data + 4 * workspace_height_stride, input_bank_c_reg, 1);

          int16x8_t acc_s16_0_1;
          uint8x8_t acc_u8_0_1;
          // Iterate over input width shifts within 4x4 blocks.
          {
            acc0 = adjusted_bias_data_s_0;
            acc1 = adjusted_bias_data_s_0;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg, 2);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg, 2);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_c_reg, 0);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            vst1_lane_8x4(output_data, acc_u8_0_1, 0);
            vst1_lane_8x4(output_data + output_height_stride, acc_u8_0_1, 1);

            acc0 = adjusted_bias_data_s_1;
            acc1 = adjusted_bias_data_s_1;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_b, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_b, input_bank_a_reg, 2);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_b, input_bank_b_reg, 2);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_b, input_bank_c_reg, 0);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            vst1_lane_8x4(output_data + 4, acc_u8_0_1, 0);
            vst1_lane_8x4(output_data + 4 + output_height_stride, acc_u8_0_1,
                          1);

            input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
            input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);
            input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 16);

            output_data += output_depth;
          }

          // output_width == four_over_stride.
          acc0 = adjusted_bias_data_s_0;
          acc1 = adjusted_bias_data_s_0;

          acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg, 0);
          acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg, 2);
          acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_b_reg, 0);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg, 2);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_c_reg, 0);

          // Fixed-point multiplication.
          acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
          acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
              acc0, -output_shift);
          acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
          acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
              acc1, -output_shift);
          // Add the output offset.
          acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
          acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
          // Apply the activation function.
          acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
          acc_u8_0_1 =
              vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
          acc_u8_0_1 =
              vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

          vst1_lane_8x4(output_data, acc_u8_0_1, 0);
          vst1_lane_8x4(output_data + output_height_stride, acc_u8_0_1, 1);

          acc0 = adjusted_bias_data_s_1;
          acc1 = adjusted_bias_data_s_1;

          acc0 = vdotq_four_lane_s32(acc0, filter_reg_0_b, input_bank_a_reg, 0);
          acc0 = vdotq_four_lane_s32(acc0, filter_reg_1_b, input_bank_a_reg, 2);
          acc0 = vdotq_four_lane_s32(acc0, filter_reg_2_b, input_bank_b_reg, 0);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_0_b, input_bank_b_reg, 0);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_1_b, input_bank_b_reg, 2);
          acc1 = vdotq_four_lane_s32(acc1, filter_reg_2_b, input_bank_c_reg, 0);

          // Fixed-point multiplication.
          acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
          acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
              acc0, -output_shift);
          acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
          acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
              acc1, -output_shift);
          // Add the output offset.
          acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
          acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
          // Apply the activation function.
          acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
          acc_u8_0_1 =
              vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
          acc_u8_0_1 =
              vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

          vst1_lane_8x4(output_data + 4, acc_u8_0_1, 0);
          vst1_lane_8x4(output_data + 4 + output_height_stride, acc_u8_0_1, 1);

          input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
          input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);
          input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 16);

          output_data += output_depth;
        }
        for (; i_width < output_width_overall_micro_repeats; ++i_width) {
          // output_width == 1.
          const int8* input_data = scratch_data + 4 + 4 * i_width;

          // Load next sub-micro block of data.
          input_bank_a_reg = vld1q_lane_8x4(input_data, input_bank_a_reg, 1);
          input_bank_a_reg = vld1q_lane_8x4(
              input_data + workspace_height_stride, input_bank_a_reg, 3);
          input_bank_b_reg = vld1q_lane_8x4(
              input_data + 2 * workspace_height_stride, input_bank_b_reg, 1);
          input_bank_b_reg = vld1q_lane_8x4(
              input_data + 3 * workspace_height_stride, input_bank_b_reg, 3);
          input_bank_c_reg = vld1q_lane_8x4(
              input_data + 4 * workspace_height_stride, input_bank_c_reg, 1);

          int16x8_t acc_s16_0_1;
          uint8x8_t acc_u8_0_1;
          // Iterate over input width shifts within 4x4 blocks.
          {
            acc0 = adjusted_bias_data_s_0;
            acc1 = adjusted_bias_data_s_0;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg, 2);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_a, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_a, input_bank_b_reg, 2);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_a, input_bank_c_reg, 0);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            vst1_lane_8x4(output_data, acc_u8_0_1, 0);
            vst1_lane_8x4(output_data + output_height_stride, acc_u8_0_1, 1);

            acc0 = adjusted_bias_data_s_1;
            acc1 = adjusted_bias_data_s_1;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_b, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_b, input_bank_a_reg, 2);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_b, input_bank_b_reg, 2);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_b, input_bank_c_reg, 0);

            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);
            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            vst1_lane_8x4(output_data + 4, acc_u8_0_1, 0);
            vst1_lane_8x4(output_data + 4 + output_height_stride, acc_u8_0_1,
                          1);

            input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
            input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);
            input_bank_c_reg = vshrq_n_u64(input_bank_c_reg, 16);

            output_data += output_depth;
          }
        }
      } else {
        TFLITE_DCHECK_EQ(block_height, 1);
        // Work through one slice, by row, at a time.
        const int8* scratch_data = scratch_block_data;
        uint8* output_data = output_block_data + 8 * j_depth;

        //
        int8x16_t input_bank_a_reg;  //  left 0, right 0, left 1, right 1.
        int8x16_t input_bank_b_reg;  //  left 2, right 2, xxx, xxx.

        // Load first sub-micro block of data into operational banks.
        input_bank_a_reg =
            vld1q_dup_s8x4(scratch_data);  // Load lane 0, avoiding
                                           // uninitialized variable.
        input_bank_a_reg = vld1q_lane_8x4(
            scratch_data + workspace_height_stride, input_bank_a_reg, 2);
        input_bank_b_reg = vld1q_dup_s8x4(
            scratch_data +
            2 * workspace_height_stride);  // Load lane 0, avoiding
                                           // uninitialized variable.

        int32x4_t acc0;
        int32x4_t acc1;

        for (int i_width = 0; i_width < output_width_overall_micro_repeats;
             ++i_width) {
          const int output_width =
              i_width == output_width_micro_repeats ? residual_width : 2;

          TFLITE_DCHECK_LE(output_width, 2);
          TFLITE_DCHECK_GE(output_width, 1);
          TFLITE_DCHECK_LE(output_width * stride_val, 4);
          const int8* input_data = scratch_data + 4 + 4 * i_width;

          // Load next sub-micro block of data.
          input_bank_a_reg = vld1q_lane_8x4(input_data, input_bank_a_reg, 1);
          input_bank_a_reg = vld1q_lane_8x4(
              input_data + workspace_height_stride, input_bank_a_reg, 3);
          input_bank_b_reg = vld1q_lane_8x4(
              input_data + 2 * workspace_height_stride, input_bank_b_reg, 1);

          int16x8_t acc_s16_0_1;
          uint8x8_t acc_u8_0_1;

          // Iterate over input width shifts within 4x4 blocks.
          {
            acc0 = adjusted_bias_data_s_0;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg, 2);

            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);

            // Second sub-block accumulation.
            acc1 = adjusted_bias_data_s_1;

            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_b, input_bank_a_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_b, input_bank_a_reg, 2);

            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);

            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            // This stores the results for both sub-blocks together.
            vst1_u8(output_data, acc_u8_0_1);

            input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
            input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);

            output_data += output_depth;
          }
          if (output_width == 2) {
            acc0 = adjusted_bias_data_s_0;

            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_2_a, input_bank_b_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_0_a, input_bank_a_reg, 0);
            acc0 =
                vdotq_four_lane_s32(acc0, filter_reg_1_a, input_bank_a_reg, 2);

            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc0 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0, -output_shift);

            // Second sub-block accumulation.
            acc1 = adjusted_bias_data_s_1;

            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_2_b, input_bank_b_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_0_b, input_bank_a_reg, 0);
            acc1 =
                vdotq_four_lane_s32(acc1, filter_reg_1_b, input_bank_a_reg, 2);

            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            acc1 = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc1, -output_shift);

            // Add the output offset.
            acc_s16_0_1 = vcombine_s16(vqmovn_s32(acc0), vqmovn_s32(acc1));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            acc_u8_0_1 = vqmovun_s16(acc_s16_0_1);
            acc_u8_0_1 =
                vmax_u8(acc_u8_0_1, vget_low_u8(output_activation_min_vec));
            acc_u8_0_1 =
                vmin_u8(acc_u8_0_1, vget_low_u8(output_activation_max_vec));

            // This stores the results for both sub-blocks together.
            vst1_u8(output_data, acc_u8_0_1);

            input_bank_a_reg = vshrq_n_u64(input_bank_a_reg, 16);
            input_bank_b_reg = vshrq_n_u64(input_bank_b_reg, 16);

            output_data += output_depth;
          }
        }
      }
    }
  }

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockNeon(scratch_block_data, filter_workspace, bias_data,
                         output_block_data, function_params);
  }
};

#undef DP_OFFSET_INPUT_DEPTH
#undef DP_OFFSET_OUTPUT_DEPTH
#undef DP_OFFSET_STRIDE
#undef DP_OFFSET_BIAS_INCREMENT
//
#undef DP_OFFSET_INPUT_OFFSET
#undef DP_OFFSET_OUTPUT_OFFSET
#undef DP_OFFSET_OUTPUT_MULTIPLIER
#undef DP_OFFSET_OUTPUT_SHIFT
#undef DP_OFFSET_QUANTIZED_ACTIVATION_MIN
#undef DP_OFFSET_QUANTIZED_ACTIVATION_MAX
//
#undef DP_OFFSET_PADDING_LEFT
#undef DP_OFFSET_PADDING_RIGHT
#undef DP_OFFSET_PADDING_TOP
#undef DP_OFFSET_PADDING_BOTTOM
//
#undef DP_OFFSET_DEPTH_MICRO_REPEATS
//
#undef DP_OFFSET_WIDTH_MACRO_COUNT
#undef DP_OFFSET_INPUT_WIDTH_OVERALL_MICRO_REPEATS
#undef DP_OFFSET_INPUT_WIDTH_MICRO_REPEATS
#undef DP_OFFSET_RESIDUAL_WIDTH
#undef DP_OFFSET_OUTPUT_WIDTH_OVERALL_MICRO_REPEATS
#undef DP_OFFSET_OUTPUT_WIDTH_MICRO_REPEATS
#undef DP_OFFSET_OUTPUT_RESIDUAL_WIDTH
#undef DP_OFFSET_WORKSPACE_WIDTH_MICRO_REPEATS
//
#undef DP_OFFSET_HEIGHT_MACRO_COUNT
#undef DP_OFFSET_INBOUND_BLOCK_HEIGHT
#undef DP_OFFSET_OUTBOUND_BLOCK_HEIGHT
#undef DP_OFFSET_INPUT_HEIGHT_STRIDE
#undef DP_OFFSET_OUTPUT_HEIGHT_STRIDE
#undef DP_OFFSET_WORKSPACE_HEIGHT_STRIDE
//
#undef DP_OFFSET_FOUR_OVER_STRIDE

#endif  // __ARM_FEATURE_DOTPROD && !GOOGLE_L4T

// Top-level implementation function for 3x3 depthwise convolution using NEON
// dot-product instructions.
//
// MACRO & MICRO BLOCKS
//
// The task is divided into macro blocks. Data is copied first into a macro
// block in a workspace. This has two purposes: (a) bringing data into
// cache, and (b) permuting data so that it can be used much more easily in
// a dot-product filter.
//
// When there is no depth multiplication:
//
// The permutations required for dot-products are local, within 4 data points
// down the depth and 4 across the width. We want to pull in input data at least
// 8-bytes at a time, down the depth, and so we divide the macro blocks into
// 1x4x8 (height, width, depth) and further divide the micro blocks into
// sub-blocks with shape (1x4x4).
//
// Each macro-block is constructed from micro-blocks that are internally
// rearranged during loading into the macro-block workspace.
//
// In other words, the micro-block shape is
//     {1, 1, 4, 8}
// Each macro block is typically shape
//     {1, height_block_size, 4 * workspace_width_micro_repeats, 64}
// and workspace_width_micro_repeats is chosen so it fits into the workspace.
//
// However, if depth < 64, we decrease the macro block depth, enabling us to
// increase the macro-block width.
//
// When there is depth multiplication:
//
// We require input-depth = 1 and exploit that instead.  Note that output data
// is still full-depth, *as is the filter and bias data after certain
// adjustments*, and so the filter stage in this case still proceeds in terms of
// sub-blocks.
//
// The Magic of these numbers:
//     4 is the number of input elements used in each dot-product.
//     8 is the number of inputs we load at a time into a register.
//     64 is min amount of data to be loaded in a stretch (when possible).
//
// FILTER DATA PREPARATION
//
// Filter data needs to be permuted in a fashion like that of input data, and
// this is done in a preprocessing stage. In addition, this stage extends the
// filter in the direction of width from 3 to 4. The extra filter taps are set
// to zero so that input data does not have to be zeroed before applying
// dot-products.
//
// OVERALL COUNTS: HANDLING TRAILING ITERATION
//
// Often it is necessary to handle the last iteration in a loop differently,
// generally because the final item is shorter. The logic to detect the
// special case can be a bit expensive. We use a scheme in which there are
// two counts, in a pattern like xxx_yyy_repeats and
// xxx_overall_yyy_repeats. The first gives the count of "normal"
// iterations. The loop iterates over the second count, and the induction
// variable is checked to see if it reaches xxx_yyy_repeats. If there is no
// special trailing iteration, xxx_yyy_repeats = xxx_overall_yyy_repeats,
// and the special code is not executed.
//
// Example:
// Suppose that we characterize a size s as
// f(s) -> (block-4-repetitions, remainder, overall_repetitions):
// f(11) -> (2, 3, 3)
// f(12) -> (3, 0, 3)
// f(13) -> (3, 1, 4)
//
// POINTING OUTSIDE OF INPUT ARRAY.
//
// When there is padding, the input data pointer passed to the fill routines
// points outside of the input array and into a kind-of virtual padded
// margin. It turns out that this simplifies the code and removes
// conditional statements. It is hard to explain why without comparing two
// versions of the code. In summary, this way the adjustment into the margin
// can be made unconditionally, and the correction back into the input array
// is done where there is a conditional already.
//
// OVERLAP
//
// Since this is *depthwise* conv, neither the batch nor the depth have overlap.
// The height and depth overlap by (filter_size - 1). Thus some data is used
// twice on the borders of macro blocks.
//
template <DepthwiseConvImplementation implementation>
inline void DepthwiseConvDotProduct3x3(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  // Check kernel restrictions.
  constexpr int filter_size = 3;
  constexpr int kMaxStride = 2;
  constexpr int kMaxPadding = 1;
  constexpr int kSymmetricZeroPoint = 128;
  TFLITE_DCHECK_EQ(params.weights_offset, -kSymmetricZeroPoint);
  TFLITE_DCHECK_LE(params.stride_width, kMaxStride);
  TFLITE_DCHECK_EQ(params.stride_height, params.stride_width);
  TFLITE_DCHECK_EQ(params.dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(params.dilation_height_factor, 1);
  TFLITE_DCHECK_LE(params.padding_values.width, kMaxPadding);
  TFLITE_DCHECK_LE(params.padding_values.height, kMaxPadding);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  // Key kernel parameters (along with padding handled later).
  const int stride = params.stride_width;
  const int depth_multiplier = params.depth_multiplier;
  const bool has_depth_multiplication = depth_multiplier > 1;

  // Extract task dimensions.
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  TFLITE_DCHECK(!has_depth_multiplication || input_depth == 1);
  TFLITE_DCHECK(has_depth_multiplication || input_depth == output_depth);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  TFLITE_DCHECK_EQ(input_depth * depth_multiplier, output_depth);
  TFLITE_DCHECK_EQ(MatchingDim(filter_shape, 1, filter_shape, 2), filter_size);

  // Return now if nothing to do.
  if (output_width == 0 || output_height == 0) {
    return;
  }

  // Kernel parameter structure: set basic fields.
  //
  // In asm it is easier to pass a structure than more than, say, 8 parameters.
  DepthwiseConvDotProdParams function_params;
  function_params.input_depth = input_depth;
  function_params.output_depth = output_depth;
  function_params.input_offset = params.input_offset;
  function_params.output_offset = params.output_offset;
  function_params.output_multiplier = params.output_multiplier;
  function_params.output_shift = params.output_shift;
  function_params.quantized_activation_min = params.quantized_activation_min;
  function_params.quantized_activation_max = params.quantized_activation_max;
  function_params.stride = stride;

  // Handle inbound bias data.
  //
  // Note that this data is adjusted in a per-depth process before the main
  // filters. The adjustment accounts for a non-symmetric input offset.
  //
  // Kernel subroutines need to be able to operate consistently on an bias
  // array. Where there is no bias, we provide one filled with zeros.
  constexpr int kMinBiasLoad = 8;
  int32 zero_bias_data[kMinBiasLoad];
  int32 bias_increment;
  if (bias_data) {
    bias_increment = 4;
  } else {
    memset(zero_bias_data, 0, sizeof(zero_bias_data));
    bias_data = &zero_bias_data[0];
    bias_increment = 0;
  }
  function_params.bias_increment = bias_increment;
  TFLITE_DCHECK_LE(2 * function_params.bias_increment, kMinBiasLoad);

  // Process padding.
  //
  // Whether "correct" or not, this matches ComputeConvSizes. When there is
  // stride > 1 there can be padding on the bottom or top, and therefore
  // we need to consider padding. This is true even if one or other of the
  // padding_values is 0.
  const int padded_width = (output_width - 1) * stride + filter_size;
  {
    const int padding_left = params.padding_values.width;
    // Right padding would be -1 if discarding input because of stride.
    const int padding_right =
        std::max(padded_width - input_width - padding_left, 0);
    const int padding_top = params.padding_values.height;
    const int padded_height = (output_height - 1) * stride + filter_size;
    const int padding_bottom =
        std::max(padded_height - input_height - padding_top, 0);

    function_params.padding_left = padding_left;
    function_params.padding_right = padding_right;
    function_params.padding_top = padding_top;
    function_params.padding_bottom = padding_bottom;

    TFLITE_DCHECK_LE(padding_left, padding_right);
    TFLITE_DCHECK_LE(padding_top, padding_bottom);
  }
  // When stride == 1 left or top padding may only be non-zero.
  // This is when padding is specified but not needed on a trailing dimension.
  // When stride == 2 right or bottom padding may only be non-zero.
  // This is a result of the details of the padding calculations.
  const bool padding_required =
      function_params.padding_left > 0 || function_params.padding_top > 0 ||
      function_params.padding_right > 0 || function_params.padding_bottom > 0;

  // Choose parameter-specific kernel subroutines.
  //
  // The main part of the kernel has two stages. First, a temporary workspace is
  // filled with padded and permuted data. Second, the filter is applied to the
  // workspace data to generate output.
  //
  // The workspace fill stage handles padding so that the filter stage does not
  // need to account for it. The workspace fill stage does not need to
  // understand striding, and implicitly handles striding through the parameters
  // that it is given.
  using pack_macro_block_func_t = decltype(
      &PackMacroBlock<implementation,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      0>::Run);
  using kernel_macro_block_func_t = decltype(
      &KernelMacroBlock<implementation,
                        DepthwiseConvDepthMultiplication::kNoMultiplication,
                        1>::Run);
  pack_macro_block_func_t pack_macro_block_func;
  kernel_macro_block_func_t kernel_macro_block_func;
  {
    if (has_depth_multiplication) {
      if (padding_required) {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kUnitInputDepth,
                           /*max_padding=*/1>::Run;
      } else {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kUnitInputDepth,
                           /*max_padding=*/0>::Run;
      }
      if (stride == 1) {
        kernel_macro_block_func =
            KernelMacroBlock<implementation,
                             DepthwiseConvDepthMultiplication::kUnitInputDepth,
                             /*stride=*/1>::Run;
      } else {
        kernel_macro_block_func =
            KernelMacroBlock<implementation,
                             DepthwiseConvDepthMultiplication::kUnitInputDepth,
                             /*stride=*/2>::Run;
      }
    } else {
      if (padding_required) {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kNoMultiplication,
                           /*max_padding=*/1>::Run;
      } else {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kNoMultiplication,
                           /*max_padding=*/0>::Run;
      }
      if (stride == 1) {
        kernel_macro_block_func = KernelMacroBlock<
            implementation, DepthwiseConvDepthMultiplication::kNoMultiplication,
            /*stride=*/1>::Run;
      } else {
        kernel_macro_block_func = KernelMacroBlock<
            implementation, DepthwiseConvDepthMultiplication::kNoMultiplication,
            /*stride=*/2>::Run;
      }
    }
  }

  // Stride-only variables.
  //
  // stride == 1 ? 4 : 2:
  const int output_height_per_macro = 6 - 2 * stride;
  // output_height_per_macro * stride:
  constexpr int input_height_per_macro = 4;
  // Number of rows per micro block (= rows per macro block) is
  //   (output_height_per_macro - 1) * stride + 1 + (filter_size - 1)
  //   = stride == 1 ? 3 + filter_size : 2 + filter_size:
  const int height_block_size = 4 + filter_size - stride;
  const int input_height_overlap = filter_size - stride;
  // stride == 1 ? 4 : 2:
  function_params.four_over_stride = output_height_per_macro;

  TFLITE_DCHECK_EQ(stride * function_params.four_over_stride, 4);
  TFLITE_DCHECK_EQ(height_block_size,
                   input_height_per_macro + input_height_overlap);

  // Create workspaces.
  //
  // Filter workspace is for shuffle: only first depth/8 is used.
  // indexed as [depth/8][sub-block][height][depth][width].
  TFLITE_DCHECK_EQ(kDepthwiseConvAdjustedBiasLimit % 8, 0);
  int8 macroblock_workspace[kDepthwiseConvScratchWorkspaceSize];
  int32 adjusted_bias_data[kDepthwiseConvAdjustedBiasLimit];
  int8 filter_workspace[kDepthwiseConvAdjustedBiasLimit >> 3][3][2][4][4];

  // Output depth characterization.
  //
  const int depth_macro_count = output_depth / 64;
  const int depth_overall_macro_count = (output_depth + 63) / 64;
  // Number of micro blocks down the depth in a final incomplete macro block.
  const int depth_trailing_micro_repeats = output_depth / 8 % 8;
  // The output_depth may not have a remainder: it must be a multiple of 8.
  TFLITE_DCHECK_EQ(output_depth,
                   64 * depth_macro_count + 8 * depth_trailing_micro_repeats);

  // Characterize the first macro block depth, the largest.
  //
  // We base treatment of the width on the trailing macro block if there are
  // no full blocks, in order to do more work together (that is, increase
  // workspace_width_micro_repeats when largest_macro_depth < 64).
  const int largest_macro_depth =
      has_depth_multiplication
          ? 1
          : (depth_macro_count > 0 ? 64 : 8 * depth_trailing_micro_repeats);

  // Characterize width, consumption of input and generation of output.
  //
  // In the case of depth multiplication, we ensure that some of the workspace
  // at the end remains unused. This enables the filter routines to load the
  // "next" data, of at least 16 bytes, even when at the end of the workspace.
  // It is relatively expensive to detect the end micro block. It is also very
  // difficult to test for (to trigger) erroneous reads (past end of array) in
  // the depth multplication case.
  int workspace_width_micro_repeats =
      (has_depth_multiplication
           ? kDepthwiseConvScratchWorkspaceSize - kWorkspaceExtension
           : kDepthwiseConvScratchWorkspaceSize) /
      (4 * largest_macro_depth * height_block_size);
  // When there is no depth multiplication, the workspace depth is a multiple of
  // 8, which ensures that workspace rows are 16-byte aligned. (Actually 32,
  // because of the micro width of 4.) This is not necessarily the case under
  // depth multiplication, so we adjust now to impose this restriction.
  if (has_depth_multiplication) {
    workspace_width_micro_repeats = (workspace_width_micro_repeats / 4) * 4;
  }
  TFLITE_DCHECK_EQ((workspace_width_micro_repeats * largest_macro_depth) % 4,
                   0);
  // Discount 1 of the micro-block repeats in each macro block to account for
  // overlap.
  const int consumed_width_per_macro_block =
      4 * (workspace_width_micro_repeats - 1);
  const int output_width_per_macro_block =
      function_params.four_over_stride * (workspace_width_micro_repeats - 1);
  TFLITE_DCHECK_GT(workspace_width_micro_repeats, 1);
  TFLITE_DCHECK_EQ(output_width_per_macro_block * stride,
                   consumed_width_per_macro_block);

  // Width repetitions and residuals.
  //
  // Use of the workspace is characterized primarily in terms of *padded input*.
  // Striding only matters in a few places.
  //
  // Simplifications: We require that there always be at least one full
  // micro-block across the width. Since the maximum padding is 1, the trailing
  // padding cannot span two micro blocks.
  const int residual_micro_width = padded_width % 4;
  // We base the count of macro blocks on the amount of padded input data each
  // one consumes.
  int width_overall_macro_count = (padded_width - residual_micro_width +
                                   consumed_width_per_macro_block - 1) /
                                  consumed_width_per_macro_block;
  // Recall that we left a micro block at the end of each macro block for use as
  // overlap. There is a special case in which we can use one fewer macro
  // blocks, with the last one consuming extra input. (But not if the
  // calculation thinks that we can use zero blocks.)
  if (padded_width <=
      ((width_overall_macro_count - 1) * consumed_width_per_macro_block + 4)) {
    width_overall_macro_count -= 1;
  }
  width_overall_macro_count = std::max(width_overall_macro_count, 1);
  // We always have to treat the final macro block along width as trailing,
  // because even if it is full in terms of padded input, it will be incomplete
  // in terms of output.
  const int width_macro_count = width_overall_macro_count - 1;
  // Micro blocks are traversed in terms of input in fill routines.
  const int width_trailing_micro_repeats =
      (padded_width - consumed_width_per_macro_block * width_macro_count) / 4;
  const int width_overall_trailing_micro_repeats =
      (padded_width - consumed_width_per_macro_block * width_macro_count + 3) /
      4;
  // Micro blocks are traversed in terms of output in filtering routines.
  const int residual_output_micro_width =
      (output_width - 1) % function_params.four_over_stride + 1;
  const int output_width_trailing_micro_repeats =
      residual_micro_width > (filter_size - 1)
          ? width_trailing_micro_repeats
          : width_trailing_micro_repeats - 1;
  // Check results.
  TFLITE_DCHECK_GT(width_overall_trailing_micro_repeats, 0);
  TFLITE_DCHECK_EQ(padded_width,
                   residual_micro_width +
                       consumed_width_per_macro_block * width_macro_count +
                       4 * width_trailing_micro_repeats);
  TFLITE_DCHECK_LE(width_overall_macro_count, width_macro_count + 1);
  TFLITE_DCHECK_GE(width_overall_macro_count, width_macro_count);

  // Height repetitions and residuals.
  //
  const int height_macro_count = output_height / output_height_per_macro;
  const int residual_output_height = output_height % output_height_per_macro;
  const int height_overall_macro_count =
      (output_height + output_height_per_macro - 1) / output_height_per_macro;
  TFLITE_DCHECK_EQ(
      output_height,
      residual_output_height + output_height_per_macro * height_macro_count);
  TFLITE_DCHECK_LE(height_overall_macro_count, height_macro_count + 1);
  TFLITE_DCHECK_GE(height_overall_macro_count, height_macro_count);

  // Data strides.
  //
  const int input_height_stride = input_width * input_depth;
  const int output_height_stride = output_width * output_depth;
  const int input_batch_stride = input_height_stride * input_height;
  const int output_batch_stride = output_height_stride * output_height;
  const int input_depth_macro_stride = has_depth_multiplication ? 0 : 64;
  const int input_width_macro_stride =
      input_depth * consumed_width_per_macro_block;
  const int output_width_macro_stride =
      output_depth * output_width_per_macro_block;

  // Store parameters that do not vary across macro blocks.
  //
  function_params.workspace_width_micro_repeats = workspace_width_micro_repeats;
  function_params.height_macro_count = height_overall_macro_count;
  function_params.width_macro_count = width_overall_macro_count;
  function_params.input_height_stride = input_height_stride;
  function_params.output_height_stride = output_height_stride;
  function_params.residual_width = residual_micro_width;

  // Main process.
  //
  // Most kernels are nested batch-height-width-depth. Here we proceed over
  // macro blocks batch-width-depth-height.
  //
  // Example of handling of trailing iteration: when there is trailing depth,
  // depth_overall_macro_count = depth_macro_count + 1, so we can adjust the
  // dimensions for trailing macro blocks by looking for
  // j_depth == depth_macro_count.
  for (int b = 0; b < batches; ++b) {
    for (int k_width = 0; k_width < width_overall_macro_count; ++k_width) {
      // Figure out the work to be done for this macro block. If it trails in
      // any dimension, the work in that dimension is adjusted.
      // The work to be done across widths has 3 cases:
      // (a) A full macro block,
      // (b) Partial terminal macro block, with input and output ending in
      //     same micro block, and
      // (c) Partial terminal macro block, with output corresponding to one
      //     fewer micro blocks, because filter extends across micro-block
      //     boundary.
      if (k_width != width_macro_count) {
        function_params.output_residual_width = 0;
        function_params.input_width_micro_repeats =
            workspace_width_micro_repeats;
        function_params.input_width_overall_micro_repeats =
            workspace_width_micro_repeats;
        function_params.output_width_micro_repeats =
            workspace_width_micro_repeats - 1;
      } else {
        function_params.output_residual_width = residual_output_micro_width;
        function_params.input_width_micro_repeats =
            width_trailing_micro_repeats;
        function_params.input_width_overall_micro_repeats =
            width_overall_trailing_micro_repeats;
        function_params.output_width_micro_repeats =
            output_width_trailing_micro_repeats;
      }
      function_params.output_width_overall_micro_repeats =
          function_params.output_residual_width == 0
              ? function_params.output_width_micro_repeats
              : function_params.output_width_micro_repeats + 1;

      for (int j_depth = 0; j_depth < depth_overall_macro_count; ++j_depth) {
        const uint8* input_data_block =
            input_data + b * input_batch_stride +
            j_depth * input_depth_macro_stride +
            k_width * input_width_macro_stride -
            function_params.padding_left * input_depth -
            function_params.padding_top * input_height_stride;
        uint8* output_data_block = output_data + b * output_batch_stride +
                                   j_depth * 64 +
                                   k_width * output_width_macro_stride;

        // Process filter and bias data.
        //
        function_params.depth_micro_repeats =
            j_depth == depth_macro_count ? depth_trailing_micro_repeats : 8;
        ProcessPerDepth<implementation>::Run(
            filter_data + 64 * j_depth,
            bias_data + 8 * 2 * bias_increment * j_depth,
            filter_workspace[0][0][0][0], adjusted_bias_data, &function_params);

        // Under depth multiplication the workspace_height_stride does not have
        // to depend on input_width_overall_micro_repeats, but this improves the
        // compactness of workspace use.
        const int workspace_height_stride =
            has_depth_multiplication
                ? 16 * ((function_params.input_width_overall_micro_repeats +
                         3) >>
                        2)
                : 4 * function_params.input_width_overall_micro_repeats * 8 *
                      function_params.depth_micro_repeats;
        TFLITE_DCHECK_EQ(workspace_height_stride % 16, 0);
        function_params.workspace_height_stride = workspace_height_stride;

        // For the first macro block for output rows we fill in the first few
        // rows.  After this we will copy them (see below in loop.)
        function_params.inbound_block_height = input_height_overlap;
        pack_macro_block_func(-1, k_width, input_data_block,
                              macroblock_workspace, &function_params);
        input_data_block += input_height_stride * input_height_overlap;

        for (int i_height = 0; i_height < height_overall_macro_count;
             ++i_height) {
          if (i_height != height_macro_count) {
            function_params.inbound_block_height = input_height_per_macro;
            function_params.outbound_block_height = output_height_per_macro;
          } else {
            function_params.inbound_block_height =
                residual_output_height * stride;
            function_params.outbound_block_height = residual_output_height;
          }
          TFLITE_DCHECK_LT(i_height * output_height_per_macro, output_height);
          TFLITE_DCHECK_LT(i_height * input_height_per_macro, input_height);
          TFLITE_DCHECK_LT(k_width * output_width_per_macro_block,
                           output_width);
          TFLITE_DCHECK_LT(k_width * consumed_width_per_macro_block,
                           input_width);

          // Macro blocks overlap by input_height_overlap rows, so we copy
          // those instead of filling in afresh.  The first macro block across
          // output rows was filled in outside of the loop (above).
          if (i_height > 0) {
            memcpy(macroblock_workspace,
                   macroblock_workspace +
                       input_height_per_macro * workspace_height_stride,
                   input_height_overlap * workspace_height_stride);
          }

          pack_macro_block_func(
              i_height, k_width, input_data_block,
              macroblock_workspace +
                  input_height_overlap * workspace_height_stride,
              &function_params);

          kernel_macro_block_func(
              macroblock_workspace, filter_workspace[0][0][0][0],
              adjusted_bias_data, output_data_block, &function_params);

          input_data_block += input_height_stride * input_height_per_macro;
          output_data_block += output_height_stride * output_height_per_macro;
        }
      }
    }
  }
}

#undef vst1_lane_8x4
#undef vst1q_lane_8x4
#undef vld1q_lane_s8x8
#undef vld1_lane_8x4
#undef vld1q_lane_8x4
#undef vld1q_dup_s8x4

#undef STR
#undef STR_UNEXPANDED

}  // namespace depthwise_conv
}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_

#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
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
      return vdotq_lane_s32(acc, lhs, vget_low_s8(rhs), 0);
    case 1:
      return vdotq_lane_s32(acc, lhs, vget_low_s8(rhs), 1);
    case 2:
      return vdotq_lane_s32(acc, lhs, vget_high_s8(rhs), 0);
    case 3:
    default:
      return vdotq_lane_s32(acc, lhs, vget_high_s8(rhs), 1);
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

//  This structure is typically used for reducing the magnitude of outputs, and
//  the historical name reflects that.
template <DepthwiseConvOutputRounding output_rounding>
struct DivideByPOT {};

template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kAwayFromZero> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
    return RoundingDivideByPOT(x, exponent);
  }
  // Mult versions use the exponents directly, rather than negated.
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, int exponent) {
    return RoundingDivideByPOT(x, -exponent);
  }
};

#ifdef USE_NEON
template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kUpward> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
    return vqrshlq_s32(x, vdupq_n_s32(static_cast<int32_t>(-exponent)));
  }
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, IntegerType exponent) {
    return vqrshlq_s32(x, exponent);
  }
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, int exponent) {
    return vqrshlq_s32(x, vdupq_n_s32(static_cast<int32_t>(exponent)));
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

enum class QuantizationType {
  kNonPerChannelUint8 = 0,
  kPerChannelInt8 = 1,
};

template <QuantizationType quantization_type>
struct QuantizationTypeImpl {};

template <>
struct QuantizationTypeImpl<QuantizationType::kNonPerChannelUint8> {
  typedef uint8_t ExternalType;

  static constexpr int kIntSymmetricZeroPoint = 128;
  static constexpr uint8_t kUint8SignBit = 0x80;
};

template <>
struct QuantizationTypeImpl<QuantizationType::kPerChannelInt8> {
  typedef int8_t ExternalType;

  static constexpr int kIntSymmetricZeroPoint = 0;
  static constexpr uint8_t kUint8SignBit = 0x0;
};

template <
    QuantizationType quantization_type = QuantizationType::kNonPerChannelUint8>
inline DotProduct3x3KernelType CategorizeDotProductKernel(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    const RuntimeShape& output_shape, const DepthwiseParams& params,
    const int32_t* output_shift_ptr = nullptr) {
  constexpr int kSymmetricZeroPoint =
      QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
  const int padding =
      std::max(params.padding_values.width, params.padding_values.height);
  const int stride = params.stride_width;
  const int32_t input_depth = input_shape.Dims(3);
  const int32_t depth_multiplier = params.depth_multiplier;
  const int32_t filter_height = filter_shape.Dims(1);
  const int32_t filter_width = filter_shape.Dims(2);

  bool supported = stride == params.stride_height && stride <= 2 &&
                   padding <= 1 && filter_width == 3 && filter_height == 3 &&
                   params.dilation_width_factor == 1 &&
                   params.dilation_height_factor == 1 &&
                   (((input_depth % 8) == 0 && depth_multiplier == 1) ||
                    (input_depth == 1 && depth_multiplier > 1));

  if (!supported) {
    return DotProduct3x3KernelType::kNone;
  }

  if (params.weights_offset != -kSymmetricZeroPoint) {
    return DotProduct3x3KernelType::kNone;
  }

  if (quantization_type == QuantizationType::kPerChannelInt8) {
    if (output_shift_ptr == nullptr) {
      return DotProduct3x3KernelType::kNone;
    }
  } else if (params.output_shift > 0) {
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

// Encapsulates constant parameters used in DepthwiseConv.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
struct DepthwiseConvParams {
  int64_t input_depth;
  int64_t input_row_size;
  int64_t output_depth;
  int64_t output_row_size;
  int64_t filter_row_size;
  int32_t input_offset;
  int32_t output_offset;
  int32_t filter_offset;
  int32_t output_multiplier;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_right_shift;
  int32_t input_width;
  int32_t input_height;
  int32_t stride_width;
  int32_t stride_height;
  int32_t output_width;
  int32_t output_height;
  float float_output_activation_min;
  float float_output_activation_max;
};

// Encapsulates constant parameters used in DepthwiseConv using dot-product ops.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
//
// This structure is specifically designed for use in asm.
struct DepthwiseConvDotProdParams {
  int64_t input_depth;
  int64_t output_depth;
  int32_t stride;
  int32_t bias_increment;
  //
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  //
  int32_t padding_left;
  int32_t padding_right;
  int32_t padding_top;
  int32_t padding_bottom;
  //
  int32_t depth_micro_repeats;
  //
  int32_t width_macro_count;
  int32_t input_width_overall_micro_repeats;
  int32_t input_width_micro_repeats;
  int32_t residual_width;
  int32_t output_width_overall_micro_repeats;
  int32_t output_width_micro_repeats;
  int32_t output_residual_width;
  int32_t workspace_width_micro_repeats;
  //
  int32_t height_macro_count;
  int32_t inbound_block_height;
  int32_t outbound_block_height;
  int32_t input_height_stride;
  int32_t output_height_stride;
  int32_t workspace_height_stride;
  //
  int32_t four_over_stride;
  //
  const int32_t* output_multiplier_per_channel;
  const int32_t* output_shift_per_channel;
};

template <DepthwiseConvOutputRounding output_rounding, int32_t kDepth,
          int32_t kStrideWidth, int32_t kStrideHeight>
struct DepthwiseConvWindow {};

template <DepthwiseConvOutputRounding output_rounding, int32_t kDepth,
          int32_t kStrideWidth, int32_t kStrideHeight>
struct DepthwiseConvWindowPerChannel {};

enum class EdgeType { kCorner, kHorizontal, kVertical, kCenter };

template <DepthwiseConvOutputRounding output_rounding, EdgeType kEdgeType,
          int kPadWidth, int kPadHeight>
struct DepthwiseConvPartial {};

template <DepthwiseConvOutputRounding output_rounding, EdgeType kEdgeType,
          int kPadWidth, int kPadHeight>
struct DepthwiseConvPartialPerChannel {};

// Copies a subset of the input designated by |input_ptr| into |output_ptr|
// with the specified output dimensions. Supports output depths of 64 only as
// this is the cache line size.
template <typename T>
inline void ShuffleInput(const T* input_ptr, int64_t input_depth,
                         int32_t input_width, int32_t input_height,
                         int64_t output_depth, int32_t output_width,
                         int32_t output_height, T* output_ptr) {
  const int64_t input_row_size = input_depth * input_width;
  for (int32_t y = 0; y < output_height; y++) {
    const T* ptr = input_ptr;
    for (int32_t x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

// Calculates the input size depending on stride and output.
inline int32_t get_shuffle_input_size(int32_t stride, int32_t output) {
  return stride * (output - 1) + 3;
}

// Indicates the input and output dimensions used when shuffling input
// activations.
struct ShuffleParams {
  int32_t output_width;
  int32_t output_height;
  int32_t input_width;
  int32_t input_height;

  ShuffleParams() = default;
  ShuffleParams(int32_t output_width, int32_t output_height,
                int32_t stride_width, int32_t stride_height)
      : output_width(output_width),
        output_height(output_height),
        input_width(get_shuffle_input_size(stride_width, output_width)),
        input_height(get_shuffle_input_size(stride_height, output_height)) {}
};

template <
    QuantizationType quantization_type = QuantizationType::kNonPerChannelUint8>
inline bool Fast3x3FilterKernelSupported(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    int32_t stride_width, int32_t stride_height, int32_t dilation_width_factor,
    int32_t dilation_height_factor, int32_t pad_width, int32_t pad_height,
    int32_t depth_multiplier, const RuntimeShape& output_shape,
    int32_t output_shift, const int32_t* output_shift_ptr = nullptr) {
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t input_depth = input_shape.Dims(3);
  const int32_t filter_height = filter_shape.Dims(1);
  const int32_t filter_width = filter_shape.Dims(2);
  const int32_t output_height = output_shape.Dims(1);
  const int32_t output_width = output_shape.Dims(2);

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

  const int32_t out_x = output_width - 1;
  const int32_t out_y = output_height - 1;

  const int32_t in_x_origin = (out_x * stride_width) - pad_width;
  const int32_t in_y_origin = (out_y * stride_height) - pad_height;

  const int32_t in_x_end = in_x_origin + filter_width;
  const int32_t in_y_end = in_y_origin + filter_height;

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

// Permute filter data, and adjust bias data to account for symmetric input
// offset. Details are provided in the implementation of the
// kUseCModel3x3DotProduct version.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          QuantizationType quantization_type>
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
          QuantizationType quantization_type,
          DepthwiseConvDepthMultiplication depth_multiplication,
          int32_t max_padding>
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
          QuantizationType quantization_type,
          DepthwiseConvDepthMultiplication depth_multiplication, int32_t stride>
struct KernelMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

#if defined(__aarch64__)
// Experiments suggest that a modest performance improvement is seen, at least
// on 855 chipset big cores, with cache hints.
template <typename T>
inline void PreloadInputBlock(
    const T* input_block_data,
    const DepthwiseConvDotProdParams* function_params) {
  // Preload.
  const int input_width_micro_repeats =
      function_params->input_width_micro_repeats;
  const int block_height = function_params->inbound_block_height;
  const int residual_width = function_params->residual_width;
  const int input_height_stride = function_params->input_height_stride;
  const int input_depth = function_params->input_depth;

  const int total_width = 4 * input_width_micro_repeats + residual_width;
  const T* row_ptr = input_block_data;
  for (int k_height = 0; k_height < block_height; ++k_height) {
    const T* ptr = row_ptr;
    for (int j = 0; j < total_width; ++j) {
      // Input data is loaded once.
      optimized_ops_preload_l1_keep(ptr);
      ptr += input_depth;
    }
    row_ptr += input_height_stride;
  }
}
#endif  // __aarch64__

}  // namespace depthwise_conv
}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_

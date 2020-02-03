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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_

// This file provides kernel implementations that are not used in shipped
// inference code, but rather (a) show how model C++ code is designed and then
// transformed into asm code, and (b) aid with maintenance and later development
// of variations. Many projects (even including, say, the classic NAG libraries)
// develop highly optimized code, but do not maintain intermediate versions.
// Often the result is incomprehensible final-version code.

#include <algorithm>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace depthwise_conv {

#ifdef USE_NEON

inline int8x16_t util_vld1q_x8(const uint8* data_addr) {
  return vreinterpretq_s8_u8(vld1q_u8(data_addr));
}
inline int8x16_t util_vld1q_x8(const int8* data_addr) {
  return vld1q_s8(data_addr);
}
inline int8x8_t util_vld1_x8(const uint8* data_addr) {
  return vreinterpret_s8_u8(vld1_u8(data_addr));
}
inline int8x8_t util_vld1_x8(const int8* data_addr) {
  return vld1_s8(data_addr);
}

// Lane operations are for clarity and convenience. We want to load and store
// 4 8-bit lanes together. So these are treated much like 32-bit loads and
// 32-bit stores. Stores require 32-bit alignment.

#define vst1_lane_8x4(dst, reg, lane_num)                         \
  TFLITE_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % 4, 0); \
  vst1_lane_u32(reinterpret_cast<uint32_t*>(dst), reg, lane_num)
#define vst1q_lane_8x4(dst, reg, lane_num)                        \
  TFLITE_DCHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % 4, 0); \
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(dst), reg, lane_num)

// Important! Most compilation configurations will compile and run without
// reinterpret_cast. Sanitizers may fail silently on lane-loading, with an
// obscure bug or mis-feature probably in unhygienic macro expansion.
#define vld1q_lane_s8x8(src, reg, lane_num) \
  vld1q_lane_u64(reinterpret_cast<const uint64_t*>(src), reg, lane_num)
#define vld1_lane_8x4(src, reg, lane_num) \
  vld1_lane_s32(reinterpret_cast<const int32*>(src), reg, lane_num)
#define vld1q_lane_8x4(src, reg, lane_num) \
  vld1q_lane_s32(reinterpret_cast<const int32*>(src), reg, lane_num)
#define vld1q_dup_s8x4(src) vld1q_dup_s32(reinterpret_cast<const int32*>(src))

#endif  // USE_NEON

template <QuantizationType quantization_type>
struct ProcessPerDepth<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                       quantization_type> {
  // Filter data is provided as filter_block[3][3][depth/8][2][4]: height 3,
  // width 3,  sub-block 0 or 1, depth 4. Filter data is written as
  // filter_bank[3][2][4][4]; height 3, sub-block, depth 4, width 4.
  //
  // Note that this rearrangement is much like that performed on input data when
  // filling the workspace, and optimized versions will be similar.
  static inline void FillFilterBank(int depth, const uint8* filter_block,
                                    int8 filter_bank[3][2][4][4]) {
    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    // Load filter data in, 8-bytes down depth / sub-block at a time.
    //
    // loaded_filter has dimensions height 3, width 4, sub-block 0 or 1,
    // depth 4.
    uint8 loaded_filter[3][4][2][4];
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        memcpy(loaded_filter[y][x][0], &filter_block[3 * y * depth + x * depth],
               8);
      }
      // Pad the filter with symmetric representation of 0, so that the values
      // become 0 when the zero-poing is added below. Thus these filter taps are
      // effectively disregarded in later filtering.
      memset(loaded_filter[y][3][0], kSymmetricZeroPoint, 8);
    }
    for (int y = 0; y < 3; ++y) {
      for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
          filter_bank[y][0][z][x] =
              loaded_filter[y][x][0][z] - kSymmetricZeroPoint;
          filter_bank[y][1][z][x] =
              loaded_filter[y][x][1][z] - kSymmetricZeroPoint;
        }
      }
    }
  }

  // Adjust the bias (weights) data according to the input offset.
  //
  // The output calculation is
  // out[h][w][d] = bias[d] + sum_ij (in[h+i][w+j][d] + in_offset) *
  //                                 (filter[i][j][d] + filter_offset)
  // (where offsets are expressed as differences from 128).
  //
  // Since we cannot efficiently handle varying offsets / bias across the image,
  // we insist on filter_offset = 0.
  //
  // This function calculates
  // adjusted_bias[d] = bias[d] + sum_ij in_offset * filter[i][j][d]
  // which accounts for input offset. If the bias is constant over the depth,
  // the adjusted bias will vary.
  static inline void AdjustBias(int32 input_offset,
                                const int8 filter_bank[3][2][4][4],
                                const int32* bias_data,
                                int32 adjusted_bias_block[2][4]) {
    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    TFLITE_DCHECK_GE(input_offset, -255);
    TFLITE_DCHECK_LE(input_offset, 0);
    // For instance, if input_offset == 128, no adjustment is needed.
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;

    for (int s = 0; s < 2; ++s) {
      for (int z = 0; z < 4; ++z) {
        adjusted_bias_block[s][z] = bias_data[4 * s + z];
        for (int i = 0; i < 9; ++i) {
          adjusted_bias_block[s][z] +=
              input_offset_difference * filter_bank[i % 3][s][z][i / 3];
        }
      }
    }
  }

  static void Run(const uint8* filter_data, const int32* bias_data,
                  int8* shuffled_filter_data, int32* adjusted_bias_data,
                  const DepthwiseConvDotProdParams* function_params) {
    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;
    const int depth = function_params->output_depth;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int bias_increment = function_params->bias_increment;
    const int32 input_offset = function_params->input_offset;

    int8 filter_bank[3][2][4][4];
    int32 adjusted_bias_block[2][4];

    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      FillFilterBank(depth, filter_data + 8 * j_depth, filter_bank);
      AdjustBias(input_offset, filter_bank,
                 bias_data + 2 * bias_increment * j_depth, adjusted_bias_block);

      memcpy(shuffled_filter_data, filter_bank[0][0][0],
             shuffled_filter_increment);
      shuffled_filter_data += shuffled_filter_increment;
      memcpy(adjusted_bias_data, adjusted_bias_block[0],
             8 * sizeof(adjusted_bias_block[0][0]));
      adjusted_bias_data += 8;
    }
  }
};

template <QuantizationType quantization_type>
struct ProcessPerDepth<DepthwiseConvImplementation::kUseUnwound3x3DotProduct,
                       quantization_type> {
  static inline void Run(const uint8* filter_data, const int32* bias_data,
                         int8* shuffled_filter_data, int32* adjusted_bias_data,
                         const DepthwiseConvDotProdParams* function_params) {
    const int depth = function_params->output_depth;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int bias_increment = function_params->bias_increment;

    // Simulate NEON-register transposition of subset of filter.
    int8 filter_bank_a_0[4][4];  // Depth 4, width 4.
    int8 filter_bank_a_1[4][4];
    int8 filter_bank_a_2[4][4];
    int8 filter_bank_b_0[4][4];
    int8 filter_bank_b_1[4][4];
    int8 filter_bank_b_2[4][4];

    // Load filter data in, essentially dropping the [depth/8] dimension, which
    // is equivalent to loading just the depth needed for one micro-block.
    //
    // loaded_filter has dimensions height 3, width 4, sub-block 0 or 1,
    // depth 4.
    uint8 loaded_filter_0[4][2][4];
    uint8 loaded_filter_1[4][2][4];
    uint8 loaded_filter_2[4][2][4];

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    const int32 input_offset = function_params->input_offset;
    TFLITE_DCHECK_GE(input_offset, -255);
    TFLITE_DCHECK_LE(input_offset, 0);
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;

    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      const uint8* filter_block = filter_data + 8 * j_depth;

      // Filter data is provided as filter_block[3][3][depth/8][2][4].
      // height 3, width 3, micro-blocks, sub-block 0 or 1, depth 4.
      // filter_bank[3][2][4][4]; Sub-block, height 3, depth 4, width 4.
      for (int x = 0; x < 3; ++x) {
        memcpy(loaded_filter_0[x][0], &filter_block[3 * 0 * depth + x * depth],
               8);
        memcpy(loaded_filter_1[x][0], &filter_block[3 * 1 * depth + x * depth],
               8);
        memcpy(loaded_filter_2[x][0], &filter_block[3 * 2 * depth + x * depth],
               8);
      }
      // Pad the filter with -filter_offset, so that the values become 0 when
      // the filter_offset is later added, and so the filter tap is effectively
      // disregarded.
      memset(loaded_filter_0[3][0], kSymmetricZeroPoint, 8);
      memset(loaded_filter_1[3][0], kSymmetricZeroPoint, 8);
      memset(loaded_filter_2[3][0], kSymmetricZeroPoint, 8);

      for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
          filter_bank_a_0[z][x] =
              loaded_filter_0[x][0][z] - kSymmetricZeroPoint;
          filter_bank_b_0[z][x] =
              loaded_filter_0[x][1][z] - kSymmetricZeroPoint;
          filter_bank_a_1[z][x] =
              loaded_filter_1[x][0][z] - kSymmetricZeroPoint;
          filter_bank_b_1[z][x] =
              loaded_filter_1[x][1][z] - kSymmetricZeroPoint;
          filter_bank_a_2[z][x] =
              loaded_filter_2[x][0][z] - kSymmetricZeroPoint;
          filter_bank_b_2[z][x] =
              loaded_filter_2[x][1][z] - kSymmetricZeroPoint;
        }
      }

      memcpy(shuffled_filter_data, filter_bank_a_0, 16);
      shuffled_filter_data += 16;
      memcpy(shuffled_filter_data, filter_bank_b_0, 16);
      shuffled_filter_data += 16;
      memcpy(shuffled_filter_data, filter_bank_a_1, 16);
      shuffled_filter_data += 16;
      memcpy(shuffled_filter_data, filter_bank_b_1, 16);
      shuffled_filter_data += 16;
      memcpy(shuffled_filter_data, filter_bank_a_2, 16);
      shuffled_filter_data += 16;
      memcpy(shuffled_filter_data, filter_bank_b_2, 16);
      shuffled_filter_data += 16;

      int32 adjusted_bias_data_0[4];
      int32 adjusted_bias_data_1[4];
      // For instance, if input_offset == 128, no adjustment is needed.
      for (int z = 0; z < 4; ++z) {
        adjusted_bias_data_0[z] = bias_data[z];
        adjusted_bias_data_1[z] = bias_data[4 + z];
        for (int x = 0; x < 4; ++x) {
          adjusted_bias_data_0[z] +=
              input_offset_difference * filter_bank_a_0[z][x];
          adjusted_bias_data_0[z] +=
              input_offset_difference * filter_bank_a_1[z][x];
          adjusted_bias_data_0[z] +=
              input_offset_difference * filter_bank_a_2[z][x];
          adjusted_bias_data_1[z] +=
              input_offset_difference * filter_bank_b_0[z][x];
          adjusted_bias_data_1[z] +=
              input_offset_difference * filter_bank_b_1[z][x];
          adjusted_bias_data_1[z] +=
              input_offset_difference * filter_bank_b_2[z][x];

          adjusted_bias_data[z] = adjusted_bias_data_0[z];
          adjusted_bias_data[4 + z] = adjusted_bias_data_1[z];
        }
      }
      bias_data += 2 * bias_increment;
      adjusted_bias_data += 8;
    }
  }
};

#ifdef USE_NEON
template <QuantizationType quantization_type>
struct ProcessPerDepth<DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
                       quantization_type> {
  static void ProcessPerDepthIntrinsics(
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          filter_data,
      const int32* bias_data, int8* shuffled_filter_data,
      int32* adjusted_bias_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int depth = function_params->output_depth;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int bias_increment = function_params->bias_increment;

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    constexpr uint8 kSignBit =
        QuantizationTypeImpl<quantization_type>::kUint8SignBit;
    const int32 input_offset = function_params->input_offset;
    if (quantization_type == QuantizationType::kNonPerChannelUint8) {
      TFLITE_DCHECK_GE(input_offset, -255);
      TFLITE_DCHECK_LE(input_offset, 0);
    }
    const int32 input_offset_difference = input_offset + kSymmetricZeroPoint;
    const int8x16_t ones_vector = vdupq_n_s8(1);

    // Simulate NEON-register transposition of subset of filter.
    int8x16_t input_0_a;
    int8x16_t input_0_b;
    int8x16_t input_0_c;
    int8x16_t input_1_a;
    int8x16_t input_1_b;
    int8x16_t input_1_c;
    int8x16_t input_2_a;
    int8x16_t input_2_b;
    int8x16_t input_2_c;

    int8x16_t filter_0_a;
    int8x16_t filter_0_b;
    int8x16_t filter_1_a;
    int8x16_t filter_1_b;
    int8x16_t filter_2_a;
    int8x16_t filter_2_b;

    // For uint8, effect subtraction of zero-point = 128 by XOR of sign bit.
    const uint8x16_t sign_bit = vdupq_n_u8(kSignBit);

    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_block = filter_data;
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      // Filter data is provided as filter_block[3][3][depth/8][2][4].
      // height 3, width 3, micro-blocks, sub-block 0 or 1, depth 4.
      // filter_bank[3][2][4][4]; Sub-block, height 3, depth 4, width 4.

      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          filter_block_ptr = filter_block;
      input_0_a = vld1q_lane_s8x8(filter_block_ptr, input_0_a, 0);
      filter_block_ptr += depth;
      input_0_b = vld1q_lane_s8x8(filter_block_ptr, input_0_b, 0);
      filter_block_ptr += depth;
      input_0_c = vld1q_lane_s8x8(filter_block_ptr, input_0_c, 0);
      filter_block_ptr += depth;
      input_1_a = vld1q_lane_s8x8(filter_block_ptr, input_1_a, 0);
      filter_block_ptr += depth;
      input_1_b = vld1q_lane_s8x8(filter_block_ptr, input_1_b, 0);
      filter_block_ptr += depth;
      input_1_c = vld1q_lane_s8x8(filter_block_ptr, input_1_c, 0);
      filter_block_ptr += depth;
      input_2_a = vld1q_lane_s8x8(filter_block_ptr, input_2_a, 0);
      filter_block_ptr += depth;
      input_2_b = vld1q_lane_s8x8(filter_block_ptr, input_2_b, 0);
      filter_block_ptr += depth;
      input_2_c = vld1q_lane_s8x8(filter_block_ptr, input_2_c, 0);

      filter_0_a = vzip1q_s8(input_0_a, input_0_b);
      filter_0_b = vzip1q_s8(input_0_c, sign_bit);
      filter_1_a = vzip1q_s8(input_1_a, input_1_b);
      filter_1_b = vzip1q_s8(input_1_c, sign_bit);
      filter_2_a = vzip1q_s8(input_2_a, input_2_b);
      filter_2_b = vzip1q_s8(input_2_c, sign_bit);
      if (quantization_type == QuantizationType::kNonPerChannelUint8) {
        filter_0_a = veorq_s8(filter_0_a, sign_bit);
        filter_0_b = veorq_s8(filter_0_b, sign_bit);
        filter_1_a = veorq_s8(filter_1_a, sign_bit);
        filter_1_b = veorq_s8(filter_1_b, sign_bit);
        filter_2_a = veorq_s8(filter_2_a, sign_bit);
        filter_2_b = veorq_s8(filter_2_b, sign_bit);
      }
      vzipq_s8x2_in_place(&filter_0_a, &filter_0_b);
      vzipq_s8x2_in_place(&filter_1_a, &filter_1_b);
      vzipq_s8x2_in_place(&filter_2_a, &filter_2_b);

      vst1q_s8(shuffled_filter_data, filter_0_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_0_b);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_1_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_1_b);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_2_a);
      shuffled_filter_data += 16;
      vst1q_s8(shuffled_filter_data, filter_2_b);
      shuffled_filter_data += 16;

      int32x4_t adjusted_bias_data_a = vld1q_s32(bias_data);
      bias_data += bias_increment;
      int32x4_t adjusted_bias_data_b = vld1q_s32(bias_data);
      bias_data += bias_increment;
      // For instance, if input_offset is kIntSymmetricZeroPoint, no adjustment
      // is needed.

      int32x4_t filter_sum_a = vdupq_n_s32(0);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_0_a, ones_vector);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_1_a, ones_vector);
      filter_sum_a = vdotq_s32(filter_sum_a, filter_2_a, ones_vector);
      int32x4_t filter_sum_b = vdupq_n_s32(0);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_0_b, ones_vector);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_1_b, ones_vector);
      filter_sum_b = vdotq_s32(filter_sum_b, filter_2_b, ones_vector);

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

  static inline void Run(const typename QuantizationTypeImpl<
                             quantization_type>::ExternalType* filter_data,
                         const int32* bias_data, int8* shuffled_filter_data,
                         int32* adjusted_bias_data,
                         const DepthwiseConvDotProdParams* function_params) {
    ProcessPerDepthIntrinsics(filter_data, bias_data, shuffled_filter_data,
                              adjusted_bias_data, function_params);
  }
};
#endif

template <QuantizationType quantization_type, int32 max_padding>
struct PackMacroBlock<
    DepthwiseConvImplementation::kUseCModel3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kNoMultiplication, max_padding> {
  // A straight copy of a macro block of input data into a scratch buffer.
  //
  // Requirement: depth_micro_repeats > 0.
  static inline void CopyMacroBlock(
      int32 height_block_number, int32 width_block_number,
      const DepthwiseConvDotProdParams& function_params,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data) {
    TFLITE_DCHECK_LE(max_padding, 1);

    // Strides.
    // The input depth and count of micro blocks provide the width strides.
    const int input_height_stride = function_params.input_height_stride;
    const int workspace_height_stride = function_params.workspace_height_stride;
    const int input_depth = function_params.input_depth;
    const int depth_micro_repeats = function_params.depth_micro_repeats;
    TFLITE_DCHECK_GT(depth_micro_repeats, 0);

    // Remaining iteration and dimension parameters.
    //
    // If width_overall_micro_repeats = input_width_micro_repeats + 1, then the
    // final micro block is incomplete.
    const int width_overall_micro_repeats =
        function_params.input_width_overall_micro_repeats;
    int input_width_micro_repeats = function_params.input_width_micro_repeats;
    const int residual_width = function_params.residual_width;
    const int block_height = function_params.inbound_block_height;

    const int padding_left = function_params.padding_left;
    const int padding_right = function_params.padding_right;
    const int padding_top = function_params.padding_top;
    const int padding_bottom = function_params.padding_bottom;

    const bool leading_width_padding =
        padding_left > 0 && width_block_number == 0;
    const bool trailing_width_padding =
        padding_right > 0 &&
        width_block_number == (function_params.width_macro_count - 1);
    const bool leading_height_padding =
        padding_top > 0 && height_block_number < 0;
    const bool trailing_height_padding =
        padding_bottom > 0 &&
        height_block_number == (function_params.height_macro_count - 1);

    // Modify the trailing case to reflect the input width.
    int input_residual_width =
        input_width_micro_repeats < width_overall_micro_repeats ? residual_width
                                                                : 4;
    if (trailing_width_padding) {
      input_residual_width -= 1;
      input_width_micro_repeats = width_overall_micro_repeats - 1;
    }

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    const int32 input_offset_difference =
        function_params.input_offset + kSymmetricZeroPoint;

    // We load data into a temporary buffer and then save, to match subsequent
    // processing. This will make it easier to combine stages into one ASM
    // routine.
    int8 tmp_load[4][2][4];

    int copy_block_height = block_height;
    if (leading_height_padding) {
      memset(scratch_block_data, -input_offset_difference,
             workspace_height_stride);
      scratch_block_data += workspace_height_stride;
      input_block_data += input_height_stride;
      copy_block_height -= 1;
    }
    if (trailing_height_padding) {
      copy_block_height -= 1;
    }

    // The outer 3 loops go through all the micro blocks in a macro block.
    for (int k_height = 0; k_height < copy_block_height; ++k_height) {
      for (int j_width = 0; j_width < width_overall_micro_repeats; ++j_width) {
        // Figure out division of work (available input vs trailing padding).
        int adjusted_residual_width =
            j_width == input_width_micro_repeats ? input_residual_width : 4;

        int start_width = 0;
        if (leading_width_padding && j_width == 0) {
          start_width = 1;
          memset(tmp_load[0][0], -input_offset_difference, 8);
        }
        if (adjusted_residual_width < 4) {
          for (int x = adjusted_residual_width; x < 4; ++x) {
            memset(tmp_load[x][0], -input_offset_difference, 8);
          }
        }

        for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
          // The inner 3 loops go through the sub-block, depth and width within
          // each micro block.

          // Load, and apply symmetric offset.
          int8* scratch_data =
              scratch_block_data + k_height * workspace_height_stride +
              j_width * 4 * 8 + i_depth * 4 * 8 * width_overall_micro_repeats;
          const typename QuantizationTypeImpl<quantization_type>::ExternalType*
              input_data = input_block_data + k_height * input_height_stride +
                           j_width * 4 * input_depth + i_depth * 8;
          // Full-size macro blocks are 2*4*4 = 32 bytes.
          for (int x = start_width; x < adjusted_residual_width; ++x) {
            for (int s = 0; s < 2; ++s) {
              for (int d = 0; d < 4; ++d) {
                tmp_load[x][s][d] = input_data[x * input_depth + 4 * s + d] -
                                    kSymmetricZeroPoint;
              }
            }
          }

          // Save results.
          memcpy(&scratch_data[0], tmp_load[0][0], 8);
          memcpy(&scratch_data[8], tmp_load[1][0], 8);
          memcpy(&scratch_data[16], tmp_load[2][0], 8);
          memcpy(&scratch_data[24], tmp_load[3][0], 8);
        }
      }
    }

    if (trailing_height_padding) {
      memset(scratch_block_data + copy_block_height * workspace_height_stride,
             -input_offset_difference, workspace_height_stride);
    }
  }

  // Transpose 4x4 blocks within each sub-micro-block.
  //
  // Implemented somewhat like NEON register manipulation, so that we can see
  // equivalence of the two approaches.
  static inline void MicroTransposeBlocks(
      const DepthwiseConvDotProdParams& function_params,
      int8* scratch_block_data) {
    const int workspace_height_stride = function_params.workspace_height_stride;
    const int width_overall_micro_repeats =
        function_params.input_width_overall_micro_repeats;
    const int depth_micro_repeats = function_params.depth_micro_repeats;
    const int block_height = function_params.inbound_block_height;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in the
    // NEON code we are simulating.
    int8 tmp_load[4][2][4];         // [width][sub-block][depth]
    int8 tmp_transposed[4][2][4];   // [depth][sub-block][width]
    int8 tmp_interleaved[2][4][4];  // [sub-block][depth][width]

    // The outer 3 loops go through all the micro blocks in a macro block.
    for (int k_height = 0; k_height < block_height; ++k_height) {
      for (int j_width = 0; j_width < width_overall_micro_repeats; ++j_width) {
        for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
          int8* scratch_data =
              scratch_block_data + k_height * workspace_height_stride +
              j_width * 4 * 8 + i_depth * 4 * 8 * width_overall_micro_repeats;
          // A. Load data
          memcpy(tmp_load[0][0], &scratch_data[0], 8);
          memcpy(tmp_load[1][0], &scratch_data[8], 8);
          memcpy(tmp_load[2][0], &scratch_data[16], 8);
          memcpy(tmp_load[3][0], &scratch_data[24], 8);

          // B. Simulate between-register transposition.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_transposed[x][0][y] = tmp_load[y][0][x];
              tmp_transposed[x][1][y] = tmp_load[y][1][x];
            }
          }

          // C. Simulate between-register interleaving.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_interleaved[0][x][y] = tmp_transposed[x][0][y];
              tmp_interleaved[1][x][y] = tmp_transposed[x][1][y];
            }
          }
          // D. Simulate mangled storage arrangement.
          memcpy(&scratch_data[0], tmp_interleaved[0][0], 16);
          memcpy(&scratch_data[16], tmp_interleaved[1][0], 16);
        }
      }
    }
  }

  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    CopyMacroBlock(height_block_number, width_block_number, *function_params,
                   input_block_data, scratch_block_data);
    MicroTransposeBlocks(*function_params, scratch_block_data);
  }
};

template <QuantizationType quantization_type, int32 max_padding>
struct PackMacroBlock<
    DepthwiseConvImplementation::kUseCModel3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kUnitInputDepth, max_padding> {
  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    // Currently support for padding is limited to 1 on any side.
    TFLITE_DCHECK_LE(max_padding, 1);

    // Strides.
    // The count of micro blocks (below) provides the width strides.
    const int input_height_stride = function_params->input_height_stride;
    const int workspace_height_stride =
        function_params->workspace_height_stride;

    // Remaining iteration and dimension parameters.
    //
    // If width_overall_micro_repeats = input_width_micro_repeats + 1, then the
    // final micro block is incomplete.
    const int width_overall_micro_repeats =
        function_params->input_width_overall_micro_repeats;
    const int input_width_micro_repeats =
        function_params->input_width_micro_repeats;
    const int residual_width = function_params->residual_width;
    const int block_height = function_params->inbound_block_height;
    TFLITE_DCHECK_GE(workspace_height_stride, 4 * width_overall_micro_repeats);

    const int padding_left = function_params->padding_left;
    const int padding_right = function_params->padding_right;
    const int padding_top = function_params->padding_top;
    const int padding_bottom = function_params->padding_bottom;

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

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    const int32 input_offset_difference =
        function_params->input_offset + kSymmetricZeroPoint;

    int copy_block_height = block_height;
    if (leading_height_padding) {
      memset(scratch_block_data, -input_offset_difference,
             workspace_height_stride + kWorkspaceExtension);
      scratch_block_data += workspace_height_stride;
      input_block_data += input_height_stride;
      copy_block_height -= 1;
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

    TFLITE_DCHECK_LE(
        copy_size,
        input_height_stride - width_block_number * input_width_micro_repeats);
    // We may drop up to stride-1 of trailing input.
    TFLITE_DCHECK_GE(copy_size, input_height_stride - 1);

    // When there is unit input depth, the micro-block iteration need only be
    // through the height. The micro blocks are contiguous across the width.
    for (int k_height = 0; k_height < copy_block_height; ++k_height) {
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_data = input_block_data + k_height * input_height_stride;
      int8* scratch_data =
          scratch_block_data + k_height * workspace_height_stride;

      // Handle leading padding. This is overwritten if there is no padding.
      scratch_data[0] = -input_offset_difference;

      memcpy(&scratch_data[start_width], input_data, copy_size);
      for (int i = 0; i < copy_size; ++i) {
        scratch_data[start_width + i] += -kSymmetricZeroPoint;
      }

      // Handle trailing padding, and fill in remainder of micro block.
      memset(&scratch_data[start_width + copy_size], -input_offset_difference,
             4 - adjusted_residual_width + kWorkspaceExtension);
    }

    if (trailing_height_padding) {
      memset(scratch_block_data + copy_block_height * workspace_height_stride,
             -input_offset_difference,
             workspace_height_stride + kWorkspaceExtension);
    }
  }
};

// Beginning of code section containing intermediate code transformation.
//
// This section is only compiled when kUseUnwound3x3DotProduct versions of
// templated functions are selected.
template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseUnwound3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/0> {
  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
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
    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
    const int micro_block_size = 4 * 8;
    const int depth_advance = width_overall_micro_repeats * micro_block_size;
    const int width_advance =
        micro_block_size *
        (1 - depth_micro_repeats * width_overall_micro_repeats);
    const int height_advance = workspace_height_stride -
                               width_overall_micro_repeats * micro_block_size;
    const int input_depth_skip = 4 * input_depth - 8 * depth_micro_repeats;

    // Transpositions are 4x4, but doing 2 at a time is more efficient in the
    // NEON code we are simulating. Note the blocks of 4x4 are still interleaved
    // down the depth.
    int8 tmp_load[4][2][4];
    int8 tmp_transposed[4][2][4];
    int8 tmp_interleaved[2][4][4];

    // Work through one slice, by row, at a time.
    int8* scratch_data = scratch_block_data;
    for (int k_height = 0; k_height < block_height; ++k_height) {
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_data = input_block_data;
      input_block_data += input_height_stride;

      // Traverse the width one point at a time, but the depth in (micro) blocks
      // of size 8.
      //
      // The depth and width margins, which are filled with "zeros", may be
      // larger than is strictly needed to calculate output. This is because the
      // conv calculation is performed across complete micro blocks.
      for (int j_width = 0; j_width < input_width_micro_repeats; ++j_width) {
        // Load, then zero.
        for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
          // A. Simulate register loading.
          for (int x = 0; x < 4; ++x) {
            for (int s = 0; s < 2; ++s) {
              for (int d = 0; d < 4; ++d) {
                tmp_load[x][s][d] = input_data[x * input_depth + 4 * s + d] -
                                    kSymmetricZeroPoint;
              }
            }
          }
          // B. Simulate between-register transposition.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_transposed[x][0][y] = tmp_load[y][0][x];
              tmp_transposed[x][1][y] = tmp_load[y][1][x];
            }
          }

          // C and D are to be performed together as 4-byte stores in NEON code.
          // C. Simulate between-register interleaving.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_interleaved[0][x][y] = tmp_transposed[x][0][y];
              tmp_interleaved[1][x][y] = tmp_transposed[x][1][y];
            }
          }
          // D. Simulate mangled storage arrangement.
          memcpy(&scratch_data[0], tmp_interleaved[0][0], 8);
          memcpy(&scratch_data[8], tmp_interleaved[0][2], 8);
          memcpy(&scratch_data[16], tmp_interleaved[1][0], 8);
          memcpy(&scratch_data[24], tmp_interleaved[1][2], 8);

          scratch_data += depth_advance;
          input_data += 8;
        }
        scratch_data += width_advance;
        input_data += input_depth_skip;
      }
      if (width_overall_micro_repeats > input_width_micro_repeats) {
        TFLITE_DCHECK_EQ(width_overall_micro_repeats,
                         input_width_micro_repeats + 1);
        TFLITE_DCHECK_GT(residual_width, 0);
        // Figure out division of work (available input vs zero-ed).
        const int adjusted_residual_width = residual_width;
        // Load, then zero.
        for (int i_depth = 0; i_depth < depth_micro_repeats; ++i_depth) {
          // A. Simulate register loading.
          for (int x = 0; x < adjusted_residual_width; ++x) {
            for (int s = 0; s < 2; ++s) {
              for (int d = 0; d < 4; ++d) {
                tmp_load[x][s][d] = input_data[x * input_depth + 4 * s + d] -
                                    kSymmetricZeroPoint;
              }
            }
          }
          for (int x = adjusted_residual_width; x < 4; ++x) {
            for (int s = 0; s < 2; ++s) {
              for (int d = 0; d < 4; ++d) {
                tmp_load[x][s][d] = 0;
              }
            }
          }
          // B. Simulate between-register transposition.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_transposed[x][0][y] = tmp_load[y][0][x];
              tmp_transposed[x][1][y] = tmp_load[y][1][x];
            }
          }

          // C and D are to be performed together as 4-byte stores in NEON code.
          // C. Simulate between-register interleaving.
          for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
              tmp_interleaved[0][x][y] = tmp_transposed[x][0][y];
              tmp_interleaved[1][x][y] = tmp_transposed[x][1][y];
            }
          }
          // D. Simulate mangled storage arrangement.
          memcpy(&scratch_data[0], tmp_interleaved[0][0], 8);
          memcpy(&scratch_data[8], tmp_interleaved[0][2], 8);
          memcpy(&scratch_data[16], tmp_interleaved[1][0], 8);
          memcpy(&scratch_data[24], tmp_interleaved[1][2], 8);

          scratch_data += depth_advance;
          input_data += 8;
        }
        scratch_data += width_advance;
        input_data += input_depth_skip;
      }
      scratch_data += height_advance;
    }

    TFLITE_DCHECK_EQ(scratch_data, scratch_block_data +
                                       block_height * workspace_height_stride);
  }
};

template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseUnwound3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/1> {
  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    // Just use C model code for case of padding. Optimized versions merge the
    // modifications therein to handle padding.
    PackMacroBlock<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                   quantization_type,
                   DepthwiseConvDepthMultiplication::kNoMultiplication,
                   /*max_padding=*/1>::Run(height_block_number,
                                           width_block_number, input_block_data,
                                           scratch_block_data, function_params);
  }
};

template <QuantizationType quantization_type, int32 max_padding>
struct PackMacroBlock<
    DepthwiseConvImplementation::kUseUnwound3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kUnitInputDepth, max_padding> {
  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
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

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;

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

    // This is used to simulate what should happen in registers.
    int8 tmp_data[16];

    int scratch_data_offset = 0;
    int input_block_offset = 0;

    if (copy_size >= 16) {
      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // The surrounding condition ensures that we always need at least one
        // iteration of the main copy loop. In the case of leading width
        // padding, we unroll this specially.
        if (leading_width_padding) {
          memcpy(tmp_data + 1, input_block_data + input_block_offset, 15);
          for (int i = 0; i < 16; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          tmp_data[0] = -input_offset_difference;
          memcpy(scratch_data, tmp_data, 16);
          copy_done += 15;
        }

        // Main copy loop.
        for (; (copy_done + 16) <= copy_size; copy_done += 16) {
          memcpy(tmp_data, input_block_data + input_block_offset + copy_done,
                 16);
          for (int i = 0; i < 16; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          TFLITE_DCHECK_EQ((start_width + copy_done) % 16, 0);
          memcpy(&scratch_data[start_width + copy_done], tmp_data, 16);
        }

        const int copy_remaining = copy_size - copy_done;
        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.

        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size_adjusted) {
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          memcpy(tmp_data,
                 input_block_data + input_block_offset + copy_done -
                     (16 - copy_remaining),
                 16);
          // Shift to select the part that we need.
          for (int i = 0; i < copy_remaining; ++i) {
            tmp_data[i] = tmp_data[(16 - copy_remaining) + i];
          }
          for (int i = 0; i < 16; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          // Apply padding to remainder, some unnecessary but costless in regs.
          for (int i = copy_remaining; i < 16; ++i) {
            tmp_data[i] = -input_offset_difference;
          }
          const int final_repeats =
              width_overall_micro_repeats - (start_width + copy_done) / 4;
          for (int i = 0; i < final_repeats; ++i) {
            memcpy(&scratch_data[start_width + copy_done], tmp_data + 4 * i, 4);
            copy_done += 4;
          }
        }
        memset(scratch_data + start_width + copy_done, -input_offset_difference,
               kWorkspaceExtension);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else if (copy_size >= 4) {
      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Work through one slice, by row, at a time.
        int8* scratch_data = scratch_data_base + scratch_data_offset;

        int copy_done = 0;

        // The surrounding condition ensures that we always need at least one
        // iteration of the main copy loop. In the case of leading width
        // padding, we unroll this specially.
        if (leading_width_padding) {
          memcpy(tmp_data + 1, input_block_data + input_block_offset, 3);
          for (int i = 0; i < 4; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          tmp_data[0] = -input_offset_difference;
          memcpy(scratch_data, tmp_data, 4);
          copy_done += 3;
        }

        for (; (copy_done + 4) <= copy_size; copy_done += 4) {
          memcpy(tmp_data, input_block_data + input_block_offset + copy_done,
                 4);
          for (int i = 0; i < 4; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          // Perform as 4 int32 stores, because that is our alignment.
          memcpy(&scratch_data[start_width + copy_done], tmp_data, 4);
        }

        // Total amount
        // = copy_size - copy_done + 4 - adjusted_residual_width
        // = width_overall_micro_repeats * 4 - start_width - copy_done.
        // Undone micro blocks
        // = width_overall_micro_repeats - (start_width + copy_done) / 4.
        const int copy_remaining = copy_size - copy_done;
        // Conditional is (copy_remaining > 0 || trailing_width_padding).
        if (copy_done < copy_size_adjusted) {
          TFLITE_DCHECK_LT(copy_remaining, 4);
          // Employ overlapping-load strategy in order to load full register,
          // but use only part.
          memcpy(tmp_data,
                 input_block_data + input_block_offset + copy_done -
                     (4 - copy_remaining),
                 4);
          // Shift to select the part that we need.
          for (int i = 0; i < copy_remaining; ++i) {
            tmp_data[i] = tmp_data[(4 - copy_remaining) + i];
          }
          for (int i = 0; i < 4; ++i) {
            tmp_data[i] += -kSymmetricZeroPoint;
          }
          // Apply padding to remainder, some unnecessary but costless in regs.
          for (int i = copy_remaining; i < 4; ++i) {
            tmp_data[i] = -input_offset_difference;
          }
          memcpy(&scratch_data[start_width + copy_done], tmp_data, 4);
          copy_done += 4;
        }
        memset(scratch_data + start_width + copy_done, -input_offset_difference,
               kWorkspaceExtension);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else if (width_overall_micro_repeats == 2) {
      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Apply padding by quick fill of whole reg.
        for (int i = 0; i < 8; ++i) {
          tmp_data[i] = -input_offset;
        }
        for (int i = 0; i < copy_size; ++i) {
          // Apply shift-left insert, tmp_data as both operands.
          // The zero-index byte is left unchanged.
          for (int i = 7; i > 0; --i) {
            tmp_data[i] = tmp_data[i - 1];
          }
          tmp_data[1] =
              input_block_data[input_block_offset + (copy_size - 1 - i)];
        }
        if (!leading_width_padding) {
          // Remove leading padding, junking trailing byte, OK because max size
          // is less than 8.
          TFLITE_DCHECK_LT(copy_size_adjusted + start_width, 8);
          for (int i = 0; i < 7; ++i) {
            tmp_data[i] = tmp_data[i + 1];
          }
        }
        for (int i = 0; i < 8; ++i) {
          tmp_data[i] += -kSymmetricZeroPoint;
        }
        memcpy(scratch_data_base + scratch_data_offset, tmp_data, 8);
        memset(scratch_data_base + scratch_data_offset + 8,
               -input_offset_difference, kWorkspaceExtension);

        scratch_data_offset += workspace_height_stride;
        input_block_offset += input_height_stride;
      }
    } else {
      TFLITE_DCHECK_EQ(width_overall_micro_repeats, 1);
      // This path is basically the same as the preceding, 2-micro-block one,
      // but here we simply store fewer bytes.
      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        // Apply padding by quick fill of whole reg.
        for (int i = 0; i < 8; ++i) {
          tmp_data[i] = -input_offset;
        }
        for (int i = 0; i < copy_size; ++i) {
          // Apply shift-left insert, tmp_data as both operands.
          // The zero-index byte is left unchanged.
          for (int i = 7; i > 0; --i) {
            tmp_data[i] = tmp_data[i - 1];
          }
          tmp_data[1] =
              input_block_data[input_block_offset + (copy_size - 1 - i)];
        }
        if (!leading_width_padding) {
          // Remove leading padding, junking trailing byte, OK because max size
          // is less than 8.
          TFLITE_DCHECK_LT(copy_size_adjusted + start_width, 8);
          for (int i = 0; i < 7; ++i) {
            tmp_data[i] = tmp_data[i + 1];
          }
        }
        for (int i = 0; i < 8; ++i) {
          tmp_data[i] += -kSymmetricZeroPoint;
        }
        memcpy(scratch_data_base + scratch_data_offset, tmp_data, 4);
        memset(scratch_data_base + scratch_data_offset + 4,
               -input_offset_difference, kWorkspaceExtension);

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
};
// The preceding section is only compiled when kUseUnwound3x3DotProduct versions
// of templated functions are selected.
//
// End of code section containing intermediate code transformation.

#ifdef USE_NEON
template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/0> {
  static inline void PackMacroBlockIntrinsics(
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
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
    constexpr uint8 kSignBit =
        QuantizationTypeImpl<quantization_type>::kUint8SignBit;
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
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_data_0 = input_block_data;
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

          input_data_a = util_vld1q_x8(input_data_0);
          input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
          input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
          input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
          input_data_0 += 16;

          for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
            work_reg_a = vzip1q_s8(input_data_a, input_data_b);
            work_reg_b = vzip1q_s8(input_data_c, input_data_d);
            vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
            if (quantization_type == QuantizationType::kNonPerChannelUint8) {
              work_reg_a = veorq_s8(work_reg_a, sign_bit);
              work_reg_b = veorq_s8(work_reg_b, sign_bit);
            }

            work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
            work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
            vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

            input_data_a = util_vld1q_x8(input_data_0);
            input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
            vst1q_s8(scratch_data_0, work_reg_a);
            vst1q_s8(scratch_data_0 + 16, work_reg_b);

            scratch_data_0 += depth_advance;

            if (quantization_type == QuantizationType::kNonPerChannelUint8) {
              work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
              work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
            }

            input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
            input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
            vst1q_s8(scratch_data_0, work_reg_a_sp);
            vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

            scratch_data_0 += depth_advance;
            input_data_0 += 16;
          }

          work_reg_a = vzip1q_s8(input_data_a, input_data_b);
          work_reg_b = vzip1q_s8(input_data_c, input_data_d);
          vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg_a = veorq_s8(work_reg_a, sign_bit);
            work_reg_b = veorq_s8(work_reg_b, sign_bit);
          }
          vst1q_s8(scratch_data_0, work_reg_a);
          vst1q_s8(scratch_data_0 + 16, work_reg_b);

          scratch_data_0 += depth_advance;

          work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
          work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
          vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
            work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
          }

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
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg_a = veorq_s8(work_reg_a, sign_bit);
            work_reg_b = veorq_s8(work_reg_b, sign_bit);
          }

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

          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg_a = veorq_s8(work_reg_a, sign_bit);
            work_reg_b = veorq_s8(work_reg_b, sign_bit);
          }
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

  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
#ifdef __aarch64__
    PreloadInputBlock(input_block_data, function_params);
#endif
    PackMacroBlockIntrinsics(input_block_data, scratch_block_data,
                             function_params);
  }
};

template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      /*max_padding=*/1> {
  static inline void PackMacroBlockIntrinsics(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    constexpr uint8 kSignBit =
        QuantizationTypeImpl<quantization_type>::kUint8SignBit;

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
    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;

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
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_data_0 = input_block_data;
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

              input_data_a = util_vld1q_x8(input_data_0);
              input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
              input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
              input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
              input_data_0 += 16;

              for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
                work_reg_a = vzip1q_s8(input_data_a, input_data_b);
                work_reg_b = vzip1q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
                if (quantization_type ==
                    QuantizationType::kNonPerChannelUint8) {
                  work_reg_a = veorq_s8(work_reg_a, sign_bit);
                  work_reg_b = veorq_s8(work_reg_b, sign_bit);
                }

                work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
                work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

                input_data_a = util_vld1q_x8(input_data_0);
                input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a);
                vst1q_s8(scratch_data_0 + 16, work_reg_b);

                scratch_data_0 += depth_advance;

                if (quantization_type ==
                    QuantizationType::kNonPerChannelUint8) {
                  work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                  work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
                }

                input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
                input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a_sp);
                vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

                scratch_data_0 += depth_advance;
                input_data_0 += 16;
              }

              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }
              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;

              work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
              work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
              }

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
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }

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

              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }
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

              input_data_a = vdupq_n_u8(-input_offset);
              input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
              input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
              input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
              input_data_0 += 16;

              for (; i_depth < depth_micro_repeats - 1; i_depth += 2) {
                work_reg_a = vzip1q_s8(input_data_a, input_data_b);
                work_reg_b = vzip1q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
                if (quantization_type ==
                    QuantizationType::kNonPerChannelUint8) {
                  work_reg_a = veorq_s8(work_reg_a, sign_bit);
                  work_reg_b = veorq_s8(work_reg_b, sign_bit);
                }

                work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
                work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
                vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);

                input_data_a = vdupq_n_u8(-input_offset);
                input_data_b = util_vld1q_x8(input_data_0 + 1 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a);
                vst1q_s8(scratch_data_0 + 16, work_reg_b);

                scratch_data_0 += depth_advance;

                if (quantization_type ==
                    QuantizationType::kNonPerChannelUint8) {
                  work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                  work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
                }

                input_data_c = util_vld1q_x8(input_data_0 + 2 * input_depth);
                input_data_d = util_vld1q_x8(input_data_0 + 3 * input_depth);
                vst1q_s8(scratch_data_0, work_reg_a_sp);
                vst1q_s8(scratch_data_0 + 16, work_reg_b_sp);

                scratch_data_0 += depth_advance;
                input_data_0 += 16;
              }

              work_reg_a = vzip1q_s8(input_data_a, input_data_b);
              work_reg_b = vzip1q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a, &work_reg_b);
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }
              vst1q_s8(scratch_data_0, work_reg_a);
              vst1q_s8(scratch_data_0 + 16, work_reg_b);

              scratch_data_0 += depth_advance;

              work_reg_a_sp = vzip2q_s8(input_data_a, input_data_b);
              work_reg_b_sp = vzip2q_s8(input_data_c, input_data_d);
              vzipq_s8x2_in_place(&work_reg_a_sp, &work_reg_b_sp);
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a_sp = veorq_s8(work_reg_a_sp, sign_bit);
                work_reg_b_sp = veorq_s8(work_reg_b_sp, sign_bit);
              }

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
              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }

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

              if (quantization_type == QuantizationType::kNonPerChannelUint8) {
                work_reg_a = veorq_s8(work_reg_a, sign_bit);
                work_reg_b = veorq_s8(work_reg_b, sign_bit);
              }
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

  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
#ifdef __aarch64__
    PreloadInputBlock(input_block_data, function_params);
#endif

    PackMacroBlockIntrinsics(height_block_number, width_block_number,
                             input_block_data, scratch_block_data,
                             function_params);
  }
};

template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kUnitInputDepth,
                      /*max_padding=*/1> {
  static inline void PackMacroBlockIntrinsics(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
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

    constexpr int kSymmetricZeroPoint =
        QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;

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

    constexpr uint8 kSignBit =
        QuantizationTypeImpl<quantization_type>::kUint8SignBit;

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
          work_reg = util_vld1q_x8(input_block_data + input_block_offset);
          work_reg = vextq_s8(padding_reg, work_reg, 15);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg = veorq_s8(work_reg, sign_bit);
          }
          vst1q_s8(scratch_data, work_reg);
          copy_done += 15;
        }

        // Main copy loop.
        for (; (copy_done + 16) <= copy_size; copy_done += 16) {
          work_reg =
              util_vld1q_x8(input_block_data + input_block_offset + copy_done);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg = veorq_s8(work_reg, sign_bit);
          }
          TFLITE_DCHECK_EQ((start_width + copy_done) % 16, 0);
          vst1q_s8(scratch_data + start_width + copy_done, work_reg);
        }

        if (copy_done + 8 <= copy_size) {
          half_work_reg =
              util_vld1_x8(input_block_data + input_block_offset + copy_done);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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
          half_work_reg = util_vld1_x8(input_block_data + input_block_offset +
                                       copy_size - 8);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (8 - copy_remaining)));
          half_work_reg =
              vbsl_s8(padding_mask, vget_low_s8(padding_reg), half_work_reg);

          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
          vst1_lane_8x4(scratch_data, half_work_reg, 0);
          copy_done += 3;
        }

        // Main copy loop.
        for (; (copy_done + 4) <= copy_size; copy_done += 4) {
          half_work_reg =
              vld1_lane_8x4(input_block_data + input_block_offset + copy_done,
                            half_work_reg, 0);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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

          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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

      for (int k_height = 0; k_height < copy_block_height; ++k_height) {
        half_work_reg = vdup_n_u8(-input_offset);
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

        if (quantization_type == QuantizationType::kNonPerChannelUint8) {
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
        }
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

        if (quantization_type == QuantizationType::kNonPerChannelUint8) {
          half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
        }
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

  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
#ifdef __aarch64__
    PreloadInputBlock(input_block_data, function_params);
#endif

    PackMacroBlockIntrinsics(height_block_number, width_block_number,
                             input_block_data, scratch_block_data,
                             function_params);
  }
};

template <QuantizationType quantization_type>
struct PackMacroBlock<DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
                      quantization_type,
                      DepthwiseConvDepthMultiplication::kUnitInputDepth,
                      /*max_padding=*/0> {
  static inline void PackMacroBlockIntrinsics(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
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

    constexpr uint8 kSignBit =
        QuantizationTypeImpl<quantization_type>::kUint8SignBit;

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
              util_vld1q_x8(input_block_data + input_block_offset + copy_done);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            work_reg = veorq_s8(work_reg, sign_bit);
          }
          TFLITE_DCHECK_EQ(copy_done % 16, 0);
          vst1q_s8(scratch_data + copy_done, work_reg);
        }

        if (copy_done + 8 <= copy_size) {
          half_work_reg =
              util_vld1_x8(input_block_data + input_block_offset + copy_done);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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
          half_work_reg = util_vld1_x8(input_block_data + input_block_offset +
                                       copy_size - 8);

          half_work_reg =
              vshl_u64(half_work_reg, vdup_n_s64(-8 * (8 - copy_remaining)));

          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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
          half_work_reg =
              vld1_lane_8x4(input_block_data + input_block_offset + copy_done,
                            half_work_reg, 0);
          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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

          if (quantization_type == QuantizationType::kNonPerChannelUint8) {
            half_work_reg = veor_s8(half_work_reg, vget_low_s8(sign_bit));
          }
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

  static inline void Run(
      int32 height_block_number, int32 width_block_number,
      const typename QuantizationTypeImpl<quantization_type>::ExternalType*
          input_block_data,
      int8* scratch_block_data,
      const DepthwiseConvDotProdParams* function_params) {
#ifdef __aarch64__
    PreloadInputBlock(input_block_data, function_params);
#endif

    PackMacroBlockIntrinsics(height_block_number, width_block_number,
                             input_block_data, scratch_block_data,
                             function_params);
  }
};

#endif  // ARM NEON

// Apply filter to macro block of input data and store results.
//
// Requirement: depth_micro_repeats > 0 || residual_depth > 0.
template <int32 stride, QuantizationType quantization_type>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseCModel3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kNoMultiplication, stride> {
  // Construct a width-shifted combination of two input sub-blocks, effectively
  // concatenating them.
  //
  // The filter is applied using sub-blocks. These are in the needed form for
  // the first (width) offset. For subsequent offsets, the filter is applied to
  // shifted and combined data. The concatentation and shifting herein is fairly
  // straightforward, but in the optimized code is an area of creativity in
  // design because NEON instructions do not directly support the required
  // between-register permutation.
  //
  // In NEON optimized code, input data is grouped in 4-byte blocks. In order to
  // move along the width for each output point calculation, data is shifted, in
  // essence between two such blocks.
  //
  // selected_data has format height 3, depth 4, width 4.
  //
  // When the micro block is trailing (the last across the macro-block width),
  // it would be illegal to load the right (next) block, and the no_right_block
  // indicates this scenario.
  static inline void ConcatenateInputSubBlocks(int offset, int sub_block,
                                               int workspace_height_stride,
                                               int width_micro_stride,
                                               bool no_right_block,
                                               const int8* input_block,
                                               int8 selected_data[3][4][4]) {
    TFLITE_DCHECK_GE(offset, 0);
    TFLITE_DCHECK_LT(offset, 4);

    // The input banks have same format as selected_data.
    int8 left_bank[3][4][4];
    int8 right_bank[3][4][4];

    // Work through one slice, by row, at a time.
    for (int k_height = 0; k_height < 3; ++k_height) {
      // Simulate demangling of mangled storage arrangement.
      const int8* left_input_block =
          &input_block[k_height * workspace_height_stride + sub_block * 2 * 8];
      memcpy(left_bank[k_height][0], left_input_block, 16);
      if (no_right_block) {
        memset(right_bank[k_height][0], 0, 16);
      } else {
        const int8* right_input_block =
            &input_block[k_height * workspace_height_stride +
                         sub_block * 2 * 8 + width_micro_stride];
        memcpy(right_bank[k_height][0], right_input_block, 16);
      }
      for (int depth_index = 0; depth_index < 4; ++depth_index) {
        memcpy(selected_data[k_height][depth_index],
               &left_bank[k_height][depth_index][offset], 4 - offset);
        memcpy(&selected_data[k_height][depth_index][4 - offset],
               right_bank[k_height][depth_index], offset);
      }
    }
  }

  // Straight implementation of 3x3 filter within sub-micro block.
  static inline void Calculate3x3FilterOutput(
      const DepthwiseConvDotProdParams& params, int sub_block,
      const int8 selected_data[3][4][4], const int8 filter_bank[3][2][4][4],
      const int32* bias_data, uint8 output_values[4]) {
    const int32 output_activation_min = params.quantized_activation_min;
    const int32 output_activation_max = params.quantized_activation_max;
    const int32 output_multiplier = params.output_multiplier;
    const int32 output_shift = params.output_shift;
    const int32 output_offset = params.output_offset;
    for (int d = 0; d < 4; ++d) {
      int32 acc = 0;
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
          int32 input_val = selected_data[y][d][x];
          int32 filter_val = filter_bank[y][sub_block][d][x];
          acc += filter_val * input_val;
        }
      }
      acc += bias_data[d];
      acc = reference_ops::depthwise_conv::DepthwiseConvRound<
          DepthwiseConvOutputRounding::kUpward>(acc, output_multiplier,
                                                output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_values[d] = static_cast<uint8>(acc);
    }
  }

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
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

    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    constexpr int bias_increment = 4;
    TFLITE_DCHECK_EQ(function_params->bias_increment, bias_increment);

    TFLITE_DCHECK(depth_micro_repeats > 0);
    const int width_micro_stride = 4 * 8;
    const int depth_micro_stride =
        width_micro_stride * input_width_overall_micro_repeats;

    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;

    // Simulate NEON-register transposition of subset of filter.
    int8 filter_bank[3][2][4][4];  // Height 3, sub-block,  depth 4, width 4.
    // Simulate NEON-register input data concatenation + sub-selection.
    int8 sub_selected_input_data[3][4][4];  // Height 3, depth 4, width 4.
    uint8 output_values[4];                 // Depth 4.

    // The outer 3 loops go through all the micro blocks in a macro block, and
    // separately treat the two sub-blocks within each micro block.
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      memcpy(filter_bank[0][0][0],
             filter_workspace + j_depth * shuffled_filter_increment,
             shuffled_filter_increment);

      for (int s = 0; s < 2; ++s) {
        for (int k_height = 0; k_height < block_height; ++k_height) {
          const int8* scratch_data =
              scratch_block_data +
              workspace_height_stride * k_height * stride_val +
              depth_micro_stride * j_depth;
          uint8* output_data =
              output_block_data + output_height_stride * k_height + 8 * j_depth;

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            const int output_width = i_width == output_width_micro_repeats
                                         ? residual_width
                                         : four_over_stride;
            const bool no_right_block = (output_width - 1) * stride_val < 2;
            TFLITE_DCHECK_LE(output_width * stride_val, 4);
            const int8* input_data =
                scratch_data + width_micro_stride * i_width;
            // Iterate over input width shifts within sub-micro blocks.
            for (int x = 0; x < output_width; ++x) {
              ConcatenateInputSubBlocks(x * stride_val, s,
                                        workspace_height_stride,
                                        width_micro_stride, no_right_block,
                                        input_data, sub_selected_input_data);
              Calculate3x3FilterOutput(
                  *function_params, s, sub_selected_input_data, filter_bank,
                  bias_data + (2 * j_depth + s) * bias_increment,
                  output_values);
              for (int d = 0; d < 4; ++d) {
                output_data[depth * (four_over_stride * i_width + x) + 4 * s +
                            d] = output_values[d];
              }
            }
          }
        }
      }
    }
  }
};

// Apply filter to macro block of input data and store results.
//
// Parameters for repeats and residual sizes are in terms of outputs.
//
// Requirement: depth_micro_repeats > 0 || residual_depth > 0.
template <int32 stride, QuantizationType quantization_type>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseCModel3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kUnitInputDepth, stride> {
  // Construct a width-shifted combination of two input sub-blocks, effectively
  // concatenating them.
  //
  // The filter is applied using sub-blocks. These are in the needed form for
  // the first (width) offset. For subsequent offsets, the filter is applied to
  // shifted and combined data. The concatentation and shifting herein is fairly
  // straightforward, but in the optimized code is an area of creativity in
  // design because NEON instructions do not directly support the required
  // between-register permutation.
  //
  // In NEON optimized code, input data is grouped in 4-byte blocks. In order to
  // move along the width for each output point calculation, data is shifted, in
  // essence between two such blocks.
  //
  // selected_data has format height 3, width 4.
  //
  // When the micro block is trailing (the last across the macro-block width),
  // it would be illegal to load the right (next) block, and the no_right_block
  // indicates this scenario.
  static inline void ConcatenateInputSubBlocks(int offset,
                                               int workspace_height_stride,
                                               bool no_right_block,
                                               const int8* input_block,
                                               int8 selected_data[3][4]) {
    TFLITE_DCHECK_GE(offset, 0);
    TFLITE_DCHECK_LT(offset, 4);
    if (no_right_block) {
      for (int k_height = 0; k_height < 3; ++k_height) {
        memcpy(selected_data[k_height],
               &input_block[k_height * workspace_height_stride + offset],
               4 - offset);
      }
    } else {
      for (int k_height = 0; k_height < 3; ++k_height) {
        memcpy(selected_data[k_height],
               &input_block[k_height * workspace_height_stride + offset], 4);
      }
    }
  }

  // Straight implementation of 3x3 filter within sub-micro block.
  static inline void Calculate3x3FilterOutput(
      const DepthwiseConvDotProdParams& function_params, int sub_block,
      const int8 selected_data[3][4], const int8 filter_bank[3][2][4][4],
      const int32* bias_data, uint8 output_values[4]) {
    const int32 output_activation_min =
        function_params.quantized_activation_min;
    const int32 output_activation_max =
        function_params.quantized_activation_max;
    const int32 output_multiplier = function_params.output_multiplier;
    const int32 output_shift = function_params.output_shift;
    const int32 output_offset = function_params.output_offset;
    for (int d = 0; d < 4; ++d) {
      int32 acc = 0;
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
          int32 input_val = selected_data[y][x];
          int32 filter_val = filter_bank[y][sub_block][d][x];
          acc += filter_val * input_val;
        }
      }
      acc += bias_data[d];
      acc = reference_ops::depthwise_conv::DepthwiseConvRound<
          DepthwiseConvOutputRounding::kUpward>(acc, output_multiplier,
                                                output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_values[d] = static_cast<uint8>(acc);
    }
  }

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int depth = function_params->output_depth;
    const int stride_val = function_params->stride;
    const int four_over_stride = function_params->four_over_stride;

    const int workspace_width_micro_repeats =
        function_params->workspace_width_micro_repeats;
    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    constexpr int bias_increment = 4;
    TFLITE_DCHECK_EQ(function_params->bias_increment, bias_increment);

    TFLITE_DCHECK(depth_micro_repeats > 0);

    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;

    // Simulate NEON-register transposition of subset of filter.
    int8 filter_bank[3][2][4][4];  // Height 3, sub-block,  depth 4, width 4.
    // Simulate NEON-register input data concatenation + sub-selection.
    int8 sub_selected_input_data[3][4];  // Height 3, depth 4, width 4.
    uint8 output_values[4];              // Depth 4.

    // The outer 3 loops go through all the micro blocks in a macro block, and
    // separately treat the two sub-blocks within each micro block.
    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      memcpy(filter_bank[0][0][0],
             filter_workspace + j_depth * shuffled_filter_increment,
             shuffled_filter_increment);

      for (int s = 0; s < 2; ++s) {
        for (int k_height = 0; k_height < block_height; ++k_height) {
          const int8* scratch_data =
              scratch_block_data +
              workspace_height_stride * k_height * stride_val;
          uint8* output_data =
              output_block_data + output_height_stride * k_height + 8 * j_depth;

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            const int output_width = i_width == output_width_micro_repeats
                                         ? residual_width
                                         : four_over_stride;
            const bool no_right_block = i_width == output_width_micro_repeats &&
                                        output_width_overall_micro_repeats ==
                                            workspace_width_micro_repeats;
            TFLITE_DCHECK_LE(output_width * stride_val, 4);
            const int8* input_data = scratch_data + 4 * i_width;
            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              ConcatenateInputSubBlocks(x * stride_val, workspace_height_stride,
                                        no_right_block, input_data,
                                        sub_selected_input_data);
              Calculate3x3FilterOutput(
                  *function_params, s, sub_selected_input_data, filter_bank,
                  bias_data + (2 * j_depth + s) * bias_increment,
                  output_values);
              for (int d = 0; d < 4; ++d) {
                output_data[depth * (four_over_stride * i_width + x) + 4 * s +
                            d] = output_values[d];
              }
            }
          }
        }
      }
    }
  }
};

// Beginning of code section containing intermediate code transformation.
//
// This section is only compiled when kUseUnwound3x3DotProduct versions of
// templated functions are selected.
template <int32 stride, QuantizationType quantization_type>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseUnwound3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kNoMultiplication, stride> {
  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
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

    // Simulate NEON-register transposition of subset of filter.
    int8 filter_bank_a_0[4][4];  // Depth 4, width 4.
    int8 filter_bank_a_1[4][4];
    int8 filter_bank_a_2[4][4];
    int8 filter_bank_b_0[4][4];
    int8 filter_bank_b_1[4][4];
    int8 filter_bank_b_2[4][4];
    // Simulate NEON-register input data concatenation + sub-selection.
    // Also sub-block, height 3, depth 4, width 4.
    uint8 output_values[4];  // Sub-block, depth 4.
    // selected_data has format Depth 4, width 4.
    int8 left_bank_0[4][4];
    int8 left_bank_1[4][4];
    int8 left_bank_2[4][4];
    int8 right_bank_0[4][4];
    int8 right_bank_1[4][4];
    int8 right_bank_2[4][4];
    memset(right_bank_0[0], 0, 16);
    memset(right_bank_1[0], 0, 16);
    memset(right_bank_2[0], 0, 16);

    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;

    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      const int8* filter_block =
          filter_workspace + shuffled_filter_increment * j_depth;

      memcpy(filter_bank_a_0, filter_block, 16);
      memcpy(filter_bank_b_0, filter_block + 16, 16);
      memcpy(filter_bank_a_1, filter_block + 32, 16);
      memcpy(filter_bank_b_1, filter_block + 48, 16);
      memcpy(filter_bank_a_2, filter_block + 64, 16);
      memcpy(filter_bank_b_2, filter_block + 80, 16);

      for (int s = 0; s < 2; ++s) {
        // Work through one slice, by row, at a time.
        for (int k_height = 0; k_height < block_height; ++k_height) {
          const int8* scratch_data =
              scratch_block_data +
              workspace_height_stride * k_height * stride_val +
              depth_micro_stride * j_depth;
          uint8* output_data =
              output_block_data + output_height_stride * k_height + 8 * j_depth;
          const int8* input_data_0 = scratch_data + s * 2 * 8;

          // Load first sub-micro block of data into operational banks.
          memcpy(left_bank_0[0], input_data_0, 16);
          memcpy(left_bank_1[0], input_data_0 + workspace_height_stride, 16);
          memcpy(left_bank_2[0], input_data_0 + 2 * workspace_height_stride,
                 16);

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            const int output_width = i_width == output_width_micro_repeats
                                         ? residual_width
                                         : four_over_stride;
            TFLITE_DCHECK_LE(output_width * stride_val, 4);
            const int8* input_data =
                input_data_0 + width_micro_stride * i_width;
            const bool no_right_block = (output_width - 1) * stride_val < 2;

            // Load next sub-micro block of data.
            if (!no_right_block) {
              memcpy(right_bank_0[0], input_data + width_micro_stride, 16);
              memcpy(right_bank_1[0],
                     input_data + workspace_height_stride + width_micro_stride,
                     16);
              memcpy(
                  right_bank_2[0],
                  input_data + 2 * workspace_height_stride + width_micro_stride,
                  16);
            }

            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              // Operate on depth of 4 in batches.
              for (int d = 0; d < 4; ++d) {
                int32 acc = 0;
                for (int x = 0; x < 4; ++x) {
                  int32 input_val = left_bank_0[d][x];
                  int32 filter_val = filter_bank_a_0[d][x];
                  acc += filter_val * input_val;
                }
                for (int x = 0; x < 4; ++x) {
                  int32 input_val = left_bank_1[d][x];
                  int32 filter_val = filter_bank_a_1[d][x];
                  acc += filter_val * input_val;
                }
                for (int x = 0; x < 4; ++x) {
                  int32 input_val = left_bank_2[d][x];
                  int32 filter_val = filter_bank_a_2[d][x];
                  acc += filter_val * input_val;
                }
                acc += bias_data[d];
                acc = reference_ops::depthwise_conv::DepthwiseConvRound<
                    DepthwiseConvOutputRounding::kUpward>(
                    acc, output_multiplier, output_shift);
                acc += output_offset;
                acc = std::max(acc, output_activation_min);
                acc = std::min(acc, output_activation_max);
                output_values[d] = static_cast<uint8>(acc);
              }

              for (int d = 0; d < 4; ++d) {
                output_data[depth * (four_over_stride * i_width + x) + 4 * s +
                            d] = output_values[d];
              }

              // Simulate shifting instructions.
              if (stride_val == 1) {
                for (int depth_index = 0; depth_index < 4; ++depth_index) {
                  for (int z = 0; z < 3; ++z) {
                    left_bank_0[depth_index][z] =
                        left_bank_0[depth_index][z + 1];
                    left_bank_1[depth_index][z] =
                        left_bank_1[depth_index][z + 1];
                    left_bank_2[depth_index][z] =
                        left_bank_2[depth_index][z + 1];
                  }
                  left_bank_0[depth_index][3] = right_bank_0[depth_index][0];
                  left_bank_1[depth_index][3] = right_bank_1[depth_index][0];
                  left_bank_2[depth_index][3] = right_bank_2[depth_index][0];
                  for (int z = 0; z < 3; ++z) {
                    right_bank_0[depth_index][z] =
                        right_bank_0[depth_index][z + 1];
                    right_bank_1[depth_index][z] =
                        right_bank_1[depth_index][z + 1];
                    right_bank_2[depth_index][z] =
                        right_bank_2[depth_index][z + 1];
                  }
                }
              } else {
                for (int depth_index = 0; depth_index < 4; ++depth_index) {
                  for (int z = 0; z < 2; ++z) {
                    left_bank_0[depth_index][z] =
                        left_bank_0[depth_index][z + 2];
                    left_bank_1[depth_index][z] =
                        left_bank_1[depth_index][z + 2];
                    left_bank_2[depth_index][z] =
                        left_bank_2[depth_index][z + 2];
                  }
                  left_bank_0[depth_index][2] = right_bank_0[depth_index][0];
                  left_bank_1[depth_index][2] = right_bank_1[depth_index][0];
                  left_bank_2[depth_index][2] = right_bank_2[depth_index][0];
                  left_bank_0[depth_index][3] = right_bank_0[depth_index][1];
                  left_bank_1[depth_index][3] = right_bank_1[depth_index][1];
                  left_bank_2[depth_index][3] = right_bank_2[depth_index][1];
                  for (int z = 0; z < 2; ++z) {
                    right_bank_0[depth_index][z] =
                        right_bank_0[depth_index][z + 2];
                    right_bank_1[depth_index][z] =
                        right_bank_1[depth_index][z + 2];
                    right_bank_2[depth_index][z] =
                        right_bank_2[depth_index][z + 2];
                  }
                }
              }
            }
          }
        }
        bias_data += bias_increment;

        // Move filter for second sub-block into operational filter.
        for (int z = 0; z < 4; ++z) {
          for (int x = 0; x < 4; ++x) {
            filter_bank_a_0[z][x] = filter_bank_b_0[z][x];
            filter_bank_a_1[z][x] = filter_bank_b_1[z][x];
            filter_bank_a_2[z][x] = filter_bank_b_2[z][x];
          }
        }
      }
    }
  }
};

template <int32 stride, QuantizationType quantization_type>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseUnwound3x3DotProduct, quantization_type,
    DepthwiseConvDepthMultiplication::kUnitInputDepth, stride> {
  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int output_depth = function_params->output_depth;
    const int stride_val = function_params->stride;
    const int four_over_stride = function_params->four_over_stride;

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

    TFLITE_DCHECK(depth_micro_repeats > 0);

    TFLITE_DCHECK_EQ(bias_increment, 4);

    constexpr int shuffled_filter_increment = 2 * 3 * 4 * 4;

    // Simulate NEON-register transposition of subset of filter.
    int8 filter_bank_a_0[4][4];  // Depth 4, width 4.
    int8 filter_bank_a_1[4][4];
    int8 filter_bank_a_2[4][4];
    int8 filter_bank_b_0[4][4];
    int8 filter_bank_b_1[4][4];
    int8 filter_bank_b_2[4][4];
    // Simulate NEON-register input data concatenation + sub-selection.
    // Also sub-block, height 3, depth 4, width 4.

    int8 input_bank_0[8];
    int8 input_bank_1[8];
    int8 input_bank_2[8];

    TFLITE_DCHECK_GE(depth_micro_repeats, 1);

    uint8 output_values[2][4];  // Sub-block, depth 4.

    for (int j_depth = 0; j_depth < depth_micro_repeats; ++j_depth) {
      memcpy(filter_bank_a_0, filter_workspace, 16);
      memcpy(filter_bank_b_0, filter_workspace + 16, 16);
      memcpy(filter_bank_a_1, filter_workspace + 32, 16);
      memcpy(filter_bank_b_1, filter_workspace + 48, 16);
      memcpy(filter_bank_a_2, filter_workspace + 64, 16);
      memcpy(filter_bank_b_2, filter_workspace + 80, 16);

      // Work through one slice, by row, at a time.
      for (int k_height = 0; k_height < block_height; ++k_height) {
        const int8* scratch_data =
            scratch_block_data +
            workspace_height_stride * k_height * stride_val;
        uint8* output_data =
            output_block_data + output_height_stride * k_height + 8 * j_depth;

        memcpy(input_bank_0, scratch_data, 4);
        memcpy(input_bank_1, scratch_data + workspace_height_stride, 4);
        memcpy(input_bank_2, scratch_data + 2 * workspace_height_stride, 4);

        for (int i_width = 0; i_width < output_width_overall_micro_repeats;
             ++i_width) {
          const int output_width = i_width == output_width_micro_repeats
                                       ? residual_width
                                       : four_over_stride;

          TFLITE_DCHECK_LE(output_width * stride_val, 4);
          const int8* input_data = scratch_data + 4 * i_width;

          memcpy(input_bank_0 + 4, input_data + 4, 4);
          memcpy(input_bank_1 + 4, input_data + workspace_height_stride + 4, 4);
          memcpy(input_bank_2 + 4, input_data + 2 * workspace_height_stride + 4,
                 4);

          // Iterate over input width shifts within 4x4 blocks.
          for (int w = 0; w < output_width; ++w) {
            constexpr int offset =
                0;  // Shift input instead of offset in multiply-accumulate.

            {
              const int s = 0;
              for (int d = 0; d < 4; ++d) {
                int32 acc = bias_data[s * 4 + d];
                for (int x = 0; x < 4; ++x) {
                  int32 input_val_0 = input_bank_0[offset + x];
                  int32 filter_val_0 = filter_bank_a_0[d][x];
                  acc += filter_val_0 * input_val_0;
                  int32 input_val_1 = input_bank_1[offset + x];
                  int32 filter_val_1 = filter_bank_a_1[d][x];
                  acc += filter_val_1 * input_val_1;
                  int32 input_val_2 = input_bank_2[offset + x];
                  int32 filter_val_2 = filter_bank_a_2[d][x];
                  acc += filter_val_2 * input_val_2;
                }
                acc = reference_ops::depthwise_conv::DepthwiseConvRound<
                    DepthwiseConvOutputRounding::kUpward>(
                    acc, output_multiplier, output_shift);
                acc += output_offset;
                acc = std::max(acc, output_activation_min);
                acc = std::min(acc, output_activation_max);
                output_values[s][d] = static_cast<uint8>(acc);

                output_data[s * 4 + d] = output_values[s][d];
              }
            }
            {
              const int s = 1;
              for (int d = 0; d < 4; ++d) {
                int32 acc = bias_data[s * 4 + d];
                for (int x = 0; x < 4; ++x) {
                  int32 input_val_0 = input_bank_0[offset + x];
                  int32 filter_val_0 = filter_bank_b_0[d][x];
                  acc += filter_val_0 * input_val_0;
                  int32 input_val_1 = input_bank_1[offset + x];
                  int32 filter_val_1 = filter_bank_b_1[d][x];
                  acc += filter_val_1 * input_val_1;
                  int32 input_val_2 = input_bank_2[offset + x];
                  int32 filter_val_2 = filter_bank_b_2[d][x];
                  acc += filter_val_2 * input_val_2;
                }
                acc = reference_ops::depthwise_conv::DepthwiseConvRound<
                    DepthwiseConvOutputRounding::kUpward>(
                    acc, output_multiplier, output_shift);
                acc += output_offset;
                acc = std::max(acc, output_activation_min);
                acc = std::min(acc, output_activation_max);
                output_values[s][d] = static_cast<uint8>(acc);

                output_data[s * 4 + d] = output_values[s][d];
              }
            }

            // Simulate register shifts.
            for (int i = 0; i < (8 - stride_val); ++i) {
              input_bank_0[i] = input_bank_0[i + stride_val];
              input_bank_1[i] = input_bank_1[i + stride_val];
              input_bank_2[i] = input_bank_2[i + stride_val];
            }

            output_data += output_depth;
          }
        }
      }
      bias_data += 2 * bias_increment;
      filter_workspace += shuffled_filter_increment;
    }
  }
};
// The preceding section is only compiled when kUseUnwound3x3DotProduct versions
// of templated functions are selected.
//
// End of code section containing intermediate code transformation.

#ifdef USE_NEON
template <>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
    QuantizationType::kNonPerChannelUint8,
    DepthwiseConvDepthMultiplication::kNoMultiplication,
    /*stride=*/1> {
  static inline void KernelMacroBlockIntrinsics(
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
    constexpr int kBiasIncrement = 4;

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
          bias_data += kBiasIncrement;

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
              // Only needed for sanitizer checks.
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
        const int8* input_data_base = input_data_depthwise;
        uint8* output_data_base = output_data_depthwise;

        const int32x4_t adjusted_bias_data_a = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;
        const int32x4_t adjusted_bias_data_b = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;

        for (int k_height = 0; k_height < block_height; ++k_height) {
          const int8* next_input_data = input_data_base;
          uint8* output_data = output_data_base;

          // Load first sub-micro block of data into operational banks.
          int8x16_t left_bank_0_reg_a = vld1q_s8(next_input_data);
          int8x16_t left_bank_1_reg_a =
              vld1q_s8(next_input_data + workspace_height_stride);
          int8x16_t left_bank_2_reg_a =
              vld1q_s8(next_input_data + 2 * workspace_height_stride);
          int8x16_t left_bank_0_reg_b = vld1q_s8(next_input_data + 16);
          int8x16_t left_bank_1_reg_b =
              vld1q_s8(next_input_data + workspace_height_stride + 16);
          int8x16_t left_bank_2_reg_b =
              vld1q_s8(next_input_data + 2 * workspace_height_stride + 16);

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            next_input_data += width_micro_stride;
            const int output_width =
                i_width == output_width_micro_repeats ? residual_width : 4;

            int8x16_t right_bank_0_reg_a;
            int8x16_t right_bank_1_reg_a;
            int8x16_t right_bank_2_reg_a;
            int8x16_t right_bank_0_reg_b;
            int8x16_t right_bank_1_reg_b;
            int8x16_t right_bank_2_reg_b;
            // Logic: (output_width - 1) * stride_val < 2.
            const bool no_right_block = output_width < 3;

            // Load next sub-micro block of data.
            if (no_right_block) {
              // Only needed for sanitizer checks.
              right_bank_0_reg_a = vdupq_n_s8(0);
              right_bank_1_reg_a = vdupq_n_s8(0);
              right_bank_2_reg_a = vdupq_n_s8(0);
              right_bank_0_reg_b = vdupq_n_s8(0);
              right_bank_1_reg_b = vdupq_n_s8(0);
              right_bank_2_reg_b = vdupq_n_s8(0);
            } else {
              right_bank_0_reg_a = vld1q_s8(next_input_data);
              right_bank_1_reg_a =
                  vld1q_s8(next_input_data + workspace_height_stride);
              right_bank_2_reg_a =
                  vld1q_s8(next_input_data + 2 * workspace_height_stride);
              right_bank_0_reg_b = vld1q_s8(next_input_data + 16);
              right_bank_1_reg_b =
                  vld1q_s8(next_input_data + workspace_height_stride + 16);
              right_bank_2_reg_b =
                  vld1q_s8(next_input_data + 2 * workspace_height_stride + 16);
            }

            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              int32x4_t acc_a = adjusted_bias_data_a;
              int32x4_t acc_b = adjusted_bias_data_b;
              acc_a = vdotq_s32(acc_a, filter_reg_0_a, left_bank_0_reg_a);
              acc_a = vdotq_s32(acc_a, filter_reg_1_a, left_bank_1_reg_a);
              acc_a = vdotq_s32(acc_a, filter_reg_2_a, left_bank_2_reg_a);
              acc_b = vdotq_s32(acc_b, filter_reg_0_b, left_bank_0_reg_b);
              acc_b = vdotq_s32(acc_b, filter_reg_1_b, left_bank_1_reg_b);
              acc_b = vdotq_s32(acc_b, filter_reg_2_b, left_bank_2_reg_b);

              // Fixed-point multiplication.
              acc_a = vqrdmulhq_n_s32(acc_a, output_multiplier);
              acc_b = vqrdmulhq_n_s32(acc_b, output_multiplier);
              acc_a = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc_a, -output_shift);
              acc_b = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc_b, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_0 =
                  vcombine_s16(vqmovn_s32(acc_a), vqmovn_s32(acc_b));
              acc_s16_0_0 = vqaddq_s16(acc_s16_0_0, output_offset_vec);
              // Apply the activation function.
              uint8x8_t acc_u8_0_0 = vqmovun_s16(acc_s16_0_0);
              acc_u8_0_0 =
                  vmax_u8(acc_u8_0_0, vget_low_u8(output_activation_min_vec));
              acc_u8_0_0 =
                  vmin_u8(acc_u8_0_0, vget_low_u8(output_activation_max_vec));

              vst1_u8(output_data, acc_u8_0_0);

              biregister_rotate_8(&left_bank_0_reg_a, &right_bank_0_reg_a);
              biregister_rotate_8(&left_bank_1_reg_a, &right_bank_1_reg_a);
              biregister_rotate_8(&left_bank_2_reg_a, &right_bank_2_reg_a);
              biregister_rotate_8(&left_bank_0_reg_b, &right_bank_0_reg_b);
              biregister_rotate_8(&left_bank_1_reg_b, &right_bank_1_reg_b);
              biregister_rotate_8(&left_bank_2_reg_b, &right_bank_2_reg_b);

              output_data += depth;
            }
          }
          input_data_base += workspace_height_stride;
          output_data_base += output_height_stride;
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
    KernelMacroBlockIntrinsics(scratch_block_data, filter_workspace, bias_data,
                               output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
    QuantizationType::kNonPerChannelUint8,
    DepthwiseConvDepthMultiplication::kNoMultiplication,
    /*stride=*/2> {
  static inline void KernelMacroBlockIntrinsics(
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
    constexpr int kStrideVal = 2;
    constexpr int kFourOverStride = 2;
    TFLITE_DCHECK_EQ(function_params->stride, kStrideVal);
    TFLITE_DCHECK_EQ(function_params->four_over_stride, kFourOverStride);

    const int workspace_width_micro_repeats =
        function_params->workspace_width_micro_repeats;
    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    constexpr int kBiasIncrement = 4;

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
            const int output_width = kFourOverStride;
            TFLITE_DCHECK_LE(output_width * kStrideVal, 4);
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
            TFLITE_DCHECK_NE(residual_width, kFourOverStride);

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
          bias_data += kBiasIncrement;
        }
      } else {
        // block_height == 1.
        int8x16_t filter_reg_0_a;
        int8x16_t filter_reg_1_a;
        int8x16_t filter_reg_2_a;
        int8x16_t filter_reg_0_b;
        int8x16_t filter_reg_1_b;
        int8x16_t filter_reg_2_b;

        filter_reg_0_a = vld1q_s8(filter_block);
        filter_reg_1_a = vld1q_s8(filter_block + 32);
        filter_reg_2_a = vld1q_s8(filter_block + 64);
        filter_reg_0_b = vld1q_s8(filter_block + 16);
        filter_reg_1_b = vld1q_s8(filter_block + 16 + 32);
        filter_reg_2_b = vld1q_s8(filter_block + 16 + 64);

        const int8* scratch_data =
            scratch_block_data + depth_micro_stride * j_depth;
        uint8* output_data = output_block_data + 8 * j_depth;
        const int8* input_data_0 = scratch_data;

        const int32x4_t adjusted_bias_data_a = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;
        const int32x4_t adjusted_bias_data_b = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;

        // Load first sub-micro block of data into operational banks.
        int8x16_t left_bank_0_reg_a = vld1q_s8(input_data_0);
        int8x16_t left_bank_1_reg_a =
            vld1q_s8(input_data_0 + workspace_height_stride);
        int8x16_t left_bank_2_reg_a =
            vld1q_s8(input_data_0 + 2 * workspace_height_stride);
        int8x16_t left_bank_0_reg_b = vld1q_s8(input_data_0 + 16);
        int8x16_t left_bank_1_reg_b =
            vld1q_s8(input_data_0 + workspace_height_stride + 16);
        int8x16_t left_bank_2_reg_b =
            vld1q_s8(input_data_0 + 2 * workspace_height_stride + 16);

        int8x16_t right_bank_0_reg_a;
        int8x16_t right_bank_1_reg_a;
        int8x16_t right_bank_2_reg_a;
        int8x16_t right_bank_0_reg_b;
        int8x16_t right_bank_1_reg_b;
        int8x16_t right_bank_2_reg_b;

        int32x4_t acc0_a;
        int32x4_t acc0_b;

        for (int i_width = 0; i_width < output_width_overall_micro_repeats;
             ++i_width) {
          const int output_width = i_width == output_width_micro_repeats
                                       ? residual_width
                                       : kFourOverStride;
          TFLITE_DCHECK_LE(output_width * kStrideVal, 4);
          const int8* input_data = input_data_0 + width_micro_stride * i_width;
          const bool no_right_block = i_width == output_width_micro_repeats &&
                                      output_width_overall_micro_repeats ==
                                          workspace_width_micro_repeats;

          if (!no_right_block) {
            // Load next sub-micro block of data.
            right_bank_0_reg_a = vld1q_s8(input_data + width_micro_stride);
            right_bank_1_reg_a = vld1q_s8(input_data + width_micro_stride +
                                          workspace_height_stride);
            right_bank_2_reg_a = vld1q_s8(input_data + width_micro_stride +
                                          2 * workspace_height_stride);
            right_bank_0_reg_b = vld1q_s8(input_data + width_micro_stride + 16);
            right_bank_1_reg_b = vld1q_s8(input_data + width_micro_stride +
                                          workspace_height_stride + 16);
            right_bank_2_reg_b = vld1q_s8(input_data + width_micro_stride +
                                          2 * workspace_height_stride + 16);
          }

          uint8* output_data_base = output_data + depth * 2 * i_width;

          // Iterate over input width shifts within 4x4 blocks.
          {
            acc0_a = adjusted_bias_data_a;
            acc0_b = adjusted_bias_data_b;

            acc0_a = vdotq_s32(acc0_a, filter_reg_0_a, left_bank_0_reg_a);
            acc0_a = vdotq_s32(acc0_a, filter_reg_1_a, left_bank_1_reg_a);
            acc0_a = vdotq_s32(acc0_a, filter_reg_2_a, left_bank_2_reg_a);
            acc0_b = vdotq_s32(acc0_b, filter_reg_0_b, left_bank_0_reg_b);
            acc0_b = vdotq_s32(acc0_b, filter_reg_1_b, left_bank_1_reg_b);
            acc0_b = vdotq_s32(acc0_b, filter_reg_2_b, left_bank_2_reg_b);

            // Fixed-point multiplication.
            acc0_a = vqrdmulhq_n_s32(acc0_a, output_multiplier);
            acc0_b = vqrdmulhq_n_s32(acc0_b, output_multiplier);
            acc0_a = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0_a, -output_shift);
            acc0_b = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0_b, -output_shift);
            // Add the output offset.
            int16x8_t acc_s16_0_1 =
                vcombine_s16(vqmovn_s32(acc0_a), vqmovn_s32(acc0_b));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            uint8x8_t acc_u8 = vqmovun_s16(acc_s16_0_1);
            acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
            acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

            vst1_u8(output_data_base, acc_u8);

            left_bank_0_reg_a = vrev32q_u16(left_bank_0_reg_a);
            left_bank_1_reg_a = vrev32q_u16(left_bank_1_reg_a);
            left_bank_2_reg_a = vrev32q_u16(left_bank_2_reg_a);
            left_bank_0_reg_b = vrev32q_u16(left_bank_0_reg_b);
            left_bank_1_reg_b = vrev32q_u16(left_bank_1_reg_b);
            left_bank_2_reg_b = vrev32q_u16(left_bank_2_reg_b);
            vtrn1_s8x2_in_place(&left_bank_0_reg_a, &right_bank_0_reg_a);
            vtrn1_s8x2_in_place(&left_bank_1_reg_a, &right_bank_1_reg_a);
            vtrn1_s8x2_in_place(&left_bank_2_reg_a, &right_bank_2_reg_a);
            vtrn1_s8x2_in_place(&left_bank_0_reg_b, &right_bank_0_reg_b);
            vtrn1_s8x2_in_place(&left_bank_1_reg_b, &right_bank_1_reg_b);
            vtrn1_s8x2_in_place(&left_bank_2_reg_b, &right_bank_2_reg_b);
          }

          if (output_width > 1) {
            acc0_a = adjusted_bias_data_a;
            acc0_b = adjusted_bias_data_b;

            acc0_a = vdotq_s32(acc0_a, filter_reg_0_a, left_bank_0_reg_a);
            acc0_a = vdotq_s32(acc0_a, filter_reg_1_a, left_bank_1_reg_a);
            acc0_a = vdotq_s32(acc0_a, filter_reg_2_a, left_bank_2_reg_a);
            acc0_b = vdotq_s32(acc0_b, filter_reg_0_b, left_bank_0_reg_b);
            acc0_b = vdotq_s32(acc0_b, filter_reg_1_b, left_bank_1_reg_b);
            acc0_b = vdotq_s32(acc0_b, filter_reg_2_b, left_bank_2_reg_b);

            // Fixed-point multiplication.
            acc0_a = vqrdmulhq_n_s32(acc0_a, output_multiplier);
            acc0_b = vqrdmulhq_n_s32(acc0_b, output_multiplier);
            acc0_a = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0_a, -output_shift);
            acc0_b = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                acc0_b, -output_shift);
            // Add the output offset.
            int16x8_t acc_s16_0_1 =
                vcombine_s16(vqmovn_s32(acc0_a), vqmovn_s32(acc0_b));
            acc_s16_0_1 = vqaddq_s16(acc_s16_0_1, output_offset_vec);
            // Apply the activation function.
            uint8x8_t acc_u8 = vqmovun_s16(acc_s16_0_1);
            acc_u8 = vmax_u8(acc_u8, output_activation_min_vec);
            acc_u8 = vmin_u8(acc_u8, output_activation_max_vec);

            vst1_u8(output_data_base + depth, acc_u8);

            left_bank_0_reg_a = right_bank_0_reg_a;
            left_bank_1_reg_a = right_bank_1_reg_a;
            left_bank_2_reg_a = right_bank_2_reg_a;
            left_bank_0_reg_b = right_bank_0_reg_b;
            left_bank_1_reg_b = right_bank_1_reg_b;
            left_bank_2_reg_b = right_bank_2_reg_b;
          }
        }
      }
    }
  }  // NOLINT(readability/fn_size) Manually unrolled.

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockIntrinsics(scratch_block_data, filter_workspace, bias_data,
                               output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
    QuantizationType::kNonPerChannelUint8,
    DepthwiseConvDepthMultiplication::kUnitInputDepth,
    /*stride=*/1> {
  static inline void KernelMacroBlockIntrinsics(
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
    constexpr int kBiasIncrement = 4;

    TFLITE_DCHECK(depth_micro_repeats > 0);

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
          bias_data += kBiasIncrement;

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
        uint8* output_data_base = output_data_depthwise;

        const int32x4_t adjusted_bias_data_a = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;
        const int32x4_t adjusted_bias_data_b = vld1q_s32(bias_data);
        bias_data += kBiasIncrement;

        for (int k_height = 0; k_height < block_height; ++k_height) {
          const int8* next_input_data =
              scratch_block_data + k_height * workspace_height_stride;
          uint8* output_data = output_data_base;

          int8x16_t input_bank_p_reg;  //  left 0, right 0, left 1, right 1.
          int8x16_t input_bank_q_reg;  //  left 2, right 2, left 3, right 3.

          // Load first sub-micro block of data into operational banks.
          input_bank_p_reg =
              vld1q_dup_s8x4(next_input_data);  // Load lane 0, avoiding
                                                // uninitialized variable.
          input_bank_p_reg = vld1q_lane_8x4(
              next_input_data + workspace_height_stride, input_bank_p_reg, 2);
          input_bank_q_reg = vld1q_dup_s8x4(
              next_input_data +
              2 * workspace_height_stride);  // Load lane 0, avoiding
                                             // uninitialized variable.

          for (int i_width = 0; i_width < output_width_overall_micro_repeats;
               ++i_width) {
            next_input_data += 4;
            const int output_width =
                i_width == output_width_micro_repeats ? residual_width : 4;

            // Load next sub-micro block of data.
            input_bank_p_reg =
                vld1q_lane_8x4(next_input_data, input_bank_p_reg, 1);
            input_bank_p_reg = vld1q_lane_8x4(
                next_input_data + workspace_height_stride, input_bank_p_reg, 3);
            input_bank_q_reg =
                vld1q_lane_8x4(next_input_data + 2 * workspace_height_stride,
                               input_bank_q_reg, 1);
            // Iterate over input width shifts within 4x4 blocks.
            for (int x = 0; x < output_width; ++x) {
              int32x4_t acc_a = adjusted_bias_data_a;
              int32x4_t acc_b = adjusted_bias_data_b;
              acc_a = vdotq_four_lane_s32(acc_a, filter_reg_0_a,
                                          input_bank_p_reg, 0);
              acc_a = vdotq_four_lane_s32(acc_a, filter_reg_1_a,
                                          input_bank_p_reg, 2);
              acc_a = vdotq_four_lane_s32(acc_a, filter_reg_2_a,
                                          input_bank_q_reg, 0);
              acc_b = vdotq_four_lane_s32(acc_b, filter_reg_0_b,
                                          input_bank_p_reg, 0);
              acc_b = vdotq_four_lane_s32(acc_b, filter_reg_1_b,
                                          input_bank_p_reg, 2);
              acc_b = vdotq_four_lane_s32(acc_b, filter_reg_2_b,
                                          input_bank_q_reg, 0);

              // Fixed-point multiplication.
              acc_a = vqrdmulhq_n_s32(acc_a, output_multiplier);
              acc_b = vqrdmulhq_n_s32(acc_b, output_multiplier);
              acc_a = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc_a, -output_shift);
              acc_b = DivideByPOT<DepthwiseConvOutputRounding::kUpward>::Run(
                  acc_b, -output_shift);
              // Add the output offset.
              int16x8_t acc_s16_0_0 =
                  vcombine_s16(vqmovn_s32(acc_a), vqmovn_s32(acc_b));
              acc_s16_0_0 = vqaddq_s16(acc_s16_0_0, output_offset_vec);
              // Apply the activation function.
              uint8x8_t acc_u8_0_0 = vqmovun_s16(acc_s16_0_0);
              acc_u8_0_0 =
                  vmax_u8(acc_u8_0_0, vget_low_u8(output_activation_min_vec));
              acc_u8_0_0 =
                  vmin_u8(acc_u8_0_0, vget_low_u8(output_activation_max_vec));

              vst1_u8(output_data, acc_u8_0_0);

              input_bank_p_reg = vshrq_n_u64(input_bank_p_reg, 8);
              input_bank_q_reg = vshrq_n_u64(input_bank_q_reg, 8);

              output_data += output_depth;
            }
          }
          output_data_base += output_height_stride;
        }
      }
      output_data_depthwise += 8;
    }
  }  // NOLINT(readability/fn_size) Manually unrolled.

  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    KernelMacroBlockIntrinsics(scratch_block_data, filter_workspace, bias_data,
                               output_block_data, function_params);
  }
};

template <>
struct KernelMacroBlock<
    DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
    QuantizationType::kNonPerChannelUint8,
    DepthwiseConvDepthMultiplication::kUnitInputDepth,
    /*stride=*/2> {
  static inline void KernelMacroBlockIntrinsics(
      const int8* scratch_block_data, const int8* filter_workspace,
      const int32* bias_data, uint8* output_block_data,
      const DepthwiseConvDotProdParams* function_params) {
    const int workspace_height_stride =
        function_params->workspace_height_stride;
    const int output_width_micro_repeats =
        function_params->output_width_micro_repeats;
    const int depth_micro_repeats = function_params->depth_micro_repeats;
    const int output_depth = function_params->output_depth;
    constexpr int kStrideVal = 2;
    TFLITE_DCHECK_EQ(function_params->stride, kStrideVal);

    const int output_width_overall_micro_repeats =
        function_params->output_width_overall_micro_repeats;
    const int block_height = function_params->outbound_block_height;
    const int residual_width = function_params->output_residual_width;
    const int output_height_stride = function_params->output_height_stride;
    constexpr int kBiasIncrement = 4;

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

      const int32x4_t adjusted_bias_data_s_0 = vld1q_s32(bias_data);
      bias_data += kBiasIncrement;
      const int32x4_t adjusted_bias_data_s_1 = vld1q_s32(bias_data);
      bias_data += kBiasIncrement;

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
          TFLITE_DCHECK_LE(output_width * kStrideVal, 4);
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
    KernelMacroBlockIntrinsics(scratch_block_data, filter_workspace, bias_data,
                               output_block_data, function_params);
  }
};

#undef vst1_lane_8x4
#undef vst1q_lane_8x4
#undef vld1q_lane_s8x8
#undef vld1_lane_8x4
#undef vld1q_lane_8x4
#undef vld1q_dup_s8x4

#endif  //  USE_NEON

}  // namespace depthwise_conv
}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_

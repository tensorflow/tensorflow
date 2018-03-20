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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

#ifdef __aarch64__

inline void preload_l1_keep(const uint8* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
}

// Implementation of quantized DepthwiseConv for 3x3 filters.

// Below are helper structs to remove the use of arrays.
// There is an llvm bug that causes significant slowdown when using arrays for
// NEON intrinsics vector data types.
// See: https://bugs.llvm.org/show_bug.cgi?id=34945

struct Int32x16 {
  int32x4_t v0, v1, v2, v3;
};

struct Int16x16 {
  int16x8_t low, high;
};

struct Int16x16x3 {
  Int16x16 v0, v1, v2;
};

struct Filter3x3x16 {
  Int16x16x3 r0, r1, r2;
};

// Loads 3x3 filter of depth 16 and adds filter offsets.
inline Filter3x3x16 LoadFilterDepth16(const uint8* filter_ptr,
                                      int32 filter_offset, int output_depth) {
  Filter3x3x16 filter;

  uint8x8_t temp_u8_0, temp_u8_1, temp_u8_2, temp_u8_3, temp_u8_4, temp_u8_5,
      temp_u8_6, temp_u8_7, temp_u8_8, temp_u8_9, temp_u8_10, temp_u8_11,
      temp_u8_12, temp_u8_13, temp_u8_14, temp_u8_15, temp_u8_16, temp_u8_17;
  int16x8_t filter_offset_vec = vdupq_n_s16(filter_offset);

  temp_u8_0 = vld1_u8(filter_ptr + 0 * output_depth);
  temp_u8_1 = vld1_u8(filter_ptr + 0 * output_depth + 8);
  temp_u8_2 = vld1_u8(filter_ptr + 1 * output_depth);
  temp_u8_3 = vld1_u8(filter_ptr + 1 * output_depth + 8);
  temp_u8_4 = vld1_u8(filter_ptr + 2 * output_depth);
  temp_u8_5 = vld1_u8(filter_ptr + 2 * output_depth + 8);

  temp_u8_6 = vld1_u8(filter_ptr + 3 * output_depth);
  temp_u8_7 = vld1_u8(filter_ptr + 3 * output_depth + 8);
  temp_u8_8 = vld1_u8(filter_ptr + 4 * output_depth);
  temp_u8_9 = vld1_u8(filter_ptr + 4 * output_depth + 8);
  temp_u8_10 = vld1_u8(filter_ptr + 5 * output_depth);
  temp_u8_11 = vld1_u8(filter_ptr + 5 * output_depth + 8);

  temp_u8_12 = vld1_u8(filter_ptr + 6 * output_depth);
  temp_u8_13 = vld1_u8(filter_ptr + 6 * output_depth + 8);
  temp_u8_14 = vld1_u8(filter_ptr + 7 * output_depth);
  temp_u8_15 = vld1_u8(filter_ptr + 7 * output_depth + 8);
  temp_u8_16 = vld1_u8(filter_ptr + 8 * output_depth);
  temp_u8_17 = vld1_u8(filter_ptr + 8 * output_depth + 8);

  filter.r0.v0.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_0));
  filter.r0.v0.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_1));
  filter.r0.v1.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_2));
  filter.r0.v1.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_3));
  filter.r0.v2.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_4));
  filter.r0.v2.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_5));

  filter.r1.v0.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_6));
  filter.r1.v0.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_7));
  filter.r1.v1.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_8));
  filter.r1.v1.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_9));
  filter.r1.v2.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_10));
  filter.r1.v2.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_11));

  filter.r2.v0.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_12));
  filter.r2.v0.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_13));
  filter.r2.v1.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_14));
  filter.r2.v1.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_15));
  filter.r2.v2.low = vreinterpretq_s16_u16(vmovl_u8(temp_u8_16));
  filter.r2.v2.high = vreinterpretq_s16_u16(vmovl_u8(temp_u8_17));

  filter.r0.v0.low = vaddq_s16(filter.r0.v0.low, filter_offset_vec);
  filter.r0.v0.high = vaddq_s16(filter.r0.v0.high, filter_offset_vec);
  filter.r0.v1.low = vaddq_s16(filter.r0.v1.low, filter_offset_vec);
  filter.r0.v1.high = vaddq_s16(filter.r0.v1.high, filter_offset_vec);
  filter.r0.v2.low = vaddq_s16(filter.r0.v2.low, filter_offset_vec);
  filter.r0.v2.high = vaddq_s16(filter.r0.v2.high, filter_offset_vec);

  filter.r1.v0.low = vaddq_s16(filter.r1.v0.low, filter_offset_vec);
  filter.r1.v0.high = vaddq_s16(filter.r1.v0.high, filter_offset_vec);
  filter.r1.v1.low = vaddq_s16(filter.r1.v1.low, filter_offset_vec);
  filter.r1.v1.high = vaddq_s16(filter.r1.v1.high, filter_offset_vec);
  filter.r1.v2.low = vaddq_s16(filter.r1.v2.low, filter_offset_vec);
  filter.r1.v2.high = vaddq_s16(filter.r1.v2.high, filter_offset_vec);

  filter.r2.v0.low = vaddq_s16(filter.r2.v0.low, filter_offset_vec);
  filter.r2.v0.high = vaddq_s16(filter.r2.v0.high, filter_offset_vec);
  filter.r2.v1.low = vaddq_s16(filter.r2.v1.low, filter_offset_vec);
  filter.r2.v1.high = vaddq_s16(filter.r2.v1.high, filter_offset_vec);
  filter.r2.v2.low = vaddq_s16(filter.r2.v2.low, filter_offset_vec);
  filter.r2.v2.high = vaddq_s16(filter.r2.v2.high, filter_offset_vec);

  return filter;
}

// Loads 3 input cells of depth 16 and adds input offsets.
inline Int16x16x3 LoadInputRowDepth16(const uint8* ptr, int input_depth,
                                      int32 input_offset,
                                      Int16x16x3 input_row) {
  uint8x8_t temp_0, temp_1;
  int16x8_t offset_vec = vdupq_n_s16(input_offset);

  temp_0 = vld1_u8(ptr + 0 * input_depth);
  temp_1 = vld1_u8(ptr + 0 * input_depth + 8);
  input_row.v0.low = vreinterpretq_s16_u16(vmovl_u8(temp_0));
  input_row.v0.high = vreinterpretq_s16_u16(vmovl_u8(temp_1));
  input_row.v0.low = vaddq_s16(input_row.v0.low, offset_vec);
  input_row.v0.high = vaddq_s16(input_row.v0.high, offset_vec);

  temp_0 = vld1_u8(ptr + 1 * input_depth);
  temp_1 = vld1_u8(ptr + 1 * input_depth + 8);
  input_row.v1.low = vreinterpretq_s16_u16(vmovl_u8(temp_0));
  input_row.v1.high = vreinterpretq_s16_u16(vmovl_u8(temp_1));
  input_row.v1.low = vaddq_s16(input_row.v1.low, offset_vec);
  input_row.v1.high = vaddq_s16(input_row.v1.high, offset_vec);

  temp_0 = vld1_u8(ptr + 2 * input_depth);
  temp_1 = vld1_u8(ptr + 2 * input_depth + 8);
  input_row.v2.low = vreinterpretq_s16_u16(vmovl_u8(temp_0));
  input_row.v2.high = vreinterpretq_s16_u16(vmovl_u8(temp_1));
  input_row.v2.low = vaddq_s16(input_row.v2.low, offset_vec);
  input_row.v2.high = vaddq_s16(input_row.v2.high, offset_vec);

  return input_row;
}

// Performs multiply accumulate on 3 inputs of depth 16.
inline Int32x16 MultiplyAccumulateRowDepth16(Int32x16 output,
                                             const Int16x16x3& filter_row,
                                             const Int16x16x3& input_row) {
  output.v0 = vmlal_s16(output.v0, vget_low_s16(filter_row.v0.low),
                        vget_low_s16(input_row.v0.low));
  output.v1 = vmlal_s16(output.v1, vget_high_s16(filter_row.v0.low),
                        vget_high_s16(input_row.v0.low));
  output.v2 = vmlal_s16(output.v2, vget_low_s16(filter_row.v0.high),
                        vget_low_s16(input_row.v0.high));
  output.v3 = vmlal_s16(output.v3, vget_high_s16(filter_row.v0.high),
                        vget_high_s16(input_row.v0.high));

  output.v0 = vmlal_s16(output.v0, vget_low_s16(filter_row.v1.low),
                        vget_low_s16(input_row.v1.low));
  output.v1 = vmlal_s16(output.v1, vget_high_s16(filter_row.v1.low),
                        vget_high_s16(input_row.v1.low));
  output.v2 = vmlal_s16(output.v2, vget_low_s16(filter_row.v1.high),
                        vget_low_s16(input_row.v1.high));
  output.v3 = vmlal_s16(output.v3, vget_high_s16(filter_row.v1.high),
                        vget_high_s16(input_row.v1.high));

  output.v0 = vmlal_s16(output.v0, vget_low_s16(filter_row.v2.low),
                        vget_low_s16(input_row.v2.low));
  output.v1 = vmlal_s16(output.v1, vget_high_s16(filter_row.v2.low),
                        vget_high_s16(input_row.v2.low));
  output.v2 = vmlal_s16(output.v2, vget_low_s16(filter_row.v2.high),
                        vget_low_s16(input_row.v2.high));
  output.v3 = vmlal_s16(output.v3, vget_high_s16(filter_row.v2.high),
                        vget_high_s16(input_row.v2.high));

  return output;
}

// Applies activation, offset and downquantize on a set of accumulator
// registers of depth 16. Stores results to output.
inline void DownquantizeAndStoreDepth16(Int32x16 acc, int32 output_multiplier,
                                        int output_shift,
                                        int32x4_t output_offset_vec,
                                        int32x4_t output_activation_min_vec,
                                        int32x4_t output_activation_max_vec,
                                        uint8* output_ptr) {
  // Fixed-point multiplication.
  acc.v0 = vqrdmulhq_n_s32(acc.v0, output_multiplier);
  acc.v1 = vqrdmulhq_n_s32(acc.v1, output_multiplier);
  acc.v2 = vqrdmulhq_n_s32(acc.v2, output_multiplier);
  acc.v3 = vqrdmulhq_n_s32(acc.v3, output_multiplier);

  using gemmlowp::RoundingDivideByPOT;
  acc.v0 = RoundingDivideByPOT(acc.v0, output_shift);
  acc.v1 = RoundingDivideByPOT(acc.v1, output_shift);
  acc.v2 = RoundingDivideByPOT(acc.v2, output_shift);
  acc.v3 = RoundingDivideByPOT(acc.v3, output_shift);

  // Add the output offset.
  acc.v0 = vaddq_s32(acc.v0, output_offset_vec);
  acc.v1 = vaddq_s32(acc.v1, output_offset_vec);
  acc.v2 = vaddq_s32(acc.v2, output_offset_vec);
  acc.v3 = vaddq_s32(acc.v3, output_offset_vec);

  // Apply the activation function.
  acc.v0 = vmaxq_s32(acc.v0, output_activation_min_vec);
  acc.v1 = vmaxq_s32(acc.v1, output_activation_min_vec);
  acc.v2 = vmaxq_s32(acc.v2, output_activation_min_vec);
  acc.v3 = vmaxq_s32(acc.v3, output_activation_min_vec);

  acc.v0 = vminq_s32(acc.v0, output_activation_max_vec);
  acc.v1 = vminq_s32(acc.v1, output_activation_max_vec);
  acc.v2 = vminq_s32(acc.v2, output_activation_max_vec);
  acc.v3 = vminq_s32(acc.v3, output_activation_max_vec);

  // Saturating cast to uint8 and store to destination.
  int16x4_t acc_tlla_s16 = vqmovn_s32(acc.v0);
  int16x4_t acc_tllb_s16 = vqmovn_s32(acc.v1);
  int16x4_t acc_tlha_s16 = vqmovn_s32(acc.v2);
  int16x4_t acc_tlhb_s16 = vqmovn_s32(acc.v3);

  int16x8_t res_s16_0 = vcombine_s16(acc_tlla_s16, acc_tllb_s16);
  int16x8_t res_s16_1 = vcombine_s16(acc_tlha_s16, acc_tlhb_s16);
  uint8x8_t res_u8_0 = vqmovun_s16(res_s16_0);
  uint8x8_t res_u8_1 = vqmovun_s16(res_s16_1);
  vst1q_u8(output_ptr, vcombine_u8(res_u8_0, res_u8_1));
}

// A kernel that is optimized on the number of output cells in the x and y
// direction, and the stride. Assumes 3x3 filters of 16 depth.
template <int kFixedOutputX, int kFixedOutputY, int kFixedStride = 1>
struct ConvKernel3x3FilterDepth16 {};

template <>
struct ConvKernel3x3FilterDepth16<1, 2, 1> {
  static void Run(const Filter3x3x16& filter, const uint8* input_ptr,
                  int input_depth, int32 input_offset, int input_row_width,
                  const int32* bias_ptr, int32 output_offset,
                  int32 output_multiplier, int output_shift,
                  int32 output_activation_min, int32 output_activation_max,
                  uint8* output_ptr, int output_depth, int output_width) {
    // 16 depth accumulators for the 2 outputs.
    Int32x16 acc0, acc1;

    // Accumulators for top filter.
    acc0.v0 = vld1q_s32(bias_ptr);
    acc0.v1 = vld1q_s32(bias_ptr + 4);
    acc0.v2 = vld1q_s32(bias_ptr + 8);
    acc0.v3 = vld1q_s32(bias_ptr + 12);
    // Accumulators for bottom filter.
    acc1.v0 = vld1q_s32(bias_ptr);
    acc1.v1 = vld1q_s32(bias_ptr + 4);
    acc1.v2 = vld1q_s32(bias_ptr + 8);
    acc1.v3 = vld1q_s32(bias_ptr + 12);

    // Main multiply accumulate work.
    {
      // Load inputs for one filter row at a time.
      Int16x16x3 input;

      // Do first row of top filter.
      input = LoadInputRowDepth16(input_ptr, input_depth, input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r0, input);

      // Do second row of top filter.
      input = LoadInputRowDepth16(input_ptr + input_row_width, input_depth,
                                  input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r1, input);

      // The inputs to second row of the top filter are also the inputs to the
      // first row of the bottom filter.
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r0, input);

      // Do third row of top filter.
      input = LoadInputRowDepth16(input_ptr + 2 * input_row_width, input_depth,
                                  input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r2, input);

      // The inputs to third row of the top filter are also the inputs to the
      // second row of the bottom filter.
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r1, input);

      // Do third row of bottom filter.
      input = LoadInputRowDepth16(input_ptr + 3 * input_row_width, input_depth,
                                  input_offset, input);
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r2, input);
    }

    // Apply activation, downquantize and store.
    int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    int32x4_t output_activation_min_vec = vdupq_n_s32(output_activation_min);
    int32x4_t output_activation_max_vec = vdupq_n_s32(output_activation_max);

    DownquantizeAndStoreDepth16(acc0, output_multiplier, output_shift,
                                output_offset_vec, output_activation_min_vec,
                                output_activation_max_vec, output_ptr);

    DownquantizeAndStoreDepth16(acc1, output_multiplier, output_shift,
                                output_offset_vec, output_activation_min_vec,
                                output_activation_max_vec,
                                output_ptr + output_depth * output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth16<1, 2, 2> {
  static void Run(const Filter3x3x16& filter, const uint8* input_ptr,
                  int input_depth, int32 input_offset, int input_row_width,
                  const int32* bias_ptr, int32 output_offset,
                  int32 output_multiplier, int output_shift,
                  int32 output_activation_min, int32 output_activation_max,
                  uint8* output_ptr, int output_depth, int output_width) {
    // 16 depth accumulators for the 2 outputs.
    Int32x16 acc0, acc1;

    // Accumulators for top filter.
    acc0.v0 = vld1q_s32(bias_ptr);
    acc0.v1 = vld1q_s32(bias_ptr + 4);
    acc0.v2 = vld1q_s32(bias_ptr + 8);
    acc0.v3 = vld1q_s32(bias_ptr + 12);
    // Accumulators for bottom filter.
    acc1.v0 = vld1q_s32(bias_ptr);
    acc1.v1 = vld1q_s32(bias_ptr + 4);
    acc1.v2 = vld1q_s32(bias_ptr + 8);
    acc1.v3 = vld1q_s32(bias_ptr + 12);

    // Main multiply accumulate work.
    {
      // Load inputs for one filter row at a time.
      Int16x16x3 input;

      // Do first row of top filter.
      input = LoadInputRowDepth16(input_ptr, input_depth, input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r0, input);

      // Do second row of top filter.
      input = LoadInputRowDepth16(input_ptr + input_row_width, input_depth,
                                  input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r1, input);

      // Do third row of top filter.
      input = LoadInputRowDepth16(input_ptr + 2 * input_row_width, input_depth,
                                  input_offset, input);
      acc0 = MultiplyAccumulateRowDepth16(acc0, filter.r2, input);

      // The inputs to third row of the top filter are also the inputs
      // to first row of the bottom filter.
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r0, input);

      // Do second row of bottom filter.
      input = LoadInputRowDepth16(input_ptr + 3 * input_row_width, input_depth,
                                  input_offset, input);
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r1, input);

      // Do third row of bottom filter.
      input = LoadInputRowDepth16(input_ptr + 4 * input_row_width, input_depth,
                                  input_offset, input);
      acc1 = MultiplyAccumulateRowDepth16(acc1, filter.r2, input);
    }

    // Apply activation, downquantize and store.
    int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    int32x4_t output_activation_min_vec = vdupq_n_s32(output_activation_min);
    int32x4_t output_activation_max_vec = vdupq_n_s32(output_activation_max);

    DownquantizeAndStoreDepth16(acc0, output_multiplier, output_shift,
                                output_offset_vec, output_activation_min_vec,
                                output_activation_max_vec, output_ptr);

    DownquantizeAndStoreDepth16(acc1, output_multiplier, output_shift,
                                output_offset_vec, output_activation_min_vec,
                                output_activation_max_vec,
                                output_ptr + output_depth * output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth16<1, 1> {
  static void Run(const Filter3x3x16& filter, const uint8* input_ptr,
                  int input_depth, int32 input_offset, int input_row_width,
                  const int32* bias_ptr, int32 output_offset,
                  int32 output_multiplier, int output_shift,
                  int32 output_activation_min, int32 output_activation_max,
                  uint8* output_ptr, int output_depth, int output_width) {
    Int32x16 acc;
    acc.v0 = vld1q_s32(bias_ptr);
    acc.v1 = vld1q_s32(bias_ptr + 4);
    acc.v2 = vld1q_s32(bias_ptr + 8);
    acc.v3 = vld1q_s32(bias_ptr + 12);

    // Main multiply accumulate work.
    {
      // Load inputs for one filter row at a time.
      Int16x16x3 input;

      // Do first row.
      input = LoadInputRowDepth16(input_ptr, input_depth, input_offset, input);
      acc = MultiplyAccumulateRowDepth16(acc, filter.r0, input);

      // Do second row.
      input = LoadInputRowDepth16(input_ptr + input_row_width, input_depth,
                                  input_offset, input);
      acc = MultiplyAccumulateRowDepth16(acc, filter.r1, input);

      // Do third row.
      input = LoadInputRowDepth16(input_ptr + 2 * input_row_width, input_depth,
                                  input_offset, input);
      acc = MultiplyAccumulateRowDepth16(acc, filter.r2, input);
    }

    // Apply activation, downquantize and store.
    int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    int32x4_t output_activation_min_vec = vdupq_n_s32(output_activation_min);
    int32x4_t output_activation_max_vec = vdupq_n_s32(output_activation_max);

    DownquantizeAndStoreDepth16(acc, output_multiplier, output_shift,
                                output_offset_vec, output_activation_min_vec,
                                output_activation_max_vec, output_ptr);
  }
};

inline void DepthwiseConv3by3FilterDepth16(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data, const Dims<4>& bias_dims, int stride_width,
    int stride_height, int pad_width, int pad_height, int depth_multiplier,
    int32 output_offset, int32 output_multiplier, int output_shift,
    int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  // Algorithm assumes below constraints. It is optimized for depth multiplier
  // of 1, 3x3 filter, no padding, strides 1 and 2.
  TFLITE_DCHECK(output_depth == input_depth * depth_multiplier);
  TFLITE_DCHECK(depth_multiplier == 1);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 3);
  TFLITE_DCHECK(pad_height == 0);
  TFLITE_DCHECK(pad_width == 0);
  TFLITE_DCHECK(stride_width == 1);
  TFLITE_DCHECK(stride_height == 1);

  // The number of outputs to process in the main loop.
  const int num_x_outputs = 1;
  const int num_y_outputs = 2;

  const int input_row_width = output_depth * (input_width + 2 * pad_width);
  const int input_batch_size =
      input_row_width * (input_height + 2 * pad_height);
  const int output_batch_size = output_depth * output_width * output_height;
  const int input_ptr_x_increment = input_depth * stride_width;

  // Calculate extents of non-boundary loop.
  int out_x_start = 0;
  for (; out_x_start < input_width; out_x_start++) {
    int in_x = (out_x_start * stride_width) - pad_width;
    if (in_x >= 0) {
      break;
    }
  }
  int out_x_end = output_width - 1;
  for (; out_x_end >= 0; out_x_end--) {
    int in_x = (out_x_end * stride_width) - pad_width;
    int in_x_end = in_x + filter_width + (num_x_outputs - 1) * stride_width;
    if (in_x_end <= input_width) {
      out_x_end++;
      break;
    }
  }
  int out_y_start = 0;
  for (; out_y_start < input_height; out_y_start++) {
    int in_y = (out_y_start * stride_height) - pad_height;
    if (in_y >= 0) {
      break;
    }
  }
  int out_y_end = output_height - 1;
  for (; out_y_end >= 0; out_y_end--) {
    int in_y = (out_y_end * stride_height) - pad_height;
    int in_y_end = in_y + filter_height + (num_y_outputs - 1) * stride_height;
    if (in_y_end <= input_height) {
      out_y_end++;
      break;
    }
  }

  // Offsets for preloading inputs.
  const int i0 = 0;
  const int i1 = input_depth;
  const int i2 = 2 * input_depth;
  const int i3 = input_row_width;
  const int i4 = input_row_width + input_depth;
  const int i5 = input_row_width + 2 * input_depth;
  const int i6 = 2 * input_row_width;
  const int i7 = 2 * input_row_width + input_depth;
  const int i8 = 2 * input_row_width + 2 * input_depth;
  const int i9 = 3 * input_row_width;
  const int i10 = 3 * input_row_width + input_depth;
  const int i11 = 3 * input_row_width + 2 * input_depth;

  for (int b = 0; b < batches; ++b) {
    const int32* bias_ptr = bias_data;
    const uint8* filter_ptr = filter_data;

    const int in_batch_offset = b * input_batch_size;
    const int out_batch_offset = b * output_batch_size;

    int depth = 0;
    for (; depth <= output_depth - 16; depth += 16) {
      Filter3x3x16 filter =
          LoadFilterDepth16(filter_ptr, filter_offset, output_depth);

      // Handle 1x2 outputs.
      int out_y = out_y_start;
      for (; out_y < out_y_end; out_y += num_y_outputs) {
        int out_x = out_x_start;

        int in_y_offset =
            stride_height * input_row_width * (out_y + pad_height);
        int in_x_offset = stride_width * input_depth * (out_x + pad_width);

        const uint8* input_ptr =
            input_data + depth + in_x_offset + in_y_offset + in_batch_offset;

        uint8* output_ptr = output_data + depth + (out_x * output_depth) +
                            (output_depth * output_width * out_y) +
                            out_batch_offset;

        // Preload inputs. If input depth is large, preload every value of the
        // input for this depth range. Otherwise, preload only the first values
        // of each row.
        if (input_depth >= 32) {
          preload_l1_keep(input_ptr + i0);
          preload_l1_keep(input_ptr + i1);
          preload_l1_keep(input_ptr + i2);
          preload_l1_keep(input_ptr + i3);
          preload_l1_keep(input_ptr + i4);
          preload_l1_keep(input_ptr + i5);
          preload_l1_keep(input_ptr + i6);
          preload_l1_keep(input_ptr + i7);
          preload_l1_keep(input_ptr + i8);
          preload_l1_keep(input_ptr + i9);
          preload_l1_keep(input_ptr + i10);
          preload_l1_keep(input_ptr + i11);
        } else {
          preload_l1_keep(input_ptr + i0);
          preload_l1_keep(input_ptr + i3);
          preload_l1_keep(input_ptr + i6);
          preload_l1_keep(input_ptr + i9);
        }

        for (; out_x < out_x_end; out_x += num_x_outputs) {
          ConvKernel3x3FilterDepth16<1, 2, 1>::Run(
              filter, input_ptr, input_depth, input_offset, input_row_width,
              bias_ptr, output_offset, output_multiplier, output_shift,
              output_activation_min, output_activation_max, output_ptr,
              output_depth, output_width);

          input_ptr += input_ptr_x_increment * num_x_outputs;
          output_ptr += output_depth * num_x_outputs;

          // Preload the next inputs depending on stride.
          if (stride_width == 1) {
            preload_l1_keep(input_ptr + i2);
            preload_l1_keep(input_ptr + i5);
            preload_l1_keep(input_ptr + i8);
            preload_l1_keep(input_ptr + i11);
          } else if (stride_width == 2) {
            preload_l1_keep(input_ptr + i1);
            preload_l1_keep(input_ptr + i2);
            preload_l1_keep(input_ptr + i4);
            preload_l1_keep(input_ptr + i5);
            preload_l1_keep(input_ptr + i7);
            preload_l1_keep(input_ptr + i8);
            preload_l1_keep(input_ptr + i10);
            preload_l1_keep(input_ptr + i11);
          }
        }

        // Handle the rest of the right side.
        for (; out_x < output_width; out_x++) {
          // This code path can only be reached if we're handling >1 x outputs
          // at a time or support padding.
        }
      }

      // Handle the rest of the bottom side.
      for (; out_y < output_height; out_y++) {
        int out_x = out_x_start;

        int in_y_offset =
            stride_height * input_row_width * (out_y + pad_height);
        int in_x_offset = stride_width * input_depth * (out_x + pad_width);

        const uint8* input_ptr =
            input_data + depth + in_x_offset + in_y_offset + in_batch_offset;

        uint8* output_ptr = output_data + depth + (out_x * output_depth) +
                            (output_depth * output_width * out_y) +
                            out_batch_offset;

        for (; out_x < output_width; out_x++) {
          ConvKernel3x3FilterDepth16<1, 1>::Run(
              filter, input_ptr, input_depth, input_offset, input_row_width,
              bias_ptr, output_offset, output_multiplier, output_shift,
              output_activation_min, output_activation_max, output_ptr,
              output_depth, output_width);

          input_ptr += input_ptr_x_increment;
          output_ptr += output_depth;
        }
      }
      filter_ptr += 16;
      bias_ptr += 16;
    }
  }
}

#endif  // __aarch64__

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

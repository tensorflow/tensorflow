/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

struct Int32x8 {
  int32x4_t low, high;
};

struct Filter3x3x8 {
  int16x8_t f0, f1, f2, f3, f4, f5, f6, f7, f8;
};

// Loads 3x3 filter of depth 8 and adds filter offsets.
inline Filter3x3x8 Load3x3Filter(const uint8* filter_ptr, int32 filter_offset,
                                 int output_depth) {
  Filter3x3x8 filter;

  uint8x8_t temp_u8_0, temp_u8_1, temp_u8_2, temp_u8_3, temp_u8_4, temp_u8_5,
      temp_u8_6, temp_u8_7, temp_u8_8;
  int16x8_t filter_offset_vec = vdupq_n_s16(filter_offset);

  temp_u8_0 = vld1_u8(filter_ptr + 0 * output_depth);
  temp_u8_1 = vld1_u8(filter_ptr + 1 * output_depth);
  temp_u8_2 = vld1_u8(filter_ptr + 2 * output_depth);
  temp_u8_3 = vld1_u8(filter_ptr + 3 * output_depth);
  temp_u8_4 = vld1_u8(filter_ptr + 4 * output_depth);
  temp_u8_5 = vld1_u8(filter_ptr + 5 * output_depth);
  temp_u8_6 = vld1_u8(filter_ptr + 6 * output_depth);
  temp_u8_7 = vld1_u8(filter_ptr + 7 * output_depth);
  temp_u8_8 = vld1_u8(filter_ptr + 8 * output_depth);

  filter.f0 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_0));
  filter.f1 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_1));
  filter.f2 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_2));
  filter.f3 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_3));
  filter.f4 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_4));
  filter.f5 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_5));
  filter.f6 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_6));
  filter.f7 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_7));
  filter.f8 = vreinterpretq_s16_u16(vmovl_u8(temp_u8_8));

  filter.f0 = vaddq_s16(filter.f0, filter_offset_vec);
  filter.f1 = vaddq_s16(filter.f1, filter_offset_vec);
  filter.f2 = vaddq_s16(filter.f2, filter_offset_vec);
  filter.f3 = vaddq_s16(filter.f3, filter_offset_vec);
  filter.f4 = vaddq_s16(filter.f4, filter_offset_vec);
  filter.f5 = vaddq_s16(filter.f5, filter_offset_vec);
  filter.f6 = vaddq_s16(filter.f6, filter_offset_vec);
  filter.f7 = vaddq_s16(filter.f7, filter_offset_vec);
  filter.f8 = vaddq_s16(filter.f8, filter_offset_vec);

  return filter;
}

// Applies activation, offset and downquantize on a set of accumulator
// registers that correspond to a 2x2 output of depth 8.
// Stores results to output.
inline void DownquantizeAndStore2x2Output(
    Int32x8 acc_0, Int32x8 acc_1, Int32x8 acc_2, Int32x8 acc_3,
    int32 output_offset, int32 output_multiplier, int output_shift,
    int32 output_activation_min, int32 output_activation_max, uint8* output_ptr,
    int output_depth, int output_width) {
  using gemmlowp::RoundingDivideByPOT;
  const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
  const int32x4_t output_activation_min_vec =
      vdupq_n_s32(output_activation_min);
  const int32x4_t output_activation_max_vec =
      vdupq_n_s32(output_activation_max);

  // Fixed-point multiplication.
  acc_0.low = vqrdmulhq_n_s32(acc_0.low, output_multiplier);
  acc_0.high = vqrdmulhq_n_s32(acc_0.high, output_multiplier);
  acc_1.low = vqrdmulhq_n_s32(acc_1.low, output_multiplier);
  acc_1.high = vqrdmulhq_n_s32(acc_1.high, output_multiplier);
  acc_2.low = vqrdmulhq_n_s32(acc_2.low, output_multiplier);
  acc_2.high = vqrdmulhq_n_s32(acc_2.high, output_multiplier);
  acc_3.low = vqrdmulhq_n_s32(acc_3.low, output_multiplier);
  acc_3.high = vqrdmulhq_n_s32(acc_3.high, output_multiplier);

  acc_0.low = RoundingDivideByPOT(acc_0.low, output_shift);
  acc_0.high = RoundingDivideByPOT(acc_0.high, output_shift);
  acc_1.low = RoundingDivideByPOT(acc_1.low, output_shift);
  acc_1.high = RoundingDivideByPOT(acc_1.high, output_shift);
  acc_2.low = RoundingDivideByPOT(acc_2.low, output_shift);
  acc_2.high = RoundingDivideByPOT(acc_2.high, output_shift);
  acc_3.low = RoundingDivideByPOT(acc_3.low, output_shift);
  acc_3.high = RoundingDivideByPOT(acc_3.high, output_shift);

  // Add the output offset.
  acc_0.low = vaddq_s32(acc_0.low, output_offset_vec);
  acc_0.high = vaddq_s32(acc_0.high, output_offset_vec);
  acc_1.low = vaddq_s32(acc_1.low, output_offset_vec);
  acc_1.high = vaddq_s32(acc_1.high, output_offset_vec);
  acc_2.low = vaddq_s32(acc_2.low, output_offset_vec);
  acc_2.high = vaddq_s32(acc_2.high, output_offset_vec);
  acc_3.low = vaddq_s32(acc_3.low, output_offset_vec);
  acc_3.high = vaddq_s32(acc_3.high, output_offset_vec);

  // Apply the activation function.
  acc_0.low = vmaxq_s32(acc_0.low, output_activation_min_vec);
  acc_0.high = vmaxq_s32(acc_0.high, output_activation_min_vec);
  acc_1.low = vmaxq_s32(acc_1.low, output_activation_min_vec);
  acc_1.high = vmaxq_s32(acc_1.high, output_activation_min_vec);
  acc_2.low = vmaxq_s32(acc_2.low, output_activation_min_vec);
  acc_2.high = vmaxq_s32(acc_2.high, output_activation_min_vec);
  acc_3.low = vmaxq_s32(acc_3.low, output_activation_min_vec);
  acc_3.high = vmaxq_s32(acc_3.high, output_activation_min_vec);

  acc_0.low = vminq_s32(acc_0.low, output_activation_max_vec);
  acc_0.high = vminq_s32(acc_0.high, output_activation_max_vec);
  acc_1.low = vminq_s32(acc_1.low, output_activation_max_vec);
  acc_1.high = vminq_s32(acc_1.high, output_activation_max_vec);
  acc_2.low = vminq_s32(acc_2.low, output_activation_max_vec);
  acc_2.high = vminq_s32(acc_2.high, output_activation_max_vec);
  acc_3.low = vminq_s32(acc_3.low, output_activation_max_vec);
  acc_3.high = vminq_s32(acc_3.high, output_activation_max_vec);

  // Saturating cast to uint8 and store to destination.
  int16x4_t acc_0_low_s16 = vqmovn_s32(acc_0.low);
  int16x4_t acc_0_high_s16 = vqmovn_s32(acc_0.high);
  int16x4_t acc_1_low_s16 = vqmovn_s32(acc_1.low);
  int16x4_t acc_1_high_s16 = vqmovn_s32(acc_1.high);
  int16x4_t acc_2_low_s16 = vqmovn_s32(acc_2.low);
  int16x4_t acc_2_high_s16 = vqmovn_s32(acc_2.high);
  int16x4_t acc_3_low_s16 = vqmovn_s32(acc_3.low);
  int16x4_t acc_3_high_s16 = vqmovn_s32(acc_3.high);

  int16x8_t res_0_s16 = vcombine_s16(acc_0_low_s16, acc_0_high_s16);
  int16x8_t res_1_s16 = vcombine_s16(acc_1_low_s16, acc_1_high_s16);
  int16x8_t res_2_s16 = vcombine_s16(acc_2_low_s16, acc_2_high_s16);
  int16x8_t res_3_s16 = vcombine_s16(acc_3_low_s16, acc_3_high_s16);

  uint8x8_t res_0_u8 = vqmovun_s16(res_0_s16);
  uint8x8_t res_1_u8 = vqmovun_s16(res_1_s16);
  uint8x8_t res_2_u8 = vqmovun_s16(res_2_s16);
  uint8x8_t res_3_u8 = vqmovun_s16(res_3_s16);

  vst1_u8(output_ptr, res_0_u8);
  vst1_u8(output_ptr + output_depth, res_1_u8);
  vst1_u8(output_ptr + output_depth * output_width, res_2_u8);
  vst1_u8(output_ptr + output_depth * output_width + output_depth, res_3_u8);
}

inline void DownquantizeAndStore(Int32x8 acc, int32 output_offset,
                                 int32 output_multiplier, int output_shift,
                                 int32 output_activation_min,
                                 int32 output_activation_max,
                                 uint8* output_ptr) {
  using gemmlowp::RoundingDivideByPOT;
  const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
  const int32x4_t output_activation_min_vec =
      vdupq_n_s32(output_activation_min);
  const int32x4_t output_activation_max_vec =
      vdupq_n_s32(output_activation_max);

  acc.low = vqrdmulhq_n_s32(acc.low, output_multiplier);
  acc.high = vqrdmulhq_n_s32(acc.high, output_multiplier);

  acc.low = RoundingDivideByPOT(acc.low, output_shift);
  acc.high = RoundingDivideByPOT(acc.high, output_shift);

  acc.low = vaddq_s32(acc.low, output_offset_vec);
  acc.high = vaddq_s32(acc.high, output_offset_vec);

  acc.low = vmaxq_s32(acc.low, output_activation_min_vec);
  acc.high = vmaxq_s32(acc.high, output_activation_min_vec);

  acc.low = vminq_s32(acc.low, output_activation_max_vec);
  acc.high = vminq_s32(acc.high, output_activation_max_vec);

  int16x4_t acc_low_s16 = vqmovn_s32(acc.low);
  int16x4_t acc_high_s16 = vqmovn_s32(acc.high);

  int16x8_t res_s16 = vcombine_s16(acc_low_s16, acc_high_s16);
  uint8x8_t res_u8 = vqmovun_s16(res_s16);
  vst1_u8(output_ptr, res_u8);
}

inline void DownquantizeAndStore2Output(
    Int32x8 acc_0, Int32x8 acc_1, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_ptr, int output_ptr_offset) {
  {
    using gemmlowp::RoundingDivideByPOT;
    const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    const int32x4_t output_activation_min_vec =
        vdupq_n_s32(output_activation_min);
    const int32x4_t output_activation_max_vec =
        vdupq_n_s32(output_activation_max);

    // Fixed-point multiplication.
    acc_0.low = vqrdmulhq_n_s32(acc_0.low, output_multiplier);
    acc_0.high = vqrdmulhq_n_s32(acc_0.high, output_multiplier);
    acc_1.low = vqrdmulhq_n_s32(acc_1.low, output_multiplier);
    acc_1.high = vqrdmulhq_n_s32(acc_1.high, output_multiplier);

    acc_0.low = RoundingDivideByPOT(acc_0.low, output_shift);
    acc_0.high = RoundingDivideByPOT(acc_0.high, output_shift);
    acc_1.low = RoundingDivideByPOT(acc_1.low, output_shift);
    acc_1.high = RoundingDivideByPOT(acc_1.high, output_shift);

    // Add the output offset.
    acc_0.low = vaddq_s32(acc_0.low, output_offset_vec);
    acc_0.high = vaddq_s32(acc_0.high, output_offset_vec);
    acc_1.low = vaddq_s32(acc_1.low, output_offset_vec);
    acc_1.high = vaddq_s32(acc_1.high, output_offset_vec);

    // Apply the activation function.
    acc_0.low = vmaxq_s32(acc_0.low, output_activation_min_vec);
    acc_0.high = vmaxq_s32(acc_0.high, output_activation_min_vec);
    acc_1.low = vmaxq_s32(acc_1.low, output_activation_min_vec);
    acc_1.high = vmaxq_s32(acc_1.high, output_activation_min_vec);

    acc_0.low = vminq_s32(acc_0.low, output_activation_max_vec);
    acc_0.high = vminq_s32(acc_0.high, output_activation_max_vec);
    acc_1.low = vminq_s32(acc_1.low, output_activation_max_vec);
    acc_1.high = vminq_s32(acc_1.high, output_activation_max_vec);
  }

  // Saturating cast to uint8 and store to destination.
  int16x8_t res_0_s16;
  {
    int16x4_t acc_0_low_s16 = vqmovn_s32(acc_0.low);
    int16x4_t acc_0_high_s16 = vqmovn_s32(acc_0.high);
    res_0_s16 = vcombine_s16(acc_0_low_s16, acc_0_high_s16);
  }

  int16x8_t res_1_s16;
  {
    int16x4_t acc_1_low_s16 = vqmovn_s32(acc_1.low);
    int16x4_t acc_1_high_s16 = vqmovn_s32(acc_1.high);
    res_1_s16 = vcombine_s16(acc_1_low_s16, acc_1_high_s16);
  }

  uint8x8_t res_0_u8 = vqmovun_s16(res_0_s16);
  uint8x8_t res_1_u8 = vqmovun_s16(res_1_s16);
  vst1_u8(output_ptr, res_0_u8);
  vst1_u8(output_ptr + output_ptr_offset, res_1_u8);
}

// Performs multiply accumulate on 3 inputs of depth 8.
inline Int32x8 MultiplyAccumulateRow(Int32x8 accum, int16x8_t f0, int16x8_t f1,
                                     int16x8_t f2, int16x8_t i0, int16x8_t i1,
                                     int16x8_t i2) {
  accum.low = vmlal_s16(accum.low, vget_low_s16(f0), vget_low_s16(i0));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f0), vget_high_s16(i0));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f1), vget_low_s16(i1));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f1), vget_high_s16(i1));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f2), vget_low_s16(i2));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f2), vget_high_s16(i2));
  return accum;
}

// Performs multiply accumulate on 3 inputs of depth 8.
inline Int32x8 MultiplyAccumulate3x3Filter(const Filter3x3x8& f, int16x8_t i0,
                                           int16x8_t i1, int16x8_t i2,
                                           int16x8_t i3, int16x8_t i4,
                                           int16x8_t i5, int16x8_t i6,
                                           int16x8_t i7, int16x8_t i8,
                                           Int32x8 accum) {
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f0), vget_low_s16(i0));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f0), vget_high_s16(i0));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f1), vget_low_s16(i1));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f1), vget_high_s16(i1));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f2), vget_low_s16(i2));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f2), vget_high_s16(i2));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f3), vget_low_s16(i3));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f3), vget_high_s16(i3));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f4), vget_low_s16(i4));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f4), vget_high_s16(i4));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f5), vget_low_s16(i5));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f5), vget_high_s16(i5));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f6), vget_low_s16(i6));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f6), vget_high_s16(i6));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f7), vget_low_s16(i7));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f7), vget_high_s16(i7));
  accum.low = vmlal_s16(accum.low, vget_low_s16(f.f8), vget_low_s16(i8));
  accum.high = vmlal_s16(accum.high, vget_high_s16(f.f8), vget_high_s16(i8));
  return accum;
}

inline void DotProductAndStore(const Filter3x3x8& filter, int16x8_t i0,
                               int16x8_t i1, int16x8_t i2, int16x8_t i3,
                               int16x8_t i4, int16x8_t i5, int16x8_t i6,
                               int16x8_t i7, int16x8_t i8,
                               const int32* bias_ptr, int32 output_offset,
                               int32 output_multiplier, int output_shift,
                               int32 output_activation_min,
                               int32 output_activation_max, uint8* output_ptr) {
  Int32x8 acc;
  acc.low = vld1q_s32(bias_ptr);
  acc.high = vld1q_s32(bias_ptr + 4);

  acc = MultiplyAccumulate3x3Filter(filter, i0, i1, i2, i3, i4, i5, i6, i7, i8,
                                    acc);

  DownquantizeAndStore(acc, output_offset, output_multiplier, output_shift,
                       output_activation_min, output_activation_max,
                       output_ptr);
}

// Performs multiply-accumulate on a 3x4 input for 2 horizontal outputs.
inline void DotProductAndStore2xStride1(
    const Filter3x3x8& filter, int16x8_t i0, int16x8_t i1, int16x8_t i2,
    int16x8_t i3, int16x8_t i4, int16x8_t i5, int16x8_t i6, int16x8_t i7,
    int16x8_t i8, int16x8_t i9, int16x8_t i10, int16x8_t i11,
    const int32* bias_ptr, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_ptr, int output_ptr_offset) {
  Int32x8 acc_0, acc_1;
  acc_0.low = vld1q_s32(bias_ptr);
  acc_1.low = vld1q_s32(bias_ptr);
  acc_0.high = vld1q_s32(bias_ptr + 4);
  acc_1.high = vld1q_s32(bias_ptr + 4);

  acc_0 = MultiplyAccumulate3x3Filter(filter, i0, i1, i2, i4, i5, i6, i8, i9,
                                      i10, acc_0);
  acc_1 = MultiplyAccumulate3x3Filter(filter, i1, i2, i3, i5, i6, i7, i9, i10,
                                      i11, acc_1);
  DownquantizeAndStore2Output(acc_0, acc_1, output_offset, output_multiplier,
                              output_shift, output_activation_min,
                              output_activation_max, output_ptr,
                              output_ptr_offset);
}

// Performs multiply-accumulate on a 4x3 input for 2 vertical outputs.
inline void DotProductAndStore2yStride1(
    const Filter3x3x8& filter, int16x8_t i0, int16x8_t i1, int16x8_t i2,
    int16x8_t i3, int16x8_t i4, int16x8_t i5, int16x8_t i6, int16x8_t i7,
    int16x8_t i8, int16x8_t i9, int16x8_t i10, int16x8_t i11,
    const int32* bias_ptr, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_ptr, int output_ptr_offset) {
  Int32x8 acc_0, acc_1;
  acc_0.low = vld1q_s32(bias_ptr);
  acc_1.low = vld1q_s32(bias_ptr);
  acc_0.high = vld1q_s32(bias_ptr + 4);
  acc_1.high = vld1q_s32(bias_ptr + 4);

  acc_0 = MultiplyAccumulate3x3Filter(filter, i0, i1, i2, i3, i4, i5, i6, i7,
                                      i8, acc_0);
  acc_1 = MultiplyAccumulate3x3Filter(filter, i3, i4, i5, i6, i7, i8, i9, i10,
                                      i11, acc_1);
  DownquantizeAndStore2Output(acc_0, acc_1, output_offset, output_multiplier,
                              output_shift, output_activation_min,
                              output_activation_max, output_ptr,
                              output_ptr_offset);
}

// A kernel that is optimized on the number of output cells in the x and y
// direction, and the stride. Assumes 3x3 filters of 8 depth.
template <int kFixedOutputY, int kFixedOutputX, int kFixedStrideWidth,
          int kFixedStrideHeight>
struct ConvKernel3x3FilterDepth8 {};

template <>
struct ConvKernel3x3FilterDepth8<8, 8, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int output_row_size = output_depth * output_width;

    // To process 8x8 outputs using a 3x3 filter, we require 10x10 inputs.
    // Load inputs for the first 2 filters on the top left, then slide to
    // the right, down, left, down, right, etc. in a snake-like path. This
    // minimizes the total number of loads.
    //
    //        INPUT                          OUTPUT
    //   |\----------------\               |\------------\
    //   | \                \              | \            \
    //   |  \----------------\             |  \------------\
    //   |  | 0    ...     9 |             |  | 0  ...   7 |
    //   |  | 10   ...    19 |     --->    |  | 8  ...  15 |
    //   |  | 20   ...    29 |              \ | .. ...  .. |
    //    \ | ..   ...    .. |               \| 56 ...  63 |
    //     \| 90   ...   109 |                |------------|
    //      |----------------|
    //
    // The first set of loads corresponds to:
    //
    //        INPUT                          OUTPUT
    //   |\-----------------                |\-----------
    //   | \                                | \
    //   |  \-----------------              |  \----------
    //   |  | 0  1   2  3 ...               |  | 0  1 ...
    //   |  | 10 11 12 13 ...     --->      |  | ..   ...
    //   |  | 20 21 22 23 ...                  | ..   ...
    //   |  | ..   ...    ...
    //
    // The next set of loads correspond to a sliding window to the right.
    // It loads inputs 4, 5, 14, 15, 23, 24 and keeps 2, 3, 12, 13, and 22:
    //
    //        INPUT                          OUTPUT
    //   |\-------------------                |\-------------
    //   | \                                  | \
    //   |  \-------------------              |  \------------
    //   |  | .. 2  3   4  5 ...              |  | .. 2  3 ...
    //   |  | .. 12 13 14 15 ...     --->     |  | ..      ...
    //   |  | .. 21 22 23 24 ...                 | ..      ...
    //   |  | ..    ...      ...
    //
    // And so on...

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the top left. Referring to the
    // indexes in the diagram above, this corresponds to outputs (0) and (1).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    // Slide to the right for outputs x = [2, 3], y = 0. Referring to the
    // indexes in the diagram above, this corresponds to outputs (2) and (3).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth, output_depth);

    // Slide to the right again for outputs x = [4, 5], y = 0. Referring to the
    // indexes in the diagram above, this corresponds to outputs (4) and (5).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 6 * input_depth;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 4 * output_depth, output_depth);

    // Slide to the right one last time for outputs x = [6, 7], y = 0.
    // Referring to the indexes in the diagram above, this corresponds to
    // outputs (6) and (7).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 8 * input_depth;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 6 * output_depth, output_depth);

    // Slide to down for outputs x = [6, 7], y = 1. Referring to the indexes in
    // the diagram above, this corresponds to outputs (14) and (15).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 6 * input_depth + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 6 * output_depth + output_row_size,
        output_depth);

    // Slide left for outputs x = [4, 5], y = 1. Referring to the indexes in
    // the diagram above, this corresponds to outputs (12) and (13).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 4 * output_depth + output_row_size,
        output_depth);

    // Slide left again for outputs x = [2, 3], y = 1. Referring to the indexes
    // in the diagram above, this corresponds to outputs (10) and (11).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 2 * input_depth + input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth + output_row_size,
        output_depth);

    // Slide left one more time for outputs x = [0, 1], y = 1. Referring to the
    // indexes in the diagram above, this corresponds to outputs (8) and (9).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + output_row_size, output_depth);

    // Slide down for outputs x = [0, 1], y = 2. Referring to the
    // indexes in the diagram above, this corresponds to outputs (16) and (17).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_row_size, output_depth);

    // Slide right for outputs x = [2, 3], y = 2. Referring to the
    // indexes in the diagram above, this corresponds to outputs (18) and (19).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 2 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_10, input_11, input_8, input_9, input_2, input_3, input_0,
        input_1, input_6, input_7, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 2 * output_row_size, output_depth);

    // Slide right for outputs x = [4, 5], y = 2. Referring to the
    // indexes in the diagram above, this corresponds to outputs (20) and (21).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 6 * input_depth + 2 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 2 * output_row_size, output_depth);

    // Slide right one more time for outputs x = [6, 7], y = 2. Referring to the
    // indexes in the diagram above, this corresponds to outputs (22) and (23).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 8 * input_depth + 2 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_10, input_11, input_8, input_9, input_2, input_3, input_0,
        input_1, input_6, input_7, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 2 * output_row_size, output_depth);

    // Slide down for outputs x = [6, 7], y = 3. Referring to the indexes in
    // the diagram above, this corresponds to outputs (30) and (31).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 6 * input_depth + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 3 * output_row_size, output_depth);

    // Slide left for outputs x = [4, 5], y = 3. Referring to the indexes in
    // the diagram above, this corresponds to outputs (28) and (29).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 3 * output_row_size, output_depth);

    // Slide left for outputs x = [2, 3], y = 3. Referring to the indexes in
    // the diagram above, this corresponds to outputs (26) and (27).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 2 * input_depth + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 3 * output_row_size, output_depth);

    // Slide left one more time for outputs x = [0, 1], y = 3. Referring to the
    // indexes in the diagram above, this corresponds to outputs (24) and (25).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 3 * output_row_size, output_depth);

    // Slide down for outputs x = [0, 1], y = 4. Referring to the indexes in
    // the diagram above, this corresponds to outputs (32) and (33).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 6 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 4 * output_row_size, output_depth);

    // Slide right for outputs x = [2, 3], y = 4. Referring to the indexes in
    // the diagram above, this corresponds to outputs (34) and (35).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 4 * output_row_size, output_depth);

    // Slide right for outputs x = [4, 5], y = 4. Referring to the indexes in
    // the diagram above, this corresponds to outputs (36) and (37).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 6 * input_depth + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 4 * output_row_size, output_depth);

    // Slide right one more time for outputs x = [6, 7], y = 4. Referring to the
    // indexes in the diagram above, this corresponds to outputs (38) and (39).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 8 * input_depth + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 4 * output_row_size, output_depth);

    // Slide down for outputs x = [6, 7], y = 5. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (46) and (47).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 6 * input_depth + 7 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_10, input_11, input_8, input_9, input_2, input_3, input_0,
        input_1, input_6, input_7, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 5 * output_row_size, output_depth);

    // Slide left for outputs x = [4, 5], y = 5. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (44) and (45).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 5 * output_row_size, output_depth);

    // Slide left for outputs x = [2, 3], y = 5. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (42) and (43).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 2 * input_depth + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_10, input_11, input_8, input_9, input_2, input_3, input_0,
        input_1, input_6, input_7, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 5 * output_row_size, output_depth);

    // Slide left one more time for outputs x = [0, 1], y = 5. Referring to the
    // indexes in the diagram above, this corresponds to outputs (40) and (41).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 5 * output_row_size, output_depth);

    // Slide down for outputs x = [0, 1], y = 6. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (48) and (49).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 8 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 6 * output_row_size, output_depth);

    // Slide right for outputs x = [2, 3], y = 6. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (50) and (51).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 6 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 6 * output_row_size, output_depth);

    // Slide right for outputs x = [4, 5], y = 6. Referring to the  indexes in
    // the diagram above, this corresponds to outputs (52) and (53).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 6 * input_depth + 6 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 6 * output_row_size, output_depth);

    // Slide right one more time for outputs x = [6, 7], y = 6. Referring to the
    // indexes in the diagram above, this corresponds to outputs (54) and (55).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 8 * input_depth + 6 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 6 * output_row_size, output_depth);

    // Slide down for outputs x = [6, 7], y = 7. Referring to the indexes in the
    // diagram above, this corresponds to outputs (62) and (63).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 6 * input_depth + 9 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 6 * output_depth + 7 * output_row_size, output_depth);

    // Slide left for outputs x = [4, 5], y = 7. Referring to the indexes in the
    // diagram above, this corresponds to outputs (60) and (61).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 7 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 4 * output_depth + 7 * output_row_size, output_depth);

    // Slide left for outputs x = [2, 3], y = 7. Referring to the indexes in the
    // diagram above, this corresponds to outputs (58) and (59).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 2 * input_depth + 7 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 7 * output_row_size, output_depth);

    // Slide left one more time for outputs x = [0, 1], y = 7. Referring to the
    // indexes in the diagram above, this corresponds to outputs (56) and (57).
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 7 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 7 * output_row_size, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 4, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int output_row_size = output_depth * output_width;

    // To process 4x4 outputs using a 3x3 filter, we require 6x6 inputs.
    // Load inputs for the first 2 filters on the top left, then slide to
    // the right, down, left, down, right, etc. in a snake-like path. This
    // minimizes the total number of loads.
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the top left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    // Now load 1x2 inputs on the top right.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth, output_depth);

    // Now load next inputs when sliding window down.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 2 * input_depth + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth + output_row_size,
        output_depth);

    // Now load next inputs when sliding window left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + output_row_size, output_depth);

    // Now load next inputs when sliding window down.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_row_size, output_depth);

    // Now load next inputs when sliding window right.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth + 2 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_10, input_11, input_8, input_9, input_2, input_3, input_0,
        input_1, input_6, input_7, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 2 * output_row_size, output_depth);

    // Now load next inputs when sliding window down.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 2 * input_depth + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max,
        output_ptr + 2 * output_depth + 3 * output_row_size, output_depth);

    // Now load next inputs when sliding window left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 3 * output_row_size, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 2, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int output_row_size = output_depth * output_width;

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the top.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Now load next inputs one row down.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Now load next row.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_8, input_9, input_10, input_11, input_0, input_1, input_2,
        input_3, input_4, input_5, input_6, input_7, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Now load last row.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 5 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 1, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int output_row_size = output_depth * output_width;

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 2x1 outputs starting from the top.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      ptr += input_row_size;
      temp_3 = vld1_u8(ptr);
      temp_4 = vld1_u8(ptr + input_depth);
      temp_5 = vld1_u8(ptr + 2 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      ptr += input_row_size;
      temp_3 = vld1_u8(ptr);
      temp_4 = vld1_u8(ptr + input_depth);
      temp_5 = vld1_u8(ptr + 2 * input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2yStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_row_size);

    // Load inputs for bottom 2 rows.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      ptr += input_row_size;
      temp_3 = vld1_u8(ptr);
      temp_4 = vld1_u8(ptr + input_depth);
      temp_5 = vld1_u8(ptr + 2 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
    }

    DotProductAndStore2yStride1(
        filter, input_6, input_7, input_8, input_9, input_10, input_11, input_0,
        input_1, input_2, input_3, input_4, input_5, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_row_size,
        output_row_size);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 2, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    Int32x8 acc_0, acc_1, acc_2, acc_3;

    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_2.low = vld1q_s32(bias_ptr);
    acc_3.low = vld1q_s32(bias_ptr);

    bias_ptr += 4;
    acc_0.high = vld1q_s32(bias_ptr);
    acc_1.high = vld1q_s32(bias_ptr);
    acc_2.high = vld1q_s32(bias_ptr);
    acc_3.high = vld1q_s32(bias_ptr);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);

    // Add scope for input registers to help the compiler know that it is
    // not needed.
    {
      // To process 2x2 outputs using a 3x3 filter, we require 4x4 inputs.
      // Load inputs for the top two filters first.
      int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
          input_7, input_8, input_9, input_10, input_11;

      const uint8* ptr = input_ptr;

      // Load top 3 rows.
      {
        uint8x8_t temp_0, temp_1, temp_2, temp_3;

        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);

        input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

        input_0 = vaddq_s16(input_0, input_offset_vec);
        input_1 = vaddq_s16(input_1, input_offset_vec);
        input_2 = vaddq_s16(input_2, input_offset_vec);
        input_3 = vaddq_s16(input_3, input_offset_vec);

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);

        input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

        input_4 = vaddq_s16(input_4, input_offset_vec);
        input_5 = vaddq_s16(input_5, input_offset_vec);
        input_6 = vaddq_s16(input_6, input_offset_vec);
        input_7 = vaddq_s16(input_7, input_offset_vec);

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);

        input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

        input_8 = vaddq_s16(input_8, input_offset_vec);
        input_9 = vaddq_s16(input_9, input_offset_vec);
        input_10 = vaddq_s16(input_10, input_offset_vec);
        input_11 = vaddq_s16(input_11, input_offset_vec);
      }

      // Multiply-accum for top-left output.
      acc_0 = MultiplyAccumulate3x3Filter(filter, input_0, input_1, input_2,
                                          input_4, input_5, input_6, input_8,
                                          input_9, input_10, acc_0);

      // Multiply-accum for top-right output.
      acc_1 = MultiplyAccumulate3x3Filter(filter, input_1, input_2, input_3,
                                          input_5, input_6, input_7, input_9,
                                          input_10, input_11, acc_1);

      // Now load the bottom row.
      {
        uint8x8_t temp_0, temp_1, temp_2, temp_3;

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);

        input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

        input_0 = vaddq_s16(input_0, input_offset_vec);
        input_1 = vaddq_s16(input_1, input_offset_vec);
        input_2 = vaddq_s16(input_2, input_offset_vec);
        input_3 = vaddq_s16(input_3, input_offset_vec);
      }

      // Multiply-accum for bottom-left output.
      acc_2 = MultiplyAccumulate3x3Filter(filter, input_4, input_5, input_6,
                                          input_8, input_9, input_10, input_0,
                                          input_1, input_2, acc_2);

      // Multiply-accum for bottom-right output.
      acc_3 = MultiplyAccumulate3x3Filter(filter, input_5, input_6, input_7,
                                          input_9, input_10, input_11, input_1,
                                          input_2, input_3, acc_3);
    }

    DownquantizeAndStore2x2Output(acc_0, acc_1, acc_2, acc_3, output_offset,
                                  output_multiplier, output_shift,
                                  output_activation_min, output_activation_max,
                                  output_ptr, output_depth, output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 4, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int output_row_size = output_depth * output_width;

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the top left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    // Now load 1x2 inputs on the top right.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + 4 * input_depth;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth, output_depth);

    // Now load next inputs when sliding window down.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr + 2 * input_depth + 3 * input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_6, input_7, input_4, input_5, input_10, input_11, input_8,
        input_9, input_2, input_3, input_0, input_1, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth + output_row_size,
        output_depth);

    // Now load next inputs when sliding window left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_0, input_1, input_2, input_3, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + output_row_size, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<1, 4, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the left.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      temp_3 = vld1_u8(ptr + 3 * input_depth);

      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_3));

      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth);

    // Now load 1x2 inputs on the right.
    {
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr + input_depth * 4;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_2 = vld1_u8(ptr);
      temp_3 = vld1_u8(ptr + input_depth);

      ptr += input_row_size;
      temp_4 = vld1_u8(ptr);
      temp_5 = vld1_u8(ptr + input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
    }

    DotProductAndStore2xStride1(
        filter, input_2, input_3, input_0, input_1, input_6, input_7, input_4,
        input_5, input_10, input_11, input_8, input_9, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr + 2 * output_depth, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 1, 1, 1> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    // To process 2x1 outputs using a 3x3 filter, we require 4x3 inputs.
    // Load all inputs at the beginning.
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11;

    // Load inputs for 1x2 outputs starting from the top left.
    {
      const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
      uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

      const uint8* ptr = input_ptr;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      ptr += input_row_size;
      temp_3 = vld1_u8(ptr);
      temp_4 = vld1_u8(ptr + input_depth);
      temp_5 = vld1_u8(ptr + 2 * input_depth);

      input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_0 = vaddq_s16(input_0, input_offset_vec);
      input_1 = vaddq_s16(input_1, input_offset_vec);
      input_2 = vaddq_s16(input_2, input_offset_vec);
      input_3 = vaddq_s16(input_3, input_offset_vec);
      input_4 = vaddq_s16(input_4, input_offset_vec);
      input_5 = vaddq_s16(input_5, input_offset_vec);

      ptr += input_row_size;
      temp_0 = vld1_u8(ptr);
      temp_1 = vld1_u8(ptr + input_depth);
      temp_2 = vld1_u8(ptr + 2 * input_depth);
      ptr += input_row_size;
      temp_3 = vld1_u8(ptr);
      temp_4 = vld1_u8(ptr + input_depth);
      temp_5 = vld1_u8(ptr + 2 * input_depth);

      input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
      input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
      input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
      input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
      input_10 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
      input_11 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

      input_6 = vaddq_s16(input_6, input_offset_vec);
      input_7 = vaddq_s16(input_7, input_offset_vec);
      input_8 = vaddq_s16(input_8, input_offset_vec);
      input_9 = vaddq_s16(input_9, input_offset_vec);
      input_10 = vaddq_s16(input_10, input_offset_vec);
      input_11 = vaddq_s16(input_11, input_offset_vec);
    }

    DotProductAndStore2yStride1(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9, input_10, input_11, bias_ptr, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_ptr, output_depth * output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 2, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    const int output_row_size = output_depth * output_width;

    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    Int32x8 acc_0, acc_1;
    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_0.high = vld1q_s32(bias_ptr + 4);
    acc_1.high = vld1q_s32(bias_ptr + 4);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, input_9;

    const uint8* ptr = input_ptr;
    uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4;

    // Load first 2 rows.
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_9 = vaddq_s16(input_9, input_offset_vec);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f0, filter.f1, filter.f2,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f0, filter.f1, filter.f2,
                                  input_2, input_3, input_4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f3, filter.f4, filter.f5,
                                  input_5, input_6, input_7);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f3, filter.f4, filter.f5,
                                  input_7, input_8, input_9);

    // Load next 2 rows.
    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_9 = vaddq_s16(input_9, input_offset_vec);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f6, filter.f7, filter.f8,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f6, filter.f7, filter.f8,
                                  input_2, input_3, input_4);

    DownquantizeAndStore2Output(
        acc_0, acc_1, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Moving onto the next row of outputs.
    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_0.high = vld1q_s32(bias_ptr + 4);
    acc_1.high = vld1q_s32(bias_ptr + 4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f0, filter.f1, filter.f2,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f0, filter.f1, filter.f2,
                                  input_2, input_3, input_4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f3, filter.f4, filter.f5,
                                  input_5, input_6, input_7);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f3, filter.f4, filter.f5,
                                  input_7, input_8, input_9);

    // Load next 2 rows.
    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_9 = vaddq_s16(input_9, input_offset_vec);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f6, filter.f7, filter.f8,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f6, filter.f7, filter.f8,
                                  input_2, input_3, input_4);

    DownquantizeAndStore2Output(
        acc_0, acc_1, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Moving onto the next row of outputs.
    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_0.high = vld1q_s32(bias_ptr + 4);
    acc_1.high = vld1q_s32(bias_ptr + 4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f0, filter.f1, filter.f2,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f0, filter.f1, filter.f2,
                                  input_2, input_3, input_4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f3, filter.f4, filter.f5,
                                  input_5, input_6, input_7);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f3, filter.f4, filter.f5,
                                  input_7, input_8, input_9);

    // Load next 2 rows.
    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_9 = vaddq_s16(input_9, input_offset_vec);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f6, filter.f7, filter.f8,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f6, filter.f7, filter.f8,
                                  input_2, input_3, input_4);

    DownquantizeAndStore2Output(
        acc_0, acc_1, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth);

    output_ptr += output_row_size;

    // Moving onto the next row of outputs.
    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_0.high = vld1q_s32(bias_ptr + 4);
    acc_1.high = vld1q_s32(bias_ptr + 4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f0, filter.f1, filter.f2,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f0, filter.f1, filter.f2,
                                  input_2, input_3, input_4);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f3, filter.f4, filter.f5,
                                  input_5, input_6, input_7);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f3, filter.f4, filter.f5,
                                  input_7, input_8, input_9);

    // Load last row.
    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    temp_3 = vld1_u8(ptr + 3 * input_depth);
    temp_4 = vld1_u8(ptr + 4 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);

    acc_0 = MultiplyAccumulateRow(acc_0, filter.f6, filter.f7, filter.f8,
                                  input_0, input_1, input_2);

    acc_1 = MultiplyAccumulateRow(acc_1, filter.f6, filter.f7, filter.f8,
                                  input_2, input_3, input_4);

    DownquantizeAndStore2Output(
        acc_0, acc_1, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 4, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    // Reuse 4x2 kernel twice.
    ConvKernel3x3FilterDepth8<4, 2, 2, 2>::Run(
        input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
        filter_offset, bias_ptr, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth,
        output_width);

    ConvKernel3x3FilterDepth8<4, 2, 2, 2>::Run(
        input_ptr + 4 * input_depth, input_depth, input_offset, input_row_size,
        filter_ptr, filter_offset, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_ptr + 2 * output_depth, output_depth, output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<4, 1, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    const int output_row_size = output_depth * output_width;

    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8;
    uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7,
        temp_8;

    const uint8* ptr = input_ptr;

    // Load all inputs for top output.
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Second output.
    output_ptr += output_row_size;

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);

    DotProductAndStore(
        filter, input_6, input_7, input_8, input_0, input_1, input_2, input_3,
        input_4, input_5, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Third output.
    output_ptr += output_row_size;

    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);

    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));
    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));

    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);

    DotProductAndStore(
        filter, input_3, input_4, input_5, input_6, input_7, input_8, input_0,
        input_1, input_2, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Fourth output.
    output_ptr += output_row_size;

    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);

    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 2, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    Int32x8 acc_0, acc_1, acc_2, acc_3;
    acc_0.low = vld1q_s32(bias_ptr);
    acc_1.low = vld1q_s32(bias_ptr);
    acc_2.low = vld1q_s32(bias_ptr);
    acc_3.low = vld1q_s32(bias_ptr);

    bias_ptr += 4;
    acc_0.high = vld1q_s32(bias_ptr);
    acc_1.high = vld1q_s32(bias_ptr);
    acc_2.high = vld1q_s32(bias_ptr);
    acc_3.high = vld1q_s32(bias_ptr);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);

    // Add scope for input registers to help the compiler know that it is
    // not needed.
    {
      // To process 2x2 outputs using a 3x3 filter at stride 2, we require
      // 5x5 inputs. We load the first 5x2 inputs at a time.
      int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
          input_7, input_8, input_9;

      const uint8* ptr = input_ptr;

      // Load inputs.
      {
        uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4;

        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);
        temp_4 = vld1_u8(ptr + 4 * input_depth);

        input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
        input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

        input_0 = vaddq_s16(input_0, input_offset_vec);
        input_1 = vaddq_s16(input_1, input_offset_vec);
        input_2 = vaddq_s16(input_2, input_offset_vec);
        input_3 = vaddq_s16(input_3, input_offset_vec);
        input_4 = vaddq_s16(input_4, input_offset_vec);

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);
        temp_4 = vld1_u8(ptr + 4 * input_depth);

        input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
        input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

        input_5 = vaddq_s16(input_5, input_offset_vec);
        input_6 = vaddq_s16(input_6, input_offset_vec);
        input_7 = vaddq_s16(input_7, input_offset_vec);
        input_8 = vaddq_s16(input_8, input_offset_vec);
        input_9 = vaddq_s16(input_9, input_offset_vec);
      }

      acc_0 = MultiplyAccumulateRow(acc_0, filter.f0, filter.f1, filter.f2,
                                    input_0, input_1, input_2);

      acc_1 = MultiplyAccumulateRow(acc_1, filter.f0, filter.f1, filter.f2,
                                    input_2, input_3, input_4);

      acc_0 = MultiplyAccumulateRow(acc_0, filter.f3, filter.f4, filter.f5,
                                    input_5, input_6, input_7);

      acc_1 = MultiplyAccumulateRow(acc_1, filter.f3, filter.f4, filter.f5,
                                    input_7, input_8, input_9);

      // Load next inputs.
      {
        uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4;

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);
        temp_4 = vld1_u8(ptr + 4 * input_depth);

        input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
        input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

        input_0 = vaddq_s16(input_0, input_offset_vec);
        input_1 = vaddq_s16(input_1, input_offset_vec);
        input_2 = vaddq_s16(input_2, input_offset_vec);
        input_3 = vaddq_s16(input_3, input_offset_vec);
        input_4 = vaddq_s16(input_4, input_offset_vec);

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);
        temp_4 = vld1_u8(ptr + 4 * input_depth);

        input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
        input_9 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

        input_5 = vaddq_s16(input_5, input_offset_vec);
        input_6 = vaddq_s16(input_6, input_offset_vec);
        input_7 = vaddq_s16(input_7, input_offset_vec);
        input_8 = vaddq_s16(input_8, input_offset_vec);
        input_9 = vaddq_s16(input_9, input_offset_vec);
      }

      acc_0 = MultiplyAccumulateRow(acc_0, filter.f6, filter.f7, filter.f8,
                                    input_0, input_1, input_2);

      acc_1 = MultiplyAccumulateRow(acc_1, filter.f6, filter.f7, filter.f8,
                                    input_2, input_3, input_4);

      // Moving onto the two bottom outputs.
      acc_2 = MultiplyAccumulateRow(acc_2, filter.f0, filter.f1, filter.f2,
                                    input_0, input_1, input_2);

      acc_3 = MultiplyAccumulateRow(acc_3, filter.f0, filter.f1, filter.f2,
                                    input_2, input_3, input_4);

      acc_2 = MultiplyAccumulateRow(acc_2, filter.f3, filter.f4, filter.f5,
                                    input_5, input_6, input_7);

      acc_3 = MultiplyAccumulateRow(acc_3, filter.f3, filter.f4, filter.f5,
                                    input_7, input_8, input_9);

      // Load last input row.
      {
        uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4;

        ptr += input_row_size;
        temp_0 = vld1_u8(ptr);
        temp_1 = vld1_u8(ptr + input_depth);
        temp_2 = vld1_u8(ptr + 2 * input_depth);
        temp_3 = vld1_u8(ptr + 3 * input_depth);
        temp_4 = vld1_u8(ptr + 4 * input_depth);

        input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
        input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
        input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
        input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
        input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));

        input_0 = vaddq_s16(input_0, input_offset_vec);
        input_1 = vaddq_s16(input_1, input_offset_vec);
        input_2 = vaddq_s16(input_2, input_offset_vec);
        input_3 = vaddq_s16(input_3, input_offset_vec);
        input_4 = vaddq_s16(input_4, input_offset_vec);
      }

      acc_2 = MultiplyAccumulateRow(acc_2, filter.f6, filter.f7, filter.f8,
                                    input_0, input_1, input_2);

      acc_3 = MultiplyAccumulateRow(acc_3, filter.f6, filter.f7, filter.f8,
                                    input_2, input_3, input_4);
    }

    DownquantizeAndStore2x2Output(acc_0, acc_1, acc_2, acc_3, output_offset,
                                  output_multiplier, output_shift,
                                  output_activation_min, output_activation_max,
                                  output_ptr, output_depth, output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 4, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    // Reuse 2x2 kernel twice.
    ConvKernel3x3FilterDepth8<2, 2, 2, 2>::Run(
        input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
        filter_offset, bias_ptr, output_offset, output_multiplier, output_shift,
        output_activation_min, output_activation_max, output_ptr, output_depth,
        output_width);

    ConvKernel3x3FilterDepth8<2, 2, 2, 2>::Run(
        input_ptr + 4 * input_depth, input_depth, input_offset, input_row_size,
        filter_ptr, filter_offset, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_ptr + 2 * output_depth, output_depth, output_width);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<2, 1, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    const int output_row_size = output_depth * output_width;

    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8;
    uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7,
        temp_8;

    const uint8* ptr = input_ptr;

    // Load all inputs for top output.
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Second output.
    output_ptr += output_row_size;

    ptr += input_row_size;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);

    DotProductAndStore(
        filter, input_6, input_7, input_8, input_0, input_1, input_2, input_3,
        input_4, input_5, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<1, 2, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8;
    uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7,
        temp_8;

    const uint8* ptr = input_ptr;

    // Load all inputs for top output.
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Second output.
    output_ptr += output_depth;

    ptr = input_ptr + 3 * input_depth;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);

    DotProductAndStore(
        filter, input_2, input_0, input_1, input_5, input_3, input_4, input_8,
        input_6, input_7, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);
  }
};

template <>
struct ConvKernel3x3FilterDepth8<1, 4, 2, 2> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8;
    uint8x8_t temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7,
        temp_8;

    const uint8* ptr = input_ptr;

    // Load all inputs for top output.
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    temp_2 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    temp_5 = vld1_u8(ptr + 2 * input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);
    temp_8 = vld1_u8(ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Second output.
    output_ptr += output_depth;

    ptr = input_ptr + 3 * input_depth;
    temp_0 = vld1_u8(ptr);
    temp_1 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_3 = vld1_u8(ptr);
    temp_4 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_6 = vld1_u8(ptr);
    temp_7 = vld1_u8(ptr + input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));

    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);

    DotProductAndStore(
        filter, input_2, input_0, input_1, input_5, input_3, input_4, input_8,
        input_6, input_7, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Third output.
    output_ptr += output_depth;

    ptr = input_ptr + 5 * input_depth;
    temp_2 = vld1_u8(ptr);
    temp_0 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_5 = vld1_u8(ptr);
    temp_3 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_8 = vld1_u8(ptr);
    temp_6 = vld1_u8(ptr + input_depth);

    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));

    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);

    DotProductAndStore(
        filter, input_1, input_2, input_0, input_4, input_5, input_3, input_7,
        input_8, input_6, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);

    // Fourth output.
    output_ptr += output_depth;

    ptr = input_ptr + 7 * input_depth;
    temp_1 = vld1_u8(ptr);
    temp_2 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_4 = vld1_u8(ptr);
    temp_5 = vld1_u8(ptr + input_depth);
    ptr += input_row_size;
    temp_7 = vld1_u8(ptr);
    temp_8 = vld1_u8(ptr + input_depth);

    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);
  }
};

template <int kFixedStrideWidth, int kFixedStrideHeight>
struct ConvKernel3x3FilterDepth8<1, 1, kFixedStrideWidth, kFixedStrideHeight> {
  static inline void Run(const uint8* input_ptr, int input_depth,
                         int32 input_offset, int input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int output_depth, int output_width) {
    Filter3x3x8 filter = Load3x3Filter(filter_ptr, filter_offset, output_depth);

    int16x8_t input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8;

    uint8x8_t temp_0 = vld1_u8(input_ptr);
    uint8x8_t temp_1 = vld1_u8(input_ptr + input_depth);
    uint8x8_t temp_2 = vld1_u8(input_ptr + 2 * input_depth);

    input_ptr += input_row_size;
    uint8x8_t temp_3 = vld1_u8(input_ptr);
    uint8x8_t temp_4 = vld1_u8(input_ptr + input_depth);
    uint8x8_t temp_5 = vld1_u8(input_ptr + 2 * input_depth);

    input_ptr += input_row_size;
    uint8x8_t temp_6 = vld1_u8(input_ptr);
    uint8x8_t temp_7 = vld1_u8(input_ptr + input_depth);
    uint8x8_t temp_8 = vld1_u8(input_ptr + 2 * input_depth);

    input_0 = vreinterpretq_s16_u16(vmovl_u8(temp_0));
    input_1 = vreinterpretq_s16_u16(vmovl_u8(temp_1));
    input_2 = vreinterpretq_s16_u16(vmovl_u8(temp_2));
    input_3 = vreinterpretq_s16_u16(vmovl_u8(temp_3));
    input_4 = vreinterpretq_s16_u16(vmovl_u8(temp_4));
    input_5 = vreinterpretq_s16_u16(vmovl_u8(temp_5));
    input_6 = vreinterpretq_s16_u16(vmovl_u8(temp_6));
    input_7 = vreinterpretq_s16_u16(vmovl_u8(temp_7));
    input_8 = vreinterpretq_s16_u16(vmovl_u8(temp_8));

    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    input_0 = vaddq_s16(input_0, input_offset_vec);
    input_1 = vaddq_s16(input_1, input_offset_vec);
    input_2 = vaddq_s16(input_2, input_offset_vec);
    input_3 = vaddq_s16(input_3, input_offset_vec);
    input_4 = vaddq_s16(input_4, input_offset_vec);
    input_5 = vaddq_s16(input_5, input_offset_vec);
    input_6 = vaddq_s16(input_6, input_offset_vec);
    input_7 = vaddq_s16(input_7, input_offset_vec);
    input_8 = vaddq_s16(input_8, input_offset_vec);

    DotProductAndStore(
        filter, input_0, input_1, input_2, input_3, input_4, input_5, input_6,
        input_7, input_8, bias_ptr, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_ptr);
  }
};

inline void ShuffleInput(const uint8* input_ptr, int input_depth,
                         int input_width, int input_height, int output_depth,
                         int output_width, int output_height,
                         uint8* output_ptr) {
  const int input_row_size = input_depth * input_width;

  for (int y = 0; y < output_height; y++) {
    const uint8* ptr = input_ptr;
    for (int x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

template <int kFixedHeight, int kFixedStrideWidth, int kFixedStrideHeight>
struct ConvRow3x3FilterDepth8 {};

template <int kFixedStrideWidth, int kFixedStrideHeight>
struct ConvRow3x3FilterDepth8<1, kFixedStrideWidth, kFixedStrideHeight> {
  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    int out_x = start_x;

    // 1x4 at a time.
    for (; out_x <= output_width - 4; out_x += 4) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<1, 4, kFixedStrideWidth, kFixedStrideHeight>::
            Run(input_ptr, input_depth, input_offset, input_row_size,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 4 * kFixedStrideWidth * input_depth;
      output_data += 4 * output_depth;
    }

    // 1x1 at a time.
    for (; out_x < output_width; out_x++) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<1, 1, kFixedStrideWidth, kFixedStrideHeight>::
            Run(input_ptr, input_depth, input_offset, input_row_size,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += kFixedStrideWidth * input_depth;
      output_data += output_depth;
    }
  }
};

template <int kFixedStrideWidth, int kFixedStrideHeight>
struct ConvRow3x3FilterDepth8<2, kFixedStrideWidth, kFixedStrideHeight> {
  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    int out_x = start_x;

    // 2x4 at a time.
    for (; out_x <= output_width - 4; out_x += 4) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<2, 4, kFixedStrideWidth, kFixedStrideHeight>::
            Run(input_ptr, input_depth, input_offset, input_row_size,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 4 * kFixedStrideWidth * input_depth;
      output_data += 4 * output_depth;
    }

    // 2x2 at a time.
    for (; out_x <= output_width - 2; out_x += 2) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<2, 2, kFixedStrideWidth, kFixedStrideHeight>::
            Run(input_ptr, input_depth, input_offset, input_row_size,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 2 * kFixedStrideWidth * input_depth;
      output_data += 2 * output_depth;
    }

    // 2x1 at a time.
    for (; out_x < output_width; out_x++) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<2, 1, kFixedStrideWidth, kFixedStrideHeight>::
            Run(input_ptr, input_depth, input_offset, input_row_size,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += kFixedStrideWidth * input_depth;
      output_data += output_depth;
    }
  }
};

template <>
struct ConvRow3x3FilterDepth8<4, 1, 1> {
  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    int out_x = start_x;

    // 4x4 at a time.
    for (; out_x <= output_width - 4; out_x += 4) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 4, 1, 1>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 4 * input_depth;
      output_data += 4 * output_depth;
    }

    // Handle the rest of the right side.
    // 4x2 at a time.
    for (; out_x <= output_width - 2; out_x += 2) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 2, 1, 1>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 2 * input_depth;
      output_data += 2 * output_depth;
    }

    // 4x1 at a time.
    for (; out_x < output_width; out_x++) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 1, 1, 1>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += input_depth;
      output_data += output_depth;
    }
  }
};

template <>
struct ConvRow3x3FilterDepth8<4, 2, 2> {
  // The buffer size of the shuffled input.
  static inline constexpr int ShuffleWorkspaceSize() { return 64 * 9 * 9; }

  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    // Branch and cache misses increase substantially with stride 2 kernels.
    // Adding prefetching reduces latency by as much as 2x.
    const int i0 = 0;
    const int i1 = input_depth;
    const int i2 = 2 * input_depth;
    const int i3 = 3 * input_depth;
    const int i4 = 4 * input_depth;
    const int i5 = 5 * input_depth;
    const int i6 = 6 * input_depth;
    const int i7 = 7 * input_depth;
    const int i8 = 8 * input_depth;

#define DEPTHWISECONV_PRELOAD_ROW(input_ptr, i)         \
  preload_l1_keep(input_ptr + i * input_row_size + i0); \
  preload_l1_keep(input_ptr + i * input_row_size + i1); \
  preload_l1_keep(input_ptr + i * input_row_size + i2); \
  preload_l1_keep(input_ptr + i * input_row_size + i3); \
  preload_l1_keep(input_ptr + i * input_row_size + i4); \
  preload_l1_keep(input_ptr + i * input_row_size + i5); \
  preload_l1_keep(input_ptr + i * input_row_size + i6); \
  preload_l1_keep(input_ptr + i * input_row_size + i7); \
  preload_l1_keep(input_ptr + i * input_row_size + i8);

    int out_x = start_x;
    // 4x4 at a time.
    for (; out_x <= output_width - 4; out_x += 4) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      int depth = 0;
      for (; depth <= output_depth - 64; depth += 64) {
        // Preload 9x9 input.
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 0);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 1);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 2);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 3);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 4);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 5);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 6);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 7);
        DEPTHWISECONV_PRELOAD_ROW(input_ptr, 8);

        // For a large input window (64x9x9) that is small enough to fit in L1
        // cache, copy the input into a separate buffer and run the kernel on
        // this new buffer. This reduces the likelihood of cache misses when
        // the kernel is loading input data. If this size is ever changed,
        // update the ShuffleWorkspaceSize() function to return the new size.
        ShuffleInput(input_ptr, input_depth, input_width, input_height, 64, 9,
                     9, shuffle_workspace);
        const uint8* shuffled_ptr = &shuffle_workspace[0];

        for (int micro_depth = 0; micro_depth <= 64 - 8; micro_depth += 8) {
          ConvKernel3x3FilterDepth8<4, 4, 2, 2>::Run(
              shuffled_ptr, 64, input_offset, 64 * 9, filter_ptr, filter_offset,
              bias_ptr, output_offset, output_multiplier, output_shift,
              output_activation_min, output_activation_max, output_ptr,
              output_depth, output_width);

          shuffled_ptr += 8;
          output_ptr += 8;
          filter_ptr += 8;
          bias_ptr += 8;
        }
        input_ptr += 64;
      }

      // Preload 9x9 input one more time for the rest of the depth.
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 0);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 1);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 2);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 3);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 4);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 5);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 6);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 7);
      DEPTHWISECONV_PRELOAD_ROW(input_ptr, 8);

      for (; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 4, 2, 2>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 4 * 2 * input_depth;
      output_data += 4 * output_depth;
    }

#undef DEPTHWISECONV_PRELOAD_ROW

    // Handle the rest of the right side.
    // 4x2 at a time.
    for (; out_x <= output_width - 2; out_x += 2) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 2, 2, 2>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 2 * 2 * input_depth;
      output_data += 2 * output_depth;
    }

    // 4x1 at a time.
    for (; out_x < output_width; out_x++) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int depth = 0; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<4, 1, 2, 2>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 2 * input_depth;
      output_data += output_depth;
    }
  }
};

template <>
struct ConvRow3x3FilterDepth8<8, 2, 2> {
  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    // Reuse 4 row kernels twice.
    ConvRow3x3FilterDepth8<4, 2, 2>::Run(
        input_data, start_x, start_y, input_depth, input_width, input_height,
        input_row_size, input_offset, filter_data, filter_offset, bias_data,
        output_offset, output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_data, output_depth, output_width,
        shuffle_workspace);

    ConvRow3x3FilterDepth8<4, 2, 2>::Run(
        input_data + 2 * 4 * input_row_size, start_x, start_y + 4, input_depth,
        input_width, input_height, input_row_size, input_offset, filter_data,
        filter_offset, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_data + 4 * output_depth * output_width, output_depth,
        output_width, shuffle_workspace);
  }
};

template <>
struct ConvRow3x3FilterDepth8<8, 1, 1> {
  // The buffer size of the shuffled input.
  static inline constexpr int ShuffleWorkspaceSize() { return 64 * 10 * 10; }

  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int input_depth, int input_width, int input_height,
                         int input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int output_depth, int output_width,
                         uint8* shuffle_workspace) {
    int out_x = start_x;
    // 8x8 at a time.
    for (; out_x <= output_width - 8; out_x += 8) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;

      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      int depth = 0;
      for (; depth <= output_depth - 64; depth += 64) {
        // For a large input window (64x10x10) that is small enough to fit in L1
        // cache, copy the input into a separate buffer and run the kernel on
        // this new buffer. This reduces the likelihood of cache misses when
        // the kernel is loading input data. If the size of the input window
        // changes, update the function ShuffleWorkspaceSize() with the new
        // size.
        ShuffleInput(input_ptr, input_depth, input_width, input_height, 64, 10,
                     10, shuffle_workspace);
        const uint8* shuffled_ptr = shuffle_workspace;

        for (int micro_depth = 0; micro_depth <= 64 - 8; micro_depth += 8) {
          ConvKernel3x3FilterDepth8<8, 8, 1, 1>::Run(
              shuffled_ptr, 64, input_offset, 64 * 10, filter_ptr,
              filter_offset, bias_ptr, output_offset, output_multiplier,
              output_shift, output_activation_min, output_activation_max,
              output_ptr, output_depth, output_width);

          shuffled_ptr += 8;
          output_ptr += 8;
          filter_ptr += 8;
          bias_ptr += 8;
        }
        input_ptr += 64;
      }

      for (; depth <= output_depth - 8; depth += 8) {
        ConvKernel3x3FilterDepth8<8, 8, 1, 1>::Run(
            input_ptr, input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }

      input_data += 8 * input_depth;
      output_data += 8 * output_depth;
    }

    // Handle the rest of the right side by re-using 4 row kernels twice.
    ConvRow3x3FilterDepth8<4, 1, 1>::Run(
        input_data, out_x, start_y, input_depth, input_width, input_height,
        input_row_size, input_offset, filter_data, filter_offset, bias_data,
        output_offset, output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_data, output_depth, output_width,
        shuffle_workspace);

    ConvRow3x3FilterDepth8<4, 1, 1>::Run(
        input_data + 4 * input_row_size, out_x, start_y + 4, input_depth,
        input_width, input_height, input_row_size, input_offset, filter_data,
        filter_offset, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_data + 4 * output_depth * output_width, output_depth,
        output_width, shuffle_workspace);
  }
};

inline bool Fast3x3FilterKernelSupported(const Dims<4>& input_dims,
                                         const Dims<4>& filter_dims,
                                         int stride_width, int stride_height,
                                         int pad_width, int pad_height,
                                         int depth_multiplier,
                                         const Dims<4>& output_dims) {
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  bool supported = filter_width == 3 && filter_height == 3 &&
                   depth_multiplier == 1 &&
                   (stride_width == 1 || stride_width == 2) &&
                   (stride_height == 1 || stride_height == 2) &&
                   (stride_width == stride_height) && pad_width == 0 &&
                   pad_height == 0 && (input_depth % 8) == 0;

  if (!supported) {
    return false;
  }

  // Handle case where padding is zero but padding type is not kValid.
  // This would require special boundary case handling that is not supported.

  const int out_x = output_width - 1;
  const int out_y = output_height - 1;

  const int in_x_origin = (out_x * stride_width) - pad_width;
  const int in_y_origin = (out_y * stride_height) - pad_height;

  const int in_x_end = in_x_origin + filter_width;
  const int in_y_end = in_y_origin + filter_height;

  // Supported only if filter on the right and bottom boundary lies completely
  // within the input.
  return in_x_end <= input_width && in_y_end <= input_height;
}

inline void DepthwiseConv3x3Filter(
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
  // of 1, 3x3 filter, no padding and strides 1 and 2.
  TFLITE_DCHECK(output_depth == input_depth * depth_multiplier);
  TFLITE_DCHECK(depth_multiplier == 1);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 3);
  TFLITE_DCHECK(pad_height == 0);
  TFLITE_DCHECK(pad_width == 0);
  TFLITE_DCHECK(stride_height == 1 || stride_height == 2);
  TFLITE_DCHECK(stride_width == 1 || stride_width == 2);
  TFLITE_DCHECK(stride_width == stride_height);

  const int input_row_size = input_depth * (input_width + 2 * pad_width);
  const int output_row_size = output_depth * output_width;
  const int input_batch_size = input_row_size * (input_height + 2 * pad_height);
  const int output_batch_size = output_depth * output_width * output_height;

  using conv_row_func_t = decltype(&ConvRow3x3FilterDepth8<1, 1, 1>::Run);
  conv_row_func_t conv_1_output_row = ConvRow3x3FilterDepth8<1, 1, 1>::Run;
  conv_row_func_t conv_2_output_rows = ConvRow3x3FilterDepth8<2, 1, 1>::Run;
  conv_row_func_t conv_4_output_rows = ConvRow3x3FilterDepth8<4, 1, 1>::Run;
  conv_row_func_t conv_8_output_rows = ConvRow3x3FilterDepth8<8, 1, 1>::Run;

  if (stride_width == 2) {
    conv_1_output_row = ConvRow3x3FilterDepth8<1, 2, 2>::Run;
    conv_2_output_rows = ConvRow3x3FilterDepth8<2, 2, 2>::Run;
    conv_4_output_rows = ConvRow3x3FilterDepth8<4, 2, 2>::Run;
    conv_8_output_rows = ConvRow3x3FilterDepth8<8, 2, 2>::Run;
  }

  // Allocate maximum memory needed for shuffled input.
  // TODO(mariewhite): The size of this workspace is small enough to be
  // allocated on the stack. Eventually we will want to move it to the heap
  // and have it allocated outside of this function, like the im2col_array used
  // in gemmlowp.
#define DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE 10 * 10 * 64
  uint8 shuffle_workspace[DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE];

  // Make sure the kernels using this buffer will not run out of bounds.
  static_assert(ConvRow3x3FilterDepth8<8, 1, 1>::ShuffleWorkspaceSize() <=
                    DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE,
                "Shuffle workspace size is too small.");
  static_assert(ConvRow3x3FilterDepth8<4, 2, 2>::ShuffleWorkspaceSize() <=
                    DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE,
                "Shuffle workspace size is too small.");

#undef DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE

  for (int b = 0; b < batches; ++b) {
    const uint8* input_ptr = input_data + b * input_batch_size;
    uint8* output_ptr = output_data + b * output_batch_size;

    int out_y = 0;

    // Handle 8 rows at a time.
    for (; out_y <= output_height - 8; out_y += 8) {
      conv_8_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                         input_height, input_row_size, input_offset,
                         filter_data, filter_offset, bias_data, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max, output_ptr, output_depth,
                         output_width, shuffle_workspace);

      input_ptr += 8 * stride_height * input_row_size;
      output_ptr += 8 * output_row_size;
    }

    // Handle 4 rows at a time.
    for (; out_y <= output_height - 4; out_y += 4) {
      conv_4_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                         input_height, input_row_size, input_offset,
                         filter_data, filter_offset, bias_data, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max, output_ptr, output_depth,
                         output_width, shuffle_workspace);

      input_ptr += 4 * stride_height * input_row_size;
      output_ptr += 4 * output_row_size;
    }

    // Handle 2 rows at a time.
    for (; out_y <= output_height - 2; out_y += 2) {
      conv_2_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                         input_height, input_row_size, input_offset,
                         filter_data, filter_offset, bias_data, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max, output_ptr, output_depth,
                         output_width, shuffle_workspace);

      input_ptr += 2 * stride_height * input_row_size;
      output_ptr += 2 * output_row_size;
    }

    // Handle one row at a time.
    for (; out_y < output_height; out_y++) {
      conv_1_output_row(input_ptr, 0, out_y, input_depth, input_width,
                        input_height, input_row_size, input_offset, filter_data,
                        filter_offset, bias_data, output_offset,
                        output_multiplier, output_shift, output_activation_min,
                        output_activation_max, output_ptr, output_depth,
                        output_width, shuffle_workspace);

      input_ptr += stride_height * input_row_size;
      output_ptr += output_row_size;
    }
  }
}

#endif  // __aarch64__

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

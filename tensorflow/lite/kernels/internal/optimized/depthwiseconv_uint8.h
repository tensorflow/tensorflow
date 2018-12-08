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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_H_

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Implementation of quantized DepthwiseConv

template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
struct QuantizedDepthwiseConvKernel {};

#ifdef USE_NEON
template <>
struct QuantizedDepthwiseConvKernel<true, 8, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8x2_t filter_u8;
    filter_u8.val[0] = vld1_u8(filter_ptr);
    filter_u8.val[1] = vld1_u8(filter_ptr + 8);
    int16x8_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(filter_u8.val[i])),
                            vdupq_n_s16(filter_offset));
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4x2_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i].val[0] = vld1q_s32(acc_buffer_ptr + 4 * i);
        acc[i].val[1] = vld1q_s32(acc_buffer_ptr + 4 * i + 8);
      }
      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += input_ptr_increment;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x8x2_t input_dup2 = vzipq_s16(input, input);
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[0].val[i] = vmlal_s16(acc[0].val[i], vget_low_s16(filter[i]),
                                  vget_low_s16(input_dup2.val[i]));
        acc[1].val[i] = vmlal_s16(acc[1].val[i], vget_high_s16(filter[i]),
                                  vget_high_s16(input_dup2.val[i]));
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i].val[0]);
        vst1q_s32(acc_buffer_ptr + 4 * i + 8, acc[i].val[1]);
      }
      acc_buffer_ptr += 16;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 8, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    const uint8x8_t filter_u8 = vld1_u8(filter_ptr);
    const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
    const int16x8_t filter = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));

    int outp = 0;
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      uint8x8_t input_u8[2];
      for (int i = 0; i < 2; i++) {
        input_u8[i] = vld1_u8(input_ptr + 8 * i);
      }
      input_ptr += 16;
      int16x8_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vreinterpretq_s16_u16(vmovl_u8(input_u8[i]));
      }
      for (int i = 0; i < 2; i++) {
        input[i] = vaddq_s16(input[i], vdupq_n_s16(input_offset));
      }
      // Multiply-accumulate.
      acc[0] = vmlal_s16(acc[0], vget_low_s16(filter), vget_low_s16(input[0]));
      acc[1] =
          vmlal_s16(acc[1], vget_high_s16(filter), vget_high_s16(input[0]));
      acc[2] = vmlal_s16(acc[2], vget_low_s16(filter), vget_low_s16(input[1]));
      acc[3] =
          vmlal_s16(acc[3], vget_high_s16(filter), vget_high_s16(input[1]));
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle 1 output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[2];
      acc[0] = vld1q_s32(acc_buffer_ptr);
      acc[1] = vld1q_s32(acc_buffer_ptr + 4);

      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Multiply-accumulate.
      acc[0] = vmlal_s16(acc[0], vget_low_s16(filter), vget_low_s16(input));
      acc[1] = vmlal_s16(acc[1], vget_high_s16(filter), vget_high_s16(input));
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr, acc[0]);
      vst1q_s32(acc_buffer_ptr + 4, acc[1]);
      acc_buffer_ptr += 8;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 4, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    const uint8x8_t filter_u8 = vld1_u8(filter_ptr);
    const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
    const int16x8_t filter = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));

    int outp = 0;
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x8x2_t input_dup2 = vzipq_s16(input, input);
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[2 * i + 0] = vmlal_s16(acc[2 * i + 0], vget_low_s16(filter),
                                   vget_low_s16(input_dup2.val[i]));
        acc[2 * i + 1] = vmlal_s16(acc[2 * i + 1], vget_high_s16(filter),
                                   vget_high_s16(input_dup2.val[i]));
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x4x2_t input_dup2 = vzip_s16(input, input);
      // Multiply-accumulate
      acc[0] = vmlal_s16(acc[0], vget_low_s16(filter), input_dup2.val[0]);
      acc[1] = vmlal_s16(acc[1], vget_high_s16(filter), input_dup2.val[1]);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 2, 8> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    int16x8_t filter[2];
    for (int i = 0; i < 2; i++) {
      const uint8x8_t filter_u8 = vld1_u8(filter_ptr + 8 * i);
      const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
      filter[i] = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
    }
    int outp = 0;
    // Handle two output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[8];
      for (int i = 0; i < 8; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
      // Multiply-accumulate.
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(filter[0]), input, 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(filter[0]), input, 0);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(filter[1]), input, 1);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(filter[1]), input, 1);
      acc[4] = vmlal_lane_s16(acc[4], vget_low_s16(filter[0]), input, 2);
      acc[5] = vmlal_lane_s16(acc[5], vget_high_s16(filter[0]), input, 2);
      acc[6] = vmlal_lane_s16(acc[6], vget_low_s16(filter[1]), input, 3);
      acc[7] = vmlal_lane_s16(acc[7], vget_high_s16(filter[1]), input, 3);
      // Store the accumulators back to acc_buffer.
      for (int i = 0; i < 8; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 32;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_ptr += 2;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate.
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(filter[0]), input, 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(filter[0]), input, 0);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(filter[1]), input, 1);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(filter[1]), input, 1);

      // Store the accumulators back to acc_buffer.
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 2, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[2], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[3], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;
    // Handle 4 output pixels at a time.
    for (; outp <= num_output_pixels - 4; outp += 4) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x8x2_t input_dup2 = vzipq_s16(input, input);
      // Multiply-accumulate
      acc[0] = vmlal_s16(acc[0], filter, vget_low_s16(input_dup2.val[0]));
      acc[1] = vmlal_s16(acc[1], filter, vget_high_s16(input_dup2.val[0]));
      acc[2] = vmlal_s16(acc[2], filter, vget_low_s16(input_dup2.val[1]));
      acc[3] = vmlal_s16(acc[3], filter, vget_high_s16(input_dup2.val[1]));
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc = vld1q_s32(acc_buffer_ptr);

      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_ptr += 2;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x4_t input_dup2 = vzip_s16(input, input).val[0];
      // Multiply-accumulate
      acc = vmlal_s16(acc, filter, input_dup2);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 2, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;
    // Handle 8 output pixels at a time.
    for (; outp <= num_output_pixels - 8; outp += 8) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      uint8x8_t input_u8[2];
      for (int i = 0; i < 2; i++) {
        input_u8[i] = vld1_u8(input_ptr + 8 * i);
      }
      input_ptr += 16;
      int16x8_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vreinterpretq_s16_u16(vmovl_u8(input_u8[i]));
      }
      for (int i = 0; i < 2; i++) {
        input[i] = vaddq_s16(input[i], vdupq_n_s16(input_offset));
      }

      // Multiply-accumulate.
      acc[0] = vmlal_s16(acc[0], filter, vget_low_s16(input[0]));
      acc[1] = vmlal_s16(acc[1], filter, vget_high_s16(input[0]));
      acc[2] = vmlal_s16(acc[2], filter, vget_low_s16(input[1]));
      acc[3] = vmlal_s16(acc[3], filter, vget_high_s16(input[1]));
      // Store the accumulators back to acc_buffer.
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle 4 output pixels at a time.
    for (; outp <= num_output_pixels - 4; outp += 4) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));

      // Multiply-accumulate.
      acc[0] = vmlal_s16(acc[0], filter, vget_low_s16(input));
      acc[1] = vmlal_s16(acc[1], filter, vget_high_s16(input));
      // Store the accumulators back to acc_buffer.
      for (int i = 0; i < 2; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc = vld1q_s32(acc_buffer_ptr);
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate.
      acc = vmlal_s16(acc, filter, input);
      // Store the accumulators back to acc_buffer.
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
    // Handle 1 output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer.
      int32x2_t acc = vld1_s32(acc_buffer_ptr);
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_ptr += 2;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate.
      acc = vget_low_s32(vmlal_s16(vcombine_s32(acc, acc), filter, input));
      // Store the accumulators back to acc_buffer.
      vst1_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 2;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 1, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;
    // Handle 8 output pixels at a time.
    for (; outp <= num_output_pixels - 8; outp += 8) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Duplicate the input values, 2-fold
      const int16x8x2_t input_dup2 = vzipq_s16(input, input);
      // Multiply-accumulate
      acc[0] = vmlal_s16(acc[0], filter, vget_low_s16(input_dup2.val[0]));
      acc[1] = vmlal_s16(acc[1], filter, vget_high_s16(input_dup2.val[0]));
      acc[2] = vmlal_s16(acc[2], filter, vget_low_s16(input_dup2.val[1]));
      acc[3] = vmlal_s16(acc[3], filter, vget_high_s16(input_dup2.val[1]));
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x2_t acc = vld1_s32(acc_buffer_ptr);

      // Load the inputs, add input_offset.
      const uint32 input = *input_ptr++ + input_offset;

      // Multiply-accumulate
      acc = vget_low_s32(vmlal_n_s16(vcombine_s32(acc, acc), filter, input));
      // Store the accumulators back to acc_buffer
      vst1_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 2;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 1, 4> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[2], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[3], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;
    // Handle 8 output pixels at a time.
    for (; outp <= num_output_pixels - 8; outp += 8) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[8];
      for (int i = 0; i < 8; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));

      // Multiply-accumulate
      acc[0] = vmlal_lane_s16(acc[0], filter, vget_low_s16(input), 0);
      acc[1] = vmlal_lane_s16(acc[1], filter, vget_low_s16(input), 1);
      acc[2] = vmlal_lane_s16(acc[2], filter, vget_low_s16(input), 2);
      acc[3] = vmlal_lane_s16(acc[3], filter, vget_low_s16(input), 3);
      acc[4] = vmlal_lane_s16(acc[4], filter, vget_high_s16(input), 0);
      acc[5] = vmlal_lane_s16(acc[5], filter, vget_high_s16(input), 1);
      acc[6] = vmlal_lane_s16(acc[6], filter, vget_high_s16(input), 2);
      acc[7] = vmlal_lane_s16(acc[7], filter, vget_high_s16(input), 3);

      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 8; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 32;
    }
    // Handle 4 output pixels at a time.
    for (; outp <= num_output_pixels - 4; outp += 4) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate
      acc[0] = vmlal_lane_s16(acc[0], filter, input, 0);
      acc[1] = vmlal_lane_s16(acc[1], filter, input, 1);
      acc[2] = vmlal_lane_s16(acc[2], filter, input, 2);
      acc[3] = vmlal_lane_s16(acc[3], filter, input, 3);

      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc = vld1q_s32(acc_buffer_ptr);

      // Load the inputs, add input_offset.
      const uint32 input = *input_ptr++ + input_offset;

      // Multiply-accumulate
      acc = vmlal_n_s16(acc, filter, input);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 4, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[2], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[3], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;
    // Handle 4 output pixels at a time.
    for (; outp <= num_output_pixels - 4; outp += 4) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Load the inputs, add input_offset.
      int16x8_t input[2];
      for (int i = 0; i < 2; i++) {
        const uint8x8_t input_u8 = vld1_u8(input_ptr + 8 * i);
        const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
        input[i] = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      }
      input_ptr += 16;
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[2 * i + 0] =
            vmlal_s16(acc[2 * i + 0], filter, vget_low_s16(input[i]));
        acc[2 * i + 1] =
            vmlal_s16(acc[2 * i + 1], filter, vget_high_s16(input[i]));
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc;
      acc = vld1q_s32(acc_buffer_ptr);

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
      // Multiply-accumulate
      acc = vmlal_s16(acc, filter, input);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 4, 4> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    int16x8_t filter[2];
    for (int i = 0; i < 2; i++) {
      const uint8x8_t filter_u8 = vld1_u8(filter_ptr + 8 * i);
      const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
      filter[i] = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
    }

    int outp = 0;
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[8];
      for (int i = 0; i < 8; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += 8;
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));

      // Multiply-accumulate
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(filter[0]),
                              vget_low_s16(input), 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(filter[0]),
                              vget_low_s16(input), 1);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(filter[1]),
                              vget_low_s16(input), 2);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(filter[1]),
                              vget_low_s16(input), 3);
      acc[4] = vmlal_lane_s16(acc[4], vget_low_s16(filter[0]),
                              vget_high_s16(input), 0);
      acc[5] = vmlal_lane_s16(acc[5], vget_high_s16(filter[0]),
                              vget_high_s16(input), 1);
      acc[6] = vmlal_lane_s16(acc[6], vget_low_s16(filter[1]),
                              vget_high_s16(input), 2);
      acc[7] = vmlal_lane_s16(acc[7], vget_high_s16(filter[1]),
                              vget_high_s16(input), 3);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 8; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 32;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
      input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
      input_ptr += 4;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(filter[0]), input, 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(filter[0]), input, 1);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(filter[1]), input, 2);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(filter[1]), input, 3);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 0, 3> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // We will have to duplicate bytes in a NEON register, 3-fold.
    // We will do that by register-level table-look-up using VTBL instructions.
    // Here we prepare the registers containing the table-lookup indices.
    static const uint8 dup3_indices_array[3][8] = {{0, 0, 0, 1, 1, 1, 2, 2},
                                                   {2, 3, 3, 3, 4, 4, 4, 5},
                                                   {5, 5, 6, 6, 6, 7, 7, 7}};
    uint8x8_t dup3_indices[3];
    for (int i = 0; i < 3; i++) {
      dup3_indices[i] = vld1_u8(dup3_indices_array[i]);
    }

    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const uint8* local_filter_ptr = filter_ptr;
      const uint8* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 8 input channels at a time.
      for (; ic <= input_depth - 8; ic += 8) {
        // Load the filters, add filter_offset.
        int16x8_t filter[3];
        uint8x8x3_t filter_u8;
        filter_u8.val[0] = vld1_u8(local_filter_ptr);
        filter_u8.val[1] = vld1_u8(local_filter_ptr + 8);
        filter_u8.val[2] = vld1_u8(local_filter_ptr + 16);
        local_filter_ptr += 24;
        for (int i = 0; i < 3; i++) {
          const int16x8_t filter_s16 =
              vreinterpretq_s16_u16(vmovl_u8(filter_u8.val[i]));
          filter[i] = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
        }
        // Load the inputs, duplicate 3-fold, add input_offset.
        const uint8x8_t input_u8 = vld1_u8(local_input_ptr);
        local_input_ptr += 8;

        uint8x8_t input_u8_dup3[3];
        for (int i = 0; i < 3; i++) {
          input_u8_dup3[i] = vtbl1_u8(input_u8, dup3_indices[i]);
        }
        int16x8_t input_dup3[3];
        for (int i = 0; i < 3; i++) {
          const int16x8_t input_s16_dup3 =
              vreinterpretq_s16_u16(vmovl_u8(input_u8_dup3[i]));
          input_dup3[i] = vaddq_s16(input_s16_dup3, vdupq_n_s16(input_offset));
        }
        // Load the accumulators from acc_buffer
        int32x4x3_t acc[2];
        for (int i = 0; i < 2; i++) {
          acc[i].val[0] = vld1q_s32(acc_buffer_ptr + 4 * i);
          acc[i].val[1] = vld1q_s32(acc_buffer_ptr + 4 * i + 8);
          acc[i].val[2] = vld1q_s32(acc_buffer_ptr + 4 * i + 16);
        }
        // Multiply-accumulate
        for (int j = 0; j < 3; j++) {
          acc[0].val[j] = vmlal_s16(acc[0].val[j], vget_low_s16(input_dup3[j]),
                                    vget_low_s16(filter[j]));
          acc[1].val[j] = vmlal_s16(acc[1].val[j], vget_high_s16(input_dup3[j]),
                                    vget_high_s16(filter[j]));
        }
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 2; i++) {
          vst1q_s32(acc_buffer_ptr + 4 * i, acc[i].val[0]);
          vst1q_s32(acc_buffer_ptr + 4 * i + 8, acc[i].val[1]);
          vst1q_s32(acc_buffer_ptr + 4 * i + 16, acc[i].val[2]);
        }
        acc_buffer_ptr += 24;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        const int16 input_val = *local_input_ptr++ + input_offset;
        for (int i = 0; i < 3; i++) {
          const int16 filter_val = local_filter_ptr[i] + filter_offset;
          *acc_buffer_ptr++ += static_cast<int32>(filter_val) * input_val;
        }
        local_filter_ptr += 3;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 0, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const uint8* local_filter_ptr = filter_ptr;
      const uint8* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 8 input channels at a time.
      for (; ic <= input_depth - 8; ic += 8) {
        // Load the filters, add filter_offset.
        int16x8_t filter[2];
        uint8x8x2_t filter_u8;
        filter_u8.val[0] = vld1_u8(local_filter_ptr);
        filter_u8.val[1] = vld1_u8(local_filter_ptr + 8);
        local_filter_ptr += 16;
        for (int i = 0; i < 2; i++) {
          const int16x8_t filter_s16 =
              vreinterpretq_s16_u16(vmovl_u8(filter_u8.val[i]));
          filter[i] = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
        }
        // Load the inputs, add input_offset, duplicate 2-fold.
        const uint8x8_t input_u8 = vld1_u8(local_input_ptr);
        local_input_ptr += 8;
        const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
        const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
        const int16x8x2_t input_dup2 = vzipq_s16(input, input);
        // Load the accumulators from acc_buffer.
        int32x4x2_t acc[2];
        for (int i = 0; i < 2; i++) {
          acc[i].val[0] = vld1q_s32(acc_buffer_ptr + 4 * i);
          acc[i].val[1] = vld1q_s32(acc_buffer_ptr + 4 * i + 8);
        }
        // Multiply-accumulate.
        for (int j = 0; j < 2; j++) {
          acc[0].val[j] = vmlal_s16(acc[0].val[j], vget_low_s16(filter[j]),
                                    vget_low_s16(input_dup2.val[j]));
          acc[1].val[j] = vmlal_s16(acc[1].val[j], vget_high_s16(filter[j]),
                                    vget_high_s16(input_dup2.val[j]));
        }
        // Store the accumulators back to acc_buffer.
        for (int i = 0; i < 2; i++) {
          vst1q_s32(acc_buffer_ptr + 4 * i, acc[i].val[0]);
          vst1q_s32(acc_buffer_ptr + 4 * i + 8, acc[i].val[1]);
        }
        acc_buffer_ptr += 16;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        // Load the inputs.
        const int16 input_val = *local_input_ptr++ + input_offset;
        for (int i = 0; i < 2; i++) {
          const int16 filter_val = local_filter_ptr[i] + filter_offset;
          *acc_buffer_ptr++ += static_cast<int32>(filter_val) * input_val;
        }
        local_filter_ptr += 2;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 0, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const uint8* local_filter_ptr = filter_ptr;
      const uint8* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 16 input channels at a time.
      for (; ic <= input_depth - 16; ic += 16) {
        // Load the filters, add filter_offset.
        uint8x8_t filter_u8_0 = vld1_u8(local_filter_ptr + 8 * 0);
        uint8x8_t filter_u8_1 = vld1_u8(local_filter_ptr + 8 * 1);
        local_filter_ptr += 16;
        int16x8_t filter_0 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_0));
        int16x8_t filter_1 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_1));
        filter_0 = vaddq_s16(filter_0, vdupq_n_s16(filter_offset));
        filter_1 = vaddq_s16(filter_1, vdupq_n_s16(filter_offset));
        // Load the inputs, add input_offset.
        uint8x8_t input_u8_0 = vld1_u8(local_input_ptr + 8 * 0);
        uint8x8_t input_u8_1 = vld1_u8(local_input_ptr + 8 * 1);
        local_input_ptr += 16;
        int16x8_t input_0 = vreinterpretq_s16_u16(vmovl_u8(input_u8_0));
        int16x8_t input_1 = vreinterpretq_s16_u16(vmovl_u8(input_u8_1));
        input_0 = vaddq_s16(input_0, vdupq_n_s16(input_offset));
        input_1 = vaddq_s16(input_1, vdupq_n_s16(input_offset));
        // Load the accumulators from acc_buffer
        int32x4_t acc_0 = vld1q_s32(acc_buffer_ptr + 4 * 0);
        int32x4_t acc_1 = vld1q_s32(acc_buffer_ptr + 4 * 1);
        int32x4_t acc_2 = vld1q_s32(acc_buffer_ptr + 4 * 2);
        int32x4_t acc_3 = vld1q_s32(acc_buffer_ptr + 4 * 3);
        acc_0 = vmlal_s16(acc_0, vget_low_s16(input_0), vget_low_s16(filter_0));
        acc_1 =
            vmlal_s16(acc_1, vget_high_s16(input_0), vget_high_s16(filter_0));
        acc_2 = vmlal_s16(acc_2, vget_low_s16(input_1), vget_low_s16(filter_1));
        acc_3 =
            vmlal_s16(acc_3, vget_high_s16(input_1), vget_high_s16(filter_1));
        // Store the accumulators back to acc_buffer
        vst1q_s32(acc_buffer_ptr + 4 * 0, acc_0);
        vst1q_s32(acc_buffer_ptr + 4 * 1, acc_1);
        vst1q_s32(acc_buffer_ptr + 4 * 2, acc_2);
        vst1q_s32(acc_buffer_ptr + 4 * 3, acc_3);
        acc_buffer_ptr += 16;
      }
      // Handle 8 input channels at a time.
      for (; ic <= input_depth - 8; ic += 8) {
        // Load the filters, add filter_offset.
        const uint8x8_t filter_u8 = vld1_u8(local_filter_ptr);
        local_filter_ptr += 8;
        const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
        const int16x8_t filter =
            vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
        // Load the inputs, add input_offset.
        const uint8x8_t input_u8 = vld1_u8(local_input_ptr);
        local_input_ptr += 8;
        const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
        const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
        // Load the accumulators from acc_buffer
        int32x4_t acc[2];
        for (int i = 0; i < 2; i++) {
          acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
        }
        // Multiply-accumulate
        acc[0] = vmlal_s16(acc[0], vget_low_s16(input), vget_low_s16(filter));
        acc[1] = vmlal_s16(acc[1], vget_high_s16(input), vget_high_s16(filter));
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 2; i++) {
          vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
        }
        acc_buffer_ptr += 8;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        const int16 input_val = *local_input_ptr++ + input_offset;
        const int16 filter_val = *local_filter_ptr++ + filter_offset;
        *acc_buffer_ptr++ += static_cast<int32>(filter_val) * input_val;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 16, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8[2];
    for (int i = 0; i < 2; i++) {
      filter_u8[i] = vld1_u8(filter_ptr + 8 * i);
    }
    int16x8_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vreinterpretq_s16_u16(vmovl_u8(filter_u8[i]));
    }
    for (int i = 0; i < 2; i++) {
      filter[i] = vaddq_s16(filter[i], vdupq_n_s16(filter_offset));
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs, add input_offset.
      uint8x8_t input_u8[2];
      for (int i = 0; i < 2; i++) {
        input_u8[i] = vld1_u8(input_ptr + 8 * i);
      }
      input_ptr += input_ptr_increment;
      int16x8_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vreinterpretq_s16_u16(vmovl_u8(input_u8[i]));
      }
      for (int i = 0; i < 2; i++) {
        input[i] = vaddq_s16(input[i], vdupq_n_s16(input_offset));
      }
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[2 * i + 0] = vmlal_s16(acc[2 * i + 0], vget_low_s16(input[i]),
                                   vget_low_s16(filter[i]));
        acc[2 * i + 1] = vmlal_s16(acc[2 * i + 1], vget_high_s16(input[i]),
                                   vget_high_s16(filter[i]));
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 8, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    const uint8x8_t filter_u8 = vld1_u8(filter_ptr);
    const int16x8_t filter_s16 = vreinterpretq_s16_u16(vmovl_u8(filter_u8));
    const int16x8_t filter = vaddq_s16(filter_s16, vdupq_n_s16(filter_offset));
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs, add input_offset.
      const uint8x8_t input_u8 = vld1_u8(input_ptr);
      const int16x8_t input_s16 = vreinterpretq_s16_u16(vmovl_u8(input_u8));
      const int16x8_t input = vaddq_s16(input_s16, vdupq_n_s16(input_offset));
      // Load the accumulators from acc_buffer
      int32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      acc[0] = vmlal_s16(acc[0], vget_low_s16(input), vget_low_s16(filter));
      acc[1] = vmlal_s16(acc[1], vget_high_s16(input), vget_high_s16(filter));
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 1, 16> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8[2];
    for (int i = 0; i < 2; i++) {
      filter_u8[i] = vld1_u8(filter_ptr + 8 * i);
    }
    int16x8_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vreinterpretq_s16_u16(vmovl_u8(filter_u8[i]));
    }
    for (int i = 0; i < 2; i++) {
      filter[i] = vaddq_s16(filter[i], vdupq_n_s16(filter_offset));
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      uint8 input_u8 = *input_ptr;
      input_ptr += input_ptr_increment;
      int16 input = static_cast<int16>(input_u8 + input_offset);
      // Load the accumulators from acc_buffer
      int32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[2 * i + 0] =
            vmlal_n_s16(acc[2 * i + 0], vget_low_s16(filter[i]), input);
        acc[2 * i + 1] =
            vmlal_n_s16(acc[2 * i + 1], vget_high_s16(filter[i]), input);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 1, 32> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8_0 = vld1_u8(filter_ptr + 8 * 0);
    uint8x8_t filter_u8_1 = vld1_u8(filter_ptr + 8 * 1);
    uint8x8_t filter_u8_2 = vld1_u8(filter_ptr + 8 * 2);
    uint8x8_t filter_u8_3 = vld1_u8(filter_ptr + 8 * 3);
    int16x8_t filter_0 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_0));
    int16x8_t filter_1 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_1));
    int16x8_t filter_2 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_2));
    int16x8_t filter_3 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_3));
    filter_0 = vaddq_s16(filter_0, vdupq_n_s16(filter_offset));
    filter_1 = vaddq_s16(filter_1, vdupq_n_s16(filter_offset));
    filter_2 = vaddq_s16(filter_2, vdupq_n_s16(filter_offset));
    filter_3 = vaddq_s16(filter_3, vdupq_n_s16(filter_offset));
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      uint8 input_u8 = *input_ptr;
      input_ptr += input_ptr_increment;
      int16 input = static_cast<int16>(input_u8 + input_offset);
      // Load the accumulators from acc_buffer
      int32x4_t acc_0 = vld1q_s32(acc_buffer_ptr + 4 * 0);
      int32x4_t acc_1 = vld1q_s32(acc_buffer_ptr + 4 * 1);
      int32x4_t acc_2 = vld1q_s32(acc_buffer_ptr + 4 * 2);
      int32x4_t acc_3 = vld1q_s32(acc_buffer_ptr + 4 * 3);
      int32x4_t acc_4 = vld1q_s32(acc_buffer_ptr + 4 * 4);
      int32x4_t acc_5 = vld1q_s32(acc_buffer_ptr + 4 * 5);
      int32x4_t acc_6 = vld1q_s32(acc_buffer_ptr + 4 * 6);
      int32x4_t acc_7 = vld1q_s32(acc_buffer_ptr + 4 * 7);
      // Multiply-accumulate
      acc_0 = vmlal_n_s16(acc_0, vget_low_s16(filter_0), input);
      acc_1 = vmlal_n_s16(acc_1, vget_high_s16(filter_0), input);
      acc_2 = vmlal_n_s16(acc_2, vget_low_s16(filter_1), input);
      acc_3 = vmlal_n_s16(acc_3, vget_high_s16(filter_1), input);
      acc_4 = vmlal_n_s16(acc_4, vget_low_s16(filter_2), input);
      acc_5 = vmlal_n_s16(acc_5, vget_high_s16(filter_2), input);
      acc_6 = vmlal_n_s16(acc_6, vget_low_s16(filter_3), input);
      acc_7 = vmlal_n_s16(acc_7, vget_high_s16(filter_3), input);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr + 4 * 0, acc_0);
      vst1q_s32(acc_buffer_ptr + 4 * 1, acc_1);
      vst1q_s32(acc_buffer_ptr + 4 * 2, acc_2);
      vst1q_s32(acc_buffer_ptr + 4 * 3, acc_3);
      vst1q_s32(acc_buffer_ptr + 4 * 4, acc_4);
      vst1q_s32(acc_buffer_ptr + 4 * 5, acc_5);
      vst1q_s32(acc_buffer_ptr + 4 * 6, acc_6);
      vst1q_s32(acc_buffer_ptr + 4 * 7, acc_7);
      acc_buffer_ptr += 32;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 1, 20> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    // NEON wants to load 8 bytes at a time, but 20 is not divisible by 8.
    // We load the first 16 bytes into filter_u8_{0,1} as usual.
    // Then we load the 8 last bytes into filter_u8_x  (x for 'extra').
    // This is redundant: the first 4 bytes of filter_u8_x are the same
    // as the last 4 bytes of filter_u8_x.
    uint8x8_t filter_u8_0 = vld1_u8(filter_ptr + 8 * 0);
    uint8x8_t filter_u8_1 = vld1_u8(filter_ptr + 8 * 1);
    uint8x8_t filter_u8_x = vld1_u8(filter_ptr + 8 * 1 + 4);
    int16x8_t filter_0 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_0));
    int16x8_t filter_1 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_1));
    int16x8_t filter_x = vreinterpretq_s16_u16(vmovl_u8(filter_u8_x));
    filter_0 = vaddq_s16(filter_0, vdupq_n_s16(filter_offset));
    filter_1 = vaddq_s16(filter_1, vdupq_n_s16(filter_offset));
    filter_x = vaddq_s16(filter_x, vdupq_n_s16(filter_offset));
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      uint8 input_u8 = *input_ptr;
      input_ptr += input_ptr_increment;
      int16 input = static_cast<int16>(input_u8 + input_offset);
      // Load the accumulators from acc_buffer
      int32x4_t acc_0 = vld1q_s32(acc_buffer_ptr + 4 * 0);
      int32x4_t acc_1 = vld1q_s32(acc_buffer_ptr + 4 * 1);
      int32x4_t acc_2 = vld1q_s32(acc_buffer_ptr + 4 * 2);
      int32x4_t acc_3 = vld1q_s32(acc_buffer_ptr + 4 * 3);
      int32x4_t acc_4 = vld1q_s32(acc_buffer_ptr + 4 * 4);
      // Multiply-accumulate
      acc_0 = vmlal_n_s16(acc_0, vget_low_s16(filter_0), input);
      acc_1 = vmlal_n_s16(acc_1, vget_high_s16(filter_0), input);
      acc_2 = vmlal_n_s16(acc_2, vget_low_s16(filter_1), input);
      acc_3 = vmlal_n_s16(acc_3, vget_high_s16(filter_1), input);
      acc_4 = vmlal_n_s16(acc_4, vget_high_s16(filter_x), input);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr + 4 * 0, acc_0);
      vst1q_s32(acc_buffer_ptr + 4 * 1, acc_1);
      vst1q_s32(acc_buffer_ptr + 4 * 2, acc_2);
      vst1q_s32(acc_buffer_ptr + 4 * 3, acc_3);
      vst1q_s32(acc_buffer_ptr + 4 * 4, acc_4);
      acc_buffer_ptr += 20;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 1, 8> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    const uint8x8_t filter_u8 = vld1_u8(filter_ptr);
    const int16x8_t filter = vaddq_s16(
        vreinterpretq_s16_u16(vmovl_u8(filter_u8)), vdupq_n_s16(filter_offset));
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      uint8 input_u8 = *input_ptr;
      input_ptr += input_ptr_increment;
      int16 input = static_cast<int16>(input_u8 + input_offset);
      // Load the accumulators from acc_buffer
      int32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_s32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      acc[0] = vmlal_n_s16(acc[0], vget_low_s16(filter), input);
      acc[1] = vmlal_n_s16(acc[1], vget_high_s16(filter), input);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_s32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 2, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;

    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the accumulators from acc_buffer.
      int32x4_t acc = vld1q_s32(acc_buffer_ptr);
      // Load the inputs, add input_offset.
      uint16x4_t input_u16 = vdup_n_u16(0);
      input_u16 = vset_lane_u16((reinterpret_cast<const uint16*>(input_ptr))[0],
                                input_u16, 0);
      input_ptr += input_ptr_increment;
      input_u16 = vset_lane_u16((reinterpret_cast<const uint16*>(input_ptr))[0],
                                input_u16, 1);
      input_ptr += input_ptr_increment;
      const int16x4_t input_s16 = vreinterpret_s16_u16(
          vget_low_u16(vmovl_u8(vreinterpret_u8_u16(input_u16))));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate.
      acc = vmlal_s16(acc, filter, input);
      // Store the accumulators back to acc_buffer.
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }

    // Handle 1 output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the accumulators from acc_buffer.
      int32x2_t acc = vld1_s32(acc_buffer_ptr);
      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vdup_n_u8(0);
      input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
      input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
      input_ptr += input_ptr_increment;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));

      // Multiply-accumulate.
      acc = vget_low_s32(vmlal_s16(vcombine_s32(acc, acc), filter, input));
      // Store the accumulators back to acc_buffer.
      vst1_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 2;
    }
  }
};

template <>
struct QuantizedDepthwiseConvKernel<true, 4, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    if (num_output_pixels <= 0) {
      return;
    }

    // Load the filters, add filter_offset.
    uint8x8_t filter_u8 = vdup_n_u8(0);
    filter_u8 = vset_lane_u8(filter_ptr[0], filter_u8, 0);
    filter_u8 = vset_lane_u8(filter_ptr[1], filter_u8, 1);
    filter_u8 = vset_lane_u8(filter_ptr[2], filter_u8, 2);
    filter_u8 = vset_lane_u8(filter_ptr[3], filter_u8, 3);
    const int16x4_t filter_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(filter_u8)));
    const int16x4_t filter = vadd_s16(filter_s16, vdup_n_s16(filter_offset));

    int outp = 0;

    // Handle one output pixel at a time until second to the last pixel. Second
    // to the last because we read eight input pixels while only processing
    // four.
    for (; outp < num_output_pixels - 1; outp++) {
      // Load the accumulators from acc_buffer
      int32x4_t acc;
      acc = vld1q_s32(acc_buffer_ptr);

      // Load the inputs, add input_offset.
      uint8x8_t input_u8 = vld1_u8(input_ptr);
      input_ptr += input_ptr_increment;
      const int16x4_t input_s16 =
          vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
      const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
      // Multiply-accumulate
      acc = vmlal_s16(acc, filter, input);
      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }

    // Handle the last output pixel.
    // Load the accumulators from acc_buffer
    int32x4_t acc;
    acc = vld1q_s32(acc_buffer_ptr);

    // Load the inputs, add input_offset.
    uint8x8_t input_u8 = vdup_n_u8(0);
    input_u8 = vset_lane_u8(input_ptr[0], input_u8, 0);
    input_u8 = vset_lane_u8(input_ptr[1], input_u8, 1);
    input_u8 = vset_lane_u8(input_ptr[2], input_u8, 2);
    input_u8 = vset_lane_u8(input_ptr[3], input_u8, 3);
    const int16x4_t input_s16 =
        vreinterpret_s16_u16(vget_low_u16(vmovl_u8(input_u8)));
    const int16x4_t input = vadd_s16(input_s16, vdup_n_s16(input_offset));
    // Multiply-accumulate
    acc = vmlal_s16(acc, filter, input);
    // Store the accumulators back to acc_buffer
    vst1q_s32(acc_buffer_ptr, acc);
  }
};

template <>
struct QuantizedDepthwiseConvKernel<false, 12, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const uint8* input_ptr, int16 input_offset,
                  int input_ptr_increment, const uint8* filter_ptr,
                  int16 filter_offset, int32* acc_buffer_ptr) {
    // Load the filters, add filter_offset.
    uint8x8_t filter_u8_0 = vld1_u8(filter_ptr);
    uint8x8_t filter_u8_1 = vld1_u8(filter_ptr + 4);
    int16x8_t filter_s16_0 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_0));
    int16x8_t filter_s16_1 = vreinterpretq_s16_u16(vmovl_u8(filter_u8_1));
    filter_s16_0 = vaddq_s16(filter_s16_0, vdupq_n_s16(filter_offset));
    filter_s16_1 = vaddq_s16(filter_s16_1, vdupq_n_s16(filter_offset));
    int16x4_t filter_0 = vget_low_s16(filter_s16_0);
    int16x4_t filter_1 = vget_high_s16(filter_s16_0);
    int16x4_t filter_2 = vget_high_s16(filter_s16_1);

    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs, add input_offset.
      uint8x8_t input_u8_0 = vld1_u8(input_ptr);
      uint8x8_t input_u8_1 = vld1_u8(input_ptr + 4);
      input_ptr += input_ptr_increment;
      int16x8_t input_0 = vreinterpretq_s16_u16(vmovl_u8(input_u8_0));
      int16x8_t input_1 = vreinterpretq_s16_u16(vmovl_u8(input_u8_1));
      input_0 = vaddq_s16(input_0, vdupq_n_s16(input_offset));
      input_1 = vaddq_s16(input_1, vdupq_n_s16(input_offset));

      // Load the accumulators from acc_buffer
      int32x4_t acc_0 = vld1q_s32(acc_buffer_ptr + 4 * 0);
      int32x4_t acc_1 = vld1q_s32(acc_buffer_ptr + 4 * 1);
      int32x4_t acc_2 = vld1q_s32(acc_buffer_ptr + 4 * 2);

      // Multiply-accumulate
      acc_0 = vmlal_s16(acc_0, vget_low_s16(input_0), filter_0);
      acc_1 = vmlal_s16(acc_1, vget_high_s16(input_0), filter_1);
      acc_2 = vmlal_s16(acc_2, vget_high_s16(input_1), filter_2);

      // Store the accumulators back to acc_buffer
      vst1q_s32(acc_buffer_ptr + 4 * 0, acc_0);
      vst1q_s32(acc_buffer_ptr + 4 * 1, acc_1);
      vst1q_s32(acc_buffer_ptr + 4 * 2, acc_2);

      acc_buffer_ptr += 12;
    }
  }
};
#endif

// Accumulates the effect of one row of the filter, on a segment of one row
// of the output, accessing the corresponding one row of the input.
template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
void QuantizedDepthwiseConvAccumRow(int stride, int dilation_factor,
                                    int input_depth, int input_width,
                                    const uint8* input_data, int16 input_offset,
                                    int pad_width, int depth_multiplier,
                                    int filter_width, const uint8* filter_data,
                                    int16 filter_offset, int out_x_buffer_start,
                                    int out_x_buffer_end, int output_depth,
                                    int32* acc_buffer) {
#ifdef GEMMLOWP_PROFILING
  gemmlowp::ScopedProfilingLabel label(__PRETTY_FUNCTION__);
#endif
  // Sanity check parameters. This is important in particular to ensure
  // that we keep the number of template instantiations minimal, so we don't
  // increase binary size unnecessarily.
  static_assert(kFixedDepthMultiplier || !kFixedInputDepth, "");
  static_assert(kFixedInputDepth || kAllowStrided, "");
  TFLITE_DCHECK(stride == 1 || kAllowStrided);
  if (kFixedInputDepth) {
    TFLITE_DCHECK_EQ(input_depth, kFixedInputDepth);
  }
  if (kFixedDepthMultiplier) {
    TFLITE_DCHECK_EQ(depth_multiplier, kFixedDepthMultiplier);
  }
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  const int input_ptr_increment = stride * input_depth;
  const uint8* filter_base_ptr = filter_data;
  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
    // For the current (filter_x, filter_y) point in the filter,
    // compute the boundaries of the corresponding output row segment.
    int out_x_loop_start_unclampled = 0;
    int out_x_loop_end_unclampled = 0;
    if (kAllowStrided) {
      if (stride == 2) {
        out_x_loop_start_unclampled =
            (pad_width - dilation_factor * filter_x + 1) / 2;
        out_x_loop_end_unclampled =
            (pad_width + input_width - dilation_factor * filter_x + 1) / 2;
      } else if (stride == 4) {
        out_x_loop_start_unclampled =
            (pad_width - dilation_factor * filter_x + 3) / 4;
        out_x_loop_end_unclampled =
            (pad_width + input_width - dilation_factor * filter_x + 3) / 4;
      } else {
        out_x_loop_start_unclampled =
            (pad_width - dilation_factor * filter_x + stride - 1) / stride;
        out_x_loop_end_unclampled = (pad_width + input_width -
                                     dilation_factor * filter_x + stride - 1) /
                                    stride;
      }
    } else {
      out_x_loop_start_unclampled = pad_width - dilation_factor * filter_x;
      out_x_loop_end_unclampled =
          pad_width + input_width - dilation_factor * filter_x;
    }
    // The kernel will have to iterate on the segment of the
    // output row that starts at out_x_loop_start and out_x_loop_end.
    const int out_x_loop_start =
        std::max(out_x_buffer_start, out_x_loop_start_unclampled);
    const int out_x_loop_end =
        std::min(out_x_buffer_end, out_x_loop_end_unclampled);

    int32* acc_buffer_ptr =
        acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin =
        (out_x_loop_start * stride) - pad_width + dilation_factor * filter_x;
    const uint8* input_ptr = input_data + in_x_origin * input_depth;
    const int num_output_pixels = out_x_loop_end - out_x_loop_start;
    QuantizedDepthwiseConvKernel<
        kAllowStrided, kFixedInputDepth,
        kFixedDepthMultiplier>::Run(num_output_pixels, input_depth,
                                    depth_multiplier, input_ptr, input_offset,
                                    input_ptr_increment, filter_base_ptr,
                                    filter_offset, acc_buffer_ptr);
    filter_base_ptr += output_depth;
  }
}

// generic fallback of DepthwiseConvAccumRow, portable, non-templatized.
inline void QuantizedDepthwiseConvAccumRowGeneric(
    int stride, int dilation_factor, int input_depth, int input_width,
    const uint8* input_data, int16 input_offset, int pad_width,
    int depth_multiplier, int filter_width, const uint8* filter_data,
    int16 filter_offset, int out_x_buffer_start, int out_x_buffer_end,
    int output_depth, int32* acc_buffer) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConvAccumRowGeneric (slow)");
  const uint8* filter_base_ptr = filter_data;
  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
    const int out_x_loop_start = std::max(
        out_x_buffer_start,
        (pad_width - dilation_factor * filter_x + stride - 1) / stride);
    const int out_x_loop_end = std::min(
        out_x_buffer_end,
        (pad_width + input_width - dilation_factor * filter_x + stride - 1) /
            stride);

    int32* acc_buffer_ptr =
        acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin =
        (out_x_loop_start * stride) - pad_width + dilation_factor * filter_x;
    const uint8* input_ptr = input_data + in_x_origin * input_depth;
    const int input_ptr_increment = (stride - 1) * input_depth;
    for (int out_x = out_x_loop_start; out_x < out_x_loop_end; out_x++) {
      const uint8* filter_ptr = filter_base_ptr;
      for (int ic = 0; ic < input_depth; ++ic) {
        const int16 input_val = *input_ptr++ + input_offset;
        for (int m = 0; m < depth_multiplier; m++) {
          const int16 filter_val = *filter_ptr++ + filter_offset;
          *acc_buffer_ptr++ += static_cast<int32>(filter_val) * input_val;
        }
      }
      input_ptr += input_ptr_increment;
    }
    filter_base_ptr += output_depth;
  }
}

// Initializes the accumulator buffer with bias values.
inline void DepthwiseConvInitAccBuffer(int num_output_pixels, int output_depth,
                                       const int32* bias_data,
                                       int32* acc_buffer) {
  int i = 0;
#ifdef USE_NEON
  if (output_depth == 1) {
    const int32x4_t b = vdupq_n_s32(bias_data[0]);
    for (; i <= num_output_pixels - 16; i += 16) {
      vst1q_s32(acc_buffer + i + 0, b);
      vst1q_s32(acc_buffer + i + 4, b);
      vst1q_s32(acc_buffer + i + 8, b);
      vst1q_s32(acc_buffer + i + 12, b);
    }
    for (; i <= num_output_pixels - 4; i += 4) {
      vst1q_s32(acc_buffer + i, b);
    }
  } else if (output_depth == 2) {
    int32x4_t b = vdupq_n_s32(bias_data[0]);
    b = vsetq_lane_s32(bias_data[1], b, 1);
    b = vsetq_lane_s32(bias_data[1], b, 3);
    for (; i <= num_output_pixels - 8; i += 8) {
      vst1q_s32(acc_buffer + 2 * i + 0, b);
      vst1q_s32(acc_buffer + 2 * i + 4, b);
      vst1q_s32(acc_buffer + 2 * i + 8, b);
      vst1q_s32(acc_buffer + 2 * i + 12, b);
    }
    for (; i <= num_output_pixels - 2; i += 2) {
      vst1q_s32(acc_buffer + 2 * i, b);
    }
  } else if (output_depth == 4) {
    const int32x4_t b = vld1q_s32(bias_data);
    for (; i <= num_output_pixels - 4; i += 4) {
      vst1q_s32(acc_buffer + 4 * i + 0, b);
      vst1q_s32(acc_buffer + 4 * i + 4, b);
      vst1q_s32(acc_buffer + 4 * i + 8, b);
      vst1q_s32(acc_buffer + 4 * i + 12, b);
    }
    for (; i < num_output_pixels; i++) {
      vst1q_s32(acc_buffer + 4 * i, b);
    }
  } else if (output_depth == 8) {
    const int32x4_t b0 = vld1q_s32(bias_data);
    const int32x4_t b1 = vld1q_s32(bias_data + 4);
    for (; i <= num_output_pixels - 2; i += 2) {
      vst1q_s32(acc_buffer + 8 * i + 0, b0);
      vst1q_s32(acc_buffer + 8 * i + 4, b1);
      vst1q_s32(acc_buffer + 8 * i + 8, b0);
      vst1q_s32(acc_buffer + 8 * i + 12, b1);
    }
    for (; i < num_output_pixels; i++) {
      vst1q_s32(acc_buffer + 8 * i + 0, b0);
      vst1q_s32(acc_buffer + 8 * i + 4, b1);
    }
  } else if (output_depth == 16) {
    const int32x4_t b0 = vld1q_s32(bias_data);
    const int32x4_t b1 = vld1q_s32(bias_data + 4);
    const int32x4_t b2 = vld1q_s32(bias_data + 8);
    const int32x4_t b3 = vld1q_s32(bias_data + 12);
    for (; i < num_output_pixels; i++) {
      vst1q_s32(acc_buffer + 16 * i + 0, b0);
      vst1q_s32(acc_buffer + 16 * i + 4, b1);
      vst1q_s32(acc_buffer + 16 * i + 8, b2);
      vst1q_s32(acc_buffer + 16 * i + 12, b3);
    }
  }
#endif
  for (; i < num_output_pixels; i++) {
    memcpy(acc_buffer + i * output_depth, bias_data,
           sizeof(acc_buffer[0]) * output_depth);
  }
}

inline void DepthwiseConvGeneral(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
#ifdef USE_NEON
  const bool shift_left = (output_shift > 0);
  const int32 multiplier_power_of_two = shift_left ? (1 << output_shift) : 1;
#endif

  static const int kAccBufferMaxSize = 2048;
  int32 acc_buffer[kAccBufferMaxSize];
  TFLITE_DCHECK_GE(kAccBufferMaxSize, output_depth);
  const int kOutputPixelsInAccBuffer = kAccBufferMaxSize / output_depth;
  const int kAccBufferActualSize = kOutputPixelsInAccBuffer * output_depth;
  TFLITE_DCHECK_LE(kOutputPixelsInAccBuffer * output_depth,
                   kAccBufferActualSize);
  TFLITE_DCHECK_LE(kAccBufferActualSize, kAccBufferMaxSize);
  TFLITE_DCHECK_GE(kOutputPixelsInAccBuffer, 1);

  // row_accum_func will point to the core accumulation function to be used
  // for this DepthwiseConv op.
  using row_accum_func_t = decltype(&QuantizedDepthwiseConvAccumRowGeneric);
  row_accum_func_t row_accum_func = nullptr;

#define TFMINI_USE_DEPTHWISECONV_KERNEL(ALLOW_STRIDED, FIXED_INPUT_DEPTH, \
                                        FIXED_DEPTH_MULTIPLIER)           \
  if (!row_accum_func && (stride_width == 1 || ALLOW_STRIDED) &&          \
      (input_depth == FIXED_INPUT_DEPTH || FIXED_INPUT_DEPTH == 0) &&     \
      depth_multiplier == FIXED_DEPTH_MULTIPLIER) {                       \
    row_accum_func =                                                      \
        QuantizedDepthwiseConvAccumRow<ALLOW_STRIDED, FIXED_INPUT_DEPTH,  \
                                       FIXED_DEPTH_MULTIPLIER>;           \
  }

#ifdef USE_NEON
  // We go over our list of kernels by decreasing order of preference
  // for the cases where multiple kernels could apply.

  // Start with the fastest kernels: AllowStrided=false, fixed input depth.

  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 1, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 1, 4)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 4)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 12, 1)

  // Next come the strided kernels: AllowStrided=true, fixed input depth.
  // They are a bit less efficient, but allow stride!=1.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 8, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 16, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 16)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 20)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 32)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 2, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 4, 1)

  // Finally, the kernels allowing a variable input depth,
  // these are the least efficient but most general kernels.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 3)
#endif  // USE_NEON

  // No matching fast kernel found, use slow fallback.
  if (!row_accum_func) {
    row_accum_func = QuantizedDepthwiseConvAccumRowGeneric;
  }

#undef TFMINI_USE_DEPTHWISECONV_KERNEL

  const int input_height_stride = input_shape.Dims(3) * input_shape.Dims(2);
  const int input_batch_stride = input_height_stride * input_shape.Dims(1);
  const int filter_height_stride = filter_shape.Dims(3) * filter_shape.Dims(2);

  // Now that we have determined row_accum_func, we can start work.
  uint8* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int filter_y_start =
          std::max(0, (-in_y_origin + dilation_height_factor - 1) /
                          dilation_height_factor);
      const int filter_y_end =
          std::min(filter_height,
                   (input_height - in_y_origin + dilation_height_factor - 1) /
                       dilation_height_factor);
      for (int out_x_buffer_start = 0; out_x_buffer_start < output_width;
           out_x_buffer_start += kOutputPixelsInAccBuffer) {
        const int out_x_buffer_end = std::min(
            output_width, out_x_buffer_start + kOutputPixelsInAccBuffer);
        // We call a 'pixel' a group of activation that share all but the
        // 'depth'/'channel' coordinate. num_output_pixels is the number of
        // output pixels that we will accumulate in this loop iteration.
        const int num_output_pixels = out_x_buffer_end - out_x_buffer_start;
        // Initialize our local accumulator with the bias values, so we don't
        // have to add them later.
        DepthwiseConvInitAccBuffer(num_output_pixels, output_depth, bias_data,
                                   acc_buffer);
        // Accumulation loop. Most of the time should be spent in here.
        for (int filter_y = filter_y_start; filter_y < filter_y_end;
             ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          row_accum_func(
              stride_width, dilation_width_factor, input_depth, input_width,
              input_data + in_y * input_height_stride + b * input_batch_stride,
              input_offset, pad_width, depth_multiplier, filter_width,
              filter_data + filter_y * filter_height_stride, filter_offset,
              out_x_buffer_start, out_x_buffer_end, output_depth, acc_buffer);
        }
        // Finished accumulating int32 values. Now need to convert them to
        // the final 8bit form and store them.
        gemmlowp::ScopedProfilingLabel label("downquantize+store");
        const int num_output_values = output_depth * num_output_pixels;
        int i = 0;
#ifdef USE_NEON
        using gemmlowp::RoundingDivideByPOT;
        const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
        const int32x4_t output_activation_min_vec =
            vdupq_n_s32(output_activation_min);
        const int32x4_t output_activation_max_vec =
            vdupq_n_s32(output_activation_max);
        // Handle 16 values at once.
        // This allows us to issue 4 mutually independent int32
        // multiplications (vqrdmulh), which should alleviate most of their
        // high latency.
        for (; i <= num_output_values - 16; i += 16) {
          int32x4_t acc[4];
          for (int j = 0; j < 4; j++) {
            acc[j] = vld1q_s32(acc_buffer + i + 4 * j);
          }

          if (!shift_left) {
            // Fixed-point multiplication.
            for (int j = 0; j < 4; j++) {
              acc[j] = vqrdmulhq_n_s32(acc[j], output_multiplier);
            }
            for (int j = 0; j < 4; j++) {
              acc[j] = RoundingDivideByPOT(acc[j], -output_shift);
            }
          } else {
            // Fixed-point multiplication.
            for (int j = 0; j < 4; j++) {
              acc[j] = vmulq_n_s32(acc[j], multiplier_power_of_two);
              acc[j] = vqrdmulhq_n_s32(acc[j], output_multiplier);
            }
          }
          // Add the output offset.
          for (int j = 0; j < 4; j++) {
            acc[j] = vaddq_s32(acc[j], output_offset_vec);
          }
          // Apply the activation function.
          for (int j = 0; j < 4; j++) {
            acc[j] = vmaxq_s32(acc[j], output_activation_min_vec);
          }
          for (int j = 0; j < 4; j++) {
            acc[j] = vminq_s32(acc[j], output_activation_max_vec);
          }
          // Saturating cast to uint8 and store to destination.
          int16x4_t acc_s16[4];
          for (int j = 0; j < 4; j++) {
            acc_s16[j] = vqmovn_s32(acc[j]);
          }
          const int16x8_t res_s16_0 = vcombine_s16(acc_s16[0], acc_s16[1]);
          const int16x8_t res_s16_1 = vcombine_s16(acc_s16[2], acc_s16[3]);
          const uint8x8_t res_u8_0 = vqmovun_s16(res_s16_0);
          const uint8x8_t res_u8_1 = vqmovun_s16(res_s16_1);
          vst1q_u8(output_ptr, vcombine_u8(res_u8_0, res_u8_1));
          output_ptr += 16;
        }
        // Handle 8 values at once.
        // Not as good as 16 (now we're only issuing 2 mutually independent
        // vqrdmulh instructions, so we're probably paying for their high
        // latency).
        for (; i <= num_output_values - 8; i += 8) {
          int32x4_t acc0 = vld1q_s32(acc_buffer + i);
          int32x4_t acc1 = vld1q_s32(acc_buffer + i + 4);
          if (!shift_left) {
            // Fixed-point multiplication.
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
            // Rounding right shift.
            acc0 = RoundingDivideByPOT(acc0, -output_shift);
            acc1 = RoundingDivideByPOT(acc1, -output_shift);
          } else {
            // Fixed-point multiplication.
            acc0 = vmulq_n_s32(acc0, multiplier_power_of_two);
            acc0 = vqrdmulhq_n_s32(acc0, output_multiplier);

            acc1 = vmulq_n_s32(acc1, multiplier_power_of_two);
            acc1 = vqrdmulhq_n_s32(acc1, output_multiplier);
          }
          // Add the output offset.
          acc0 = vaddq_s32(acc0, output_offset_vec);
          acc1 = vaddq_s32(acc1, output_offset_vec);
          // Apply the activation function.
          acc0 = vmaxq_s32(acc0, output_activation_min_vec);
          acc1 = vmaxq_s32(acc1, output_activation_min_vec);
          acc0 = vminq_s32(acc0, output_activation_max_vec);
          acc1 = vminq_s32(acc1, output_activation_max_vec);
          // Saturating cast to uint8 and store to destination.
          const int16x4_t acc0_s16 = vqmovn_s32(acc0);
          const int16x4_t acc1_s16 = vqmovn_s32(acc1);
          const int16x8_t res_s16 = vcombine_s16(acc0_s16, acc1_s16);
          const uint8x8_t res_u8 = vqmovun_s16(res_s16);
          vst1_u8(output_ptr, res_u8);
          output_ptr += 8;
        }
        // Handle 4 values at once. Now we're paying the full price of the
        // high latency of vqrdmulh. Also, storing only 4 bytes at the end
        // (without any alignment) can only be done 1 byte at a time.
        // Yet, that is still worth doing to minimize the amount of leftover
        // that will have to go through the very slow scalar code.
        for (; i <= num_output_values - 4; i += 4) {
          int32x4_t acc = vld1q_s32(acc_buffer + i);
          if (!shift_left) {
            // Fixed-point multiplication.
            acc = vqrdmulhq_n_s32(acc, output_multiplier);
            // Rounding right shift.
            acc = RoundingDivideByPOT(acc, -output_shift);
          } else {
            // Fixed-point multiplication.
            acc = vmulq_n_s32(acc, multiplier_power_of_two);
            acc = vqrdmulhq_n_s32(acc, output_multiplier);
          }
          // Add the output offset.
          acc = vaddq_s32(acc, output_offset_vec);
          // Apply the activation function.
          acc = vmaxq_s32(acc, output_activation_min_vec);
          acc = vminq_s32(acc, output_activation_max_vec);
          // Saturating cast to uint8 and store to destination.
          const int16x4_t acc_s16 = vqmovn_s32(acc);
          const int16x8_t res_s16 = vcombine_s16(acc_s16, acc_s16);
          const uint8x8_t res_u8 = vqmovun_s16(res_s16);
          vst1_lane_u8(output_ptr + 0, res_u8, 0);
          vst1_lane_u8(output_ptr + 1, res_u8, 1);
          vst1_lane_u8(output_ptr + 2, res_u8, 2);
          vst1_lane_u8(output_ptr + 3, res_u8, 3);
          output_ptr += 4;
        }
#endif  // USE_NEON

        // Handle leftover values, one by one. This is very slow.
        for (; i < num_output_values; i++) {
          int32 acc = acc_buffer[i];
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          *output_ptr++ = static_cast<uint8>(acc);
        }
      }
    }
  }
}

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConv/8bit");
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_GE(dilation_width_factor, 1);
  TFLITE_DCHECK_GE(dilation_height_factor, 1);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_depth = input_shape.Dims(3);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__) && !defined(GOOGLE_L4T)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int output_shift = params.output_shift;

  // Call kernel optimized for depthwise convolutions using 3x3 filters if
  // parameters are supported.
  if (Fast3x3FilterKernelSupported(
          input_shape, filter_shape, stride_width, stride_height,
          dilation_width_factor, dilation_height_factor, pad_width, pad_height,
          depth_multiplier, output_shape, output_shift)) {
    DepthwiseConv3x3Filter(params, input_shape, input_data, filter_shape,
                           filter_data, bias_shape, bias_data, output_shape,
                           output_data);
    return;
  }
#endif

  DepthwiseConvGeneral(params, input_shape, input_data, filter_shape,
                       filter_data, bias_shape, bias_data, output_shape,
                       output_data);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_H_

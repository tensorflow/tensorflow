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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_FLOAT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_FLOAT_H_

#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Implementation of float DepthwiseConv

template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
struct FloatDepthwiseConvKernel {};

#ifdef USE_NEON

template <>
struct FloatDepthwiseConvKernel<false, 8, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vld1q_f32(filter_ptr + 4 * i);
    }
    int outp = 0;
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the inputs
      float32x4_t input[4];
      for (int i = 0; i < 4; i++) {
        input[i] = vld1q_f32(input_ptr + 4 * i);
      }
      input_ptr += 16;
      // Load the accumulators from acc_buffer
      float32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      acc[0] = vmlaq_f32(acc[0], input[0], filter[0]);
      acc[1] = vmlaq_f32(acc[1], input[1], filter[1]);
      acc[2] = vmlaq_f32(acc[2], input[2], filter[0]);
      acc[3] = vmlaq_f32(acc[3], input[3], filter[1]);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the inputs
      float32x4_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vld1q_f32(input_ptr + 4 * i);
      }
      input_ptr += 8;
      // Load the accumulators from acc_buffer
      float32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[i] = vmlaq_f32(acc[i], input[i], filter[i]);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<false, 2, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    const float32x2_t filters = vld1_f32(filter_ptr);
    const float32x4_t filters_dup2 = vcombine_f32(filters, filters);
    int outp = 0;
    // Handle 8 output pixels at a time.
    for (; outp <= num_output_pixels - 8; outp += 8) {
      // Load the inputs
      float32x4_t input[4];
      for (int i = 0; i < 4; i++) {
        input[i] = vld1q_f32(input_ptr + 4 * i);
      }
      input_ptr += 16;
      // Load the accumulators from acc_buffer
      float32x4_t acc[4];
      for (int i = 0; i < 4; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 4; i++) {
        acc[i] = vmlaq_f32(acc[i], input[i], filters_dup2);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 4; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 16;
    }
    // Handle 4 output pixels at a time.
    for (; outp <= num_output_pixels - 4; outp += 4) {
      // Load the inputs
      float32x4_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vld1q_f32(input_ptr + 4 * i);
      }
      input_ptr += 8;
      // Load the accumulators from acc_buffer
      float32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[i] = vmlaq_f32(acc[i], input[i], filters_dup2);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
    // Handle 2 output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the inputs
      const float32x4_t input = vld1q_f32(input_ptr);
      input_ptr += 4;
      // Load the accumulators from acc_buffer
      float32x4_t acc = vld1q_f32(acc_buffer_ptr);
      // Multiply-accumulate
      acc = vmlaq_f32(acc, input, filters_dup2);
      // Store the accumulators back to acc_buffer
      vst1q_f32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
    // Handle 1 output pixel at a time
    for (; outp < num_output_pixels; outp++) {
      // Load the inputs
      const float32x2_t input = vld1_f32(input_ptr);
      input_ptr += 2;
      // Load the accumulators from acc_buffer
      float32x2_t acc = vld1_f32(acc_buffer_ptr);
      // Multiply-accumulate
      acc = vmla_f32(acc, input, filters);
      // Store the accumulators back to acc_buffer
      vst1_f32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 2;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 0, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const float* local_filter_ptr = filter_ptr;
      const float* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 16 input channels at a time.
      for (; ic <= input_depth - 16; ic += 16) {
        // Load the filters
        float32x4_t filter_0 = vld1q_f32(local_filter_ptr + 4 * 0);
        float32x4_t filter_1 = vld1q_f32(local_filter_ptr + 4 * 1);
        float32x4_t filter_2 = vld1q_f32(local_filter_ptr + 4 * 2);
        float32x4_t filter_3 = vld1q_f32(local_filter_ptr + 4 * 3);
        local_filter_ptr += 16;
        // Load the inputs
        float32x4_t input_0 = vld1q_f32(local_input_ptr + 4 * 0);
        float32x4_t input_1 = vld1q_f32(local_input_ptr + 4 * 1);
        float32x4_t input_2 = vld1q_f32(local_input_ptr + 4 * 2);
        float32x4_t input_3 = vld1q_f32(local_input_ptr + 4 * 3);
        local_input_ptr += 16;
        // Load the accumulators from acc_buffer
        float32x4_t acc_0 = vld1q_f32(acc_buffer_ptr + 4 * 0);
        float32x4_t acc_1 = vld1q_f32(acc_buffer_ptr + 4 * 1);
        float32x4_t acc_2 = vld1q_f32(acc_buffer_ptr + 4 * 2);
        float32x4_t acc_3 = vld1q_f32(acc_buffer_ptr + 4 * 3);
        // Multiply-accumulate
        acc_0 = vmlaq_f32(acc_0, input_0, filter_0);
        acc_1 = vmlaq_f32(acc_1, input_1, filter_1);
        acc_2 = vmlaq_f32(acc_2, input_2, filter_2);
        acc_3 = vmlaq_f32(acc_3, input_3, filter_3);
        // Store the accumulators back to acc_buffer
        vst1q_f32(acc_buffer_ptr + 4 * 0, acc_0);
        vst1q_f32(acc_buffer_ptr + 4 * 1, acc_1);
        vst1q_f32(acc_buffer_ptr + 4 * 2, acc_2);
        vst1q_f32(acc_buffer_ptr + 4 * 3, acc_3);
        acc_buffer_ptr += 16;
      }
      // Handle 4 input channels at a time.
      for (; ic <= input_depth - 4; ic += 4) {
        // Load the filters
        float32x4_t filter;
        filter = vld1q_f32(local_filter_ptr);
        local_filter_ptr += 4;
        // Load the inputs
        float32x4_t input;
        input = vld1q_f32(local_input_ptr);
        local_input_ptr += 4;
        // Load the accumulators from acc_buffer
        float32x4_t acc;
        acc = vld1q_f32(acc_buffer_ptr);
        // Multiply-accumulate
        acc = vmlaq_f32(acc, input, filter);
        // Store the accumulators back to acc_buffer
        vst1q_f32(acc_buffer_ptr, acc);
        acc_buffer_ptr += 4;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        const float input_val = *local_input_ptr++;
        const float filter_val = *local_filter_ptr++;
        *acc_buffer_ptr++ += filter_val * input_val;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 0, 8> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const float* local_filter_ptr = filter_ptr;
      const float* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 2 input channels at a time.
      for (; ic <= input_depth - 2; ic += 2) {
        // Load the filters
        float32x4_t filter[4];
        for (int i = 0; i < 4; i++) {
          filter[i] = vld1q_f32(local_filter_ptr + 4 * i);
        }
        local_filter_ptr += 16;
        // Load the inputs
        const float32x2_t input = vld1_f32(local_input_ptr);
        local_input_ptr += 2;
        // Load the accumulators from acc_buffer
        float32x4_t acc[4];
        for (int i = 0; i < 4; i++) {
          acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
        }
        // Multiply-accumulate
        acc[0] = vmlaq_lane_f32(acc[0], filter[0], input, 0);
        acc[1] = vmlaq_lane_f32(acc[1], filter[1], input, 0);
        acc[2] = vmlaq_lane_f32(acc[2], filter[2], input, 1);
        acc[3] = vmlaq_lane_f32(acc[3], filter[3], input, 1);
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 4; i++) {
          vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
        }
        acc_buffer_ptr += 16;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        // Load the filters
        float32x4_t filter[2];
        for (int i = 0; i < 2; i++) {
          filter[i] = vld1q_f32(local_filter_ptr + 4 * i);
        }
        local_filter_ptr += 8;
        // Load the inputs
        const float input_val = *local_input_ptr++;
        // Load the accumulators from acc_buffer
        float32x4_t acc[2];
        for (int i = 0; i < 2; i++) {
          acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
        }
        // Multiply-accumulate
        for (int i = 0; i < 2; i++) {
          acc[i] = vmlaq_n_f32(acc[i], filter[i], input_val);
        }
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 2; i++) {
          vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
        }
        acc_buffer_ptr += 8;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

// Note this implementation is very slow for input_depths < 8
// (e.g. comparable to reference implementation) see, specializations for
// input_depth=3 below.
template <>
struct FloatDepthwiseConvKernel<true, 0, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const float* local_filter_ptr = filter_ptr;
      const float* local_input_ptr = input_ptr;
      int ic = 0;
      // Handle 8 input channels at a time.
      for (; ic <= input_depth - 8; ic += 8) {
        // Load the filters
        float32x4_t filter[4];
        for (int i = 0; i < 4; i++) {
          filter[i] = vld1q_f32(local_filter_ptr + 4 * i);
        }
        local_filter_ptr += 16;
        // Load the inputs
        float32x4x2_t input_dup2[2];
        for (int i = 0; i < 2; i++) {
          const float32x4_t input = vld1q_f32(local_input_ptr + 4 * i);
          input_dup2[i] = vzipq_f32(input, input);
        }
        local_input_ptr += 8;
        // Load the accumulators from acc_buffer
        float32x4_t acc[4];
        for (int i = 0; i < 4; i++) {
          acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
        }
        // Multiply-accumulate
        acc[0] = vmlaq_f32(acc[0], filter[0], input_dup2[0].val[0]);
        acc[1] = vmlaq_f32(acc[1], filter[1], input_dup2[0].val[1]);
        acc[2] = vmlaq_f32(acc[2], filter[2], input_dup2[1].val[0]);
        acc[3] = vmlaq_f32(acc[3], filter[3], input_dup2[1].val[1]);
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 4; i++) {
          vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
        }
        acc_buffer_ptr += 16;
      }
      // Handle 4 input channels at a time.
      for (; ic <= input_depth - 4; ic += 4) {
        // Load the filters
        float32x2_t filter[4];
        for (int i = 0; i < 4; i++) {
          filter[i] = vld1_f32(local_filter_ptr + 2 * i);
        }
        local_filter_ptr += 8;
        // Load the inputs
        const float32x4_t input = vld1q_f32(local_input_ptr);
        local_input_ptr += 4;
        // Load the accumulators from acc_buffer
        float32x2_t acc[4];
        for (int i = 0; i < 4; i++) {
          acc[i] = vld1_f32(acc_buffer_ptr + 2 * i);
        }
        // Multiply-accumulate
        acc[0] = vmla_lane_f32(acc[0], filter[0], vget_low_f32(input), 0);
        acc[1] = vmla_lane_f32(acc[1], filter[1], vget_low_f32(input), 1);
        acc[2] = vmla_lane_f32(acc[2], filter[2], vget_high_f32(input), 0);
        acc[3] = vmla_lane_f32(acc[3], filter[3], vget_high_f32(input), 1);
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 4; i++) {
          vst1_f32(acc_buffer_ptr + 2 * i, acc[i]);
        }
        acc_buffer_ptr += 8;
      }
      // Handle 2 input channels at a time.
      for (; ic <= input_depth - 2; ic += 2) {
        // Load the filters
        const float32x4_t filter = vld1q_f32(local_filter_ptr);
        local_filter_ptr += 4;
        // Load the inputs
        const float32x2_t input = vld1_f32(local_input_ptr);
        local_input_ptr += 2;
        // Load the accumulators from acc_buffer
        float32x2_t acc[2];
        for (int i = 0; i < 2; i++) {
          acc[i] = vld1_f32(acc_buffer_ptr + 2 * i);
        }
        // Multiply-accumulate
        acc[0] = vmla_lane_f32(acc[0], vget_low_f32(filter), input, 0);
        acc[1] = vmla_lane_f32(acc[1], vget_high_f32(filter), input, 1);
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 2; i++) {
          vst1_f32(acc_buffer_ptr + 2 * i, acc[i]);
        }
        acc_buffer_ptr += 4;
      }
      // Handle one input channel at a time.
      for (; ic < input_depth; ic++) {
        // Load the inputs
        const float input_val = *local_input_ptr++;
        // Multiply-accumulate
        for (int i = 0; i < 2; i++) {
          acc_buffer_ptr[i] += local_filter_ptr[i] * input_val;
        }
        local_filter_ptr += 2;
        acc_buffer_ptr += 2;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 3, 2> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x2_t filter[3];
    for (int i = 0; i < 3; i++) {
      filter[i] = vld1_f32(filter_ptr + 2 * i);
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const float32x2_t input01 = vld1_f32(input_ptr);
      const float32x2_t input2 = vld1_dup_f32(input_ptr + 2);
      // Load the accumulators from acc_buffer
      float32x2_t acc[3];
      for (int i = 0; i < 3; i++) {
        acc[i] = vld1_f32(acc_buffer_ptr + 2 * i);
      }
      // Multiply-accumulate for each input channel there 2 outputs
      acc[0] = vmla_lane_f32(acc[0], filter[0], input01, 0);
      acc[1] = vmla_lane_f32(acc[1], filter[1], input01, 1);
      acc[2] = vmla_lane_f32(acc[2], filter[2], input2, 0);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 3; i++) {
        vst1_f32(acc_buffer_ptr + 2 * i, acc[i]);
      }
      acc_buffer_ptr += 6;
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 3, 4> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter[3];
    for (int i = 0; i < 3; i++) {
      filter[i] = vld1q_f32(filter_ptr + 4 * i);
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // NOTE: we only want 3 values, so we read it as two ops where
      // the second op just duplicates the lane
      const float32x2_t input01 = vld1_f32(input_ptr);
      const float32x2_t input2 = vld1_dup_f32(input_ptr + 2);
      // Load the accumulators from acc_buffer
      float32x4_t acc[3];
      for (int i = 0; i < 3; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate all outputs.
      acc[0] = vmlaq_lane_f32(acc[0], filter[0], input01, 0);
      acc[1] = vmlaq_lane_f32(acc[1], filter[1], input01, 1);
      acc[2] = vmlaq_lane_f32(acc[2], filter[2], input2, 0);
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 3; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 12;
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 1, 8> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vld1q_f32(filter_ptr + 4 * i);
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs
      const float input_val = *input_ptr;
      input_ptr += input_ptr_increment;
      // Load the accumulators from acc_buffer
      float32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[i] = vmlaq_n_f32(acc[i], filter[i], input_val);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 1, 32> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter_0 = vld1q_f32(filter_ptr + 4 * 0);
    float32x4_t filter_1 = vld1q_f32(filter_ptr + 4 * 1);
    float32x4_t filter_2 = vld1q_f32(filter_ptr + 4 * 2);
    float32x4_t filter_3 = vld1q_f32(filter_ptr + 4 * 3);
    float32x4_t filter_4 = vld1q_f32(filter_ptr + 4 * 4);
    float32x4_t filter_5 = vld1q_f32(filter_ptr + 4 * 5);
    float32x4_t filter_6 = vld1q_f32(filter_ptr + 4 * 6);
    float32x4_t filter_7 = vld1q_f32(filter_ptr + 4 * 7);

    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs
      const float input_val = *input_ptr;
      input_ptr += input_ptr_increment;
      // Load the accumulators from acc_buffer
      float32x4_t acc_0 = vld1q_f32(acc_buffer_ptr + 4 * 0);
      float32x4_t acc_1 = vld1q_f32(acc_buffer_ptr + 4 * 1);
      float32x4_t acc_2 = vld1q_f32(acc_buffer_ptr + 4 * 2);
      float32x4_t acc_3 = vld1q_f32(acc_buffer_ptr + 4 * 3);
      float32x4_t acc_4 = vld1q_f32(acc_buffer_ptr + 4 * 4);
      float32x4_t acc_5 = vld1q_f32(acc_buffer_ptr + 4 * 5);
      float32x4_t acc_6 = vld1q_f32(acc_buffer_ptr + 4 * 6);
      float32x4_t acc_7 = vld1q_f32(acc_buffer_ptr + 4 * 7);
      // Multiply-accumulate
      acc_0 = vmlaq_n_f32(acc_0, filter_0, input_val);
      acc_1 = vmlaq_n_f32(acc_1, filter_1, input_val);
      acc_2 = vmlaq_n_f32(acc_2, filter_2, input_val);
      acc_3 = vmlaq_n_f32(acc_3, filter_3, input_val);
      acc_4 = vmlaq_n_f32(acc_4, filter_4, input_val);
      acc_5 = vmlaq_n_f32(acc_5, filter_5, input_val);
      acc_6 = vmlaq_n_f32(acc_6, filter_6, input_val);
      acc_7 = vmlaq_n_f32(acc_7, filter_7, input_val);
      // Store the accumulators back to acc_buffer
      vst1q_f32(acc_buffer_ptr + 4 * 0, acc_0);
      vst1q_f32(acc_buffer_ptr + 4 * 1, acc_1);
      vst1q_f32(acc_buffer_ptr + 4 * 2, acc_2);
      vst1q_f32(acc_buffer_ptr + 4 * 3, acc_3);
      vst1q_f32(acc_buffer_ptr + 4 * 4, acc_4);
      vst1q_f32(acc_buffer_ptr + 4 * 5, acc_5);
      vst1q_f32(acc_buffer_ptr + 4 * 6, acc_6);
      vst1q_f32(acc_buffer_ptr + 4 * 7, acc_7);
      acc_buffer_ptr += 32;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 1, 20> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter_0 = vld1q_f32(filter_ptr + 4 * 0);
    float32x4_t filter_1 = vld1q_f32(filter_ptr + 4 * 1);
    float32x4_t filter_2 = vld1q_f32(filter_ptr + 4 * 2);
    float32x4_t filter_3 = vld1q_f32(filter_ptr + 4 * 3);
    float32x4_t filter_4 = vld1q_f32(filter_ptr + 4 * 4);

    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs
      const float input_val = *input_ptr;
      input_ptr += input_ptr_increment;
      // Load the accumulators from acc_buffer
      float32x4_t acc_0 = vld1q_f32(acc_buffer_ptr + 4 * 0);
      float32x4_t acc_1 = vld1q_f32(acc_buffer_ptr + 4 * 1);
      float32x4_t acc_2 = vld1q_f32(acc_buffer_ptr + 4 * 2);
      float32x4_t acc_3 = vld1q_f32(acc_buffer_ptr + 4 * 3);
      float32x4_t acc_4 = vld1q_f32(acc_buffer_ptr + 4 * 4);
      // Multiply-accumulate
      acc_0 = vmlaq_n_f32(acc_0, filter_0, input_val);
      acc_1 = vmlaq_n_f32(acc_1, filter_1, input_val);
      acc_2 = vmlaq_n_f32(acc_2, filter_2, input_val);
      acc_3 = vmlaq_n_f32(acc_3, filter_3, input_val);
      acc_4 = vmlaq_n_f32(acc_4, filter_4, input_val);
      // Store the accumulators back to acc_buffer
      vst1q_f32(acc_buffer_ptr + 4 * 0, acc_0);
      vst1q_f32(acc_buffer_ptr + 4 * 1, acc_1);
      vst1q_f32(acc_buffer_ptr + 4 * 2, acc_2);
      vst1q_f32(acc_buffer_ptr + 4 * 3, acc_3);
      vst1q_f32(acc_buffer_ptr + 4 * 4, acc_4);
      acc_buffer_ptr += 20;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 0, 16> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      const float* local_filter_ptr = filter_ptr;
      const float* local_input_ptr = input_ptr;
      for (int ic = 0; ic < input_depth; ic++) {
        // Load the filters
        float32x4_t filter[4];
        for (int i = 0; i < 4; i++) {
          filter[i] = vld1q_f32(local_filter_ptr + 4 * i);
        }
        local_filter_ptr += 16;
        // Load the inputs
        const float input_val = *local_input_ptr++;
        // Load the accumulators from acc_buffer
        float32x4_t acc[4];
        for (int i = 0; i < 4; i++) {
          acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
        }
        // Multiply-accumulate
        for (int i = 0; i < 4; i++) {
          acc[i] = vmlaq_n_f32(acc[i], filter[i], input_val);
        }
        // Store the accumulators back to acc_buffer
        for (int i = 0; i < 4; i++) {
          vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
        }
        acc_buffer_ptr += 16;
      }
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 8, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    // Load the filters
    float32x4_t filter[2];
    for (int i = 0; i < 2; i++) {
      filter[i] = vld1q_f32(filter_ptr + 4 * i);
    }
    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs
      float32x4_t input[2];
      for (int i = 0; i < 2; i++) {
        input[i] = vld1q_f32(input_ptr + 4 * i);
      }
      // Load the accumulators from acc_buffer
      float32x4_t acc[2];
      for (int i = 0; i < 2; i++) {
        acc[i] = vld1q_f32(acc_buffer_ptr + 4 * i);
      }
      // Multiply-accumulate
      for (int i = 0; i < 2; i++) {
        acc[i] = vmlaq_f32(acc[i], input[i], filter[i]);
      }
      // Store the accumulators back to acc_buffer
      for (int i = 0; i < 2; i++) {
        vst1q_f32(acc_buffer_ptr + 4 * i, acc[i]);
      }
      acc_buffer_ptr += 8;
      input_ptr += input_ptr_increment;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 2, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    float32x2_t filter = vld1_f32(filter_ptr);
    float32x4_t filter_x4 = vcombine_f32(filter, filter);
    int outp = 0;

    // Handle two output pixels at a time.
    for (; outp <= num_output_pixels - 2; outp += 2) {
      // Load the inputs
      float32x2_t input_1 = vld1_f32(input_ptr);
      input_ptr += input_ptr_increment;
      float32x2_t input_2 = vld1_f32(input_ptr);
      input_ptr += input_ptr_increment;
      float32x4_t input = vcombine_f32(input_1, input_2);

      // Load the accumulators from acc_buffer
      float32x4_t acc = vld1q_f32(acc_buffer_ptr);

      // Multiply-accumulate
      acc = vmlaq_f32(acc, input, filter_x4);

      // Store the accumulators back to acc_buffer
      vst1q_f32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
    }
    // Handle one output pixel at a time.
    for (; outp < num_output_pixels; outp++) {
      // Load the inputs
      float32x2_t input = vld1_f32(input_ptr);
      input_ptr += input_ptr_increment;

      // Load the accumulators from acc_buffer
      float32x2_t acc = vld1_f32(acc_buffer_ptr);

      // Multiply-accumulate
      acc = vmla_f32(acc, input, filter);

      // Store the accumulators back to acc_buffer
      vst1_f32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 2;
    }
  }
};

template <>
struct FloatDepthwiseConvKernel<true, 4, 1> {
  static void Run(int num_output_pixels, int input_depth, int depth_multiplier,
                  const float* input_ptr, int input_ptr_increment,
                  const float* filter_ptr, float* acc_buffer_ptr) {
    float32x4_t filter = vld1q_f32(filter_ptr);

    // Handle one output pixel at a time.
    for (int outp = 0; outp < num_output_pixels; outp++) {
      // Load the inputs
      float32x4_t input = vld1q_f32(input_ptr);
      // Load the accumulators from acc_buffer
      float32x4_t acc = vld1q_f32(acc_buffer_ptr);
      // Multiply-accumulate
      acc = vmlaq_f32(acc, input, filter);
      // Store the accumulators back to acc_buffer
      vst1q_f32(acc_buffer_ptr, acc);
      acc_buffer_ptr += 4;
      input_ptr += input_ptr_increment;
    }
  }
};
#endif

// Accumulates the effect of one row of the filter, on a segment of one row
// of the output, accessing the corresponding one row of the input.
template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
void FloatDepthwiseConvAccumRow(int stride, int dilation_factor,
                                int input_depth, int input_width,
                                const float* input_data, int pad_width,
                                int depth_multiplier, int filter_width,
                                const float* filter_data,
                                int out_x_buffer_start, int out_x_buffer_end,
                                int output_depth, float* acc_buffer) {
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
  const float* filter_base_ptr = filter_data;
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

    float* acc_buffer_ptr =
        acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin =
        (out_x_loop_start * stride) - pad_width + dilation_factor * filter_x;
    const float* input_ptr = input_data + in_x_origin * input_depth;
    const int num_output_pixels = out_x_loop_end - out_x_loop_start;
    FloatDepthwiseConvKernel<kAllowStrided, kFixedInputDepth,
                             kFixedDepthMultiplier>::Run(num_output_pixels,
                                                         input_depth,
                                                         depth_multiplier,
                                                         input_ptr,
                                                         input_ptr_increment,
                                                         filter_base_ptr,
                                                         acc_buffer_ptr);
    filter_base_ptr += output_depth;
  }
}

// generic fallback of FloatDepthwiseConvAccumRow, portable, non-templatized.
inline void FloatDepthwiseConvAccumRowGeneric(
    int stride, int dilation_factor, int input_depth, int input_width,
    const float* input_data, int pad_width, int depth_multiplier,
    int filter_width, const float* filter_data, int out_x_buffer_start,
    int out_x_buffer_end, int output_depth, float* acc_buffer) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConvAccumRowGeneric (slow)");
  const float* filter_base_ptr = filter_data;
  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
    const int out_x_loop_start = std::max(
        out_x_buffer_start,
        (pad_width - dilation_factor * filter_x + stride - 1) / stride);
    const int out_x_loop_end = std::min(
        out_x_buffer_end,
        (pad_width + input_width - dilation_factor * filter_x + stride - 1) /
            stride);

    float* acc_buffer_ptr =
        acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin =
        (out_x_loop_start * stride) - pad_width + dilation_factor * filter_x;
    const float* input_ptr = input_data + in_x_origin * input_depth;
    const int input_ptr_increment = (stride - 1) * input_depth;
    for (int out_x = out_x_loop_start; out_x < out_x_loop_end; out_x++) {
      const float* filter_ptr = filter_base_ptr;
      for (int ic = 0; ic < input_depth; ++ic) {
        const float input_val = *input_ptr++;
        for (int m = 0; m < depth_multiplier; m++) {
          const float filter_val = *filter_ptr++;
          *acc_buffer_ptr++ += filter_val * input_val;
        }
      }
      input_ptr += input_ptr_increment;
    }
    filter_base_ptr += output_depth;
  }
}

// Initializes the accumulator buffer with bias values.
inline void DepthwiseConvInitAccBuffer(int num_output_pixels, int output_depth,
                                       const float* bias_data,
                                       float* acc_buffer) {
  // TODO(benoitjacob): This might need optimized specializations
  // for small output_depth values, if that ever becomes an important
  // case (like it was for some quantized DepthwiseConv cases).
  for (int i = 0; i < num_output_pixels; i++) {
    memcpy(acc_buffer + i * output_depth, bias_data,
           sizeof(acc_buffer[0]) * output_depth);
  }
}

// DepthwiseConv can run with multi threads on the dim specified by thread_dim.
// Each thread processes output elements on dim, thread_dim, in the range of
// [thread_start, thread_end).
// For example, assume thread_start = 2, thread_end = 6, and thread_dim = 1, it
// means that it will calculate DepthwiseConv for output_data[:, 2:5, :, :].
//
// The cpu_backend_context may be supplied as a nullptr by some callers. This
// parameter is included so that the signature matches that required by a
// templated function. Other versions, such as quantized, need this parameter.
inline void DepthwiseConvImpl(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data, CpuBackendContext* cpu_backend_context,
    int thread_start, int thread_end, int thread_dim) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConv/float/DepthwiseConvImpl");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK(thread_dim == 0 || thread_dim == 1);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  static const int kAccBufferMaxSize = 4832;
  float acc_buffer[kAccBufferMaxSize];
  TFLITE_DCHECK_GE(kAccBufferMaxSize, output_depth);
  const int kOutputPixelsInAccBuffer = kAccBufferMaxSize / output_depth;
  const int kAccBufferActualSize = kOutputPixelsInAccBuffer * output_depth;
  TFLITE_DCHECK_LE(kOutputPixelsInAccBuffer * output_depth,
                   kAccBufferActualSize);
  TFLITE_DCHECK_LE(kAccBufferActualSize, kAccBufferMaxSize);
  TFLITE_DCHECK_GE(kOutputPixelsInAccBuffer, 1);

  // row_accum_func will point to the core accumulation function to be used
  // for this DepthwiseConv op.
  using row_accum_func_t = decltype(&FloatDepthwiseConvAccumRowGeneric);
  row_accum_func_t row_accum_func = nullptr;

#define TFMINI_USE_DEPTHWISECONV_KERNEL(ALLOW_STRIDED, FIXED_INPUT_DEPTH, \
                                        FIXED_DEPTH_MULTIPLIER)           \
  if (!row_accum_func && (stride_width == 1 || ALLOW_STRIDED) &&          \
      (input_depth == FIXED_INPUT_DEPTH || FIXED_INPUT_DEPTH == 0) &&     \
      depth_multiplier == FIXED_DEPTH_MULTIPLIER) {                       \
    row_accum_func =                                                      \
        FloatDepthwiseConvAccumRow<ALLOW_STRIDED, FIXED_INPUT_DEPTH,      \
                                   FIXED_DEPTH_MULTIPLIER>;               \
  }

#ifdef USE_NEON
  // We go over our list of kernels by decreasing order of preference
  // for the cases where multiple kernels could apply.

  // Start with the fastest kernels: AllowStrided=false, fixed input depth.

  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 1)

  // Next come the strided kernels: AllowStrided=true, fixed input depth.
  // They are a bit less efficient, but allow stride!=1.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 20)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 32)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 2, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 3, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 3, 4)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 4, 1)

  // Finally, the kernels allowing a variable input depth,
  // these are the least efficient but most general kernels.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 16)

#endif  // USE_NEON

#undef TFMINI_USE_DEPTHWISECONV_KERNEL

  // No matching fast kernel found, use slow fallback.
  if (!row_accum_func) {
    row_accum_func = FloatDepthwiseConvAccumRowGeneric;
  }

  const int input_height_stride = input_shape.Dims(3) * input_shape.Dims(2);
  const int input_batch_stride = input_height_stride * input_shape.Dims(1);
  const int filter_height_stride = filter_shape.Dims(3) * filter_shape.Dims(2);

  // Now that we have determined row_accum_func, we can start work.
  int batch_start = 0;
  int batch_end = batches;
  int row_start = 0;
  int row_end = output_height;
  int output_ptr_offset = 0;

  switch (thread_dim) {
    case 0:
      // Multithread along with the batch axis
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, batches);
      batch_start = thread_start;
      batch_end = thread_end;
      output_ptr_offset = batch_start * FlatSizeSkipDim(output_shape, 0);
      break;
    case 1:
      // Multithread along with the row axis
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, output_height);
      row_start = thread_start;
      row_end = thread_end;
      output_ptr_offset = row_start * output_width * output_depth;
      break;
  }

  float* output_ptr = output_data + output_ptr_offset;
  int batch_step =
      (output_height + row_start - row_end) * output_width * output_depth;

  for (int b = batch_start; b < batch_end; ++b) {
    for (int out_y = row_start; out_y < row_end; ++out_y) {
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
              pad_width, depth_multiplier, filter_width,
              filter_data + filter_y * filter_height_stride, out_x_buffer_start,
              out_x_buffer_end, output_depth, acc_buffer);
        }
        // Finished accumulating. Now store to destination.
        const int num_output_values = output_depth * num_output_pixels;
        int i = 0;
// TODO(benoitjacob) optimized code goes here
#ifdef USE_NEON
        // Handle 16 values at a time
        for (; i <= num_output_values - 16; i += 16) {
          float32x4_t acc[4];
          for (int k = 0; k < 4; k++) {
            acc[k] = vld1q_f32(acc_buffer + i + 4 * k);
          }
          for (int k = 0; k < 4; k++) {
            acc[k] = vmaxq_f32(
                vdupq_n_f32(output_activation_min),
                vminq_f32(vdupq_n_f32(output_activation_max), acc[k]));
          }
          for (int k = 0; k < 4; k++) {
            vst1q_f32(output_ptr + 4 * k, acc[k]);
          }
          output_ptr += 16;
        }
        // Handle 4 values at a time
        for (; i <= num_output_values - 4; i += 4) {
          float32x4_t acc = vld1q_f32(acc_buffer + i);

          acc = vmaxq_f32(vdupq_n_f32(output_activation_min),
                          vminq_f32(vdupq_n_f32(output_activation_max), acc));

          vst1q_f32(output_ptr, acc);
          output_ptr += 4;
        }
#endif
        // Handle leftover values, one by one. This is very slow.
        for (; i < num_output_values; i++) {
          float acc = acc_buffer[i];
          acc = std::max(output_activation_min,
                         std::min(output_activation_max, acc));

          *output_ptr++ = acc;
        }
      }
    }
    output_ptr += batch_step;
  }
}

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data, CpuBackendContext* cpu_backend_context) {
  DepthwiseConvImpl(params, input_shape, input_data, filter_shape, filter_data,
                    bias_shape, bias_data, output_shape, output_data,
                    cpu_backend_context,
                    /*thread_start=*/0,
                    /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_FLOAT_H_

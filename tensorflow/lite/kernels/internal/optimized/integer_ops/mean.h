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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace optimized_integer_ops {

#ifdef USE_NEON

using optimized_ops::DivideSumForMeanImpl;
using optimized_ops::RoundToNearest;

#endif  // USE_NEON

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const int8_t* input_data,
                     int32 input_zero_point, float input_scale,
                     const RuntimeShape& output_shape, int8_t* output_data,
                     int32 output_zero_point, float output_scale,
                     int start_depth, int end_depth) {
  gemmlowp::ScopedProfilingLabel label("Mean4D/Int8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  const bool ordinary_mean =
      (input_zero_point == output_zero_point && input_scale == output_scale);
  float scale = 0.0f, bias = 0.0f;
  if (!ordinary_mean) {
    scale = input_scale / output_scale;
    bias = -input_zero_point * scale + 0.5;
  }

#ifdef USE_NEON
  const float32x4_t num_elements_dup = vdupq_n_f32(num_elements_in_axis);
  // This is only an approximation as NEON does not offer division instruction.
  const float32x4_t scale_dup = vdupq_n_f32(scale);
  const float32x4_t num_elements_reverse = vrecpeq_f32(num_elements_dup);
  float32x4_t zero_point_with_bias_dup = vdupq_n_f32(output_zero_point + bias);
#endif  // USE_NEON

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d < end_depth - 8; out_d += 8) {
      float32x4_t temp_sum_1 = vdupq_n_f32(0);
      float32x4_t temp_sum_2 = vdupq_n_f32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const int8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          int8x8_t input_data_val = vld1_s8(input_data_ptr);
          int16x8_t input_data_val_shift = vmovl_s8(input_data_val);
          float32x4_t input_float_1 =
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_data_val_shift)));
          float32x4_t input_float_2 =
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_data_val_shift)));
          temp_sum_1 = vaddq_f32(temp_sum_1, input_float_1);
          temp_sum_2 = vaddq_f32(temp_sum_2, input_float_2);
        }
      }

      const float32x4_t mean_1 =
          DivideSumForMeanImpl(temp_sum_1, num_elements_reverse, ordinary_mean,
                               scale_dup, zero_point_with_bias_dup);
      const float32x4_t mean_2 =
          DivideSumForMeanImpl(temp_sum_2, num_elements_reverse, ordinary_mean,
                               scale_dup, zero_point_with_bias_dup);

      int32x4_t casted_mean_1 = RoundToNearest(mean_1);
      int16x4_t narrow_range_mean_1 = vmovn_s32(casted_mean_1);
      int32x4_t casted_mean_2 = RoundToNearest(mean_2);
      int16x4_t narrow_range_mean_2 = vmovn_s32(casted_mean_2);
      int16x8_t combined_mean =
          vcombine_s16(narrow_range_mean_2, narrow_range_mean_1);
      int8x8_t narrowed_combined_mean = vmovn_s16(combined_mean);
      int8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1_s8(output_data_ptr, narrowed_combined_mean);
    }
#endif  // USE_NEON

    for (; out_d < end_depth; ++out_d) {
      float temp_value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          temp_value +=
              input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }

      temp_value = temp_value / num_elements_in_axis;
      if (ordinary_mean) {
        output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
            static_cast<int8_t>(round(temp_value));
      } else {
        output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
            static_cast<int8_t>(round(temp_value * scale + bias)) +
            output_zero_point;
      }
    }
  }
}

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const int8_t* input_data,
                 int32 input_zero_point, float input_scale,
                 const RuntimeShape& output_shape, int8_t* output_data,
                 int32 output_zero_point, float output_scale, int start_height,
                 int end_height)
      : op_params(op_params),
        input_shape(input_shape),
        input_data(input_data),
        input_zero_point(input_zero_point),
        input_scale(input_scale),
        output_shape(output_shape),
        output_data(output_data),
        output_zero_point(output_zero_point),
        output_scale(output_scale),
        start_height(start_height),
        end_height(end_height) {}

  void Run() override {
    MeanImpl(op_params, input_shape, input_data, input_zero_point, input_scale,
             output_shape, output_data, output_zero_point, output_scale,
             start_height, end_height);
  }

 private:
  const tflite::MeanParams& op_params;
  const RuntimeShape& input_shape;
  const int8_t* input_data;
  int32 input_zero_point;
  float input_scale;
  const RuntimeShape& output_shape;
  int8_t* output_data;
  int32 output_zero_point;
  float output_scale;
  int start_height;
  int end_height;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const int8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 int8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("Mean4D/Int8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, input_zero_point, input_scale,
             output_shape, output_data, output_zero_point, output_scale, 0,
             output_depth);
  } else {
    // Instead parrallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, input_zero_point,
                         input_scale, output_shape, output_data,
                         output_zero_point, output_scale, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }
              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                float input_value =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                float filter_value = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                total += (input_value * filter_value);
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8_t* input_data, const RuntimeShape& filter_shape,
                 const uint8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape,
                 uint8_t* output_data, const RuntimeShape& im2col_shape,
                 uint8_t* im2col_data, void* cpu_backend_context) {
  (void)cpu_backend_context;  // only used in optimized code.
  (void)im2col_data;          // only used in optimized code.
  (void)im2col_shape;         // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc +=
                    (filter_val + filter_offset) * (input_val + input_offset);
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

inline void HybridConvPerChannel(
    const ConvParams& params, float* scaling_factors_ptr,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const RuntimeShape& im2col_shape, int8_t* im2col_data,
    const float* per_channel_scale, int32_t* input_offset) {
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x,
                      in_channel + group * filter_input_depth)];
                  int32_t filter_val =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  acc += filter_val * (input_val - input_offset[batch]);
                }
              }
            }
          }
          float acc_float =
              acc * per_channel_scale[out_channel] * scaling_factors_ptr[batch];
          if (bias_data) {
            acc_float += bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              ActivationFunctionWithMinMax(acc_float, output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_

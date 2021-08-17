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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {
inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
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

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8_t, even though
                  // it is represented using int32_t. int32_t += int8_t *
                  // (int8_t - int8_t) so the highest value we can get from each
                  // accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(b/174275578): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[output_channel],
                output_shift[output_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               output_channel)] = static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}

inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const std::int64_t* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
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

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            std::int64_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  // Accumulate with 64 bits accumulator.
                  // We assume maximum of 2^16 accumulations as with the 8-bit
                  // case so actually the value in the accumulator should not
                  // exceed 40 bits
                  acc += static_cast<int64_t>(filter_val) *
                         static_cast<int64_t>(input_val);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            int32_t scaled_acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[output_channel],
                output_shift[output_channel]);
            scaled_acc = std::max(scaled_acc, output_activation_min);
            scaled_acc = std::min(scaled_acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               output_channel)] =
                static_cast<int16_t>(scaled_acc);
          }
        }
      }
    }
  }
}

inline void DepthwiseConvHybridPerChannel(
    const DepthwiseParams& params, float* scaling_factors_ptr,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scale, int32_t* input_offset) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int bias_depth = bias_shape.FlatSize();
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_depth, output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  acc += filter_val * (input_val - input_offset[batch]);
                }
              }
            }
            float acc_float = static_cast<float>(acc);
            acc_float *=
                per_channel_scale[output_channel] * scaling_factors_ptr[batch];
            if (bias_data && output_channel < bias_depth) {
              acc_float += bias_data[output_channel];
            }
            output_data[Offset(output_shape, batch, out_y, out_x,
                               output_channel)] =
                ActivationFunctionWithMinMax(acc_float, output_activation_min,
                                             output_activation_max);
          }
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_

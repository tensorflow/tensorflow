/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV3D_TRANPOSE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV3D_TRANPOSE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

template <typename InputType, typename BiasType>
inline void Conv3DTransposePerChannel(
    const Conv3DTransposeParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    InputType* output_data, BiasType* scratch_buffer) {
  using AccumulatorType = BiasType;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int stride_depth = params.stride_depth;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int pad_depth = params.padding_values.depth;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_num_channels = MatchingDim(input_shape, 4, filter_shape, 4);
  const int output_num_channels = output_shape.Dims(4);
  const int input_depth = input_shape.Dims(1);
  const int input_height = input_shape.Dims(2);
  const int input_width = input_shape.Dims(3);
  const int filter_depth = filter_shape.Dims(0);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_output_depth = filter_shape.Dims(3);
  const int output_depth = output_shape.Dims(1);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(3);
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = std::numeric_limits<InputType>::min();
  const int32_t output_activation_max = std::numeric_limits<InputType>::max();
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_num_channels);
  }

  const int num_elements = output_shape.FlatSize();
  // We need to initialize scratch_buffer to all 0s, as we apply the same
  // 'scatter' based trick as in float version.
  memset(scratch_buffer, 0, num_elements * sizeof(BiasType));

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    const int in_idx_b = batch * input_depth;
    const int out_idx_b = batch * output_depth;
    for (int in_d = 0; in_d < input_depth; ++in_d) {
      const int in_idx_d = (in_idx_b + in_d) * input_height;
      for (int in_y = 0; in_y < input_height; ++in_y) {
        const int in_idx_y = (in_idx_d + in_y) * input_width;
        for (int in_x = 0; in_x < input_width; ++in_x) {
          const int in_idx_x = (in_idx_y + in_x) * input_num_channels;
          for (int in_channel = 0; in_channel < input_num_channels;
               ++in_channel) {
            const int in_idx = in_idx_x + in_channel;
            // Loop through the output elements it will influence.
            const int out_x_origin = (in_x * stride_width) - pad_width;
            const int out_y_origin = (in_y * stride_height) - pad_height;
            const int out_d_origin = (in_d * stride_depth) - pad_depth;
            for (int filter_d = 0; filter_d < filter_depth; ++filter_d) {
              const int out_d = out_d_origin + params.dilation_depth * filter_d;
              const int flt_idx_d = filter_d * filter_height;
              const int out_idx_d = (out_idx_b + out_d) * output_height;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                const int out_y =
                    out_y_origin + params.dilation_height * filter_y;
                const int flt_idx_y = (flt_idx_d + filter_y) * filter_width;
                const int out_idx_y = (out_idx_d + out_y) * output_width;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int out_x =
                      out_x_origin + params.dilation_width * filter_x;
                  const int flt_idx_x =
                      (flt_idx_y + filter_x) * filter_output_depth;
                  const int out_idx_x =
                      (out_idx_y + out_x) * output_num_channels;
                  for (int out_channel = 0; out_channel < output_num_channels;
                       ++out_channel) {
                    // Zero padding by omitting the areas outside the output.
                    const bool is_point_inside_output =
                        (out_x >= 0) && (out_x < output_width) &&
                        (out_y >= 0) && (out_y < output_height) &&
                        (out_d >= 0) && (out_d < output_depth);
                    if (!is_point_inside_output) {
                      continue;
                    }
                    const int flt_idx =
                        (flt_idx_x + out_channel) * input_num_channels +
                        in_channel;
                    const int out_idx = out_idx_x + out_channel;
                    const int32_t input_value = input_data[in_idx];
                    const int32_t filter_value = filter_data[flt_idx];
                    scratch_buffer[out_idx] +=
                        (input_value + input_offset) * filter_value;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  for (int batch = 0; batch < batches; ++batch) {
    const int out_idx_b = batch * output_depth;
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      const int out_idx_d = (out_idx_b + out_d) * output_height;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int out_idx_y = (out_idx_d + out_y) * output_width;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int out_idx_x = (out_idx_y + out_x) * output_num_channels;
          for (int out_channel = 0; out_channel < output_num_channels;
               ++out_channel) {
            const int out_idx = out_idx_x + out_channel;
            AccumulatorType acc = scratch_buffer[out_idx];
            if (bias_data) {
              acc += bias_data[out_channel];
            }
            int32_t scaled_acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_channel], output_shift[out_channel]);
            scaled_acc += output_offset;
            scaled_acc = std::max(scaled_acc, output_activation_min);
            scaled_acc = std::min(scaled_acc, output_activation_max);
            output_data[out_idx] = static_cast<InputType>(scaled_acc);
          }
        }
      }
    }
  }
}
}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV3D_TRANPOSE_H_

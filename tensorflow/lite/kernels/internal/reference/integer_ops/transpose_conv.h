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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_TRANSPOSE_CONV_H_

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization transpose convolution reference kernel.
inline void TransposeConv(
    const ConvParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& output_shape,
    int8* output_data, const RuntimeShape& im2col_shape, int8* im2col_data,
    int32* scratch_buffer) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_activation_min = std::numeric_limits<int8_t>::min();
  const int32 output_activation_max = std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int num_elements = output_shape.FlatSize();
  // We need to initialize scratch_buffer to all 0s, as we apply the same
  // 'scatter' based trick as in float version.
  memset(scratch_buffer, 0, num_elements * sizeof(int32));

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence.
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location.
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds.
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  const int8 input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  const int8 filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                        out_channel)] +=
                      (input_value + input_offset) * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32 acc = scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                            out_channel)];
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
}

// int16 input, int8 filter, int64 accumulator
inline void TransposeConv(
    const ConvParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int16* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& output_shape,
    int16* output_data, const RuntimeShape& im2col_shape, int8* im2col_data,
    int64* scratch_buffer) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_activation_min = std::numeric_limits<int16_t>::min();
  const int32 output_activation_max = std::numeric_limits<int16_t>::max();
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int num_elements = output_shape.FlatSize();
  // We need to initialize scratch_buffer to all 0s, as we apply the same
  // 'scatter' based trick as in float version.
  memset(scratch_buffer, 0, num_elements * sizeof(int64));

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence.
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location.
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds.
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  const int32 input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  const int32 filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                        out_channel)] +=
                      (input_value + input_offset) * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int64 acc = scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                            out_channel)];
          int32 scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc += output_offset;
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_TRANSPOSE_CONV_H_

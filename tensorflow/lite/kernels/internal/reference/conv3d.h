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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void Conv3D(const Conv3DParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& filter_shape,
                   const float* filter_data, const RuntimeShape& bias_shape,
                   const float* bias_data, const RuntimeShape& output_shape,
                   float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_num_channels = MatchingDim(input_shape, 4, filter_shape, 3);
  const int output_num_channels = MatchingDim(filter_shape, 4, output_shape, 4);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_num_channels);
  }

  // Only NDHWC format is currently supported.
  const int input_width = input_shape.Dims(3);
  const int input_height = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_depth = filter_shape.Dims(0);
  const int output_width = output_shape.Dims(3);
  const int output_height = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(1);
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int pad_depth = params.padding_values.depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      const int in_d_origin = (out_d * params.stride_depth) - pad_depth;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * params.stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * params.stride_width) - pad_width;
          for (int out_channel = 0; out_channel < output_num_channels;
               ++out_channel) {
            float total = 0.f;
            for (int filter_d = 0; filter_d < filter_depth; ++filter_d) {
              const int in_d = in_d_origin + params.dilation_depth * filter_d;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                const int in_y =
                    in_y_origin + params.dilation_height * filter_y;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + params.dilation_width * filter_x;

                  // Zero padding by omitting the areas outside the image.
                  const bool is_point_inside_image =
                      (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height) && (in_d >= 0) &&
                      (in_d < input_depth);

                  if (!is_point_inside_image) {
                    continue;
                  }

                  for (int in_channel = 0; in_channel < input_num_channels;
                       ++in_channel) {
                    float input_value = input_data[Offset(
                        input_shape, batch, in_d, in_y, in_x, in_channel)];
                    float filter_value =
                        filter_data[Offset(filter_shape, filter_d, filter_y,
                                           filter_x, in_channel, out_channel)];
                    total += (input_value * filter_value);
                  }
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[out_channel];
            }
            output_data[Offset(output_shape, batch, out_d, out_y, out_x,
                               out_channel)] =
                ActivationFunctionWithMinMax(total + bias_value,
                                             params.float_activation_min,
                                             params.float_activation_max);
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_TRANPOSE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_TRANPOSE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void Conv3DTranspose(
    const Conv3DTransposeParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
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
  const int output_depth = output_shape.Dims(1);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_num_channels);
  }

  // Initializes the output array to zero.
  const int num_elements = output_shape.FlatSize();
  for (int i = 0; i < num_elements; i++) {
    output_data[i] = 0.0f;
  }

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_d = 0; in_d < input_depth; ++in_d) {
      for (int in_y = 0; in_y < input_height; ++in_y) {
        for (int in_x = 0; in_x < input_width; ++in_x) {
          for (int in_channel = 0; in_channel < input_num_channels;
               ++in_channel) {
            // Loop through the output elements it will influence.
            const int out_x_origin = (in_x * stride_width) - pad_width;
            const int out_y_origin = (in_y * stride_height) - pad_height;
            const int out_d_origin = (in_d * stride_depth) - pad_depth;
            for (int filter_d = 0; filter_d < filter_depth; ++filter_d) {
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  for (int out_channel = 0; out_channel < output_num_channels;
                       ++out_channel) {
                    // Compute output element location.
                    const int out_x =
                        out_x_origin + params.dilation_width * filter_x;
                    const int out_y =
                        out_y_origin + params.dilation_height * filter_y;
                    const int out_d =
                        out_d_origin + params.dilation_depth * filter_d;
                    // We cannot accumulate out of bounds.
                    if ((out_x >= 0) && (out_x < output_width) &&
                        (out_y >= 0) && (out_y < output_height) &&
                        (out_d >= 0) && (out_d < output_depth)) {
                      float input_value = input_data[Offset(
                          input_shape, batch, in_d, in_y, in_x, in_channel)];
                      float filter_value = filter_data[Offset(
                          filter_shape, filter_d, filter_y, filter_x,
                          out_channel, in_channel)];
                      output_data[Offset(output_shape, batch, out_d, out_y,
                                         out_x, out_channel)] +=
                          input_value * filter_value;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  const float float_activation_min = params.float_activation_min;
  const float float_activation_max = params.float_activation_max;
  float* data_ptr = output_data;
  if (bias_data) {
    const int outer_size =
        batches * output_depth * output_height * output_width;
    const int num_channels = input_shape.Dims(4);
    for (int n = 0; n < outer_size; ++n) {
      for (int c = 0; c < output_num_channels; ++c) {
        data_ptr[c] = ActivationFunctionWithMinMax(data_ptr[c] + bias_data[c],
                                                   float_activation_min,
                                                   float_activation_max);
      }
      data_ptr += num_channels;
    }
  } else {
    const int flat_size = output_shape.FlatSize();
    for (int i = 0; i < flat_size; ++i) {
      data_ptr[i] = ActivationFunctionWithMinMax(
          data_ptr[i], float_activation_min, float_activation_max);
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV3D_TRANPOSE_H_

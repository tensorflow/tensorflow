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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MEAN_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MEAN_H_

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void Mean(const tflite::MeanParams& op_params, int32_t multiplier,
                 int32_t shift, const RuntimeShape& unextended_input_shape,
                 const int8_t* input_data, int32 input_zero_point,
                 const RuntimeShape& unextended_output_shape,
                 int8_t* output_data, int32 output_zero_point) {
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int num_elements_in_axis = input_width * input_height;

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  static constexpr int32_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxInt8 = std::numeric_limits<int8_t>::max();

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      int32 acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)] -
                 input_zero_point;
        }
      }
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc = acc > 0 ? (acc + num_elements_in_axis / 2) / num_elements_in_axis
                    : (acc - num_elements_in_axis / 2) / num_elements_in_axis;
      acc += output_zero_point;
      acc = std::min(std::max(acc, kMinInt8), kMaxInt8);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<int8_t>(acc);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MEAN_H_

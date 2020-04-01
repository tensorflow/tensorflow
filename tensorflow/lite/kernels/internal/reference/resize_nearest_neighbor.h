/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_NEAREST_NEIGHBOR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_NEAREST_NEIGHBOR_H_

#include <cmath>

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace tflite {

namespace reference_ops {

inline int32 GetNearestNeighbor(const int input_value, const int32 input_size,
                                const int32 output_size,
                                const bool align_corners,
                                const bool half_pixel_centers) {
  const float scale =
      (align_corners && output_size > 1)
          ? (input_size - 1) / static_cast<float>(output_size - 1)
          : input_size / static_cast<float>(output_size);
  const float offset = half_pixel_centers ? 0.5f : 0.0f;
  int32 output_value = std::min(
      align_corners
          ? static_cast<int32>(TfLiteRound((input_value + offset) * scale))
          : static_cast<int32>(std::floor((input_value + offset) * scale)),
      input_size - 1);
  if (half_pixel_centers) {
    output_value = std::max(static_cast<int32>(0), output_value);
  }
  return output_value;
}

template <typename T>
inline void ResizeNearestNeighbor(
    const tflite::ResizeNearestNeighborParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& output_size_shape, const int32* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  // The Tensorflow version of this op allows resize on the width and height
  // axis only.
  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const T* input_ptr = input_data;
  T* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32 in_y = GetNearestNeighbor(y, input_height, output_height,
                                      op_params.align_corners,
                                      op_params.half_pixel_centers);
      const T* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32 in_x = GetNearestNeighbor(x, input_width, output_width,
                                        op_params.align_corners,
                                        op_params.half_pixel_centers);
        const T* x_input_ptr = y_input_ptr + in_x * col_offset;
        memcpy(output_ptr, x_input_ptr, depth * sizeof(T));
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_NEAREST_NEIGHBOR_H_

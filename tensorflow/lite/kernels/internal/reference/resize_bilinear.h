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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void ComputeInterpolationValues(const float value, const float scale,
                                       const bool half_pixel_centers,
                                       int32_t input_size, float* scaled_value,
                                       int32_t* lower_bound,
                                       int32_t* upper_bound) {
  if (half_pixel_centers) {
    *scaled_value = (value + 0.5f) * scale - 0.5f;
  } else {
    *scaled_value = value * scale;
  }
  float scaled_value_floor = std::floor(*scaled_value);
  *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor),
                          static_cast<int32_t>(0));
  *upper_bound =
      std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32_t* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }
  const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32_t y0, y1;
      ComputeInterpolationValues(y, height_scale, op_params.half_pixel_centers,
                                 input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32_t x0, x1;
        ComputeInterpolationValues(x, width_scale, op_params.half_pixel_centers,
                                   input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0) +
                             rounding_offset);
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

inline void ComputeInterpolationValuesInteger(
    const int32_t value, const int32_t scale_10, const bool half_pixel_centers,
    int32_t input_size, int32_t* scaled_value, int32_t* lower_bound,
    int32_t* upper_bound) {
  if (half_pixel_centers) {
    *scaled_value = value * scale_10 + scale_10 / 2 - (1 << 9);
  } else {
    *scaled_value = value * scale_10;
  }
  constexpr int32_t zero = 0;
  *lower_bound = std::max(*scaled_value / (1 << 10), zero);
  *upper_bound =
      std::min((*scaled_value + (1 << 10) - 1) / (1 << 10), input_size - 1);
}

// Same as above but doesn't use any floating-point for the resize
template <typename T>
inline void ResizeBilinearInteger(
    const tflite::ResizeBilinearParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& unextended_output_size_shape,
    const int32_t* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  const int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  const int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  int32_t height_scale_10 =
      ((1 << 10) * input_height + output_height / 2) / output_height;
  int32_t width_scale_10 =
      ((1 << 10) * input_width + output_width / 2) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale_10 =
        ((1 << 10) * (input_height - 1) + (output_height - 1) / 2) /
        (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale_10 = ((1 << 10) * (input_width - 1) + (output_width - 1) / 2) /
                     (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32_t input_y, y0, y1;
      ComputeInterpolationValuesInteger(y, height_scale_10,
                                        op_params.half_pixel_centers,
                                        input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        int32_t input_x, x0, x1;
        ComputeInterpolationValuesInteger(x, width_scale_10,
                                          op_params.half_pixel_centers,
                                          input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          const int64_t output_20_ll =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y0, x0, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_lu =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y1, x0, c)]) *
              (input_y - (1 << 10) * y0) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_rl =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y0, x1, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              (input_x - (1 << 10) * x0);
          const int64_t output_20_ru =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y1, x1, c)]) *
              (input_y - (1 << 10) * y0) * (input_x - (1 << 10) * x0);
          const int64_t output_20 =
              output_20_ll + output_20_lu + output_20_rl + output_20_ru;
#if TFLITE_SINGLE_ROUNDING
          const int64_t round = 1 << 19;
          const T interpolation = static_cast<T>((output_20 + round) >> 20);
#else
          const int64_t round = (output_20 > 0) ? (1 << 19) : -(1 << 19);
          const T interpolation =
              static_cast<T>((output_20 + round) / (1 << 20));
#endif  // TFLITE_SINGLE_ROUNDING
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_

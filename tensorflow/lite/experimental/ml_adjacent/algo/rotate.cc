/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cmath>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace rotate {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;

inline float DegreesToRadians(int angle) { return angle * M_PI / 180; }

void ComputeNewSize(dim_t src_width, dim_t src_height, int angle,
                    dim_t& dst_width, dim_t& dst_height) {
  // Keep same size for 180 degree case.
  dst_width = src_width;
  dst_height = src_height;

  if (angle % 90 == 0) {
    // Define new size for 90 or 270 degree angle.
    if (angle == 90 || angle == 270) {
      dst_width = src_height;
      dst_height = src_width;
    }
  } else {
    // Calculate new size for arbitrary angle.
    const float angle_rad = DegreesToRadians(angle);
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    const int64_t edge_x = static_cast<int64_t>(src_width) / 2;
    const int64_t edge_y = static_cast<int64_t>(src_height) / 2;
    for (int64_t y : {-edge_y, edge_y}) {
      for (int64_t x : {-edge_x, edge_x}) {
        const int64_t x_transformed =
            static_cast<int64_t>(std::floor(cos_angle * x + sin_angle * y));
        const int64_t y_transformed =
            static_cast<int64_t>(std::floor(-sin_angle * x + cos_angle * y));

        if (std::abs(x_transformed) > static_cast<int64_t>(dst_width) / 2) {
          dst_width = 2 * std::abs(x_transformed);
        }
        if (std::abs(y_transformed) > static_cast<int64_t>(dst_height) / 2) {
          dst_height = 2 * std::abs(y_transformed);
        }
      }
    }
  }
}

// Rotates image for 90 degree.
void Rotate90(dim_t batches, dim_t input_height, dim_t input_width, dim_t depth,
              dim_t output_height, dim_t output_width, const float* input_data,
              float* output_data) {
  TFLITE_CHECK(input_data != nullptr);
  TFLITE_CHECK(output_data != nullptr);

  const int64_t pixel_stride = depth;
  const int64_t src_row_stride = static_cast<int64_t>(input_width) * depth;
  const int64_t dst_row_stride = static_cast<int64_t>(output_width) * depth;
  const int64_t src_batch_stride = src_row_stride * input_height;
  const int64_t dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[x][width - y - 1] = src[y][x].
  for (dim_t b = 0; b < batches; ++b) {
    const float* src_data_ptr = input_data + b * src_batch_stride;
    float* dst_data_ptr = output_data + b * dst_batch_stride;

    for (dim_t y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_ptr + y * src_row_stride;
      for (dim_t x = 0; x < input_width; ++x) {
        float* dst_ptr_row = dst_data_ptr + x * dst_row_stride;

        const float* src_ptr_pixel = src_ptr_row + x * pixel_stride;
        float* dst_pixel_ptr =
            dst_ptr_row + (output_width - y - 1) * pixel_stride;

        for (dim_t c = 0; c < depth; ++c) {
          *dst_pixel_ptr++ = *src_ptr_pixel++;
        }
      }
    }
  }
}

// Rotates image for 180 degree.
void Rotate180(dim_t batches, dim_t input_height, dim_t input_width,
               dim_t depth, dim_t output_height, dim_t output_width,
               const float* input_data, float* output_data) {
  TFLITE_CHECK(input_data != nullptr);
  TFLITE_CHECK(output_data != nullptr);

  const int64_t dst_pixel_stride = depth;
  const int64_t src_row_stride = static_cast<int64_t>(input_width) * depth;
  const int64_t dst_row_stride = static_cast<int64_t>(output_width) * depth;
  const int64_t src_batch_stride = src_row_stride * input_height;
  const int64_t dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[height - y - 1][width - x - 1] = src[y][x].
  for (dim_t b = 0; b < batches; ++b) {
    const float* src_data_ptr = input_data + b * src_batch_stride;
    float* dst_data_ptr = output_data + b * dst_batch_stride;

    for (dim_t y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_ptr + y * src_row_stride;
      float* dst_ptr_row = dst_data_ptr +
                           (output_height - y - 1) * dst_row_stride +
                           (output_width - 1) * dst_pixel_stride;
      for (dim_t x = 0; x < input_width; ++x) {
        for (dim_t c = 0; c < depth; ++c) {
          dst_ptr_row[c] = src_ptr_row[c];
        }
        dst_ptr_row -= depth;
        src_ptr_row += depth;
      }
    }
  }
}

// Rotates image for 270 degree.
void Rotate270(dim_t batches, dim_t input_height, dim_t input_width,
               dim_t depth, dim_t output_height, dim_t output_width,
               const float* input_data, float* output_data) {
  TFLITE_CHECK(input_data != nullptr);
  TFLITE_CHECK(output_data != nullptr);

  const int64_t pixel_stride = depth;
  const int64_t src_row_stride = static_cast<int64_t>(input_width) * depth;
  const int64_t dst_row_stride = static_cast<int64_t>(output_width) * depth;
  const int64_t src_batch_stride = src_row_stride * input_height;
  const int64_t dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[output_height - x - 1][y] = src[y][x].
  for (dim_t b = 0; b < batches; ++b) {
    const float* src_data_ptr = input_data + b * src_batch_stride;
    float* dst_data_ptr = output_data + b * dst_batch_stride;

    for (dim_t y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_ptr + y * src_row_stride;
      for (dim_t x = 0; x < input_width; ++x) {
        float* dst_ptr_row =
            dst_data_ptr + (output_height - x - 1) * dst_row_stride;

        const float* src_ptr_pixel = src_ptr_row + x * pixel_stride;
        float* dst_pixel_ptr = dst_ptr_row + y * pixel_stride;

        for (dim_t c = 0; c < depth; ++c) {
          *dst_pixel_ptr++ = *src_ptr_pixel++;
        }
      }
    }
  }
}

// Performs generic rotation for arbitrary angle.
void RotateGeneric(dim_t batches, dim_t input_height, dim_t input_width,
                   dim_t depth, dim_t output_height, dim_t output_width,
                   int angle, const float* input_data, float* output_data) {
  TFLITE_CHECK(input_data != nullptr);
  TFLITE_CHECK(output_data != nullptr);

  const int64_t pixel_stride = depth;
  const int64_t src_row_stride = static_cast<int64_t>(input_width) * depth;
  const int64_t dst_row_stride = static_cast<int64_t>(output_width) * depth;
  const int64_t src_batch_stride = src_row_stride * input_height;
  const int64_t dst_batch_stride = dst_row_stride * output_height;

  // Start off with dark image by initializing all pixels with zeros.
  memset(output_data, 0,
         static_cast<size_t>(batches) * output_width * output_height * depth *
             sizeof(output_data[0]));

  const float angle_rad = DegreesToRadians(angle);
  const float cos_angle = std::cos(angle_rad);
  const float sin_angle = std::sin(angle_rad);

  const int64_t half_output_height = static_cast<int64_t>(output_height) / 2;
  const int64_t half_output_width = static_cast<int64_t>(output_width) / 2;

  // Iterate over batches to perform a rotation with arbitrary angle.
  for (dim_t b = 0; b < batches; ++b) {
    const float* src_data_ptr = input_data + b * src_batch_stride;
    float* dst_data_ptr = output_data + b * dst_batch_stride;

    for (int64_t y = -half_output_height; y < half_output_height; ++y) {
      for (int64_t x = -half_output_width; x < half_output_width; ++x) {
        const float x_transformed = cos_angle * x + sin_angle * y;
        const float y_transformed = -sin_angle * x + cos_angle * y;

        // Convert to integer by computing the next smaller integer number.
        const int64_t x_transformed_integer =
            static_cast<int64_t>(std::floor(x_transformed));
        const int64_t y_transformed_integer =
            static_cast<int64_t>(std::floor(y_transformed));

        // Move into the coordinate system of input image.
        const int64_t x_src_integer =
            x_transformed_integer + static_cast<int64_t>(input_width) / 2;
        const int64_t y_src_integer =
            y_transformed_integer + static_cast<int64_t>(input_height) / 2;

        // Calculate coordinates for interpolation.
        const int64_t x0 = x_src_integer;
        const int64_t x1 = x_src_integer + 1;
        const int64_t y0 = y_src_integer;
        const int64_t y1 = y_src_integer + 1;

        // Skip further calculations if coordinates are out of bounds.
        if (x0 < 0 || x0 >= input_width) continue;
        if (x1 < 0 || x1 >= input_width) continue;
        if (y0 < 0 || y0 >= input_height) continue;
        if (y1 < 0 || y1 >= input_height) continue;

        const float x_dist = x_transformed - x_transformed_integer;
        const float y_dist = y_transformed - y_transformed_integer;
        const float one_minus_x_dist = 1.0f - x_dist;
        const float one_minus_y_dist = 1.0f - y_dist;

        // Calculate rotated pixels for all channels.
        const float* src_ptr_row0 = src_data_ptr + y0 * src_row_stride;
        const float* src_ptr_row1 = src_data_ptr + y1 * src_row_stride;
        float* dst_row_ptr =
            dst_data_ptr + (y + half_output_height) * dst_row_stride;

        const float* src_ptr_pixel00 = src_ptr_row0 + x0 * pixel_stride;
        const float* src_ptr_pixel10 = src_ptr_row0 + x1 * pixel_stride;
        const float* src_ptr_pixel01 = src_ptr_row1 + x0 * pixel_stride;
        const float* src_ptr_pixel11 = src_ptr_row1 + x1 * pixel_stride;
        float* dst_pixel_ptr =
            dst_row_ptr + (x + half_output_width) * pixel_stride;

        for (dim_t c = 0; c < depth; ++c) {
          const float v00 = *src_ptr_pixel00++;
          const float v01 = *src_ptr_pixel01++;
          const float v10 = *src_ptr_pixel10++;
          const float v11 = *src_ptr_pixel11++;

          *dst_pixel_ptr++ =
              (v10 * one_minus_y_dist + v11 * y_dist) * x_dist +
              (v00 * one_minus_y_dist + v01 * y_dist) * one_minus_x_dist;
        }
      }
    }
  }
}

// Rotate given input with arbitrary angle. Works on `float` datatype.
void ComputeRotate(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_CHECK_EQ(inputs.size(), 2);
  TFLITE_CHECK_EQ(outputs.size(), 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  TFLITE_CHECK_EQ(img->Dims().size(), 4);
  MutableDataRef* output = outputs[0];

  TFLITE_CHECK_EQ(img->Type(), etype_t::f32);
  TFLITE_CHECK_EQ(output->Type(), etype_t::f32);

  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t img_num_channels = img->Dims()[3];

  if (img_num_batches == 0 || img_height == 0 || img_width == 0) return;

  const DataRef* angle = inputs[1];
  TFLITE_CHECK_EQ(angle->NumElements(), 1);
  TFLITE_CHECK_EQ(angle->Type(), etype_t::i32);
  const int raw_angle = *reinterpret_cast<const int*>(angle->Data());
  const int angle_data = ((raw_angle % 360) + 360) % 360;

  // Resize output buffer for rotated image.
  dim_t new_width = 0;
  dim_t new_height = 0;
  ComputeNewSize(img_width, img_height, angle_data, new_width, new_height);
  output->Resize({img_num_batches, new_height, new_width, img_num_channels});
  float* output_data = reinterpret_cast<float*>(output->Data());

  // Perform rotation depending on angle.
  if (angle_data == 90) {
    Rotate90(img_num_batches, img_height, img_width, img_num_channels,
             new_height, new_width, img_data, output_data);
    return;
  }

  if (angle_data == 180) {
    Rotate180(img_num_batches, img_height, img_width, img_num_channels,
              new_height, new_width, img_data, output_data);
    return;
  }

  if (angle_data == 270) {
    Rotate270(img_num_batches, img_height, img_width, img_num_channels,
              new_height, new_width, img_data, output_data);
    return;
  }

  RotateGeneric(img_num_batches, img_height, img_width, img_num_channels,
                new_height, new_width, angle_data, img_data, output_data);
}

}  // namespace

const Algo* Impl_Rotate() {
  static constexpr Algo kRotate = {&ComputeRotate, nullptr};
  return &kRotate;
}

}  // namespace rotate
}  // namespace ml_adj

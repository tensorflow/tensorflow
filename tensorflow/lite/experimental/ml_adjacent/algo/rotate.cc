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

    const int edge_x = src_width / 2;
    const int edge_y = src_height / 2;
    for (int y : {-edge_y, edge_y}) {
      for (int x : {-edge_x, edge_x}) {
        const int x_transformed =
            static_cast<int>(std::floor(cos_angle * x + sin_angle * y));
        const int y_transformed =
            static_cast<int>(std::floor(-sin_angle * x + cos_angle * y));

        if (std::abs(x_transformed) > dst_width / 2)
          dst_width = 2 * std::abs(x_transformed);
        if (std::abs(y_transformed) > dst_height / 2)
          dst_height = 2 * std::abs(y_transformed);
      }
    }
  }
}

// Rotates image for 90 degree.
void Rotate90(int batches, int input_height, int input_width, int depth,
              int output_height, int output_width, const float* input_data,
              float* output_data) {
  TFLITE_DCHECK(input_data != nullptr);
  TFLITE_DCHECK(output_data != nullptr);

  const int pixel_stride = depth;
  const int src_row_stride = input_width * depth;
  const int dst_row_stride = output_width * depth;
  const int src_batch_stride = src_row_stride * input_height;
  const int dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[x][width - y - 1] = src[y][x].
  for (int b = 0; b < batches; ++b) {
    const float* src_data_prt = input_data + b * src_batch_stride;
    float* dst_data_prt = output_data + b * dst_batch_stride;

    for (int y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_prt + y * src_row_stride;
      for (int x = 0; x < input_width; ++x) {
        float* dst_ptr_row = dst_data_prt + x * dst_row_stride;

        const float* src_ptr_pixel = src_ptr_row + x * pixel_stride;
        float* dst_pixel_ptr =
            dst_ptr_row + (output_width - y - 1) * pixel_stride;

        for (int c = 0; c < depth; ++c) {
          *dst_pixel_ptr++ = *src_ptr_pixel++;
        }
      }
    }
  }
}

// Rotates image for 180 degree.
void Rotate180(int batches, int input_height, int input_width, int depth,
               int output_height, int output_width, const float* input_data,
               float* output_data) {
  TFLITE_DCHECK(input_data != nullptr);
  TFLITE_DCHECK(output_data != nullptr);

  const int dst_pixel_stride = depth;
  const int src_row_stride = input_width * depth;
  const int dst_row_stride = output_width * depth;
  const int src_batch_stride = src_row_stride * input_height;
  const int dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[height - y - 1][width - x - 1] = src[y][x].
  for (int b = 0; b < batches; ++b) {
    const float* src_data_prt = input_data + b * src_batch_stride;
    float* dst_data_prt = output_data + b * dst_batch_stride;

    for (int y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_prt + y * src_row_stride;
      float* dst_ptr_row = dst_data_prt +
                           (output_height - y - 1) * dst_row_stride +
                           (output_width - 1) * dst_pixel_stride;
      for (int x = 0; x < input_width; ++x) {
        for (int c = 0; c < depth; ++c) {
          dst_ptr_row[c] = src_ptr_row[c];
        }
        dst_ptr_row -= depth;
        src_ptr_row += depth;
      }
    }
  }
}

// Rotates image for 270 degree.
void Rotate270(int batches, int input_height, int input_width, int depth,
               int output_height, int output_width, const float* input_data,
               float* output_data) {
  TFLITE_DCHECK(input_data != nullptr);
  TFLITE_DCHECK(output_data != nullptr);

  const int pixel_stride = depth;
  const int src_row_stride = input_width * depth;
  const int dst_row_stride = output_width * depth;
  const int src_batch_stride = src_row_stride * input_height;
  const int dst_batch_stride = dst_row_stride * output_height;

  // Iterate over batches to perform the following transformation:
  // dst[output_height - x - 1][y] = src[y][x].
  for (int b = 0; b < batches; ++b) {
    const float* src_data_prt = input_data + b * src_batch_stride;
    float* dst_data_prt = output_data + b * dst_batch_stride;

    for (int y = 0; y < input_height; ++y) {
      const float* src_ptr_row = src_data_prt + y * src_row_stride;
      for (int x = 0; x < input_width; ++x) {
        float* dst_ptr_row =
            dst_data_prt + (output_height - x - 1) * dst_row_stride;

        const float* src_ptr_pixel = src_ptr_row + x * pixel_stride;
        float* dst_pixel_ptr = dst_ptr_row + y * pixel_stride;

        for (int c = 0; c < depth; ++c) {
          *dst_pixel_ptr++ = *src_ptr_pixel++;
        }
      }
    }
  }
}

// Performs generic rotation for arbitrary angle.
void RotateGeneric(int batches, int input_height, int input_width, int depth,
                   int output_height, int output_width, int angle,
                   const float* input_data, float* output_data) {
  TFLITE_DCHECK(input_data != nullptr);
  TFLITE_DCHECK(output_data != nullptr);

  const int pixel_stride = depth;
  const int src_row_stride = input_width * depth;
  const int dst_row_stride = output_width * depth;
  const int src_batch_stride = src_row_stride * input_height;
  const int dst_batch_stride = dst_row_stride * output_height;

  // Start off with dark image by initializing all pixels with zeros.
  memset(
      output_data, 0,
      batches * output_width * output_height * depth * sizeof(output_data[0]));

  const float angle_rad = DegreesToRadians(angle);
  const float cos_angle = std::cos(angle_rad);
  const float sin_angle = std::sin(angle_rad);

  // Iterate over batches to perform a rotation with arbitrary angle.
  for (int b = 0; b < batches; ++b) {
    const float* src_data_prt = input_data + b * src_batch_stride;
    float* dst_data_prt = output_data + b * dst_batch_stride;

    for (int y = -output_height / 2; y < output_height / 2; ++y) {
      for (int x = -output_width / 2; x < output_width / 2; ++x) {
        const float x_transformed = cos_angle * x + sin_angle * y;
        const float y_transformed = -sin_angle * x + cos_angle * y;

        // Convert to integer by computing the next smaller integer number.
        const int x_transformed_integer =
            static_cast<int>(std::floor(x_transformed));
        const int y_transformed_integer =
            static_cast<int>(std::floor(y_transformed));

        // Move into the coordinate system of input image.
        const int x_src_integer = x_transformed_integer + input_width / 2;
        const int y_src_integer = y_transformed_integer + input_height / 2;

        // Calculate coordinates for interpolation.
        const int x0 = x_src_integer;
        const int x1 = x_src_integer + 1;
        const int y0 = y_src_integer;
        const int y1 = y_src_integer + 1;

        // Skip further calculations if coordinates are out of bounds.
        if (x0 < 0 || x0 >= input_width) continue;
        if (x1 < 0 || x1 >= input_width) continue;
        if (y0 < 0 || y0 >= input_height) continue;
        if (y1 < 0 || y1 >= input_height) continue;

        const float x_dist = x_transformed - x_transformed_integer;
        const float y_dist = y_transformed - y_transformed_integer;
        const float one_minus_x_dist = 1 - x_dist;
        const float one_minus_y_dist = 1 - y_dist;

        // Calculate rotated pixels for all channels.
        const float* src_ptr_row0 = src_data_prt + y0 * src_row_stride;
        const float* src_ptr_row1 = src_data_prt + y1 * src_row_stride;
        float* dst_row_ptr =
            dst_data_prt + (y + output_height / 2) * dst_row_stride;

        const float* src_ptr_pixel00 = src_ptr_row0 + x0 * pixel_stride;
        const float* src_ptr_pixel10 = src_ptr_row0 + x1 * pixel_stride;
        const float* src_ptr_pixel01 = src_ptr_row1 + x0 * pixel_stride;
        const float* src_ptr_pixel11 = src_ptr_row1 + x1 * pixel_stride;
        float* dst_pixel_ptr =
            dst_row_ptr + (x + output_width / 2) * pixel_stride;

        for (int c = 0; c < depth; ++c) {
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
  TFLITE_DCHECK(inputs.size() == 2);
  TFLITE_DCHECK(outputs.size() == 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t img_num_channels = img->Dims()[3];

  const DataRef* angle = inputs[1];
  const int angle_data = *reinterpret_cast<const int*>(angle->Data());

  // Resize output buffer for rotated image.
  MutableDataRef* output = outputs[0];
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
  static const Algo rotate = {&ComputeRotate, nullptr};
  return &rotate;
}

}  // namespace rotate
}  // namespace ml_adj

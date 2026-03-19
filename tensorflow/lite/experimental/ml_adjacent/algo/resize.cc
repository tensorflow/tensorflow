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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace resize {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;

// Calculates bilinear interpolated values and lower and upper bounds.
inline void ComputeInterpolationValues(const float value, const float scale,
                                       int32_t input_size, float& scaled_value,
                                       int32_t& lower_bound,
                                       int32_t& upper_bound) {
  scaled_value = value * scale;

  float scaled_value_floor = std::floor(scaled_value);
  lower_bound = std::max(static_cast<int32_t>(scaled_value_floor), 0);
  upper_bound =
      std::min(static_cast<int32_t>(std::ceil(scaled_value)), input_size - 1);
}

// Applies depth-wise scaling.
inline void ScaleDepthwise(const float* input_ptr, int32_t depth, float scale,
                           float* output_ptr) {
  for (int32_t i = 0; i < depth; ++i) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}

// Calculates 1D pixel offset in NHWC data.
inline int Offset(const int32_t* dims_data, int32_t dims_num, int32_t i0,
                  int32_t i1, int32_t i2, int32_t i3) {
#ifndef NDEBUG
  TFLITE_CHECK_EQ(dims_num, 3);
#endif
  return ((i0 * dims_data[0] + i1) * dims_data[1] + i2) * dims_data[2] + i3;
}

// Generic implementation of bilinear resize.
inline void ResizeBilinear(int32_t batches, int32_t input_height,
                           int32_t input_width, int32_t depth,
                           int32_t output_height, int32_t output_width,
                           float height_scale, float width_scale,
                           const float* input_data, float* output_data) {
  memset(output_data, 0,
         batches * output_height * output_width * depth * sizeof(float));
  const int dims_data[] = {input_height, input_width, depth};
  const int dims_num = sizeof(dims_data) / sizeof(dims_data[0]);

  int32_t output_offset = 0;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = 0.0f;
      int32_t y0 = 0;
      int32_t y1 = 0;
      ComputeInterpolationValues(y, height_scale, input_height, input_y, y0,
                                 y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = 0.0f;
        int32_t x0 = 0;
        int32_t x1 = 0;
        ComputeInterpolationValues(x, width_scale, input_width, input_x, x0,
                                   x1);
        float* output_ptr = output_data + output_offset;

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32_t input_offset = Offset(dims_data, dims_num, b, y0, x0, 0);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float* input_ptr = &input_data[input_offset];
        ScaleDepthwise(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(dims_data, dims_num, b, y0, x1, 0);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = input_data + input_offset;
        ScaleDepthwise(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(dims_data, dims_num, b, y1, x0, 0);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ScaleDepthwise(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(dims_data, dims_num, b, y1, x1, 0);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ScaleDepthwise(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

// Optimized implementatin of bilinear resize for 2X upscaling case.
inline void ResizeBilinear2x2(int32_t batches, int32_t input_height,
                              int32_t input_width, int32_t depth,
                              int32_t output_height, int32_t output_width,
                              float height_scale, float width_scale,
                              const float* input_data, float* output_data) {
  const int input_dims_data[] = {input_height, input_width, depth};
  const int output_dims_data[] = {output_height, output_width, depth};
  const int dims_num = sizeof(input_dims_data) / sizeof(input_dims_data[0]);

  for (int32_t b = 0; b < batches; ++b) {
    for (int32_t y0 = 0, y = 0; y <= output_height - 2; y += 2, ++y0) {
      for (int32_t x0 = 0, x = 0; x <= output_width - 2; x += 2, ++x0) {
        int32_t x1 = std::min(x0 + 1, input_width - 1);
        int32_t y1 = std::min(y0 + 1, input_height - 1);

        const int32 input_x_offset = (x1 - x0) * depth;
        const int32 input_y_offset = (y1 - y0) * depth * input_width;
        const int32 output_x_offset = depth;
        const int32 output_y_offset = depth * output_width;

        for (int ch = 0; ch < depth; ++ch) {
          const int32 input_offset =
              Offset(input_dims_data, dims_num, b, y0, x0, ch);

          const float x0y0 = input_data[input_offset];
          const float x1y0 = input_data[input_offset + input_x_offset];
          const float x0y1 = input_data[input_offset + input_y_offset];
          const float x1y1 =
              input_data[input_offset + input_x_offset + input_y_offset];

          // Calculate top left corner value.
          const int32 output_offset =
              Offset(output_dims_data, dims_num, b, y, x, ch);
          output_data[output_offset] = x0y0;

          // Calculate top right corner value.
          output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

          // Calculate bottom left corner value.
          float output = (x0y0 + x0y1) / 2;
          output_data[output_offset + output_y_offset] = output;

          // Calculate bottom right corner value.
          output_data[output_offset + output_x_offset + output_y_offset] =
              (output + ((x1y0 + x1y1) / 2)) / 2;
        }
      }
    }
  }
}

// Resizes given input. Supports `float` datatype only.
void ComputeResize(const InputPack& inputs, const OutputPack& outputs) {
#ifndef NDEBUG
  TFLITE_CHECK(inputs.size() == 2);
  TFLITE_CHECK(outputs.size() == 1);
#endif

  // Extract input image data.
  const DataRef* img = inputs[0];
  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t img_num_channels = img->Dims()[3];

  // Extract new image size.
  const DataRef* size = inputs[1];
  const dim_t* size_data = reinterpret_cast<const dim_t*>(size->Data());
  const dim_t new_height = size_data[0];
  const dim_t new_width = size_data[1];

  // Resize output buffer for resized image.
  MutableDataRef* output = outputs[0];
  output->Resize({img_num_batches, new_height, new_width, img_num_channels});
  float* output_data = reinterpret_cast<float*>(output->Data());

  const float width_scale = static_cast<float>(img_width) / new_width;
  const float height_scale = static_cast<float>(img_height) / new_height;

  if (new_width == 2 * img_width && new_height == 2 * img_height) {
    ResizeBilinear2x2(img_num_batches, img_height, img_width, img_num_channels,
                      new_height, new_width, height_scale, width_scale,
                      img_data, output_data);
    return;
  }

  ResizeBilinear(img_num_batches, img_height, img_width, img_num_channels,
                 new_height, new_width, height_scale, width_scale, img_data,
                 output_data);
}

}  // namespace

const Algo* Impl_Resize() {
  static const Algo center_crop = {&ComputeResize, nullptr};
  return &center_crop;
}

}  // namespace resize
}  // namespace ml_adj

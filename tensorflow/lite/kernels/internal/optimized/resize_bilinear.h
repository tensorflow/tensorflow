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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_

#include <stdint.h>
#include <sys/types.h>

#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

#ifdef USE_NEON
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  int ic = 0;
  // Handle 32 input channels at a time.
  for (; ic <= depth - 32; ic += 32) {
    float32x4x2_t input[4];
    for (int i = 0; i < 4; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[4];
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 4; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 32;
  }
  // Handle 16 input channels at a time.
  for (; ic <= depth - 16; ic += 16) {
    float32x4x2_t input[2];
    for (int i = 0; i < 2; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[2];
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 2; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 16;
  }
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    float32x4x2_t input;
    input.val[0] = vld1q_f32(input_ptr);
    input.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t acc;
    acc.val[0] = vld1q_f32(output_ptr);
    acc.val[1] = vld1q_f32(output_ptr + 4);
    acc.val[0] = vmlaq_n_f32(acc.val[0], input.val[0], scale);
    acc.val[1] = vmlaq_n_f32(acc.val[1], input.val[1], scale);

    vst1q_f32(output_ptr, acc.val[0]);
    vst1q_f32(output_ptr + 4, acc.val[1]);

    input_ptr += 8;
    output_ptr += 8;
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    float32x4_t input = vld1q_f32(input_ptr);
    float32x4_t acc = vld1q_f32(output_ptr);

    acc = vmlaq_n_f32(acc, input, scale);
    vst1q_f32(output_ptr, acc);

    input_ptr += 4;
    output_ptr += 4;
  }
  // Handle 1 input channel at a time.
  for (; ic < depth; ic++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#else
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  for (int32 i = 0; i < depth; i++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#endif

inline void ResizeBilinearKernel2x2(int32 x0, int32 x1, int32 y0, int32 y1,
                                    int32 x, int32 y, int32 depth, int32 batch,
                                    const RuntimeShape& input_shape,
                                    const float* input_data,
                                    const RuntimeShape& output_shape,
                                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int32 input_width = input_shape.Dims(2);
  const int32 output_width = output_shape.Dims(2);

  const int32 input_x_offset = (x1 - x0) * depth;
  const int32 input_y_offset = (y1 - y0) * depth * input_width;
  const int32 output_x_offset = depth;
  const int32 output_y_offset = depth * output_width;

#ifdef USE_NEON
  TFLITE_DCHECK(x1 >= x0);
  TFLITE_DCHECK(y1 >= y0);

  int ic = 0;
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    const float* input_ptr = nullptr;

    float32x4x2_t x0y0;
    input_ptr = &input_data[Offset(input_shape, batch, y0, x0, ic)];
    x0y0.val[0] = vld1q_f32(input_ptr);
    x0y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y0;
    input_ptr += input_x_offset;
    x1y0.val[0] = vld1q_f32(input_ptr);
    x1y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x0y1;
    input_ptr += -input_x_offset + input_y_offset;
    x0y1.val[0] = vld1q_f32(input_ptr);
    x0y1.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y1;
    input_ptr += input_x_offset;
    x1y1.val[0] = vld1q_f32(input_ptr);
    x1y1.val[1] = vld1q_f32(input_ptr + 4);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0.val[0]);
    vst1q_f32(output_ptr + 4, x0y0.val[1]);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4x2_t tr;
    tr.val[0] = vaddq_f32(x0y0.val[0], x1y0.val[0]);
    tr.val[1] = vaddq_f32(x0y0.val[1], x1y0.val[1]);
    tr.val[0] = vmulq_n_f32(tr.val[0], 0.5f);
    tr.val[1] = vmulq_n_f32(tr.val[1], 0.5f);

    vst1q_f32(output_ptr, tr.val[0]);
    vst1q_f32(output_ptr + 4, tr.val[1]);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4x2_t bl;
    bl.val[0] = vaddq_f32(x0y0.val[0], x0y1.val[0]);
    bl.val[1] = vaddq_f32(x0y0.val[1], x0y1.val[1]);
    bl.val[0] = vmulq_n_f32(bl.val[0], 0.5f);
    bl.val[1] = vmulq_n_f32(bl.val[1], 0.5f);
    vst1q_f32(output_ptr, bl.val[0]);
    vst1q_f32(output_ptr + 4, bl.val[1]);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4x2_t br;
    br.val[0] = vaddq_f32(x1y0.val[0], x1y1.val[0]);
    br.val[1] = vaddq_f32(x1y0.val[1], x1y1.val[1]);
    br.val[0] = vmlaq_n_f32(bl.val[0], br.val[0], 0.5f);
    br.val[1] = vmlaq_n_f32(bl.val[1], br.val[1], 0.5f);
    br.val[0] = vmulq_n_f32(br.val[0], 0.5f);
    br.val[1] = vmulq_n_f32(br.val[1], 0.5f);
    vst1q_f32(output_ptr, br.val[0]);
    vst1q_f32(output_ptr + 4, br.val[1]);
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    const float* input_ptr =
        &input_data[Offset(input_shape, batch, y0, x0, ic)];
    float32x4_t x0y0 = vld1q_f32(input_ptr);
    float32x4_t x1y0 = vld1q_f32(input_ptr + input_x_offset);
    float32x4_t x0y1 = vld1q_f32(input_ptr + input_y_offset);
    float32x4_t x1y1 = vld1q_f32(input_ptr + input_x_offset + input_y_offset);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4_t tr = vaddq_f32(x0y0, x1y0);
    tr = vmulq_n_f32(tr, 0.5f);
    vst1q_f32(output_ptr, tr);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4_t bl = vaddq_f32(x0y0, x0y1);
    bl = vmulq_n_f32(bl, 0.5f);
    vst1q_f32(output_ptr, bl);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4_t br = vaddq_f32(x1y0, x1y1);
    br = vmlaq_n_f32(bl, br, 0.5f);
    br = vmulq_n_f32(br, 0.5f);
    vst1q_f32(output_ptr, br);
  }
  // Handle one input channel at a time.
  for (; ic < depth; ic++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ic);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ic);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#else
  for (int ch = 0; ch < depth; ch++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#endif
}

inline void ResizeBilinear2x2(int32 batches, int32 input_height,
                              int32 input_width, int32 depth,
                              int32 output_height, int32 output_width,
                              const RuntimeShape& input_shape,
                              const float* input_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  for (int b = 0; b < batches; b++) {
    for (int y0 = 0, y = 0; y <= output_height - 2; y += 2, y0++) {
      for (int x0 = 0, x = 0; x <= output_width - 2; x += 2, x0++) {
        int32 x1 = std::min(x0 + 1, input_width - 1);
        int32 y1 = std::min(y0 + 1, input_height - 1);
        ResizeBilinearKernel2x2(x0, x1, y0, y1, x, y, depth, b, input_shape,
                                input_data, output_shape, output_data);
      }
    }
  }
}

inline void ResizeBilinearGeneric(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data,
    const bool half_pixel_centers) {
  memset(output_data, 0,
         batches * output_height * output_width * depth * sizeof(float));

  int32 output_offset = 0;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32 y0, y1;
      reference_ops::ComputeInterpolationValues(
          y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
          &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32 x0, x1;
        reference_ops::ComputeInterpolationValues(
            x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
            &x1);
        float* output_ptr = &output_data[output_offset];

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32 input_offset = Offset(input_shape, b, y0, x0, 0);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float* input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y0, x1, 0);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x0, 0);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x1, 0);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

template <typename T>
inline void ResizeBilinearGenericSmallChannel(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const T* input_data,
    const RuntimeShape& output_shape, T* output_data,
    const bool half_pixel_centers) {
  T* output_ptr = &output_data[0];
  const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32 y0, y1;
      reference_ops::ComputeInterpolationValues(
          y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
          &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32 x0, x1;
        reference_ops::ComputeInterpolationValues(
            x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
            &x1);

        int32 input_offset[4] = {Offset(input_shape, b, y0, x0, 0),
                                 Offset(input_shape, b, y0, x1, 0),
                                 Offset(input_shape, b, y1, x0, 0),
                                 Offset(input_shape, b, y1, x1, 0)};
        float scale[4] = {(1 - (input_y - y0)) * (1 - (input_x - x0)),
                          (1 - (input_y - y0)) * (input_x - x0),
                          (input_y - y0) * (1 - (input_x - x0)),
                          (input_y - y0) * (input_x - x0)};

        for (int d = 0; d < depth; d++) {
          const T* input_ptr = &input_data[d];
          *output_ptr++ = static_cast<T>(input_ptr[input_offset[0]] * scale[0] +
                                         input_ptr[input_offset[1]] * scale[1] +
                                         input_ptr[input_offset[2]] * scale[2] +
                                         input_ptr[input_offset[3]] * scale[3] +
                                         rounding_offset);
        }
      }
    }
  }
}

inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const float* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           float* output_data) {
  ruy::profiler::ScopeLabel label("ResizeBilinear");
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
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

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // Specialize for 2x2 upsample.
  if (!op_params.align_corners && !op_params.half_pixel_centers &&
      output_height == 2 * input_height && output_width == 2 * input_width) {
    ResizeBilinear2x2(batches, input_height, input_width, depth, output_height,
                      output_width, input_shape, input_data, output_shape,
                      output_data);
  } else {
    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;
    if (op_params.align_corners && output_height > 1) {
      height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
    }
    if (op_params.align_corners && output_width > 1) {
      width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
    }

    ResizeBilinearGeneric(batches, input_height, input_width, depth,
                          output_height, output_width, height_scale,
                          width_scale, input_shape, input_data, output_shape,
                          output_data, op_params.half_pixel_centers);
  }
}

// Note: This is not a universal quantized bilinear. It does not use int8
// or int16 arithmetic.
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const uint8* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           uint8* output_data) {
  ruy::profiler::ScopeLabel label("ResizeBilinear");
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
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

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  float height_scale =
      (op_params.align_corners && output_height > 1)
          ? (static_cast<float>(input_height - 1) / (output_height - 1))
          : (static_cast<float>(input_height) / output_height);

  float width_scale =
      (op_params.align_corners && output_width > 1)
          ? (static_cast<float>(input_width - 1) / (output_width - 1))
          : (static_cast<float>(input_width) / output_width);

  ResizeBilinearGenericSmallChannel<uint8>(
      batches, input_height, input_width, depth, output_height, output_width,
      height_scale, width_scale, input_shape, input_data, output_shape,
      output_data, op_params.half_pixel_centers);
}

// TODO(b/180609127) Create optimized int8 version from uint8. Call from here.
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const int8* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           int8* output_data) {
  reference_ops::ResizeBilinearInteger(op_params, unextended_input_shape,
                                       input_data, unextended_output_size_shape,
                                       output_size_data,
                                       unextended_output_shape, output_data);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_

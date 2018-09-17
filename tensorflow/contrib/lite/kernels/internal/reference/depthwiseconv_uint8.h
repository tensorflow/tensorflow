/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConv/8bit");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32 acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32 input_val =
                      input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  int32 filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, oc)];
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[oc];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                                -output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8>(acc);
          }
        }
      }
    }
  }
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy.
inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;

  DepthwiseConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                bias_data, DimsToShape(output_dims), output_data);
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, 1, 1, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, int32 output_offset,
                   int32 output_multiplier, int output_shift,
                   int32 output_activation_min, int32 output_activation_max,
                   uint8* output_data, const Dims<4>& output_dims) {
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   int32 output_offset, int32 output_multiplier,
                   int output_shift, int32 output_activation_min,
                   int32 output_activation_max, uint8* output_data,
                   const Dims<4>& output_dims) {
  DepthwiseConv<Ac>(input_data, input_dims, input_offset, filter_data,
                    filter_dims, filter_offset, bias_data, bias_dims, stride,
                    stride, pad_width, pad_height, depth_multiplier,
                    output_offset, output_multiplier, output_shift,
                    output_activation_min, output_activation_max, output_data,
                    output_dims);
}

}  // end namespace reference_ops
}  // end namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_

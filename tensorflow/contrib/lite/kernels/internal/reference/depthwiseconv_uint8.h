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
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  TFLITE_DCHECK(output_depth == input_depth * depth_multiplier);

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
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32 input_val =
                      input_data[Offset(input_dims, ic, in_x, in_y, b)];
                  int32 filter_val = filter_data[Offset(filter_dims, oc,
                                                        filter_x, filter_y, 0)];
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[Offset(bias_dims, oc, 0, 0, 0)];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                                -output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_dims, oc, out_x, out_y, b)] =
                static_cast<uint8>(acc);
          }
        }
      }
    }
  }
}

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

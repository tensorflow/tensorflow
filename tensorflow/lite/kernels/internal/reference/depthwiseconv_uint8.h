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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Used in tests and template parameters to control which version of depthwise
// convolution is called. Primarily for reference code, and specializations
// forced in tests.
enum class DepthwiseConvImplementation {
  // Run all tests against kUseStandardEntry even if also testing another
  // kernel, since we need to be sure that the main DepthwiseConv() function in
  // optimized_ops.h dispatches to a correctly-executing kernel.
  kNone = 0,                 // The "default" option: use the normal
                             // DepthwiseConv kernel (entry) function.
  kUseGenericKernel,         // Forced use of generic kernel.
  kUseNeon3x3,               // 3x3 kernel that uses NEON when available.
  kUseNeon3x3DotProduct,     // 3x3 kernel that uses dot-product enabled NEON
                             // when available.
  kUseCModel3x3DotProduct,   // 3x3 kernel, reference C model that is intended
                             // to match overall design NEON code.
  kUseUnwound3x3DotProduct,  // 3x3 kernel, reference C model with unwound loops
                             // and some arrays.
  kUseIntrinsics3x3DotProduct,  // 3x3 kernel using NEON intrinsics.
};

// Category of depthwise convolution output rounding.
enum class DepthwiseConvOutputRounding {
  kNone = 0,      // Invalid: specific method must be specified.
  kAwayFromZero,  // Original method: exact halves rounded away from zero.
  kUpward,        // Halves towards +infinity: adds 0.5 before truncate.
  // This is where a future kNearestEven would be placed.
};

// Category of depthwise convolution depth multiplication.
enum class DepthwiseConvDepthMultiplication {
  kNoMultiplication = 0,  // Depth multiplier = 1.
  kUnitInputDepth,        // Input depth = 1, output depth = depth multiplier.
};

namespace reference_ops {
namespace depthwise_conv {

template <DepthwiseConvOutputRounding output_rounding>
inline int32_t DepthwiseConvRound(int32_t x, int32_t quantized_multiplier,
                                  int shift) {
  TFLITE_DCHECK_NE(output_rounding, DepthwiseConvOutputRounding::kNone);
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kAwayFromZero>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int rounding_offset = right_shift > 0 ? 1 << (right_shift - 1) : 0;
  return (SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                            quantized_multiplier) +
          rounding_offset) >>
         right_shift;
}

template <DepthwiseConvOutputRounding output_rounding>
struct DepthwiseConvBasicKernel {
  static inline void Run(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const uint8_t* input_data, const RuntimeShape& filter_shape,
      const uint8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      uint8_t* output_data) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
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
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + dilation_width_factor * filter_x;
                  const int in_y =
                      in_y_origin + dilation_height_factor * filter_y;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    int32_t input_val =
                        input_data[Offset(input_shape, b, in_y, in_x, ic)];
                    int32_t filter_val = filter_data[Offset(
                        filter_shape, 0, filter_y, filter_x, oc)];
                    acc += (filter_val + filter_offset) *
                           (input_val + input_offset);
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[oc];
              }
              acc = DepthwiseConvRound<output_rounding>(acc, output_multiplier,
                                                        output_shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                  static_cast<uint8_t>(acc);
            }
          }
        }
      }
    }
  }

  // TODO(b/148596273): Reconcile reference versions, perhaps with common
  // MultiplyByQuantizedMultiplier or DepthwiseConvRound function.
  static inline void RunPerChannel(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const int8_t* input_data, const RuntimeShape& filter_shape,
      const int8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      int8_t* output_data) {
    // Get parameters.
    // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t* output_multiplier = params.output_multiplier_per_channel;
    const int32_t* output_shift = params.output_shift_per_channel;

    // Check dimensions of the tensors.
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

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
            for (int m = 0; m < depth_multiplier; ++m) {
              const int output_channel = m + in_channel * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + dilation_width_factor * filter_x;
                  const int in_y =
                      in_y_origin + dilation_height_factor * filter_y;
                  // Zero padding by omitting the areas outside the image.
                  const bool is_point_inside_image =
                      (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height);
                  if (is_point_inside_image) {
                    int32_t input_val = input_data[Offset(
                        input_shape, batch, in_y, in_x, in_channel)];
                    int32_t filter_val = filter_data[Offset(
                        filter_shape, 0, filter_y, filter_x, output_channel)];
                    // Accumulate with 32 bits accumulator.
                    // In the nudging process during model quantization, we
                    // force real value of 0.0 be represented by a quantized
                    // value. This guarantees that the input_offset is a int8_t,
                    // even though it is represented using int32_t. int32_t +=
                    // int8_t
                    // * (int8_t - int8_t) so the highest value we can get from
                    // each accumulation is [-127, 127] * ([-128, 127] -
                    // [-128, 127]), which is [-32512, 32512]. log2(32512)
                    // = 14.98, which means we can accumulate at least 2^16
                    // multiplications without overflow. The accumulator is
                    // applied to a filter so the accumulation logic will hold
                    // as long as the filter size (filter_y * filter_x *
                    // in_channel) does not exceed 2^16, which is the case in
                    // all the models we have seen so far.
                    acc += filter_val * (input_val + input_offset);
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[output_channel];
              }
              acc = DepthwiseConvRound<output_rounding>(
                  acc, output_multiplier[output_channel],
                  output_shift[output_channel]);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, batch, out_y, out_x,
                                 output_channel)] = static_cast<int8_t>(acc);
            }
          }
        }
      }
    }
  }
};

}  // namespace depthwise_conv

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8_t* input_data, const RuntimeShape& filter_shape,
    const uint8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    uint8_t* output_data) {
  return depthwise_conv::DepthwiseConvBasicKernel<
      DepthwiseConvOutputRounding::kAwayFromZero>::Run(params, input_shape,
                                                       input_data, filter_shape,
                                                       filter_data, bias_shape,
                                                       bias_data, output_shape,
                                                       output_data);
}

}  // namespace reference_ops
}  // end namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_UINT8_H_

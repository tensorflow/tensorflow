/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/legacy_types.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

static constexpr int kDepthwiseReverseShift = -1;

inline void ShapeFromDims(const tflite::Dims<4>& dims, RuntimeShape* shape) {
  shape->BuildFrom(
      {dims.sizes[3], dims.sizes[2], dims.sizes[1], dims.sizes[0]});
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
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
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  DepthwiseConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                bias_data, DimsToShape(output_dims), output_data);
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
                          const Dims<4>& output_dims) {
  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, 1, 1, pad_width,
                pad_height, depth_multiplier, output_activation_min,
                output_activation_max, output_data, output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, float* output_data,
                   const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, pad_width, pad_height,
                depth_multiplier, output_activation_min, output_activation_max,
                output_data, output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   float* output_data, const Dims<4>& output_dims) {
  DepthwiseConv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
                    bias_dims, stride, stride, pad_width, pad_height,
                    depth_multiplier, output_data, output_dims);
}

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
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kDepthwiseReverseShift * output_shift;

  DepthwiseConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                bias_data, DimsToShape(output_dims), output_data);
}

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

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data);
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int dilation_width_factor,
          int dilation_height_factor, int pad_width, int pad_height,
          float* output_data, const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, dilation_width_factor,
       dilation_height_factor, pad_width, pad_height, output_activation_min,
       output_activation_max, output_data, output_dims, im2col_data,
       im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, 1, 1, pad_width, pad_height,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  Conv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
           bias_dims, stride, stride, 1, 1, pad_width, pad_height, output_data,
           output_dims, im2col_data, im2col_dims);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 int32 output_offset, int32 output_multiplier, int output_shift,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims,
                 uint8* im2col_data, const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data, gemm_context);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height, 1, 1,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const uint8* input_data, const Dims<4>& input_dims,
          int32 input_offset, const uint8* filter_data,
          const Dims<4>& filter_dims, int32 filter_offset,
          const int32* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, int32 output_offset,
          int32 output_multiplier, int output_shift,
          int32 output_activation_min, int32 output_activation_max,
          uint8* output_data, const Dims<4>& output_dims, uint8* im2col_data,
          const Dims<4>& im2col_dims, gemmlowp::GemmContext* gemm_context) {
  Conv<Ac>(input_data, input_dims, input_offset, filter_data, filter_dims,
           filter_offset, bias_data, bias_dims, stride, stride, pad_width,
           pad_height, output_offset, output_multiplier, output_shift,
           output_activation_min, output_activation_max, output_data,
           output_dims, im2col_data, im2col_dims, gemm_context);
}

inline void TransposeConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, float* output_data,
                          const Dims<4>& output_dims, float* im2col_data,
                          const Dims<4>& im2col_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;

  TransposeConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(output_dims),
                output_data, DimsToShape(im2col_dims), im2col_data);
}

inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(weights_dims), weights_data,
                 DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
                 output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemm_context);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, int16* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemm_context);
}

inline void ShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data, gemmlowp::GemmContext* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  ShuffledFullyConnected(op_params, DimsToShape(input_dims), input_data,
                         DimsToShape(weights_dims), shuffled_weights_data,
                         DimsToShape(bias_dims), bias_data,
                         DimsToShape(output_dims), output_data,
                         shuffled_input_workspace_data, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims,
                    gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims, gemm_context);
}

inline void LstmCell(const float* input_data, const Dims<4>& input_dims,
                     const float* prev_activ_data,
                     const Dims<4>& prev_activ_dims, const float* weights_data,
                     const Dims<4>& weights_dims, const float* bias_data,
                     const Dims<4>& bias_dims, const float* prev_state_data,
                     const Dims<4>& prev_state_dims, float* output_state_data,
                     const Dims<4>& output_state_dims, float* output_activ_data,
                     const Dims<4>& output_activ_dims, float* concat_temp_data,
                     const Dims<4>& concat_temp_dims, float* activ_temp_data,
                     const Dims<4>& activ_temp_dims) {
  tflite::LstmCellParams op_params;
  // Float LSTM cell does not need parameters to be set: leave untouched.

  LstmCell(op_params, DimsToShape(input_dims), input_data,
           DimsToShape(prev_activ_dims), prev_activ_data,
           DimsToShape(weights_dims), weights_data, DimsToShape(bias_dims),
           bias_data, DimsToShape(prev_state_dims), prev_state_data,
           DimsToShape(output_state_dims), output_state_data,
           DimsToShape(output_activ_dims), output_activ_data,
           DimsToShape(concat_temp_dims), concat_temp_data,
           DimsToShape(activ_temp_dims), activ_temp_data);
}

template <int StateIntegerBits>
void LstmCell(const uint8* input_data_uint8, const Dims<4>& input_dims,
              const uint8* prev_activ_data_uint8,
              const Dims<4>& prev_activ_dims, const uint8* weights_data_uint8,
              const Dims<4>& weights_dims, const int32* bias_data_int32,
              const Dims<4>& bias_dims, const int16* prev_state_data_int16,
              const Dims<4>& prev_state_dims, int16* output_state_data_int16,
              const Dims<4>& output_state_dims, uint8* output_activ_data_uint8,
              const Dims<4>& output_activ_dims, uint8* concat_temp_data_uint8,
              const Dims<4>& concat_temp_dims, int16* activ_temp_data_int16,
              const Dims<4>& activ_temp_dims, int32 weights_zero_point,
              int32 accum_multiplier, int accum_shift,
              gemmlowp::GemmContext* gemm_context) {
  tflite::LstmCellParams op_params;
  op_params.weights_zero_point = weights_zero_point;
  op_params.accum_multiplier = accum_multiplier;
  op_params.accum_shift = accum_shift;

  LstmCell<StateIntegerBits>(
      op_params, DimsToShape(input_dims), input_data_uint8,
      DimsToShape(prev_activ_dims), prev_activ_data_uint8,
      DimsToShape(weights_dims), weights_data_uint8, DimsToShape(bias_dims),
      bias_data_int32, DimsToShape(prev_state_dims), prev_state_data_int16,
      DimsToShape(output_state_dims), output_state_data_int16,
      DimsToShape(output_activ_dims), output_activ_data_uint8,
      DimsToShape(concat_temp_dims), concat_temp_data_uint8,
      DimsToShape(activ_temp_dims), activ_temp_data_int16, gemm_context);
}

template <typename T>
void BroadcastDiv(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastDiv4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Div(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Div(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
inline void Concatenation(int concat_dim, const Scalar* const* input_data,
                          const Dims<4>* const* input_dims, int inputs_count,
                          Scalar* output_data, const Dims<4>& output_dims) {
  // For now we don't have a model with a Concatenation with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.axis = 3 - concat_dim;
  op_params.inputs_count = inputs_count;

  Concatenation(op_params, input_shapes_indirect.data(), input_data,
                DimsToShape(output_dims), output_data);
}

inline void Concatenation(int concat_dim, const uint8* const* input_data,
                          const Dims<4>* const* input_dims,
                          const int32* input_zeropoint,
                          const float* input_scale, int inputs_count,
                          uint8* output_data, const Dims<4>& output_dims,
                          const int32 output_zeropoint,
                          const float output_scale) {
  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.axis = 3 - concat_dim;
  op_params.input_zeropoint = input_zeropoint;
  op_params.input_scale = input_scale;
  op_params.inputs_count = inputs_count;
  op_params.output_zeropoint = output_zeropoint;
  op_params.output_scale = output_scale;

  ConcatenationWithScaling(op_params, input_shapes_indirect.data(), input_data,
                           DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
void DepthConcatenation(const Scalar* const* input_data,
                        const Dims<4>* const* input_dims, int inputs_count,
                        Scalar* output_data, const Dims<4>& output_dims) {
  // For now we don't have a model with a Concatenation with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.inputs_count = inputs_count;

  DepthConcatenation(op_params, input_shapes_indirect.data(), input_data,
                     DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int axis, int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  std::vector<RuntimeShape> output_shapes(outputs_count);
  std::vector<const RuntimeShape*> output_shapes_indirect(outputs_count);
  for (int i = 0; i < outputs_count; ++i) {
    ShapeFromDims(*output_dims[i], &output_shapes[i]);
    output_shapes_indirect[i] = &output_shapes[i];
  }
  tflite::SplitParams op_params;
  op_params.axis = 3 - axis;
  op_params.num_split = outputs_count;

  Split(op_params, DimsToShape(input_dims), input_data,
        output_shapes_indirect.data(), output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  TFLITE_DCHECK_GE(outputs_count, 1);
  for (int i = 0; i < outputs_count; i++) {
    /* batches = */ MatchingArraySize(*output_dims[i], 3, input_dims, 3);
    /* height = */ MatchingArraySize(*output_dims[i], 2, input_dims, 2);
    /* width = */ MatchingArraySize(*output_dims[i], 1, input_dims, 1);
  }
  // For now we don't have a model with a Split with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  TensorFlowSplit(input_data, input_dims, /*axis=*/0, outputs_count,
                  output_data, output_dims);
}

inline void Softmax(const float* input_data, const RuntimeShape& input_shape,
                    float beta, float* output_data,
                    const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.beta = beta;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Softmax(const uint8* input_data, const RuntimeShape& input_shape,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.diff_min = diff_min;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const float* input_data, const RuntimeShape& input_shape,
                       float* output_data, const RuntimeShape& output_shape) {
  SoftmaxParams params;
  // No params currently used for float LogSoftmax.
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const uint8* input_data, const RuntimeShape& input_shape,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  params.reverse_scaling_divisor = reverse_scaling_divisor;
  params.reverse_scaling_right_shift = reverse_scaling_right_shift;
  params.diff_min = diff_min;
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const uint8* input_data, const RuntimeShape& input_shape,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const RuntimeShape& output_shape) {
  LogisticParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const uint8* input_data, const RuntimeShape& input_shape,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const RuntimeShape& output_shape) {
  TanhParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const int16* input_data, const RuntimeShape& input_shape,
                 int input_left_shift, int16* output_data,
                 const RuntimeShape& output_shape) {
  TanhParams params;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Dequantize(const uint8* input_data, const Dims<4>& input_dims,
                       int32 zero_point, double scale, float* output_data,
                       const Dims<4>& output_dims) {
  tflite::DequantizationParams op_params;
  op_params.zero_point = zero_point;
  op_params.scale = scale;

  Dequantize(op_params, DimsToShape(input_dims), input_data,
             DimsToShape(output_dims), output_data);
}

inline void FakeQuant(const float* input_data, const Dims<4>& input_dims,
                      float rmin, float rmax, int num_bits, float* output_data,
                      const Dims<4>& output_dims) {
  tflite::FakeQuantParams op_params;
  op_params.num_bits = num_bits;
  op_params.minmax.min = rmin;
  op_params.minmax.max = rmax;

  FakeQuant(op_params, DimsToShape(input_dims), input_data,
            DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Gather(const T* input_data, const Dims<4>& input_dims,
                   int input_rank, const int32* coords_data,
                   const Dims<4>& coords_dims, T* output_data,
                   const Dims<4>& output_dims) {
  tflite::GatherParams op_params;
  op_params.axis = 4 - input_rank;

  Gather(op_params, DimsToShape(input_dims), input_data,
         DimsToShape(coords_dims), coords_data, DimsToShape(output_dims),
         output_data);
}

inline uint32 LegacyReverseBits32(uint32 n) {
  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  return (((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) |
          ((n & 0xFF000000) >> 24));
}

inline void StridedSliceReverseIndices(tflite::StridedSliceParams* p) {
  TFLITE_CHECK_EQ(p->start_indices_count, p->stop_indices_count);
  TFLITE_CHECK_EQ(p->stop_indices_count, p->strides_count);

  std::reverse(p->start_indices, p->start_indices + p->start_indices_count);
  std::reverse(p->stop_indices, p->stop_indices + p->stop_indices_count);
  std::reverse(p->strides, p->strides + p->strides_count);

  p->begin_mask = LegacyReverseBits32(static_cast<uint32>(p->begin_mask)) >>
                  (32 - p->start_indices_count);
  p->ellipsis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->ellipsis_mask)) >>
      (32 - p->start_indices_count);
  p->end_mask = LegacyReverseBits32(static_cast<uint32>(p->end_mask)) >>
                (32 - p->start_indices_count);
  p->new_axis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->new_axis_mask)) >>
      (32 - p->start_indices_count);
  p->shrink_axis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->shrink_axis_mask)) >>
      (32 - p->start_indices_count);
}

template <typename T>
inline void StridedSlice(const T* input_data, const Dims<4>& input_dims,
                         int begin_mask, int end_mask, int shrink_axis_mask,
                         const std::vector<int>& start_indices,
                         const std::vector<int>& stop_indices,
                         const std::vector<int>& strides, T* output_data,
                         const Dims<4>& output_dims) {
  TFLITE_DCHECK_EQ(start_indices.size(), 4);
  auto op_params = strided_slice::BuildStridedSliceParams(
      begin_mask, end_mask, shrink_axis_mask, start_indices, stop_indices,
      strides);
  StridedSliceReverseIndices(&op_params);

  StridedSlice(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Mean(const T* input_data, const Dims<4>& input_dims,
                 const std::vector<int>& reduction_indices, T* output_data,
                 const Dims<4>& output_dims) {
  tflite::MeanParams op_params;
  op_params.axis_count = reduction_indices.size();
  for (int i = 0; i < op_params.axis_count; ++i) {
    op_params.axis[i] = reduction_indices[op_params.axis_count - 1 - i];
  }

  Mean(op_params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

template <typename T>
void Transpose(const T* input, const Dims<4>& input_dims, T* output,
               const Dims<4>& output_dims, const int* permuted_axes) {
  TransposeParams params;
  params.perm_count = 4;
  for (int i = 0; i < 4; ++i) {
    params.perm[i] = 3 - permuted_axes[3 - i];
  }
  Transpose(params, DimsToShape(input_dims), input, DimsToShape(output_dims),
            output);
}

template <typename T, ComparisonFn<T> F>
inline void Comparison(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, const Dims<4>& input2_dims,
                       bool* output_data, const Dims<4>& output_dims) {
  ComparisonParams op_params;
  // No parameters needed.
  ComparisonImpl<T, F>(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

template <typename T, ComparisonFn<int32> F>
inline void Comparison(int left_shift, const T* input1_data,
                       const Dims<4>& input1_dims, int32 input1_offset,
                       int32 input1_multiplier, int input1_shift,
                       const T* input2_data, const Dims<4>& input2_dims,
                       int32 input2_offset, int32 input2_multiplier,
                       int input2_shift, bool* output_data,
                       const Dims<4>& output_dims) {
  tflite::ComparisonParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input2_shift = kReverseShift * input2_shift;

  ComparisonWithScaling<T, F>(op_params, DimsToShape(input1_dims), input1_data,
                              DimsToShape(input2_dims), input2_data,
                              DimsToShape(output_dims), output_data);
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison(const T* input1_data,
                                const Dims<4>& input1_dims,
                                const T* input2_data,
                                const Dims<4>& input2_dims, bool* output_data,
                                const Dims<4>& output_dims) {
  ComparisonParams op_params;
  // No parameters needed.
  BroadcastComparison4DSlowImpl<T, F>(op_params, DimsToShape(input1_dims),
                                      input1_data, DimsToShape(input2_dims),
                                      input2_data, DimsToShape(output_dims),
                                      output_data);
}

template <typename T, ComparisonFn<int32> F>
inline void BroadcastComparison(int left_shift, const T* input1_data,
                                const Dims<4>& input1_dims, int32 input1_offset,
                                int32 input1_multiplier, int input1_shift,
                                const T* input2_data,
                                const Dims<4>& input2_dims, int32 input2_offset,
                                int32 input2_multiplier, int input2_shift,
                                bool* output_data, const Dims<4>& output_dims) {
  ComparisonParams op_params;

  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input2_shift = kReverseShift * input2_shift;

  BroadcastComparison4DSlowWithScaling<T, F>(
      op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

#define TFLITE_LEGACY_COMPARISON_OP(name)                                     \
  template <typename T>                                                       \
  inline void name(const T* input1_data, const Dims<4>& input1_dims,          \
                   const T* input2_data, const Dims<4>& input2_dims,          \
                   bool* output_data, const Dims<4>& output_dims) {           \
    gemmlowp::ScopedProfilingLabel label(#name);                              \
    Comparison<T, name##Fn>(input1_data, input1_dims, input2_data,            \
                            input2_dims, output_data, output_dims);           \
  }                                                                           \
  template <typename T>                                                       \
  inline void name(                                                           \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label(#name "/8bit");                      \
    Comparison<T, name##Fn>(left_shift, input1_data, input1_dims,             \
                            input1_offset, input1_multiplier, input1_shift,   \
                            input2_data, input2_dims, input2_offset,          \
                            input2_multiplier, input2_shift, output_data,     \
                            output_dims);                                     \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      const T* input1_data, const Dims<4>& input1_dims, const T* input2_data, \
      const Dims<4>& input2_dims, bool* output_data,                          \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label("Broadcast" #name);                  \
    BroadcastComparison<T, name##Fn>(input1_data, input1_dims, input2_data,   \
                                     input2_dims, output_data, output_dims);  \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    gemmlowp::ScopedProfilingLabel label("Broadcast" #name "/8bit");          \
    BroadcastComparison<T, name##Fn>(left_shift, input1_data, input1_dims,    \
                                     input1_offset, input1_multiplier,        \
                                     input1_shift, input2_data, input2_dims,  \
                                     input2_offset, input2_multiplier,        \
                                     input2_shift, output_data, output_dims); \
  }
TFLITE_LEGACY_COMPARISON_OP(Equal);
TFLITE_LEGACY_COMPARISON_OP(NotEqual);
TFLITE_LEGACY_COMPARISON_OP(Greater);
TFLITE_LEGACY_COMPARISON_OP(GreaterEqual);
TFLITE_LEGACY_COMPARISON_OP(Less);
TFLITE_LEGACY_COMPARISON_OP(LessEqual);
#undef TFLITE_LEGACY_COMPARISON_OP

template <typename D, typename T>
inline void Select(const D* input_condition_data,
                   const Dims<4>& input_condition_dims, const T* input_x_data,
                   const Dims<4>& input_x_dims, const T* input_y_data,
                   const Dims<4>& input_y_dims, T* output_data,
                   const Dims<4>& output_dims) {
  Select(DimsToShape(input_condition_dims), input_condition_data,
         DimsToShape(input_x_dims), input_x_data, DimsToShape(input_y_dims),
         input_y_data, DimsToShape(output_dims), output_data);
}

template <typename D, typename T>
inline void RankOneSelect(const D* input_condition_data,
                          const Dims<4>& input_condition_dims,
                          const T* input_x_data, const Dims<4>& input_x_dims,
                          const T* input_y_data, const Dims<4>& input_y_dims,
                          T* output_data, const Dims<4>& output_dims) {
  RankOneSelect(DimsToShape(input_condition_dims), input_condition_data,
                DimsToShape(input_x_dims), input_x_data,
                DimsToShape(input_y_dims), input_y_data,
                DimsToShape(output_dims), output_data);
}

template <typename T, typename TI>
inline void SparseToDense(const std::vector<std::vector<TI>>& indices,
                          const T* values, T default_value, T* output_data,
                          const Dims<4>& output_dims, bool value_is_scalar) {
  SparseToDense(indices, values, default_value, value_is_scalar,
                DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void Pack(int dim, const Scalar* const* input_data,
          const Dims<4>* const* input_dims, int inputs_count,
          Scalar* output_data, const Dims<4>& output_dims) {
  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::PackParams op_params;
  op_params.axis = 3 - dim;
  op_params.inputs_count = inputs_count;

  Pack(op_params, input_shapes_indirect.data(), input_data,
       DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void Unpack(int axis, const Scalar* input_data, const Dims<4>& input_dims,
            int dimensions, int outputs_count, Scalar* const* output_datas,
            const Dims<4>& output_dims) {
  tflite::UnpackParams op_params;
  op_params.axis = 3 - axis;
  op_params.num_split = outputs_count;

  Unpack(op_params, DimsToShape(input_dims), input_data,
         DimsToShape(output_dims), output_datas);
}

template <typename Scalar>
void Pack(int dim, const Scalar* const* input_data,
          const Dims<4>* const* input_dims, const int32* input_zeropoint,
          const float* input_scale, int inputs_count, Scalar* output_data,
          const Dims<4>& output_dims, const int32 output_zeropoint,
          const float output_scale) {
  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::PackParams op_params;
  op_params.axis = 3 - dim;
  op_params.input_zeropoint = input_zeropoint;
  op_params.input_scale = input_scale;
  op_params.inputs_count = inputs_count;
  op_params.output_zeropoint = output_zeropoint;
  op_params.output_scale = output_scale;

  PackWithScaling(op_params, input_shapes_indirect.data(), input_data,
                  DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  tflite::L2NormalizationParams op_params;
  // No params need to be set for float.

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

inline void L2Normalization(const uint8* input_data,
                            const RuntimeShape& input_shape,
                            int32 input_zero_point, uint8* output_data,
                            const RuntimeShape& output_shape) {
  tflite::L2NormalizationParams op_params;
  op_params.input_zero_point = input_zero_point;

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  L2Normalization<Ac>(input_data, DimsToShape(input_dims), output_data,
                      DimsToShape(output_dims));
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
  L2Normalization(input_data, DimsToShape(input_dims), input_zero_point,
                  output_data, DimsToShape(output_dims));
}

inline void Relu(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Relu(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Relu1(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Relu1(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

inline void Relu6(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Relu6(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

inline void ReluX(uint8 min_value, uint8 max_value, const uint8* input_data,
                  const RuntimeShape& input_shape, uint8* output_data,
                  const RuntimeShape& output_shape) {
  tflite::ActivationParams params;
  params.quantized_activation_max = max_value;
  params.quantized_activation_min = min_value;
  ReluX(params, input_shape, input_data, output_shape, output_data);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Add/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<int32>::min();
  op_params.quantized_activation_max = std::numeric_limits<int32>::max();
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  tflite::ArithmeticParams op_params;
  op_params.broadcast_category =
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.broadcast_shape[4] = y0;
  op_params.broadcast_shape[3] = y1;
  op_params.broadcast_shape[2] = y2;
  op_params.broadcast_shape[1] = y3;
  op_params.broadcast_shape[0] = y4;
  BroadcastAddFivefold(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  tflite::ArithmeticParams op_params;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<T>::min();
  op_params.quantized_activation_max = std::numeric_limits<T>::max();
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// Transitional version that will be moved shortly to legacy_reference_ops, as
// part of RuntimeShape revisions.
inline void BroadcastMul4DSlow(const uint8* input1_data,
                               const Dims<4>& input1_dims, int32 input1_offset,
                               const uint8* input2_data,
                               const Dims<4>& input2_dims, int32 input2_offset,
                               int32 output_offset, int32 output_multiplier,
                               int output_shift, int32 output_activation_min,
                               int32 output_activation_max, uint8* output_data,
                               const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  op_params.input1_offset = input1_offset;
  op_params.input2_offset = input2_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul4DSlow(
      input1_data, input1_dims, input1_offset, input2_data, input2_dims,
      input2_offset, output_offset, output_multiplier,
      //
      kReverseShift * output_shift,
      //
      output_activation_min, output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int kwidth, int kheight, float* output_data,
                 const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, kwidth, kheight, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, float* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_data, output_dims);
}

inline void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int filter_width, int filter_height,
                        int32 output_activation_min,
                        int32 output_activation_max, uint8* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int filter_width, int filter_height,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_activation_min,
                  output_activation_max, output_data, output_dims);
}

inline void MaxPool(const float* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int kwidth, int kheight,
                    float output_activation_min, float output_activation_max,
                    float* output_data, const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int kwidth, int kheight, float* output_data,
             const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             float* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_data, output_dims);
}

inline void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int filter_width, int filter_height,
                    int32 output_activation_min, int32 output_activation_max,
                    uint8* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int filter_width, int filter_height, int32 output_activation_min,
             int32 output_activation_max, uint8* output_data,
             const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, filter_width, filter_height, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             int32 output_activation_min, int32 output_activation_max,
             uint8* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

inline void L2Pool(const float* input_data, const Dims<4>& input_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int filter_width, int filter_height,
                   float output_activation_min, float output_activation_max,
                   float* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  L2Pool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
         output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int filter_width, int filter_height, float* output_data,
            const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  L2Pool(input_data, input_dims, stride_width, stride_height, pad_width,
         pad_height, filter_width, filter_height, output_activation_min,
         output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int filter_width, int filter_height,
            float* output_data, const Dims<4>& output_dims) {
  L2Pool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
             filter_width, filter_height, output_data, output_dims);
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), beta, output_data,
          DimsToShape(output_dims));
}

inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), input_beta_multiplier,
          input_beta_left_shift, diff_min, output_data,
          DimsToShape(output_dims));
}

inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), output_data,
             DimsToShape(output_dims));
}

inline void LogSoftmax(const uint8* input_data, const Dims<4>& input_dims,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), input_multiplier,
             input_left_shift, reverse_scaling_divisor,
             reverse_scaling_right_shift, diff_min, output_data,
             DimsToShape(output_dims));
}

inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Logistic(const uint8* input_data, const Dims<4>& input_dims,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), input_zero_point,
           input_range_radius, input_multiplier, input_left_shift, output_data,
           DimsToShape(output_dims));
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Tanh(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_zero_point,
       input_range_radius, input_multiplier, input_left_shift, output_data,
       DimsToShape(output_dims));
}

inline void Tanh(const int16* input_data, const Dims<4>& input_dims,
                 int input_left_shift, int16* output_data,
                 const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_left_shift, output_data,
       DimsToShape(output_dims));
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::DepthToSpaceParams op_params;
  op_params.block_size = block_size;

  DepthToSpace(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::SpaceToDepthParams op_params;
  op_params.block_size = block_size;

  SpaceToDepth(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Mul(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  // No params in this version.

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.output_offset = output_offset;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
  tflite::LocalResponseNormalizationParams op_params;
  op_params.range = range;
  op_params.bias = bias;
  op_params.alpha = alpha;
  op_params.beta = beta;

  LocalResponseNormalization(op_params, DimsToShape(input_dims), input_data,
                             DimsToShape(output_dims), output_data);
}

template <typename SrcT, typename DstT>
void Cast(const SrcT* input_data, const Dims<4>& input_dims, DstT* output_data,
          const Dims<4>& output_dims) {
  Cast(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Floor(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

template <typename T>
inline void ResizeBilinear(const T* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, T* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear<float>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear<uint8>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims,
                           const int32_t pad_value) {
  tflite::SpaceToBatchParams op_params;
  op_params.output_offset = pad_value;

  SpaceToBatchND(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(paddings_dims), paddings_data,
                 DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims) {
  tflite::SpaceToBatchParams op_params;
  op_params.output_offset = 0;

  SpaceToBatchND(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(paddings_dims), paddings_data,
                 DimsToShape(output_dims), output_data);
}

template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
  BatchToSpaceND(DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(crops_dims), crops_data, DimsToShape(output_dims),
                 output_data);
}

// Legacy signature, function covered both Pad and PadV2.
template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);
  tflite::PadParams op_params;
  op_params.left_padding_count = 4;
  op_params.right_padding_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.left_padding[i] = left_paddings[3 - i];
    op_params.right_padding[i] = right_paddings[3 - i];
  }
  // SetFloatOrInt(pad_value, &op_params.pad_value);
  const T pad_value_copy = pad_value;

  Pad(op_params, DimsToShape(input_dims), input_data, &pad_value_copy,
      DimsToShape(output_dims), output_data);
}

// Old Pad that calls legacy PadV2.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

// Old Pad that only padded with 0.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  const T pad_value = static_cast<T>(0);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, pad_value);
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Minimum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Maximum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T, typename Op>
void TensorFlowMaximumMinimum(const T* input1_data, const Dims<4>& input1_dims,
                              const T* input2_data, const Dims<4>& input2_dims,
                              T* output_data, const Dims<4>& output_dims,
                              Op op) {
  MaximumMinimumBroadcast4DSlow(DimsToShape(input1_dims), input1_data,
                                DimsToShape(input2_dims), input2_data,
                                DimsToShape(output_dims), output_data, op);
}

template <typename T1, typename T2, typename T3>
void ArgMax(const T3* axis, const T1* input_data,
            const tflite::Dims<4>& input_dims, T2* output_data,
            const tflite::Dims<4>& output_dims) {
  // Assumes the input always has 4 dimensions, and therefore,
  // output always has three dimensions.
  auto output_shape = RuntimeShape(
      {output_dims.sizes[2], output_dims.sizes[1], output_dims.sizes[0]});
  // Another way to interpret this is that output_dims.sizes[4] is always 1.
  TFLITE_DCHECK_EQ(output_shape.FlatSize(),
                   DimsToShape(output_dims).FlatSize());
  // Legacy path only supported this.
  TFLITE_DCHECK_EQ(axis[0], 3);
  ArgMinMax(DimsToShape(input_dims), input_data, axis, output_shape,
            output_data, std::greater<T1>());
}

template <typename T1, typename T2, typename T3, typename Cmp>
void ArgMinMax(const T3* axis, const T1* input_data, const Dims<4>& input_dims,
               T2* output_data, const Dims<4>& output_dims, const Cmp& cmp) {
  ArgMinMax(axis, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
            output_data, cmp);
}

template <typename T>
inline void Pow(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T* output_data, const Dims<4>& output_dims) {
  Pow(DimsToShape(input1_dims), input1_data, DimsToShape(input2_dims),
      input2_data, DimsToShape(output_dims), output_data);
}

template <typename T>
inline void BroadcastPow(const T* input1_data, const Dims<4>& input1_dims,
                         const T* input2_data, const Dims<4>& input2_dims,
                         T* output_data, const Dims<4>& output_dims) {
  BroadcastPow4DSlow(DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void Logical(const bool* input1_data, const Dims<4>& input1_dims,
                    const bool* input2_data, const Dims<4>& input2_dims,
                    bool* output_data, const Dims<4>& output_dims,
                    const std::function<bool(bool, bool)>& func) {
  Logical(DimsToShape(input1_dims), input1_data, DimsToShape(input2_dims),
          input2_data, DimsToShape(output_dims), output_data, func);
}

inline void BroadcastLogical(const bool* input1_data,
                             const Dims<4>& input1_dims,
                             const bool* input2_data,
                             const Dims<4>& input2_dims, bool* output_data,
                             const Dims<4>& output_dims,
                             const std::function<bool(bool, bool)>& func) {
  BroadcastLogical4DSlow(DimsToShape(input1_dims), input1_data,
                         DimsToShape(input2_dims), input2_data,
                         DimsToShape(output_dims), output_data, func);
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BroadcastBinaryFunction(const T1* input1_data,
                                    const Dims<4>& input1_dims,
                                    const T2* input2_data,
                                    const Dims<4>& input2_dims, R* output_data,
                                    const Dims<4>& output_dims,
                                    R (*func)(T1, T2)) {
  BroadcastBinaryFunction(DimsToShape(input1_dims), input1_data,
                          DimsToShape(input2_dims), input2_data,
                          DimsToShape(output_dims), output_data, func);
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BinaryFunction(const T1* input1_data, const Dims<4>& input1_dims,
                           const T2* input2_data, const Dims<4>& input2_dims,
                           R* output_data, const Dims<4>& output_dims,
                           R (*func)(T1, T2)) {
  BinaryFunction(DimsToShape(input1_dims), input1_data,
                 DimsToShape(input2_dims), input2_data,
                 DimsToShape(output_dims), output_data, func);
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::SliceParams op_params;
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.begin[i] = begin[3 - i];
    op_params.size[i] = size[3 - i];
  }

  Slice(op_params, DimsToShape(input_dims), input_data,
        DimsToShape(output_dims), output_data);
}

}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_

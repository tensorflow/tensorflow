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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::ArgMax;
using reference_ops::ArgMinMax;
using reference_ops::Broadcast4DSlowGreater;
using reference_ops::Broadcast4DSlowGreaterEqual;
using reference_ops::Broadcast4DSlowGreaterEqualWithScaling;
using reference_ops::Broadcast4DSlowGreaterWithScaling;
using reference_ops::Broadcast4DSlowLess;
using reference_ops::Broadcast4DSlowLessEqual;
using reference_ops::Broadcast4DSlowLessEqualWithScaling;
using reference_ops::Broadcast4DSlowLessWithScaling;
using reference_ops::BroadcastAdd4DSlow;
using reference_ops::BroadcastGreater;
using reference_ops::BroadcastGreaterEqual;
using reference_ops::BroadcastLess;
using reference_ops::BroadcastLessEqual;
using reference_ops::BroadcastMul4DSlow;
using reference_ops::BroadcastSub4DSlow;
using reference_ops::Concatenation;
using reference_ops::ConcatenationWithScaling;
using reference_ops::DepthConcatenation;
using reference_ops::Dequantize;
using reference_ops::Div;
using reference_ops::FakeQuant;
using reference_ops::Gather;
using reference_ops::Greater;
using reference_ops::GreaterEqual;
using reference_ops::GreaterEqualWithScaling;
using reference_ops::GreaterWithScaling;
using reference_ops::Less;
using reference_ops::LessEqual;
using reference_ops::LessEqualWithScaling;
using reference_ops::LessWithScaling;
using reference_ops::Mean;
using reference_ops::RankOneSelect;
using reference_ops::Relu1;
using reference_ops::Relu6;
using reference_ops::ReluX;
using reference_ops::Select;
using reference_ops::SpaceToBatchND;
using reference_ops::Split;
using reference_ops::StridedSlice;
using reference_ops::TensorFlowSplit;
using reference_ops::Transpose;

static constexpr int kDepthwiseReverseShift = -1;

template <typename Scalar, int N>
VectorMap<Scalar> MapAsVector(Scalar* data, const Dims<N>& dims) {
  const int size = FlatSize(dims);
  return VectorMap<Scalar>(data, size, 1);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar* data,
                                                const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar* data,
                                               const Dims<N>& dims) {
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++) {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
ArrayMap<Scalar> MapAsArrayWithFirstDimAsRows(Scalar* data,
                                              const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return ArrayMap<Scalar>(data, rows, cols);
}

// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar* data,
                                                   const Dims<N>& dims,
                                                   int rows) {
  const int flatsize = FlatSize(dims);
  TFLITE_DCHECK((flatsize % rows) == 0);
  const int cols = flatsize / rows;
  return MatrixMap<Scalar>(data, rows, cols);
}

inline bool AreSameDims(const Dims<4>& dims1, const Dims<4>& dims2) {
  for (int i = 0; i < 4; i++) {
    if (dims1.sizes[i] != dims2.sizes[i]) {
      return false;
    }
  }
  return true;
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

// legacy, for compatibility with old checked-in code
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

// legacy, for compatibility with old checked-in code
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

inline void AddBiasAndEvalActivationFunction(const float* bias_data,
                                             const Dims<4>& bias_dims,
                                             float* array_data,
                                             const Dims<4>& array_dims,
                                             float output_activation_min,
                                             float output_activation_max) {
  AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                   DimsToShape(bias_dims), bias_data,
                                   DimsToShape(array_dims), array_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float* bias_data,
                                      const Dims<4>& bias_dims,
                                      float* array_data,
                                      const Dims<4>& array_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  AddBiasAndEvalActivationFunction(bias_data, bias_dims, array_data, array_dims,
                                   output_activation_min,
                                   output_activation_max);
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

inline void FullyConnected(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data_int32, const Dims<4>& bias_dims, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, int16* output_data, const Dims<4>& output_dims,
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
                 bias_data_int32, DimsToShape(output_dims), output_data,
                 gemm_context);
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
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims, gemm_context);
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

template <typename T>
inline void ExtractPatchIntoBufferColumn(
    const Dims<4>& input_dims, int w, int h, int b, int kheight, int kwidth,
    int stride_width, int stride_height, int pad_width, int pad_height,
    int in_width, int in_height, int in_depth, int single_buffer_length,
    int buffer_id, const T* in_data, T* conv_buffer_data, uint8 zero_byte) {
  ExtractPatchIntoBufferColumn(
      DimsToShape(input_dims), w, h, b, kheight, kwidth, stride_width,
      stride_height, pad_width, pad_height, in_width, in_height, in_depth,
      single_buffer_length, buffer_id, in_data, conv_buffer_data, zero_byte);
}

template <typename T>
void DilatedIm2col(const T* input_data, const Dims<4>& input_dims,
                   const Dims<4>& filter_dims, int stride_width,
                   int stride_height, int dilation_width_factor,
                   int dilation_height_factor, int pad_width, int pad_height,
                   const Dims<4>& output_dims, uint8 zero_byte,
                   T* im2col_data) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;

  DilatedIm2col(op_params, zero_byte, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), DimsToShape(output_dims),
                im2col_data);
}

template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride_width,
            int stride_height, int pad_width, int pad_height, int kheight,
            int kwidth, uint8 zero_byte, T* output_data,
            const Dims<4>& output_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;

  Im2col(op_params, kheight, kwidth, zero_byte, DimsToShape(input_dims),
         input_data, DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 zero_byte, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, zero_byte, output_data, output_dims);
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

inline void HybridConv(const int8_t* input_data, const Dims<4>& input_dims,
                       const int8_t* filter_data, const Dims<4>& filter_dims,
                       const float* bias_data, const Dims<4>& bias_dims,
                       int stride_width, int stride_height, int pad_width,
                       int pad_height, float* scaling_factors_ptr,
                       float output_activation_min, float output_activation_max,
                       float* output_data, const Dims<4>& output_dims,
                       int8_t* im2col_data, const Dims<4>& im2col_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  HybridConv(op_params, scaling_factors_ptr, DimsToShape(input_dims),
             input_data, DimsToShape(filter_dims), filter_data,
             DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
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
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride, stride, pad_width,
       pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 zero_byte, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, zero_byte, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const float* input_data, const Dims<4>& input_dims,
                const float* filter_data, const Dims<4>& filter_dims,
                const float* bias_data, const Dims<4>& bias_dims,
                float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("ConvAsGemm");

  const auto input_matrix_map =
      MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data,
                                       output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const uint8* input_data, const Dims<4>& input_dims,
                int32 input_offset, const uint8* filter_data,
                const Dims<4>& filter_dims, int32 filter_offset,
                const int32* bias_data, const Dims<4>& bias_dims,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims,
                gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("ConvAsGemm/8bit");
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  const int input_rows = input_dims.sizes[0];
  const int input_cols = FlatSizeSkipDim(input_dims, 0);
  const int filter_rows = filter_dims.sizes[3];
  const int filter_cols = FlatSizeSkipDim(filter_dims, 3);
  const int output_rows = output_dims.sizes[0];
  const int output_cols = FlatSizeSkipDim(output_dims, 0);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, input_cols);
  TFLITE_DCHECK_EQ(filter_cols, input_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[0], output_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[1], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[3], 1);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, output_cols, filter_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, -output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
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

template <typename T>
void TransposeIm2col(const T* input_data, const Dims<4>& input_dims,
                     const Dims<4>& filter_dims, int stride_width,
                     int stride_height, int pad_width, int pad_height,
                     const Dims<4>& output_dims, uint8 zero_byte,
                     T* im2col_data) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;

  TransposeIm2col(op_params, zero_byte, DimsToShape(input_dims), input_data,
                  DimsToShape(filter_dims), DimsToShape(output_dims),
                  im2col_data);
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

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  tflite::L2NormalizationParams op_params;
  // No params need to be set for float, but reserved in signature for future
  // activations.

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

// legacy, for compatibility with old checked-in code
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
  constexpr int kReverseShift = -1;
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
  T output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
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
  op_params.output_shift = kReverseShift * output_shift;

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
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

inline void Softmax(const float* input_data, const RuntimeShape& input_shape,
                    float beta, float* output_data,
                    const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.beta = beta;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), beta, output_data,
          DimsToShape(output_dims));
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
inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), input_beta_multiplier,
          input_beta_left_shift, diff_min, output_data,
          DimsToShape(output_dims));
}

inline void LogSoftmax(const float* input_data, const RuntimeShape& input_shape,
                       float* output_data, const RuntimeShape& output_shape) {
  SoftmaxParams params;
  // No params currently used for float LogSoftmax.
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), output_data,
             DimsToShape(output_dims));
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

inline void Logistic(const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const int16* input_data, const RuntimeShape& input_shape,
                     int16* output_data, const RuntimeShape& output_shape) {
  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), output_data,
           DimsToShape(output_dims));
}

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Tanh(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
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

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_zero_point,
       input_range_radius, input_multiplier, input_left_shift, output_data,
       DimsToShape(output_dims));
}

inline void Tanh(const int16* input_data, const RuntimeShape& input_shape,
                 int input_left_shift, int16* output_data,
                 const RuntimeShape& output_shape) {
  TanhParams params;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
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

inline void Mul(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Mul(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

inline void Mul(const int32* input1_data, const Dims<4>& input1_dims,
                const int32* input2_data, const Dims<4>& input2_dims,
                int32 output_activation_min, int32 output_activation_max,
                int32* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);
  tflite::ArithmeticParams op_params;
  // No parameters needed.

  MulNoActivation(op_params, DimsToShape(input1_dims), input1_data,
                  DimsToShape(input2_dims), input2_data,
                  DimsToShape(output_dims), output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  // No parameters needed.

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
  op_params.output_offset = output_offset;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

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

// For compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const float* input1_data, const Dims<4>& input1_dims,
                         const float* input2_data, const Dims<4>& input2_dims,
                         float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  float float_activation_min;
  float float_activation_max;
  GetActivationMinMax(Ac, &float_activation_min, &float_activation_max);
  SetActivationParams(float_activation_min, float_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
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

inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
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
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
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

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_

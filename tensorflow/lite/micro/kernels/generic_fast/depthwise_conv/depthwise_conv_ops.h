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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_DEPTHWISE_CONV_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_DEPTHWISE_CONV_OPS_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/generic_fast/depthwise_conv/depthwise_conv_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

TfLiteStatus EvalUInt8Padding(TfLiteContext* context,
                              const TfLiteDepthwiseConvParams& params,
                              OpData* data, const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* filter,
                              const TfLiteEvalTensor* bias,
                              TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  const int32_t input_offset = -data->input_zero_point;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
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
  int32_t* acc_buf = data->acc_buf;

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = std::max(0, -in_y_origin);
      const int32_t ker_y_end =
          std::min(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = std::max(0, -in_x_origin);
        const int32_t ker_x_end =
            std::min(filter_width, input_width - in_x_origin);
        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val =
                    filter_data[filter_offset2 + output_channel];
                acc_buf[output_channel] +=
                    (input_val + input_offset) * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias_data) {
            acc_buf[i] += bias_data[i];
          }
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier, output_shift);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalUInt8(TfLiteContext* context,
                       const TfLiteDepthwiseConvParams& params, OpData* data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = std::max(0, -in_y_origin);
      const int32_t ker_y_end =
          std::min(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = std::max(0, -in_x_origin);
        const int32_t ker_x_end =
            std::min(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val =
                    filter_data[filter_offset2 + output_channel];
                acc_buf[output_channel] +=
                    input_val * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc_buf[i] += data->sum_of_filters_factor[i];
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier, output_shift);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context,
                       const TfLiteDepthwiseConvParams& params, OpData* data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params.activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are
  // +ve-means-left.
  op_params.output_shift = -data->output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalInt8Padding(TfLiteContext* context,
                             const TfLiteDepthwiseConvParams& params,
                             OpData* data, const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = -data->input_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_activation_min = std::numeric_limits<int8_t>::min();
  const int32_t output_activation_max = std::numeric_limits<int8_t>::max();

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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = std::max(0, -in_y_origin);
      const int32_t ker_y_end =
          std::min(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = std::max(0, -in_x_origin);
        const int32_t ker_x_end =
            std::min(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val = filter_data[filter_offset2 + out_channel];
                acc_buf[out_channel] += (input_val + input_offset) * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias) {
            acc_buf[i] += bias_data[i];
          }
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier[i], output_shift[i]);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalInt8(TfLiteContext* context,
                      const TfLiteDepthwiseConvParams& params, OpData* data,
                      const TfLiteEvalTensor* input,
                      const TfLiteEvalTensor* filter,
                      const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_activation_min = std::numeric_limits<int8_t>::min();
  const int32_t output_activation_max = std::numeric_limits<int8_t>::max();

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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = std::max(0, -in_y_origin);
      const int32_t ker_y_end =
          std::min(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = std::max(0, -in_x_origin);
        const int32_t ker_x_end =
            std::min(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val = filter_data[filter_offset2 + out_channel];
                acc_buf[out_channel] += input_val * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc_buf[i] += data->sum_of_filters_factor[i];
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier[i], output_shift[i]);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalInt8Reference(TfLiteContext* context,
                               const TfLiteDepthwiseConvParams& params,
                               OpData* data, const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = data->output_zero_point;
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalUInt8Reference(TfLiteContext* context,
                                const TfLiteDepthwiseConvParams& params,
                                OpData* data, const TfLiteEvalTensor* input,
                                const TfLiteEvalTensor* filter,
                                const TfLiteEvalTensor* bias,
                                TfLiteEvalTensor* output) {
  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are
  // +ve-means-left.
  op_params.output_shift = -data->output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<uint8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<uint8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<uint8_t>(output));
  return kTfLiteOk;
}

}  // namespace
}  // namespace tflite

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_DEPTHWISE_CONV_OPS_H_ */

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_CORE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_CORE_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

template <typename T, class PADDING_TRAIT>
class ConvKernelCore;

// UINT8
template <class PADDING_TRAIT>
class ConvKernelCore<uint8_t, PADDING_TRAIT> {
 public:
  static void run(const uint8_t* input, const RuntimeShape& input_shape,
                  const uint8_t* filter, const RuntimeShape& filter_shape,
                  uint8_t* output, const RuntimeShape& output_shape,
                  ConvParams& params, int batches, int32_t* sum_of_filters,
                  const int32_t* per_channel_multiplier,
                  const int32_t* per_channel_shift) {
    const int* in_dims =
        reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
    const int* fi_dims =
        reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
    const PADDING_TRAIT pad_trait(params.input_offset);

    int32_t acc;

    for (int batch = 0; batch < batches; ++batch) {
      uint32_t offset_input0 = batch * in_dims[1];
      for (int out_y = 0; out_y < output_shape.Dims(1); ++out_y) {
        for (int out_x = 0; out_x < output_shape.Dims(2); ++out_x) {
          for (int out_channel = 0; out_channel < output_shape.Dims(3);
               ++out_channel) {
            const int in_x_origin =
                out_x * params.stride_width - params.padding_values.width;
            const int in_y_origin =
                out_y * params.stride_height - params.padding_values.height;
            uint32_t offset_filter0 = out_channel * fi_dims[1];

            acc = 0;

            const int32_t ker_y_start = std::max(0, -in_y_origin);
            const int32_t ker_x_start = std::max(0, -in_x_origin);
            const int32_t ker_y_end = std::min(
                filter_shape.Dims(1), input_shape.Dims(1) - in_y_origin);
            const int32_t ker_x_end = std::min(
                filter_shape.Dims(2), input_shape.Dims(2) - in_x_origin);

            for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
              const int in_y = in_y_origin + filter_y;
              uint32_t offset_filter1 =
                  (offset_filter0 + filter_y) * fi_dims[2];
              uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];
              for (int filter_x = ker_x_start; filter_x < ker_x_end;
                   ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];
                uint32_t offset_filter2 =
                    (offset_filter1 + filter_x) * fi_dims[3];
                for (int in_channel = 0; in_channel < input_shape.Dims(3);
                     ++in_channel) {
                  int32_t input_val = pad_trait.OffsetInputValue(
                      input[offset_input2 + in_channel]);
                  int32_t filter_val = filter[offset_filter2 + in_channel] +
                                       params.weights_offset;

                  acc += filter_val * input_val;
                }
              }
            }
            acc += sum_of_filters[out_channel];
            acc = MultiplyByQuantizedMultiplier(acc, params.output_multiplier,
                                                params.output_shift);
            acc += params.output_offset;
            acc = std::max(acc, params.quantized_activation_min);
            acc = std::min(acc, params.quantized_activation_max);
            output[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                static_cast<uint8_t>(acc);
          }
        }
      }
    }
  }
};

// INT8
template <class PADDING_TRAIT>
class ConvKernelCore<int8_t, PADDING_TRAIT> {
 public:
  static void run(const int8_t* input, const RuntimeShape& input_shape,
                  const int8_t* filter, const RuntimeShape& filter_shape,
                  int8_t* output, const RuntimeShape& output_shape,
                  ConvParams& params, int batches, int32_t* sum_of_filters,
                  const int32_t* per_channel_multiplier,
                  const int32_t* per_channel_shift) {
    const int* in_dims =
        reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
    const int* fi_dims =
        reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
    const PADDING_TRAIT pad_trait(params.input_offset);

    int32_t acc;

    for (int batch = 0; batch < batches; ++batch) {
      uint32_t offset_input0 = batch * in_dims[1];
      for (int out_y = 0; out_y < output_shape.Dims(1); ++out_y) {
        for (int out_x = 0; out_x < output_shape.Dims(2); ++out_x) {
          for (int out_channel = 0; out_channel < output_shape.Dims(3);
               ++out_channel) {
            const int in_x_origin =
                (out_x * params.stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * params.stride_height) - params.padding_values.height;
            uint32_t offset_filter0 = out_channel * fi_dims[1];

            const int32_t ker_y_start = std::max(0, -in_y_origin);
            const int32_t ker_x_start = std::max(0, -in_x_origin);

            const int32_t ker_y_end = std::min(
                filter_shape.Dims(1), input_shape.Dims(1) - in_y_origin);
            const int32_t ker_x_end = std::min(
                filter_shape.Dims(2), input_shape.Dims(2) - in_x_origin);

            acc = 0;

            for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
              const int in_y = in_y_origin + filter_y;
              uint32_t offset_filter1 =
                  (offset_filter0 + filter_y) * fi_dims[2];
              uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];

              for (int filter_x = ker_x_start; filter_x < ker_x_end;
                   ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                uint32_t offset_filter2 =
                    (offset_filter1 + filter_x) * fi_dims[3];
                uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];

                for (int in_channel = 0; in_channel < input_shape.Dims(3);
                     ++in_channel) {
                  int32_t input_val = pad_trait.OffsetInputValue(
                      input[offset_input2 + in_channel]);
                  int32_t filter_val = filter[offset_filter2 + in_channel];

                  acc +=filter_val * input_val;
                }
              }
            }
            acc += sum_of_filters[out_channel];
            acc = MultiplyByQuantizedMultiplier(
                acc, per_channel_multiplier[out_channel],
                per_channel_shift[out_channel]);
            acc += params.output_offset;
            acc = std::max(acc, params.quantized_activation_min);
            acc = std::min(acc, params.quantized_activation_max);
            output[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
};

struct ConvPaddingTraits {
  struct WithPadding {
    WithPadding(const int32_t input_offset) : input_offset_(input_offset) {}

    inline int32_t OffsetInputValue(int32_t input_value) const {
      return input_offset_ + input_value;
    };

    const int32_t input_offset_;
  };

  struct WithoutPadding {
    WithoutPadding(const int input_offset) {}

    inline int32_t OffsetInputValue(int32_t input_value) const {
      return input_value;
    };
  };
};

template <typename T>
TfLiteStatus EvalConvWithPadding(TfLiteConvParams* params, OpData* data,
                                 const TfLiteEvalTensor* input,
                                 const TfLiteEvalTensor* filter,
                                 const TfLiteEvalTensor* bias,
                                 TfLiteEvalTensor* output,
                                 TfLiteContext* context) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const T* input_data = tflite::micro::GetTensorData<T>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const T* filter_data = tflite::micro::GetTensorData<T>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  T* output_data = tflite::micro::GetTensorData<T>(output);

  ConvParams op_params;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);

  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  ConvKernelCore<T, ConvPaddingTraits::WithPadding>::run(
      input_data, input_shape, filter_data, filter_shape, output_data,
      output_shape, op_params, batches, data->sum_of_filters_factor,
      output_multiplier, output_shift);

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalConvWithoutPadding(TfLiteConvParams* params, OpData* data,
                                    const TfLiteEvalTensor* input,
                                    const TfLiteEvalTensor* filter,
                                    const TfLiteEvalTensor* bias,
                                    TfLiteEvalTensor* output,
                                    TfLiteContext* context) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const T* input_data = tflite::micro::GetTensorData<T>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const T* filter_data = tflite::micro::GetTensorData<T>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  T* output_data = tflite::micro::GetTensorData<T>(output);

  ConvParams op_params;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);

  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  ConvKernelCore<T, ConvPaddingTraits::WithoutPadding>::run(
      input_data, input_shape, filter_data, filter_shape, output_data,
      output_shape, op_params, batches, data->sum_of_filters_factor,
      output_multiplier, output_shift);

  return kTfLiteOk;
}

}  // namespace
}  // namespace tflite

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_CORE_H_ */

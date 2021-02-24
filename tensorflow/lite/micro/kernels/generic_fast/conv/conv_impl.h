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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_IMPL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_IMPL_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

/*
 * Init function is called once at the beginning to initialize kernels and
 * allocate memory.
 */
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw = context->AllocatePersistentBuffer(context, sizeof(OpData));
  TFLITE_DCHECK(raw != nullptr);
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
}

/*
 * Calculates the OpData which stores all important metadata about the kernel
 * and parameters.
 */
TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels));
  }
  return kTfLiteOk;
}

/*
 * Precomputes a factor from the filter and offsets, which needs to be
 * calculated only once, not in every invocation.
 */
template <typename T>
inline void PrecomputeSumOfFilters(const int32_t* bias,
                                   const TfLiteTensor* filters,
                                   int32_t* sum_of_filters_factor,
                                   RuntimeShape filter_shape,
                                   int32_t input_offset,
                                   int32_t filter_offset = 0) {
  if (filters->type == kTfLiteInt8) {
    // Ensure that the filter offset is 0 in the signed integer case
    TFLITE_DCHECK_EQ(filter_offset, 0);
  }
  const T* filter_data = GetTensorData<T>(filters);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int input_depth = filter_shape.Dims(3);
  const int output_depth = filter_shape.Dims(0);

  int filter_size = filter_width * filter_height * input_depth;

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int32_t sum_of_filter_factor = filter_size * filter_offset;

    for (int filter_index = filter_size * out_channel;
         filter_index < filter_size + filter_size * out_channel;
         ++filter_index) {
      sum_of_filter_factor += filter_data[filter_index];
    }
    sum_of_filters_factor[out_channel] = sum_of_filter_factor * input_offset;

    if (bias) {
      sum_of_filters_factor[out_channel] += bias[out_channel];
    }
  }
};

inline void PrecomputeSumOfFiltersPadding(const int32_t* bias,
                                          int32_t* sum_of_filters_factor,
                                          RuntimeShape filter_shape) {
  const int output_depth = filter_shape.Dims(0);
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    if (bias) {
      sum_of_filters_factor[out_channel] = bias[out_channel];
    } else {
      sum_of_filters_factor[out_channel] = 0;
    }
  }
}

void Free(TfLiteContext* context, void* buffer) {}

/*
 * Evaluation function. Called in every invocation.
 */
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  return data->eval_function(params, data, input, filter, bias, output,
                             context);
}

}  // namespace
}  // namespace tflite

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_IMPL_H_ */

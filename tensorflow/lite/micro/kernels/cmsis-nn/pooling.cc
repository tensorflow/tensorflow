/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

// These are headers from the ARM CMSIS-NN library.
#include "arm_nnfunctions.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

void AverageEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));
}

void AverageEvalUint8(TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
}

TfLiteStatus AverageEvalInt8(TfLiteContext* context, const TfLiteNode* node,
                             const TfLitePoolParams* params, const OpData* data,
                             TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  TFLITE_DCHECK_LE(activation_min, activation_max);

#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
  RuntimeShape input_shape = GetTensorShape(input);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;

  const int filter_height = params->filter_height;
  const int filter_width = params->filter_width;
  const int padding_height = data->padding.height;
  const int padding_width = data->padding.width;

  int16_t* scratch_buffer = nullptr;

  auto* buffer_idx = reinterpret_cast<int*>(node->user_data);

  if (*buffer_idx > -1) {
    void* raw = context->GetScratchBuffer(context, *buffer_idx);
    scratch_buffer = reinterpret_cast<int16_t*>(raw);
  }

  TF_LITE_ENSURE_EQ(
      context,
      arm_avgpool_s8(input_height, input_width, output_height, output_width,
                     stride_height, stride_width, filter_height, filter_width,
                     padding_height, padding_width, activation_min,
                     activation_max, depth, GetTensorData<int8_t>(input),
                     scratch_buffer, GetTensorData<int8_t>(output)),
      ARM_MATH_SUCCESS);
#else
#pragma message( \
    "CMSIS-NN optimization for avg_pool not available for this target. Using reference kernel.")

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_integer_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(output), GetTensorData<int8_t>(output));

#endif
  return kTfLiteOk;
}

void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data, TfLiteTensor* input,
                  TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<float>(input), GetTensorShape(output),
                         GetTensorData<float>(output));
}

void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<uint8_t>(input), GetTensorShape(output),
                         GetTensorData<uint8_t>(output));
}

TfLiteStatus MaxEvalInt8(TfLiteContext* context, const TfLiteNode* node,
                         const TfLitePoolParams* params, const OpData* data,
                         TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  TFLITE_DCHECK_LE(activation_min, activation_max);

#if defined(__ARM_FEATURE_DSP)
  RuntimeShape input_shape = GetTensorShape(input);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;

  const int filter_height = params->filter_height;
  const int filter_width = params->filter_width;
  const int padding_height = data->padding.height;
  const int padding_width = data->padding.width;

  int16_t* scratch_buffer = nullptr;

  auto* buffer_idx = reinterpret_cast<int*>(node->user_data);

  if (*buffer_idx > -1) {
    void* raw = context->GetScratchBuffer(context, *buffer_idx);
    scratch_buffer = reinterpret_cast<int16_t*>(raw);
  }

  TF_LITE_ENSURE_EQ(
      context,
      arm_max_pool_s8_opt(input_height, input_width, output_height,
                          output_width, stride_height, stride_width,
                          filter_height, filter_width, padding_height,
                          padding_width, activation_min, activation_max, depth,
                          GetTensorData<int8_t>(input), scratch_buffer,
                          GetTensorData<int8_t>(output)),
      ARM_MATH_SUCCESS);
#else
#pragma message( \
    "CMSIS-NN optimization for max_pool not available for this target. Using reference kernel.")

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_integer_ops::MaxPool(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(output), GetTensorData<int8_t>(output));

#endif
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw;
  context->AllocatePersistentBuffer(context, sizeof(int), &raw);
  return raw;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  RuntimeShape input_shape = GetTensorShape(input);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int output_width = output_shape.Dims(2);

  const int32_t buffer_size =
      arm_avgpool_s8_get_buffer_size(output_width, depth);

  int* buffer_idx = reinterpret_cast<int*>(node->user_data);

  node->user_data = buffer_idx;
  if (buffer_size > 0) {
    TF_LITE_ENSURE_STATUS(
        context->RequestScratchBufferInArena(context, buffer_size, buffer_idx));
  } else {
    *buffer_idx = -1;
  }
#endif
  return kTfLiteOk;
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  // Todo: make 'input' const once CMSIS-reuse is fixed
  TfLiteTensor* input = &context->tensors[flatbuffers::EndianScalar(
      node->inputs->data[kInputTensor])];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      AverageEvalUint8(context, node, params, &data, input, output);
      break;
    case kTfLiteInt8:
      return AverageEvalInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  TfLiteTensor* input = &context->tensors[flatbuffers::EndianScalar(
      node->inputs->data[kInputTensor])];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8(context, node, params, &data, input, output);
      break;
    case kTfLiteInt8:
      MaxEvalInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/pooling::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/pooling::Prepare,
                                 /*invoke=*/pooling::AverageEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/pooling::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/pooling::Prepare,
                                 /*invoke=*/pooling::MaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

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
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/pooling3d.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pooling_3d {

struct OpData {
  Padding3DValues padding_values;
  // int8_t and int16_t activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePool3DParams*>(node->builtin_data);
  auto* opdata = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 5);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context,
                    input->type == kTfLiteFloat32 ||
                        input->type == kTfLiteInt16 ||
                        input->type == kTfLiteInt8,
                    true);

  int batches = input->dims->data[0];
  int depth = input->dims->data[1];
  int height = input->dims->data[2];
  int width = input->dims->data[3];
  int channels = input->dims->data[4];

  // Prevent division by 0 in optimized pooling implementations
  TF_LITE_ENSURE(context, params->stride_depth > 0);
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height, out_depth;
  opdata->padding_values = ComputePadding3DValues(
      params->stride_height, params->stride_width, params->stride_depth, 1, 1,
      1, height, width, depth, params->filter_height, params->filter_width,
      params->filter_depth, params->padding, &out_height, &out_width,
      &out_depth);

  if (input->type == kTfLiteFloat32) {
    float activation_min, activation_max;
    CalculateActivationRange(params->activation, &activation_min,
                             &activation_max);
    SetActivationParams(activation_min, activation_max, opdata);
  }
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_NEAR(context, input->params.scale, output->params.scale,
                        1.0e-6);
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      output->params.zero_point);
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }

    int32_t activation_min, activation_max;
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &activation_min, &activation_max);
    SetActivationParams(activation_min, activation_max, opdata);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(5);
  output_size->data[0] = batches;
  output_size->data[1] = out_depth;
  output_size->data[2] = out_height;
  output_size->data[3] = out_width;
  output_size->data[4] = channels;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePool3DParams*>(node->builtin_data);
  auto* opdata = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

  Pool3DParams op_params;
  op_params.padding_values = opdata->padding_values;
  op_params.stride_depth = params->stride_depth;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_depth = params->filter_depth;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.quantized_activation_min = opdata->quantized_activation_min;
  op_params.quantized_activation_max = opdata->quantized_activation_max;
  op_params.float_activation_min = opdata->float_activation_min;
  op_params.float_activation_max = opdata->float_activation_max;

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::AveragePool3D<float, float>(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::AveragePool3D<int8_t, int32_t>(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::AveragePool3D<int16_t, int32_t>(
          op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePool3DParams*>(node->builtin_data);
  auto* opdata = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

  Pool3DParams op_params;
  op_params.padding_values = opdata->padding_values;
  op_params.stride_depth = params->stride_depth;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_depth = params->filter_depth;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.quantized_activation_min = opdata->quantized_activation_min;
  op_params.quantized_activation_max = opdata->quantized_activation_max;
  op_params.float_activation_min = opdata->float_activation_min;
  op_params.float_activation_max = opdata->float_activation_max;

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::MaxPool3D<float, float>(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::MaxPool3D<int8_t, int32_t>(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::MaxPool3D<int16_t, int32_t>(
          op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace pooling_3d

TfLiteRegistration* Register_AVERAGE_POOL_3D() {
  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::GenericPrepare,
                                 pooling_3d::AverageEval};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_3D() {
  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::GenericPrepare,
                                 pooling_3d::MaxEval};
  return &r;
}

}  // namespace builtin

namespace custom {

// Keep custom registration implementations of the definitions in
// custom_ops_register.h for backward compatibility. Remove them once we get rid
// of the custom pooling 3d ops.
TfLiteRegistration* Register_AVG_POOL_3D() {
  static TfLiteRegistration r = {
      tflite::ops::builtin::pooling_3d::Init,
      tflite::ops::builtin::pooling_3d::Free,
      tflite::ops::builtin::pooling_3d::GenericPrepare,
      tflite::ops::builtin::pooling_3d::AverageEval};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_3D() {
  static TfLiteRegistration r = {
      tflite::ops::builtin::pooling_3d::Init,
      tflite::ops::builtin::pooling_3d::Free,
      tflite::ops::builtin::pooling_3d::GenericPrepare,
      tflite::ops::builtin::pooling_3d::MaxEval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite

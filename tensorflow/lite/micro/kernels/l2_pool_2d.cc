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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/padding.h"

namespace tflite {
namespace {

struct OpData {
  TfLitePaddingValues padding;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      params->filter_height, params->filter_width, padding, &out_height,
      &out_width);

  // We currently don't have a quantized implementation of L2Pool
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);

  // TODO: properly allocate and set output dims
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return kTfLiteError;
}

void L2EvalFloat(TfLiteContext* context, TfLiteNode* node,
                 TfLitePoolParams* params, OpData* data,
                 const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_L2_POOL(type)                                                 \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  type::L2Pool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
               GetTensorShape(output), GetTensorData<float>(output))
  TF_LITE_L2_POOL(reference_ops);

#undef TF_LITE_L2_POOL
}

TfLiteStatus L2Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      L2EvalFloat(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "L2_POOL_2D only supports float32 currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* Register_L2_POOL_2D() { return nullptr; }

}  // namespace tflite

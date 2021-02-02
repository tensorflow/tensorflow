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

#include "tensorflow/lite/kernels/internal/reference/conv3d.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv3d {

// Struct to carry data from Prepare to Eval.
struct OpData {
  Padding3DValues padding;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteConv3DParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Check number of inputs/outputs.
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

  // Check dimensionality of input, filter.
  TF_LITE_ENSURE_EQ(context, input->dims->size, 5);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 5);

  // Check input channels matching filter.
  TF_LITE_ENSURE_EQ(context, input->dims->data[4], filter->dims->data[3]);

  // Check types.
  TfLiteType input_type = input->type;
  TF_LITE_ENSURE_TYPES_EQ(context, input_type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  // Check bias.
  const TfLiteTensor* bias = GetInput(context, node, 2);
  if (bias) {
    TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 4));
  }

  // Filter has shape of [filter_depth, filter_height, filter_width,
  // in_channels, out_channels].
  int batches = input->dims->data[0];
  int channels_out = filter->dims->data[4];
  int depth = input->dims->data[1];
  int height = input->dims->data[2];
  int width = input->dims->data[3];
  int filter_depth = filter->dims->data[0];
  int filter_height = filter->dims->data[1];
  int filter_width = filter->dims->data[2];

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height, out_depth;
  data->padding = ComputePadding3DValues(
      params->stride_height, params->stride_width, params->stride_depth,
      params->dilation_height_factor, params->dilation_width_factor,
      params->dilation_depth_factor, height, width, depth, filter_height,
      filter_width, filter_depth, params->padding, &out_height, &out_width,
      &out_depth);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(5);
  output_size->data[0] = batches;
  output_size->data[1] = out_depth;
  output_size->data[2] = out_height;
  output_size->data[3] = out_width;
  output_size->data[4] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConv3DParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* bias = GetInput(context, node, 2);

  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  Conv3DParams runtime_params;
  runtime_params.padding_values = data->padding;
  runtime_params.stride_depth = params->stride_depth;
  runtime_params.stride_height = params->stride_height;
  runtime_params.stride_width = params->stride_width;
  runtime_params.dilation_depth = params->dilation_depth_factor;
  runtime_params.dilation_height = params->dilation_height_factor;
  runtime_params.dilation_width = params->dilation_width_factor;
  runtime_params.float_activation_min = output_activation_min;
  runtime_params.float_activation_max = output_activation_max;

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Conv3D(runtime_params, GetTensorShape(input),
                            GetTensorData<float>(input), GetTensorShape(filter),
                            GetTensorData<float>(filter), GetTensorShape(bias),
                            GetTensorData<float>(bias), GetTensorShape(output),
                            GetTensorData<float>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv3d

TfLiteRegistration* Register_CONV_3D() {
  static TfLiteRegistration r = {conv3d::Init, conv3d::Free, conv3d::Prepare,
                                 conv3d::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

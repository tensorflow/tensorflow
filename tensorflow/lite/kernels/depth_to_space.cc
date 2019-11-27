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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace depth_to_space {

// This file has two implementation of DepthToSpace. Note that DepthToSpace only
// works on 4D tensors.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthToSpaceParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  auto data_type = output->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteUInt8 ||
                     data_type == kTfLiteInt8 || data_type == kTfLiteInt32 ||
                     data_type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  const int block_size = params->block_size;
  const int input_height = input->dims->data[1];
  const int input_width = input->dims->data[2];
  const int input_channels = input->dims->data[3];
  int output_height = input_height * block_size;
  int output_width = input_width * block_size;
  int output_channels = input_channels / block_size / block_size;

  TF_LITE_ENSURE_EQ(context, input_height, output_height / block_size);
  TF_LITE_ENSURE_EQ(context, input_width, output_width / block_size);
  TF_LITE_ENSURE_EQ(context, input_channels,
                    output_channels * block_size * block_size);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = input->dims->data[0];
  output_size->data[1] = output_height;
  output_size->data[2] = output_width;
  output_size->data[3] = output_channels;

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthToSpaceParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

#define TF_LITE_DEPTH_TO_SPACE(type, scalar)                               \
  tflite::DepthToSpaceParams op_params;                                    \
  op_params.block_size = params->block_size;                               \
  type::DepthToSpace(op_params, GetTensorShape(input),                     \
                     GetTensorData<scalar>(input), GetTensorShape(output), \
                     GetTensorData<scalar>(output))
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_DEPTH_TO_SPACE(reference_ops, float);
      } else {
        TF_LITE_DEPTH_TO_SPACE(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_DEPTH_TO_SPACE(reference_ops, uint8_t);
      } else {
        TF_LITE_DEPTH_TO_SPACE(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_DEPTH_TO_SPACE(reference_ops, int8_t);
      } else {
        TF_LITE_DEPTH_TO_SPACE(optimized_ops, int8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_DEPTH_TO_SPACE(reference_ops, int32_t);
      } else {
        TF_LITE_DEPTH_TO_SPACE(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_DEPTH_TO_SPACE(reference_ops, int64_t);
      } else {
        TF_LITE_DEPTH_TO_SPACE(optimized_ops, int64_t);
      }
      break;
    default:
      context->ReportError(context, "Type '%s' not currently supported.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
#undef TF_LITE_DEPTH_TO_SPACE

  return kTfLiteOk;
}

}  // namespace depth_to_space

TfLiteRegistration* Register_DEPTH_TO_SPACE_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, depth_to_space::Prepare,
      depth_to_space::Eval<depth_to_space::kReference>};
  return &r;
}

TfLiteRegistration* Register_DEPTH_TO_SPACE_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, depth_to_space::Prepare,
      depth_to_space::Eval<depth_to_space::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_DEPTH_TO_SPACE() {
  return Register_DEPTH_TO_SPACE_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

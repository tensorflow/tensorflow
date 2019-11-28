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
namespace space_to_depth {

// This file has two implementation of SpaceToDepth. Note that SpaceToDepth
// only works on 4D tensors.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteSpaceToDepthParams*>(node->builtin_data);

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
  int output_height = input_height / block_size;
  int output_width = input_width / block_size;

  TF_LITE_ENSURE_EQ(context, input_height, output_height * block_size);
  TF_LITE_ENSURE_EQ(context, input_width, output_width * block_size);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = input->dims->data[0];
  output_size->data[1] = output_height;
  output_size->data[2] = output_width;
  output_size->data[3] = input->dims->data[3] * block_size * block_size;

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteSpaceToDepthParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

#define TF_LITE_SPACE_TO_DEPTH(type, scalar)                               \
  tflite::SpaceToDepthParams op_params;                                    \
  op_params.block_size = params->block_size;                               \
  type::SpaceToDepth(op_params, GetTensorShape(input),                     \
                     GetTensorData<scalar>(input), GetTensorShape(output), \
                     GetTensorData<scalar>(output))
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_DEPTH(reference_ops, float);
      } else {
        TF_LITE_SPACE_TO_DEPTH(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_DEPTH(reference_ops, uint8_t);
      } else {
        TF_LITE_SPACE_TO_DEPTH(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_DEPTH(reference_ops, int8_t);
      } else {
        TF_LITE_SPACE_TO_DEPTH(optimized_ops, int8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_DEPTH(reference_ops, int32_t);
      } else {
        TF_LITE_SPACE_TO_DEPTH(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_DEPTH(reference_ops, int64_t);
      } else {
        TF_LITE_SPACE_TO_DEPTH(optimized_ops, int64_t);
      }
      break;
    default:
      context->ReportError(context, "Type '%s' not currently supported.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
#undef TF_LITE_SPACE_TO_DEPTH

  return kTfLiteOk;
}

}  // namespace space_to_depth

TfLiteRegistration* Register_SPACE_TO_DEPTH_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, space_to_depth::Prepare,
      space_to_depth::Eval<space_to_depth::kReference>};
  return &r;
}

TfLiteRegistration* Register_SPACE_TO_DEPTH_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, space_to_depth::Prepare,
      space_to_depth::Eval<space_to_depth::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_SPACE_TO_DEPTH() {
  return Register_SPACE_TO_DEPTH_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

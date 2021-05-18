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
#include "tensorflow/lite/kernels/internal/reference/space_to_depth.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kBatchRank = 0;
constexpr int kHeightRank = 1;
constexpr int kWidthRank = 2;
constexpr int kDepthRank = 3;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteSpaceToDepthParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  auto data_type = output->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  const int block_size = params->block_size;
  const int input_height = input->dims->data[kHeightRank];
  const int input_width = input->dims->data[kWidthRank];
  int output_height = input_height / block_size;
  int output_width = input_width / block_size;

  TF_LITE_ENSURE_EQ(context, input_height, output_height * block_size);
  TF_LITE_ENSURE_EQ(context, input_width, output_width * block_size);

  // Relocate dims to the persistent storage arena before changing them,
  // otherwise we'd be modifying temporary copies made by the interpreters each
  // time they process the layer.
  TfLiteEvalTensor* output_eval =
      micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  output->dims->data[kBatchRank] = input->dims->data[kBatchRank];
  output->dims->data[kHeightRank] = output_height;
  output->dims->data[kWidthRank] = output_width;
  output->dims->data[kDepthRank] =
      input->dims->data[kDepthRank] * block_size * block_size;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteSpaceToDepthParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  SpaceToDepthParams op_params;
  op_params.block_size = params->block_size;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      reference_ops::SpaceToDepth(op_params, micro::GetTensorShape(input),
                                  micro::GetTensorData<float>(input),
                                  micro::GetTensorShape(output),
                                  micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::SpaceToDepth(op_params, micro::GetTensorShape(input),
                                  micro::GetTensorData<int8_t>(input),
                                  micro::GetTensorShape(output),
                                  micro::GetTensorData<int8_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "SPACE_TO_DEPTH only supports FLOAT32 and INT8, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_SPACE_TO_DEPTH() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

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
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

// Input/output tensor index.
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// required rank for input/output tensor shape
constexpr int kTensorShapeRank = 4;

// input/output tensor shape rank associations
enum { kBatchRank = 0, kHeightRank, kWidthRank, kChannelRank };

TfLiteStatus L2Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLitePoolParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), kTensorShapeRank);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), kTensorShapeRank);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int batches = SizeOfDimension(input, kBatchRank);
  int height = SizeOfDimension(input, kHeightRank);
  int width = SizeOfDimension(input, kWidthRank);
  int channels_out = SizeOfDimension(input, kChannelRank);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  params->computed.padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      params->filter_height, params->filter_width, padding, &out_height,
      &out_width);

  // We currently don't have a quantized implementation of L2Pool
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);

  // We must update the output tensor dimensions.
  // The dims storage is expected to be the same area in memory
  // for both TfLiteTensor and TfLiteEvalTensor.  This is important
  // because TfLiteTensor in the MicroInterpreter is a temporary
  // allocation.  For the KernelRunner interpreter, TfLiteEvalTensor
  // is a temporary allocation.  We must therefore relocate the dims
  // from the FlatBuffer to the persistant storage arena.
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));
  output->dims->data[kBatchRank] = batches;
  output->dims->data[kHeightRank] = out_height;
  output->dims->data[kWidthRank] = out_width;
  output->dims->data[kChannelRank] = channels_out;

  return kTfLiteOk;
}

void L2EvalFloat(const TfLitePoolParams& params, const TfLiteEvalTensor& input,
                 tflite::PoolParams* op_params, TfLiteEvalTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params.activation, &activation_min, &activation_max);

  op_params->float_activation_min = activation_min;
  op_params->float_activation_max = activation_max;
  reference_ops::L2Pool(*op_params, tflite::micro::GetTensorShape(&input),
                        tflite::micro::GetTensorData<float>(&input),
                        tflite::micro::GetTensorShape(output),
                        tflite::micro::GetTensorData<float>(output));
}

TfLiteStatus L2Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<const TfLitePoolParams*>(node->builtin_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = params->computed.padding.height;
  op_params.padding_values.width = params->computed.padding.width;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      L2EvalFloat(*params, *input, &op_params, output);
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

TfLiteRegistration Register_L2_POOL_2D() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/L2Prepare,
          /*invoke=*/L2Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

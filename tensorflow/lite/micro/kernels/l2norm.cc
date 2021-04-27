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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace l2norm {

namespace {

// This file has two implementation of L2Norm.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
  L2NormalizationParams* data =
      static_cast<L2NormalizationParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);

  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                              output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    data->input_zero_point = input->params.zero_point;
  } else if (output->type == kTfLiteFloat32) {
    data->input_zero_point = 0;
  }

  // Our implementations don't currently support activations.
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(L2NormalizationParams));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const L2NormalizationParams& data =
      *(static_cast<const L2NormalizationParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  // TODO(b/143912164): instead of hardcode the epsilon here, we should read it
  // from tensorflow, i.e., adding a params.
  // We don't compute epsilon for quantized kernel:
  //
  // epsilon_float = (epsilon_quant - zp) * scale
  // so
  // espsilon_quant = epsilon_float / scale + zp
  // We know epsilon_float is just a very small number to avoid division by
  // zero error, and scale is > 1, so the integer value of epsilon for quant
  // is just dominated by the zero point.
  // Also, GetInvSqrtQuantizedMultiplierExp handles the scenario where the sum
  // of input value squared is zero case well.
  // So we don't even need to do handle the epsilon for quantized kernel case.
  const float epsilon = 1e-6f;
  if (output->type == kTfLiteFloat32) {
    reference_ops::L2Normalization(data, tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorData<float>(input),
                                   tflite::micro::GetTensorShape(output),
                                   tflite::micro::GetTensorData<float>(output),
                                   epsilon);
  } else if (output->type == kTfLiteUInt8) {
    reference_ops::L2Normalization(
        data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<uint8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<uint8_t>(output));
  } else if (output->type == kTfLiteInt8) {
    const auto input_shape = tflite::micro::GetTensorShape(input);
    const auto output_shape = tflite::micro::GetTensorShape(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    reference_integer_ops::L2Normalization(
        data.input_zero_point, outer_size, depth,
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorData<int8_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Output type is %s, requires float.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace l2norm

TfLiteRegistration Register_L2NORM_REF() {
  return {/*init=*/l2norm::Init,
          /*free=*/nullptr,
          /*prepare=*/l2norm::Prepare,
          /*invoke=*/l2norm::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_L2_NORMALIZATION() { return Register_L2NORM_REF(); }

}  // namespace micro
}  // namespace ops
}  // namespace tflite

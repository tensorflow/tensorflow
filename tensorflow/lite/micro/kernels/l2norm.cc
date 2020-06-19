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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace l2norm {

// This file has two implementation of L2Norm.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(DEBUG)
  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);

  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                              output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, (1. / 128.));
    if (output->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 128);
    }
    if (output->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }

  // TODO(ahentz): For some reason our implementations don't support
  // activations.
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
#endif

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

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
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = 0;                                            \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<float>(input), GetTensorShape(output), \
                        GetTensorData<float>(output), epsilon)

    TF_LITE_L2NORM(reference_ops);
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteUInt8) {
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = input->params.zero_point;                     \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<uint8>(input), GetTensorShape(output), \
                        GetTensorData<uint8>(output))

    TF_LITE_L2NORM(reference_ops);
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteInt8) {
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    reference_integer_ops::L2Normalization(input->params.zero_point, outer_size,
                                           depth, GetTensorData<int8>(input),
                                           GetTensorData<int8>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Output type is %s, requires float.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace l2norm

TfLiteRegistration* Register_L2NORM_REF() {
    static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/l2norm::Prepare,
                                 /*invoke=*/l2norm::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};

  return &r;
}

TfLiteRegistration* Register_L2_NORMALIZATION() {
  return Register_L2NORM_REF();
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

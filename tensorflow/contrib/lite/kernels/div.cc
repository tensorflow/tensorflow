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
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace div {

// This file has three implementation of Div.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input1), NumDimensions(input2));
  for (int i = 0; i < NumDimensions(input1); ++i) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(input1, i),
                      SizeOfDimension(input2, i));
  }

  TF_LITE_ENSURE_EQ(context, input1->type, output->type);
  TF_LITE_ENSURE_EQ(context, input2->type, output->type);

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input1->dims);
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void EvalDivFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLiteDivParams* params, TfLiteTensor* input1,
                  TfLiteTensor* input2, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);
#define TF_LITE_DIV(type)                                        \
  type::Div(GetTensorData<float>(input1), GetTensorDims(input1), \
            GetTensorData<float>(input2), GetTensorDims(input2), \
            output_activation_min, output_activation_max,        \
            GetTensorData<float>(output), GetTensorDims(output))
  if (kernel_type == kReference) {
    TF_LITE_DIV(reference_ops);
  } else {
    TF_LITE_DIV(optimized_ops);
  }
#undef TF_LITE_DIV
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteDivParams*>(node->builtin_data);

  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32) {
    EvalDivFloat<kernel_type>(context, node, params, input1, input2, output);
  } else {
    context->ReportError(context, "Inputs and outputs not all float types.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace div

TfLiteRegistration* Register_DIV_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, div::Prepare,
                                 div::Eval<div::kReference>};
  return &r;
}

TfLiteRegistration* Register_DIV_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, div::Prepare,
                                 div::Eval<div::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_DIV_NEON_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, div::Prepare,
                                 div::Eval<div::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_DIV() {
#ifdef USE_NEON
  return Register_DIV_NEON_OPT();
#else
  return Register_DIV_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

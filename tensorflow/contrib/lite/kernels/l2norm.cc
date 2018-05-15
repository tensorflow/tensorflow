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
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace l2norm {

// This file has two implementation of L2Norm.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);

  TF_LITE_ENSURE(
      context, output->type == kTfLiteFloat32 || output->type == kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, (1. / 128.));
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 128);
  }

  // TODO(ahentz): For some reason our implementations don't support
  // activations.
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32) {
#define TF_LITE_L2NORM(type)                                 \
  type::L2Normalization<FusedActivationFunctionType::kNone>( \
      GetTensorData<float>(input), GetTensorDims(input),     \
      GetTensorData<float>(output), GetTensorDims(output))

    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteUInt8) {
#define TF_LITE_L2NORM(type)                                               \
  type::L2Normalization(GetTensorData<uint8>(input), GetTensorDims(input), \
                        input->params.zero_point,                          \
                        GetTensorData<uint8>(output), GetTensorDims(output))

    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else {
    context->ReportError(context, "Inputs and outputs not all float types.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace l2norm

TfLiteRegistration* Register_L2NORM_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kReference>};
  return &r;
}

TfLiteRegistration* Register_L2NORM_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_L2_NORMALIZATION() {
  return Register_L2NORM_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

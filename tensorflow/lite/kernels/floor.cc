/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/floor.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace floor {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
  kGenericOptimized,
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32 ||
                              input->type == kTfLiteFloat16 ||
                              input->type == kTfLiteBFloat16);
  output->type = input->type;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (input->type == kTfLiteFloat32) {
    if (type == kGenericOptimized) {
      optimized_ops::Floor(GetTensorShape(input), GetTensorData<float>(input),
                           GetTensorShape(output),
                           GetTensorData<float>(output));
    } else {
      reference_ops::Floor(GetTensorShape(input), GetTensorData<float>(input),
                           GetTensorShape(output),
                           GetTensorData<float>(output));
    }
  }
  if (input->type == kTfLiteFloat16) {
    if (type == kGenericOptimized) {
      optimized_ops::Floor(
          GetTensorShape(input), GetTensorData<Eigen::half>(input),
          GetTensorShape(output), GetTensorData<Eigen::half>(output));
    } else {
      reference_ops::Floor(
          GetTensorShape(input), GetTensorData<Eigen::half>(input),
          GetTensorShape(output), GetTensorData<Eigen::half>(output));
    }
  }
  if (input->type == kTfLiteBFloat16) {
    if (type == kGenericOptimized) {
      optimized_ops::Floor(
          GetTensorShape(input), GetTensorData<Eigen::bfloat16>(input),
          GetTensorShape(output), GetTensorData<Eigen::bfloat16>(output));
    } else {
      reference_ops::Floor(
          GetTensorShape(input), GetTensorData<Eigen::bfloat16>(input),
          GetTensorShape(output), GetTensorData<Eigen::bfloat16>(output));
    }
  }

  return kTfLiteOk;
}
}  // namespace floor

TfLiteRegistration* Register_FLOOR_REF() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr, floor::Prepare,
                                 floor::Eval<floor::kReference>};
  return &r;
}

TfLiteRegistration* Register_FLOOR() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr, floor::Prepare,
                                 floor::Eval<floor::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/stablehlo_elementwise.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
TfLiteStatus ElementwisePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_tensor1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input_tensor1));
  const TfLiteTensor* input_tensor2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input_tensor2));

  // Check the two input tensors have the same type and size.
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor1->type, input_tensor2->type);
  TF_LITE_ENSURE_EQ(context, input_tensor1->dims->size,
                    input_tensor2->dims->size);
  for (int idx = 0; idx < input_tensor1->dims->size; ++idx) {
    TF_LITE_ENSURE_EQ(context, input_tensor1->dims->data[idx],
                      input_tensor2->dims->data[idx]);
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // We need the copy since ResizeTensor takes ownership of output_size.
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_tensor1->dims);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

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
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace comparisons {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus LessPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Don't support string and bool.
  TF_LITE_ENSURE(context,
                 input1->type != kTfLiteString || input1->type != kTfLiteBool);
  // Currently only support tensors have the same type.
  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);
  output->type = kTfLiteBool;

  bool requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  bool requires_broadcast = !HaveSameShapes(input1, input2);

#define TF_LITE_LESS(type, opname)                                          \
  reference_ops::opname(GetTensorData<type>(input1), GetTensorDims(input1), \
                        GetTensorData<type>(input2), GetTensorDims(input2), \
                        GetTensorData<bool>(output), GetTensorDims(output));

  // TODO(renjieliu): Support quantized data.
  if (requires_broadcast) {
    switch (input1->type) {
      case kTfLiteFloat32:
        TF_LITE_LESS(float, BroadcastLess);
        break;
      case kTfLiteInt32:
        TF_LITE_LESS(int32_t, BroadcastLess);
        break;
      case kTfLiteInt64:
        TF_LITE_LESS(int64_t, BroadcastLess);
        break;
      default:
        context->ReportError(context,
                             "Does not support type other than float|int");
        return kTfLiteError;
    }
  } else {
    switch (input1->type) {
      case kTfLiteFloat32:
        TF_LITE_LESS(float, Less);
        break;
      case kTfLiteInt32:
        TF_LITE_LESS(int32_t, Less);
        break;
      case kTfLiteInt64:
        TF_LITE_LESS(int64_t, Less);
        break;
      default:
        context->ReportError(context,
                             "Does not support type other than float|int");
        return kTfLiteError;
    }
  }
#undef TF_LITE_LESS
  return kTfLiteOk;
}

}  // namespace comparisons

TfLiteRegistration* Register_LESS() {
  static TfLiteRegistration r = {nullptr, nullptr, comparisons::LessPrepare,
                                 comparisons::LessEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

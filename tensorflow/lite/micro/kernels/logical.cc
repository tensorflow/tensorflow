/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace logical {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus LogicalImpl(TfLiteContext* context, TfLiteNode* node,
                         bool (*func)(bool, bool)) {
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (tflite::micro::HaveSameShapes(input1, input2)) {
    reference_ops::BinaryFunction<bool, bool, bool>(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<bool>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<bool>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<bool>(output), func);
  } else {
    reference_ops::BroadcastBinaryFunction4DSlow<bool, bool, bool>(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<bool>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<bool>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<bool>(output), func);
  }

  return kTfLiteOk;
}

bool LogicalOr(bool x, bool y) { return x || y; }

TfLiteStatus LogicalOrEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalOr);
}

bool LogicalAnd(bool x, bool y) { return x && y; }

TfLiteStatus LogicalAndEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalAnd);
}

}  // namespace
}  // namespace logical

TfLiteRegistration Register_LOGICAL_OR() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/logical::LogicalOrEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOGICAL_AND() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/logical::LogicalAndEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

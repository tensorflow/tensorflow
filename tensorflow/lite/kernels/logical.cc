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
#include <stddef.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace logical {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for logical op.
struct OpData {
  bool requires_broadcast;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Reinterprete the opaque data provided by user.
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteBool) {
    context->ReportError(context, "Logical ops only support bool type.");
    return kTfLiteError;
  }
  output->type = type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus LogicalImpl(TfLiteContext* context, TfLiteNode* node,
                         bool (*func)(bool, bool)) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (data->requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<bool, bool, bool>(
        GetTensorShape(input1), GetTensorData<bool>(input1),
        GetTensorShape(input2), GetTensorData<bool>(input2),
        GetTensorShape(output), GetTensorData<bool>(output), func);
  } else {
    reference_ops::BinaryFunction<bool, bool, bool>(
        GetTensorShape(input1), GetTensorData<bool>(input1),
        GetTensorShape(input2), GetTensorData<bool>(input2),
        GetTensorShape(output), GetTensorData<bool>(output), func);
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

TfLiteRegistration* Register_LOGICAL_OR() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  static TfLiteRegistration r = {logical::Init, logical::Free, logical::Prepare,
                                 logical::LogicalOrEval};
  return &r;
}

TfLiteRegistration* Register_LOGICAL_AND() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  static TfLiteRegistration r = {logical::Init, logical::Free, logical::Prepare,
                                 logical::LogicalAndEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

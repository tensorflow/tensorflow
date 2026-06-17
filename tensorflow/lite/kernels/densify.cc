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
#include "tensorflow/lite/kernels/internal/reference/densify.h"

#include <stddef.h>

#include <cstdint>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace densify {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

struct OpData {
  bool dense_weights_initialized;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->dense_weights_initialized = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);

  TF_LITE_ENSURE(context, op_context.input != nullptr);
  TF_LITE_ENSURE(context, op_context.input->type != kTfLiteString);
  TF_LITE_ENSURE(context, IsConstantTensor(op_context.input));
  TF_LITE_ENSURE(context, op_context.input->sparsity != nullptr);

  op_context.output->type = op_context.input->type;
  op_context.output->name = "Densify_output";
  op_context.output->allocation_type = kTfLiteArenaRwPersistent;

  return context->ResizeTensor(context, op_context.output,
                               TfLiteIntArrayCopy(op_context.input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpContext op_context(context, node);
  if (op_data->dense_weights_initialized) {
    return kTfLiteOk;
  }

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      reference_ops::Densify(op_context.input->sparsity,
                             GetTensorShape(op_context.input),
                             GetTensorData<float>(op_context.input),
                             GetTensorShape(op_context.output),
                             GetTensorData<float>(op_context.output), context);
      break;
    case kTfLiteFloat16:
      reference_ops::Densify(
          op_context.input->sparsity, GetTensorShape(op_context.input),
          GetTensorData<Eigen::half>(op_context.input),
          GetTensorShape(op_context.output),
          GetTensorData<Eigen::half>(op_context.output), context);
      break;
    case kTfLiteInt8:
      reference_ops::Densify(op_context.input->sparsity,
                             GetTensorShape(op_context.input),
                             GetTensorData<int8_t>(op_context.input),
                             GetTensorShape(op_context.output),
                             GetTensorData<int8_t>(op_context.output), context);
      break;

    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not supported.",
                         op_context.input->type);
      return kTfLiteError;
  }

  op_data->dense_weights_initialized = true;
  return kTfLiteOk;
}

}  // namespace densify

TfLiteRegistration* Register_DENSIFY() {
  static TfLiteRegistration r = {densify::Init, densify::Free, densify::Prepare,
                                 densify::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

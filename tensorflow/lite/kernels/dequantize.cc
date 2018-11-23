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
#include <string.h>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dequantize {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

struct OpData {
  // This boolean value is only used when the input tensor is constant.
  bool float_dequantized_weights_initialized;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->float_dequantized_weights_initialized = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);

  TF_LITE_ENSURE(context, op_context.input->type == kTfLiteUInt8);

  op_context.output->type = kTfLiteFloat32;
  // If the input tensor is constant, we can persist the dequantized value in
  // the output tensor. Otherwise we run dequantize upon each eval.
  if (IsConstantTensor(op_context.input)) {
    op_context.output->allocation_type = kTfLiteArenaRwPersistent;
  }
  return context->ResizeTensor(context, op_context.output,
                               TfLiteIntArrayCopy(op_context.input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpContext op_context(context, node);
  if (IsConstantTensor(op_context.input) &&
      op_data->float_dequantized_weights_initialized) {
    return kTfLiteOk;
  }

  tflite::DequantizationParams op_params;
  op_params.zero_point = op_context.input->params.zero_point;
  op_params.scale = op_context.input->params.scale;
  optimized_ops::Dequantize(op_params, GetTensorShape(op_context.input),
                            GetTensorData<uint8_t>(op_context.input),
                            GetTensorShape(op_context.output),
                            GetTensorData<float>(op_context.output));

  if (IsConstantTensor(op_context.input)) {
    op_data->float_dequantized_weights_initialized = true;
  }

  return kTfLiteOk;
}

}  // namespace dequantize

TfLiteRegistration* Register_DEQUANTIZE_OPT() {
  static TfLiteRegistration r = {dequantize::Init, dequantize::Free,
                                 dequantize::Prepare, dequantize::Eval};
  return &r;
}

TfLiteRegistration* Register_DEQUANTIZE() { return Register_DEQUANTIZE_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

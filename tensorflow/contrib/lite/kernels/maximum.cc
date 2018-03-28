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
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace maximum {

// This file has a reference implemenation of TFMaximum.
enum KernelType {
  kReference,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct MaximumContext {
  MaximumContext(TfLiteContext* context, TfLiteNode* node) {
    input1 = GetInput(context, node, kInputTensor1);
    input2 = GetInput(context, node, kInputTensor2);
    output = GetOutput(context, node, kOutputTensor);
  }
  TfLiteTensor* input1;
  TfLiteTensor* input2;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MaximumContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, op_context.input1->type, op_context.input2->type);
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(op_context.input2->dims);
  op_context.output->type = op_context.input2->type;
  return context->ResizeTensor(context, op_context.output, output_dims);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  MaximumContext op_context(context, node);

#define TF_LITE_MAXIMUM(kernel_type, data_type)    \
  kernel_type::TensorFlowMaximum<data_type>(       \
      GetTensorData<data_type>(op_context.input1), \
      GetTensorDims(op_context.input1),            \
      GetTensorData<data_type>(op_context.input2), \
      GetTensorDims(op_context.input2),            \
      GetTensorData<data_type>(op_context.output), \
      GetTensorDims(op_context.output))

  if (kernel_type == kReference) {
    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TF_LITE_MAXIMUM(reference_ops, float);
        break;
      default:
        context->ReportError(context,
                             "Type %d is currently not supported by Maximum.",
                             op_context.output->type);
        return kTfLiteError;
    }
  } else {
    context->ReportError(context,
                         "Type %d is currently not supported by Maximum.",
                         op_context.output->type);
    return kTfLiteError;
  }
#undef TF_LITE_MAXIMUM
  return kTfLiteOk;
}

}  // namespace maximum

TfLiteRegistration* Register_MAXIMUM_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, maximum::Prepare,
                                 maximum::Eval<maximum::kReference>};
  return &r;
}

TfLiteRegistration* Register_MAXIMUM() { return Register_MAXIMUM_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

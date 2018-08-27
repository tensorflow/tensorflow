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
namespace exp {

// This file has reference implementation of Exp.
enum KernelType {
  kReference,
};

struct ExpContext {
  ExpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  ExpContext op_context(context, node);
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(op_context.input->dims);
  op_context.output->type = op_context.input->type;
  return context->ResizeTensor(context, op_context.output, output_dims);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ExpContext op_context(context, node);

#define TF_LITE_EXP(kernel_type, data_type)                               \
  kernel_type::Exp<data_type>(GetTensorData<data_type>(op_context.input), \
                              NumElements(op_context.input),              \
                              GetTensorData<data_type>(op_context.output))

  // TODO(kanlig): supports half, bfloat16, float64, complex64, and complex128.
  if (kernel_type == kReference) {
    switch (op_context.input->type) {
      case kTfLiteFloat32:
        TF_LITE_EXP(reference_ops, float);
        break;
      default:
        context->ReportError(context,
                             "Type %d is currently not supported by Exp.",
                             op_context.input->type);
        return kTfLiteError;
    }
  }
#undef TF_LITE_EXP
  return kTfLiteOk;
}

}  // namespace exp

TfLiteRegistration* Register_EXP_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, exp::Prepare,
                                 exp::Eval<exp::kReference>};
  return &r;
}

// TODO(kanlig): add optimized implementation of Exp.
TfLiteRegistration* Register_EXP() { return Register_EXP_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

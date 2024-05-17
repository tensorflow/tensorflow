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
#include <cmath>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/lut.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

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

struct OpData {
  union {
    int8_t lut_int8[LUTSize<int8_t>()];
    int16_t lut_int16[LUTSize<int16_t>()];
  };
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  ExpContext op_context(context, node);
  const TfLiteTensor* input = op_context.input;
  TfLiteTensor* output = op_context.output;

  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
  output->type = input->type;

  if (input->type == kTfLiteInt8) {
    LUTPopulate<int8_t>(
        input->params.scale, input->params.zero_point, output->params.scale,
        output->params.zero_point, [](float value) { return std::exp(value); },
        data->lut_int8);
  } else if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    LUTPopulate<int16_t>(
        input->params.scale, input->params.zero_point, output->params.scale,
        output->params.zero_point, [](float value) { return std::exp(value); },
        data->lut_int16);
  }

  return context->ResizeTensor(context, op_context.output, output_dims);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  ExpContext op_context(context, node);

  // TODO(kanlig): supports half, bfloat16, float64, complex64, and complex128.
  if (kernel_type == kReference) {
    switch (op_context.input->type) {
      case kTfLiteFloat32:
        reference_ops::Exp(GetTensorData<float>(op_context.input),
                           NumElements(op_context.input),
                           GetTensorData<float>(op_context.output));
        break;
      case kTfLiteInt8:
        reference_integer_ops::LookupTable(
            GetTensorData<int8_t>(op_context.input),
            NumElements(op_context.input), data->lut_int8,
            GetTensorData<int8_t>(op_context.output));
        break;
      case kTfLiteInt16:
        reference_integer_ops::LookupTable(
            GetTensorData<int16_t>(op_context.input),
            NumElements(op_context.input), data->lut_int16,
            GetTensorData<int16_t>(op_context.output));
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Type %d is currently not supported by Exp.",
                           op_context.input->type);
        return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace exp

TfLiteRegistration* Register_EXP_REF() {
  static TfLiteRegistration r = {exp::Init, exp::Free, exp::Prepare,
                                 exp::Eval<exp::kReference>};
  return &r;
}

// TODO(kanlig): add optimized implementation of Exp.
TfLiteRegistration* Register_EXP() { return Register_EXP_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

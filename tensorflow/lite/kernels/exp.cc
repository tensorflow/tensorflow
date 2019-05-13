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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/exp.h"
#include <string.h>
#include <vector>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace exp {

// This file has reference implementation of Exp.
enum KernelType {
  kReference,
};

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};

struct ExpContext {
  ExpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

ExpParams GetExprParams(const OpData* data, const TfLiteTensor* input,
                        const TfLiteTensor* output) {
  ExpParams params;
  params.input_zero_point = input->params.zero_point;
  params.input_range_radius = data->input_range_radius;
  params.input_multiplier = data->input_multiplier;
  params.input_left_shift = data->input_left_shift;
  params.output_zero_point = output->params.zero_point;
  return params;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  ExpContext op_context(context, node);
  const TfLiteTensor* input = op_context.input;
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(op_context.input->dims);
  op_context.output->type = op_context.input->type;
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    OpData* data = reinterpret_cast<OpData*>(node->user_data);
    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
        input->params.scale *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    QuantizeMultiplierGreaterThanOne(input_real_multiplier,
                                     &data->input_multiplier,
                                     &data->input_left_shift);
    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift);
  }
  return context->ResizeTensor(context, op_context.output, output_dims);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ExpContext op_context(context, node);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input = op_context.input;
  TfLiteTensor* output = op_context.output;

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
      case kTfLiteInt8: {
        reference_integer_ops::QExp<int8_t>(
            GetExprParams(data, input, output), GetTensorShape(input),
            GetTensorData<int8_t>(input), GetTensorShape(output),
            GetTensorData<int8_t>(output));
      } break;
      case kTfLiteUInt8: {
        reference_integer_ops::QExp<uint8_t>(
            GetExprParams(data, input, output), GetTensorShape(input),
            GetTensorData<uint8_t>(input), GetTensorShape(output),
            GetTensorData<uint8_t>(output));
      } break;
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
  static TfLiteRegistration r = {exp::Init, exp::Free, exp::Prepare,
                                 exp::Eval<exp::kReference>};
  return &r;
}

// TODO(kanlig): add optimized implementation of Exp.
TfLiteRegistration* Register_EXP() { return Register_EXP_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

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
namespace maximum_minimum {

// This file has a reference implemenation of TFMaximum/TFMinimum.
enum KernelType {
  kReference,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
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

  OpContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, op_context.input1->type, op_context.input2->type);
  op_context.output->type = op_context.input1->type;

  bool requires_broadcast =
      !HaveSameShapes(op_context.input1, op_context.input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(
        context, CalculateShapeForBroadcast(context, op_context.input1,
                                            op_context.input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(op_context.input1->dims);
  }

  return context->ResizeTensor(context, op_context.output, output_size);
}

struct MaximumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 > el2 ? el1 : el2;
  }
};

struct MinimumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 < el2 ? el1 : el2;
  }
};

template <typename data_type, typename op_type>
void TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                      const OpContext& op_context) {
  reference_ops::TensorFlowMaximumMinimum<data_type>(
      GetTensorData<data_type>(op_context.input1),
      GetTensorDims(op_context.input1),
      GetTensorData<data_type>(op_context.input2),
      GetTensorDims(op_context.input2),
      GetTensorData<data_type>(op_context.output),
      GetTensorDims(op_context.output), op_type::template op<data_type>);
}

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  if (kernel_type == kReference) {
    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteUInt8:
        TFLiteOperation<uint8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt32:
       TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
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
  return kTfLiteOk;
}

}  // namespace maximum_minimum

TfLiteRegistration* Register_MAXIMUM_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kReference,
                            maximum_minimum::MaximumOp>};
  return &r;
}

TfLiteRegistration* Register_MINIMUM_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, maximum_minimum::Prepare,
      maximum_minimum::Eval<maximum_minimum::kReference,
                            maximum_minimum::MinimumOp>};
  return &r;
}
TfLiteRegistration* Register_MAXIMUM() { return Register_MAXIMUM_REF(); }
TfLiteRegistration* Register_MINIMUM() { return Register_MINIMUM_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

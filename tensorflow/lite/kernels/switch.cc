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
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace switch_kernel {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  // The first check the input condition.
  const TfLiteTensor* cond = GetInput(context, node, 1);
  switch (cond->type) {
    case kTfLiteBool:
    case kTfLiteFloat32:
    case kTfLiteInt32:
    case kTfLiteInt64:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
      // all is ok just fall out safely
      break;
    default:
      context->ReportError(context, "Does not support type %d", cond->type);
      return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, NumElements(cond), 1);

  TfLiteTensor* output_false = GetOutput(context, node, 0);
  TfLiteTensor* output_true = GetOutput(context, node, 1);
  const TfLiteTensor* input = GetInput(context, node, 0);

  output_false->type = input->type;
  output_true->type = input->type;
  context->ResizeTensor(context, output_false, TfLiteIntArrayCopy(input->dims));

  context->ResizeTensor(context, output_true, TfLiteIntArrayCopy(input->dims));

  return kTfLiteOk;
}

template <typename T>
void CopyOutput(const T* in_data, int num_elements, T* out_data) {
  for (int i = 0; i < num_elements; ++i) {
    out_data[i] = in_data[i];
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  int active_branch_subgraph_index = -1;
  TfLiteTensor* output_false = GetOutput(context, node, 0);
  TfLiteTensor* output_true = GetOutput(context, node, 1);
  TfLiteTensor* output;
  const int num_elements = NumElements(input);
  const TfLiteTensor* cond_tensor = GetInput(context, node, 1);

  const bool cond = cond_tensor->data.b[0];
  ;

  output = cond ? output_true : output_false;

  switch (input->type) {
    case kTfLiteInt32:
      CopyOutput(input->data.i32, num_elements, output->data.i32);
      break;
    case kTfLiteFloat32:
      CopyOutput(input->data.f, num_elements, output->data.f);
      break;
    case kTfLiteBool:
      CopyOutput(input->data.b, num_elements, output->data.b);
      break;
    case kTfLiteUInt8:
      CopyOutput(input->data.uint8, num_elements, output->data.uint8);
      break;
    case kTfLiteInt64:
      CopyOutput(input->data.i64, num_elements, output->data.i64);
      break;
    case kTfLiteInt16:
      CopyOutput(input->data.i16, num_elements, output->data.i16);
      break;
    case kTfLiteInt8:
      CopyOutput(input->data.int8, num_elements, output->data.int8);
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace switch_kernel

TfLiteRegistration* Register_SWITCH() {
  static TfLiteRegistration r = {nullptr, nullptr, switch_kernel::Prepare,
                                 switch_kernel::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite

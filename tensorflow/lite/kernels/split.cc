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
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace split {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteSplitParams*>(node->builtin_data);
    axis = GetInput(context, node, 0);
    input = GetInput(context, node, 1);
  }
  TfLiteSplitParams* params;
  const TfLiteTensor* axis;
  const TfLiteTensor* input;
};

struct OpData {
  void* all_outputs;
  TfLiteType type;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  auto* data = reinterpret_cast<OpData*>(buffer);

  switch (data->type) {
    case kTfLiteFloat32:
      delete static_cast<VectorOfTensors<float>*>(data->all_outputs);
      break;
    case kTfLiteInt32:
      delete static_cast<VectorOfTensors<int32_t>*>(data->all_outputs);
      break;
    case kTfLiteUInt8:
      delete static_cast<VectorOfTensors<uint8_t>*>(data->all_outputs);
      break;
    case kTfLiteInt8:
      delete static_cast<VectorOfTensors<int8_t>*>(data->all_outputs);
      break;
    case kTfLiteInt16:
      delete static_cast<VectorOfTensors<int16_t>*>(data->all_outputs);
      break;
    default:
      context->ReportError(context, "Unexpected data type - [%s] received.",
                           TfLiteTypeGetName(data->type));
  }

  delete data;
}

TfLiteStatus UseDynamicOutputTensors(TfLiteContext* context, TfLiteNode* node) {
  for (int i = 0; i < NumOutputs(node); ++i) {
    SetTensorToDynamic(GetOutput(context, node, i));
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputTensors(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteTensor* axis,
                                 const TfLiteTensor* input, int num_splits) {
  int axis_value = GetTensorData<int>(axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(input);
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < NumDimensions(input));

  const int input_size = SizeOfDimension(input, axis_value);
  TF_LITE_ENSURE_MSG(context, input_size % num_splits == 0,
                     "Not an even split");
  const int slice_size = input_size / num_splits;

  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
    output_dims->data[axis_value] = slice_size;
    TfLiteTensor* output = GetOutput(context, node, i);
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_dims));
  }

  OpData* user_data = reinterpret_cast<OpData*>(node->user_data);
  user_data->type = input->type;

  switch (user_data->type) {
    case kTfLiteFloat32:
      user_data->all_outputs = reinterpret_cast<void*>(
          new VectorOfTensors<float>(*context, *node->outputs));
      break;
    case kTfLiteInt32:
      user_data->all_outputs = reinterpret_cast<void*>(
          new VectorOfTensors<int32_t>(*context, *node->outputs));
      break;
    case kTfLiteUInt8:
      user_data->all_outputs = reinterpret_cast<void*>(
          new VectorOfTensors<uint8_t>(*context, *node->outputs));
      break;
    case kTfLiteInt8:
      user_data->all_outputs = reinterpret_cast<void*>(
          new VectorOfTensors<int8_t>(*context, *node->outputs));
      break;
    case kTfLiteInt16:
      user_data->all_outputs = reinterpret_cast<void*>(
          new VectorOfTensors<int16_t>(*context, *node->outputs));
      break;

    default:
      context->ReportError(context, "Unexpected data type - [%s] received.",
                           TfLiteTypeGetName(user_data->type));
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  OpContext op_context(context, node);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), op_context.params->num_splits);

  auto input_type = op_context.input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16 ||
                     input_type == kTfLiteInt32);
  for (int i = 0; i < NumOutputs(node); ++i) {
    GetOutput(context, node, i)->type = input_type;
  }

  // If we know the contents of the 'axis' tensor, resize all outputs.
  // Otherwise, wait until Eval().
  if (IsConstantTensor(op_context.axis)) {
    return ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                               op_context.params->num_splits);
  } else {
    return UseDynamicOutputTensors(context, node);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  // When the 'axis' tensor is non-const we can't resize output tensors in
  // Prepare(), and we have to do it now.
  if (!IsConstantTensor(op_context.axis)) {
    TF_LITE_ENSURE_OK(
        context,
        ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                            op_context.params->num_splits));
  }

  int axis_value = GetTensorData<int>(op_context.axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(op_context.input);
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < NumDimensions(op_context.input));

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // TODO(ahentz): We can improve the optimized_ops version to handle other
  // cases too.
#define TF_LITE_SPLIT(scalar)                                       \
  VectorOfTensors<scalar>* all_outputs =                            \
      static_cast<VectorOfTensors<scalar>*>(data->all_outputs);     \
  all_outputs->update(*context, *node->outputs);                    \
  tflite::SplitParams op_params;                                    \
  op_params.num_split = NumOutputs(node);                           \
  op_params.axis = axis_value;                                      \
  reference_ops::Split(op_params, GetTensorShape(op_context.input), \
                       GetTensorData<scalar>(op_context.input),     \
                       all_outputs->shapes(), all_outputs->data());

  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      TF_LITE_SPLIT(float);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_SPLIT(uint8_t);
      break;
    }
    case kTfLiteInt8: {
      TF_LITE_SPLIT(int8_t);
      break;
    }
    case kTfLiteInt16: {
      TF_LITE_SPLIT(int16_t);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_SPLIT(int32_t);
      break;
    }
    default:
      context->ReportError(context, "Type %s currently not supported.",
                           TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_SPLIT

  return kTfLiteOk;
}

}  // namespace split

TfLiteRegistration* Register_SPLIT() {
  static TfLiteRegistration r = {split::Init, split::Free, split::Prepare,
                                 split::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

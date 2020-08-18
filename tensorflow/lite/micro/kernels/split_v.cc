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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
//#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace split_v {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteSplitVParams*>(node->builtin_data);
    input = tflite::micro::GetEvalInput(context, node, 0);
    size_splits = tflite::micro::GetEvalInput(context, node, 1);
    axis = tflite::micro::GetEvalInput(context, node, 2);
  }
  TfLiteSplitVParams* params;
  const TfLiteEvalTensor* input;
  const TfLiteEvalTensor* size_splits;
  const TfLiteEvalTensor* axis;
};

template <typename T>
TfLiteStatus SplitImpl(TfLiteContext* context, TfLiteNode* node,
                       OpContext* op_context) {
  const int outputs_count = NumOutputs(node);
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  
  const TfLiteIntArray* input_dims = input->dims;
  const TfLiteEvalTensor* output0 =     tflite::micro::GetEvalOutput(context, node, 0);
  const TfLiteIntArray* output_dims = output0->dims;
  int axis_value = op_context->axis->data.i32[0];
  
  const int split_dimensions = input_dims->size;
  int axis = axis_value < 0 ? axis_value + split_dimensions : axis_value;

  TFLITE_DCHECK_LT(axis, split_dimensions);
  TFLITE_DCHECK_EQ(output_dims->size, split_dimensions);

  int64_t split_size = 0;
  for (int i = 0; i < outputs_count; i++) {
      split_size += tflite::micro::GetEvalOutput(context, node, i)->dims->data[axis];
  }

  TFLITE_DCHECK_EQ(split_size, input_dims->data[axis]);
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_dims->data[i];
  }

  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_dims->data[i];
  }
  
  const T* input_ptr = tflite::micro::GetTensorData<T>(input);
  for (int k = 0; k < outer_size; ++k) {
    for (int i = 0; i < outputs_count; ++i) {
      TfLiteEvalTensor* t = tflite::micro::GetEvalOutput(context, node, i);
      T* output_data = tflite::micro::GetTensorData<T>(t);
      output_dims = t->dims;
      const int copy_size = output_dims->data[axis] * base_inner_size;
      T* output_ptr = output_data + k * copy_size;
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
      input_ptr += copy_size;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);

  OpContext op_context(context, node);

  //TF_LITE_ENSURE_EQ(context, NumOutputs(node), op_context.params->num_splits);

  auto input_type = op_context.input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteInt8);
  for (int i = 0; i < NumOutputs(node); ++i) {
    GetOutput(context, node, i)->type = input_type;
  }

   return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
 
  const TfLiteTensor* axis_tensor = GetInput(context, node, 2);
  const TfLiteTensor* size_splits_tensor = GetInput(context, node, 1);
  // if not const return error - unsupported
  if (!IsConstantTensor(axis_tensor) || !IsConstantTensor(size_splits_tensor)) {
    TF_LITE_KERNEL_LOG(
        context, "Only const tensors are supported for size_splits and axis");
    return kTfLiteError;
  }
 

  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      SplitImpl<float>(context, node, &op_context);
      break;
    }
    case kTfLiteUInt8: {
      SplitImpl<uint8_t>(context, node, &op_context);
      break;
    }
    case kTfLiteInt16: {
      SplitImpl<int16_t>(context, node, &op_context);
      break;
    }
    case kTfLiteInt32: {
      SplitImpl<int32_t>(context, node, &op_context);
      break;
    }
    case kTfLiteInt64: {
      SplitImpl<int64_t>(context, node, &op_context);
      break;
    }
    case kTfLiteInt8: {
      SplitImpl<int8_t>(context, node, &op_context);

      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace split_v

TfLiteRegistration Register_SPLIT_V() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/split_v::Prepare,
          /*invoke=*/split_v::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

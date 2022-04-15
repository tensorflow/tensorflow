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
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace split_v {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteSplitVParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    size_splits = GetInput(context, node, 1);
    axis = GetInput(context, node, 2);
  }
  TfLiteSplitVParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* size_splits;
  const TfLiteTensor* axis;
};

TfLiteStatus UseDynamicOutputTensors(TfLiteContext* context, TfLiteNode* node) {
  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));
    SetTensorToDynamic(tensor);
  }
  return kTfLiteOk;
}

template <typename T>
void GetSizeSplitsVector(const TfLiteTensor* size_splits,
                         std::vector<int64_t>* size_splits_vector) {
  const auto num_elements = NumElements(size_splits);
  for (int i = 0; i < num_elements; ++i) {
    size_splits_vector->push_back(GetTensorData<T>(size_splits)[i]);
  }
}

TfLiteStatus ResizeOutputTensors(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteTensor* input,
                                 const TfLiteTensor* size_splits,
                                 const TfLiteTensor* axis) {
  int axis_value = GetTensorData<int>(axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(input);
  }

  std::vector<int64_t> size_splits_vector;
  if (size_splits->type == kTfLiteInt32) {
    GetSizeSplitsVector<int32_t>(size_splits, &size_splits_vector);
  } else if (size_splits->type == kTfLiteInt64) {
    GetSizeSplitsVector<int64_t>(size_splits, &size_splits_vector);
  } else {
    TF_LITE_KERNEL_LOG(context, "size_splits only support type int32|int64.");
    return kTfLiteError;
  }

  int minus_one_index = -1;
  int64_t size_splits_sum = 0;

  for (int i = 0; i < size_splits_vector.size(); ++i) {
    if (size_splits_vector.at(i) == -1) {
      if (minus_one_index == -1) {
        minus_one_index = i;
      } else {
        TF_LITE_KERNEL_LOG(context,
                           "The size_splits contains more than one -1.");
        return kTfLiteError;
      }
    } else {
      size_splits_sum += size_splits_vector.at(i);
    }
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < NumDimensions(input));
  const int input_size = SizeOfDimension(input, axis_value);

  if (minus_one_index != -1) {
    if (size_splits_sum > input_size) {
      TF_LITE_KERNEL_LOG(
          context,
          "The sum of size_splits must be less than the dimension of value.");
    } else {
      size_splits_vector[minus_one_index] = input_size - size_splits_sum;
    }
  } else if (size_splits_sum != input_size) {
    TF_LITE_KERNEL_LOG(
        context,
        "The size_splits must sum to the dimension of value along axis.");
  }

  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
    output_dims->data[axis_value] = size_splits_vector.at(i);
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_dims));
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);

  OpContext op_context(context, node);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), op_context.params->num_splits);

  auto input_type = op_context.input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteInt8);
  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));
    tensor->type = input_type;
  }

  auto size_splits = op_context.size_splits;
  TF_LITE_ENSURE_EQ(context, NumDimensions(size_splits), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), NumElements(size_splits));

  // If we know the contents of the 'size_splits' tensor and the 'axis' tensor,
  // resize all outputs. Otherwise, wait until Eval().
  if (IsConstantTensor(op_context.size_splits) &&
      IsConstantTensor(op_context.axis)) {
    return ResizeOutputTensors(context, node, op_context.input,
                               op_context.size_splits, op_context.axis);
  } else {
    return UseDynamicOutputTensors(context, node);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  // When the 'size_splits' and the 'axis' tensor is non-const we can't resize
  // output tensors in Prepare(), and we have to do it now.
  if (!IsConstantTensor(op_context.axis) ||
      !IsConstantTensor(op_context.size_splits)) {
    TF_LITE_ENSURE_OK(
        context, ResizeOutputTensors(context, node, op_context.input,
                                     op_context.size_splits, op_context.axis));
  }

  int axis_value = GetTensorData<int>(op_context.axis)[0];

  // Use split function to build the outputs since they share the same logic.
#define TF_LITE_SPLIT_V(scalar)                                     \
  VectorOfTensors<scalar> all_outputs(*context, *node->outputs);    \
  tflite::SplitParams op_params;                                    \
  op_params.num_split = NumOutputs(node);                           \
  op_params.axis = axis_value;                                      \
  reference_ops::Split(op_params, GetTensorShape(op_context.input), \
                       GetTensorData<scalar>(op_context.input),     \
                       all_outputs.shapes(), all_outputs.data());
  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      TF_LITE_SPLIT_V(float);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_SPLIT_V(uint8_t);
      break;
    }
    case kTfLiteInt16: {
      TF_LITE_SPLIT_V(int16_t);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_SPLIT_V(int32_t);
      break;
    }
    case kTfLiteInt64: {
      TF_LITE_SPLIT_V(int64_t);
      break;
    }
    case kTfLiteInt8: {
      TF_LITE_SPLIT_V(int8_t);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_SPLIT_V

  return kTfLiteOk;
}

}  // namespace split_v

TfLiteRegistration* Register_SPLIT_V() {
  static TfLiteRegistration r = {nullptr, nullptr, split_v::Prepare,
                                 split_v::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

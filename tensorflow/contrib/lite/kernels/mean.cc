/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mean {

// This file has reference implementation of Mean.
enum KernelType {
  kReference,
};

struct MeanContext {
  MeanContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteMeanParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    axis = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  TfLiteMeanParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* axis;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Creates two temp tensors to store index and axis for internal
  // implementation only.
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, 3, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext* context, MeanContext* op_context,
                            TfLiteTensor* resolved_axis) {
  TfLiteIntArray* axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

// Resizes the temp tensor that stores temp sum of reduced elements.
TfLiteStatus ResizeTempSum(TfLiteContext* context, MeanContext* op_context,
                           TfLiteTensor* temp_sum) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(1);
  size->data[0] = static_cast<int>(NumElements(op_context->output));
  return context->ResizeTensor(context, temp_sum, size);
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                MeanContext* op_context) {
  size_t num_axis = NumElements(op_context->axis);
  const TfLiteIntArray* input_dims = op_context->input->dims;
  int input_num_dims = NumDimensions(op_context->input);
  const int* axis = GetTensorData<int>(op_context->axis);
  if (op_context->params->keep_dims) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          is_axis = true;
          break;
        }
      }
      if (is_axis) {
        output_dims->data[idx] = 1;
      } else {
        output_dims->data[idx] = input_dims->data[idx];
      }
    }
    return context->ResizeTensor(context, op_context->output, output_dims);
  } else {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i) {
      int current = axis[i];
      if (current < 0) {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j) {
        int previous = axis[j];
        if (previous < 0) {
          previous += input_num_dims;
        }
        if (current == previous) {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    return context->ResizeTensor(context, op_context->output, output_dims);
  }
}

// Initializes temp tensors to store index and resolved axis.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   MeanContext* op_context) {
  // Creates a temp index to iterate through input data.
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(3);
  node->temporaries->data[0] = *scratch_tensor_index;
  TfLiteTensor* scratch_tensor = GetTemporary(context, node, /*index=*/0);
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = NumDimensions(op_context->input);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, scratch_tensor, index_size));

  // Creates a temp tensor to store resolved axis given input data.
  node->temporaries->data[1] = *scratch_tensor_index + 1;
  TfLiteTensor* resolved_axis = GetTemporary(context, node, /*index=*/1);
  resolved_axis->type = kTfLiteInt32;
  // Creates a temp tensor to store temp sums when calculating mean.
  node->temporaries->data[2] = *scratch_tensor_index + 2;
  TfLiteTensor* temp_sum = GetTemporary(context, node, /*index=*/2);
  switch (op_context->input->type) {
    case kTfLiteFloat32:
      temp_sum->type = kTfLiteFloat32;
      break;
    case kTfLiteInt32:
      temp_sum->type = kTfLiteInt64;
      break;
    case kTfLiteInt64:
      temp_sum->type = kTfLiteInt64;
      break;
    case kTfLiteUInt8:
      temp_sum->type = kTfLiteInt32;
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MeanContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));

  TfLiteTensor* resolved_axis = GetTemporary(context, node, /*index=*/1);
  TfLiteTensor* temp_sum = GetTemporary(context, node, /*index=*/2);
  // Leaves work to Eval if axis is not constant; else resizes output.
  if (!IsConstantTensor(op_context.axis)) {
    SetTensorToDynamic(op_context.output);
    SetTensorToDynamic(resolved_axis);
    SetTensorToDynamic(temp_sum);
    return kTfLiteOk;
  }
  resolved_axis->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    ResizeTempAxis(context, &op_context, resolved_axis));
  TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  temp_sum->allocation_type = kTfLiteArenaRw;
  return ResizeTempSum(context, &op_context, temp_sum);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  MeanContext op_context(context, node);
  int num_axis = static_cast<int>(NumElements(op_context.axis));
  TfLiteTensor* temp_index = GetTemporary(context, node, /*index=*/0);
  TfLiteTensor* resolved_axis = GetTemporary(context, node, /*index=*/1);
  TfLiteTensor* temp_sum = GetTemporary(context, node, /*index=*/2);
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, &op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
    TF_LITE_ENSURE_OK(context, ResizeTempSum(context, &op_context, temp_sum));
  }

#define TF_LITE_MEAN(kernel_type, data_type, temp_data_type)        \
  kernel_type::Mean<>(                                              \
      GetTensorData<data_type>(op_context.input),                   \
      op_context.input->dims->data, op_context.input->dims->size,   \
      GetTensorData<data_type>(op_context.output),                  \
      op_context.output->dims->data, op_context.output->dims->size, \
      GetTensorData<int>(op_context.axis), num_axis,                \
      op_context.params->keep_dims, GetTensorData<int>(temp_index), \
      GetTensorData<int>(resolved_axis),                            \
      GetTensorData<temp_data_type>(temp_sum))

  if (kernel_type == kReference) {
    switch (op_context.input->type) {
      case kTfLiteFloat32:
        TF_LITE_ENSURE(context, TF_LITE_MEAN(reference_ops, float, float));
        break;
      case kTfLiteInt32:
        TF_LITE_ENSURE(context, TF_LITE_MEAN(reference_ops, int, int64_t));
        break;
      case kTfLiteInt64:
        TF_LITE_ENSURE(context, TF_LITE_MEAN(reference_ops, int64_t, int64_t));
        break;
      case kTfLiteUInt8:
        TF_LITE_ENSURE_EQ(context, op_context.input->params.scale,
                          op_context.output->params.scale);
        TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point,
                          op_context.output->params.zero_point);
        TF_LITE_ENSURE(context, TF_LITE_MEAN(reference_ops, uint8_t, int));
        break;
      default:
        return kTfLiteError;
    }
  }
#undef TF_LITE_MEAN
  return kTfLiteOk;
}
}  // namespace mean

TfLiteRegistration* Register_MEAN_REF() {
  static TfLiteRegistration r = {mean::Init, mean::Free, mean::Prepare,
                                 mean::Eval<mean::kReference>};
  return &r;
}

// TODO(kanlig): add optimized implementation of Mean.
TfLiteRegistration* Register_MEAN() { return Register_MEAN_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

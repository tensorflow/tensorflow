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
    output = GetOutput(context, node, 0);
  }
  TfLiteMeanParams* params;
  TfLiteTensor* input;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Creates two temp tensors to store index and axis for internal
  // implementation only.
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, 2, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MeanContext op_context(context, node);
  int input_num_dims = NumDimensions(op_context.input);
  int axis_num_dims = op_context.params->num_axis_dimensions;

  // Creates a temp index to iterate through input data.
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = *scratch_tensor_index;
  TfLiteTensor* scratch_tensor = &context->tensors[node->temporaries->data[0]];
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = input_num_dims;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, scratch_tensor, index_size));

  // Creates a temp tensor to store resolved axis given input data.
  node->temporaries->data[1] = *scratch_tensor_index + 1;
  TfLiteTensor* axis_tensor = &context->tensors[node->temporaries->data[1]];
  axis_tensor->type = kTfLiteInt32;
  axis_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = op_context.params->num_axis_dimensions;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, axis_tensor, axis_size));

  // Determines size of output tensor.
  const TfLiteIntArray* input_dims = op_context.input->dims;
  const int* axis = op_context.params->axis;
  if (op_context.params->keep_dims) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < axis_num_dims; ++axis_idx) {
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
    return context->ResizeTensor(context, op_context.output, output_dims);
  } else {
    // Calculates size of reducing axis.
    int num_reduce_axis = axis_num_dims;
    for (int i = 0; i < axis_num_dims; ++i) {
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
      for (int axis_idx = 0; axis_idx < axis_num_dims; ++axis_idx) {
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
    return context->ResizeTensor(context, op_context.output, output_dims);
  }
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  MeanContext op_context(context, node);
  TfLiteTensor* temp_index = &context->tensors[node->temporaries->data[0]];
  TfLiteTensor* resolved_axis = &context->tensors[node->temporaries->data[1]];

#define TF_LITE_MEAN(kernel_type, data_type)                           \
  kernel_type::Mean<>(                                                 \
      GetTensorData<data_type>(op_context.input),                      \
      op_context.input->dims->data, op_context.input->dims->size,      \
      GetTensorData<data_type>(op_context.output),                     \
      op_context.output->dims->data, op_context.output->dims->size,    \
      op_context.params->axis, op_context.params->num_axis_dimensions, \
      op_context.params->keep_dims, GetTensorData<int>(temp_index),    \
      GetTensorData<int>(resolved_axis))

  if (kernel_type == kReference) {
    switch (op_context.input->type) {
      case kTfLiteFloat32:
        TF_LITE_MEAN(reference_ops, float);
        break;
      case kTfLiteInt32:
        TF_LITE_MEAN(reference_ops, int);
        break;
      case kTfLiteUInt8:
        TF_LITE_MEAN(reference_ops, uint8_t);
        break;
      case kTfLiteInt64:
        TF_LITE_MEAN(reference_ops, int64_t);
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

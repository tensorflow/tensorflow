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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace one_hot {

constexpr int kIndicesTensor = 0;
constexpr int kDepthTensor = 1;
constexpr int kOnValueTensor = 2;
constexpr int kOffValueTensor = 3;
constexpr int kOutputTensor = 0;

// Convenience utility for destructuring a node into the appropriate tensors and
// data for the op. Note that this destructuring is quite cheap, so we can avoid
// allocating op-specific, persistent data on the heap.
struct OneHotContext {
  OneHotContext(TfLiteContext* context, TfLiteNode* node) {
    indices = GetInput(context, node, kIndicesTensor);
    depth = GetInput(context, node, kDepthTensor);
    on_value = GetInput(context, node, kOnValueTensor);
    off_value = GetInput(context, node, kOffValueTensor);
    output = GetOutput(context, node, kOutputTensor);

    const auto* params =
        reinterpret_cast<TfLiteOneHotParams*>(node->builtin_data);
    const int indices_dims = indices->dims->size;
    axis = (params->axis == -1) ? indices_dims : params->axis;
    output_dims = indices_dims + 1;
    dtype = on_value->type;
  }

  const TfLiteTensor* indices;
  const TfLiteTensor* depth;
  const TfLiteTensor* on_value;
  const TfLiteTensor* off_value;
  TfLiteTensor* output;
  int axis;
  int output_dims;
  TfLiteType dtype;
};

template <typename T, typename TI>
void OneHotComputeImpl(const OneHotContext& op_context) {
  // prefix_dim_size == # of elements before the axis
  // depth == # of elements per axis
  // suffix_dim_size == # of elements after the axis
  int prefix_dim_size = 1;
  for (int i = 0; i < op_context.axis; ++i) {
    prefix_dim_size *= op_context.indices->dims->data[i];
  }
  if (prefix_dim_size == 0) {
    // If indices tensor is degenerate, return a degenerate tensor, just like
    // TensorFlow does.
    return;
  }
  const int suffix_dim_size = NumElements(op_context.indices) / prefix_dim_size;
  const int depth = *op_context.depth->data.i32;

  const T on_value = *GetTensorData<T>(op_context.on_value);
  const T off_value = *GetTensorData<T>(op_context.off_value);

  // View the indices as a matrix of size:
  //     prefix_dim_size x suffix_dim_size
  // View the output as a matrix of size:
  //     prefix_dim_size x depth x suffix_dim_size
  // Then the output is:
  //     output(i, j, k) == (indices(i, k) == j) ? on : off
  T* output = GetTensorData<T>(op_context.output);
  const TI* indices = GetTensorData<TI>(op_context.indices);
  for (int i = 0; i < prefix_dim_size; ++i) {
    for (int j = 0; j < depth; ++j) {
      for (int k = 0; k < suffix_dim_size; ++k, ++output) {
        *output = static_cast<int>(indices[i * suffix_dim_size + k]) == j
                      ? on_value
                      : off_value;
      }
    }
  }
}

template <typename T>
void OneHotCompute(const OneHotContext& op_context) {
  if (op_context.indices->type == kTfLiteInt64) {
    OneHotComputeImpl<T, int64_t>(op_context);
  } else {
    OneHotComputeImpl<T, int>(op_context);
  }
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const OneHotContext& op_context) {
  TF_LITE_ENSURE(context, *op_context.depth->data.i32 >= 0);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(op_context.output_dims);
  for (int i = 0; i < op_context.output_dims; ++i) {
    if (i < op_context.axis) {
      output_size->data[i] = op_context.indices->dims->data[i];
    } else if (i == op_context.axis) {
      output_size->data[i] = *op_context.depth->data.i32;
    } else {
      output_size->data[i] = op_context.indices->dims->data[i - 1];
    }
  }
  return context->ResizeTensor(context, op_context.output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OneHotContext op_context{context, node};
  switch (op_context.dtype) {
    // TODO(b/111744875): Support uint8 and quantization.
    case kTfLiteFloat32:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteInt64:
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteBool:
      op_context.output->type = op_context.dtype;
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unknown output data type: %s",
                         TfLiteTypeGetName(op_context.dtype));
      return kTfLiteError;
  }

  TF_LITE_ENSURE(context, op_context.indices->type == kTfLiteInt32 ||
                              op_context.indices->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, op_context.axis >= 0 &&
                              op_context.axis < op_context.output_dims);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.depth), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.on_value), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.off_value), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.on_value->type, op_context.dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.off_value->type,
                          op_context.dtype);

  if (!IsConstantTensor(op_context.depth)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }

  return ResizeOutputTensor(context, op_context);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OneHotContext op_context{context, node};

  if (IsDynamicTensor(op_context.output)) {
    ResizeOutputTensor(context, op_context);
  }

  switch (op_context.output->type) {
    case kTfLiteFloat32:
      OneHotCompute<float>(op_context);
      break;
    case kTfLiteInt32:
      OneHotCompute<int>(op_context);
      break;
    case kTfLiteInt64:
      OneHotCompute<int64_t>(op_context);
      break;
    case kTfLiteInt8:
      OneHotCompute<int8_t>(op_context);
      break;
    case kTfLiteUInt8:
      OneHotCompute<uint8_t>(op_context);
      break;
    case kTfLiteBool:
      OneHotCompute<bool>(op_context);
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace one_hot

TfLiteRegistration* Register_ONE_HOT() {
  static TfLiteRegistration r = {
      nullptr,
      nullptr,
      one_hot::Prepare,
      one_hot::Eval,
  };
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

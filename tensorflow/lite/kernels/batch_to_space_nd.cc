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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace batch_to_space_nd {

// This file has two implementations of BatchToSpaceND.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct BatchToSpaceNDContext {
  BatchToSpaceNDContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    block_shape = GetInput(context, node, 1);
    crops = GetInput(context, node, 2);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* block_shape;
  const TfLiteTensor* crops;
  TfLiteTensor* output;
};

// Currently, only 3D NHC or 4D NHWC input/output op_context are supported.
// In case of 3D input,it will be converted to 4D by adding W=1 to be NH1C.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(ycling): Support arbitrary dimension in BatchToSpaceND.
const int kInputMinDimensionNum = 3;
const int kInputMaxDimensionNum = 4;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                BatchToSpaceNDContext* op_context) {
  TfLiteIntArray* input_size = op_context->input->dims;
  const int* block_shape = GetTensorData<int32>(op_context->block_shape);
  const int* crops = GetTensorData<int32>(op_context->crops);

  int spatial_dims_num = input_size->size - 2;
  // Block_shape should be a 1D tensor with dimension [spatial_dims_num].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->block_shape), 1);
  TF_LITE_ENSURE_EQ(context, op_context->block_shape->dims->data[0],
                    spatial_dims_num);
  // Crops should be a 2D tensor with dimension [spatial_dims_num, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->crops), 2);
  TF_LITE_ENSURE_EQ(context, op_context->crops->dims->data[0],
                    spatial_dims_num);
  TF_LITE_ENSURE_EQ(context, op_context->crops->dims->data[1], 2);

  for (int i = 0; i < spatial_dims_num * 2; ++i) {
    TF_LITE_ENSURE(context, crops[i] >= 0);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
  int output_batch_size = input_size->data[0];
  for (int dim = 0; dim < spatial_dims_num; ++dim) {
    // Number of batch must be multiple of (block_shape[dim]).
    TF_LITE_ENSURE(context, block_shape[dim] != 0);
    TF_LITE_ENSURE_EQ(context, output_batch_size % block_shape[dim], 0);
    output_batch_size = output_batch_size / block_shape[dim];
    output_size->data[dim + 1] = input_size->data[dim + 1] * block_shape[dim] -
                                 crops[dim * 2] - crops[dim * 2 + 1];
  }

  output_size->data[0] = output_batch_size;
  output_size->data[input_size->size - 1] =
      input_size->data[input_size->size - 1];

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  BatchToSpaceNDContext op_context(context, node);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) >= kInputMinDimensionNum);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) <= kInputMaxDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  if (!IsConstantTensor(op_context.block_shape) ||
      !IsConstantTensor(op_context.crops)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  BatchToSpaceNDContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

#define TF_LITE_BATCH_TO_SPACE_ND(type, scalar)                        \
  type::BatchToSpaceND(GetTensorShape(op_context.input),               \
                       GetTensorData<scalar>(op_context.input),        \
                       GetTensorShape(op_context.block_shape),         \
                       GetTensorData<int32_t>(op_context.block_shape), \
                       GetTensorShape(op_context.crops),               \
                       GetTensorData<int32_t>(op_context.crops),       \
                       GetTensorShape(op_context.output),              \
                       GetTensorData<scalar>(op_context.output))
  switch (op_context.input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, float);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, uint8_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int8_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int32_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int64_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int64_t);
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %d is currently not supported by BatchToSpace.",
                         op_context.input->type);
      return kTfLiteError;
  }
#undef TF_LITE_BATCH_TO_SPACE_ND
  return kTfLiteOk;
}

}  // namespace batch_to_space_nd

TfLiteRegistration* Register_BATCH_TO_SPACE_ND_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, batch_to_space_nd::Prepare,
      batch_to_space_nd::Eval<batch_to_space_nd::kReference>};
  return &r;
}

TfLiteRegistration* Register_BATCH_TO_SPACE_ND_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, batch_to_space_nd::Prepare,
      batch_to_space_nd::Eval<batch_to_space_nd::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_BATCH_TO_SPACE_ND() {
  return Register_BATCH_TO_SPACE_ND_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

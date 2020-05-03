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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace space_to_batch_nd {

// This file has two implementations of SpaceToBatchND.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct SpaceToBatchNDContext {
  SpaceToBatchNDContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    block_shape = GetInput(context, node, 1);
    paddings = GetInput(context, node, 2);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* block_shape;
  const TfLiteTensor* paddings;
  TfLiteTensor* output;
};

// Currently, only 3D NHC and 4D NHWC input/output op_context are supported.
// In case of 3D input, it will be extended to 3D NHWC by adding W=1.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(b/149952582): Support arbitrary dimension in SpaceToBatchND.
const int kInputMinDimensionNum = 3;
const int kInputMaxDimensionNum = 4;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                SpaceToBatchNDContext* op_context) {
  TfLiteIntArray* input_size = op_context->input->dims;
  const int32* block_shape = GetTensorData<int32>(op_context->block_shape);
  const int32* paddings_data = GetTensorData<int32>(op_context->paddings);

  int spatial_dims_num = input_size->size - 2;
  // Block_shape should be a 1D tensor with dimension [spatial_dims_num].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->block_shape), 1);
  TF_LITE_ENSURE_EQ(context, op_context->block_shape->dims->data[0],
                    spatial_dims_num);
  // Paddings should be a 2D tensor with dimension [spatial_dims_num, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->paddings), 2);
  TF_LITE_ENSURE_EQ(context, op_context->paddings->dims->data[0],
                    spatial_dims_num);
  TF_LITE_ENSURE_EQ(context, op_context->paddings->dims->data[1], 2);

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);

  // Ensures the input height and width (with padding) is a multiple of block
  // shape height and width.
  int output_batch_size = input_size->data[0];
  for (int dim = 0; dim < spatial_dims_num; ++dim) {
    int final_dim_size = (input_size->data[dim + 1] + paddings_data[dim * 2] +
                          paddings_data[dim * 2 + 1]);
    TF_LITE_ENSURE_EQ(context, final_dim_size % block_shape[dim], 0);
    output_size->data[dim + 1] = final_dim_size / block_shape[dim];
    output_batch_size *= block_shape[dim];
  }

  output_size->data[0] = output_batch_size;
  output_size->data[input_size->size - 1] =
      input_size->data[input_size->size - 1];

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  SpaceToBatchNDContext op_context(context, node);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) >= kInputMinDimensionNum);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) <= kInputMaxDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  if (!IsConstantTensor(op_context.block_shape) ||
      !IsConstantTensor(op_context.paddings)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  SpaceToBatchNDContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

#define TF_LITE_SPACE_TO_BATCH_ND(type, scalar, pad_value)             \
  tflite::SpaceToBatchParams op_params;                                \
  op_params.output_offset = pad_value;                                 \
  type::SpaceToBatchND(op_params, GetTensorShape(op_context.input),    \
                       GetTensorData<scalar>(op_context.input),        \
                       GetTensorShape(op_context.block_shape),         \
                       GetTensorData<int32_t>(op_context.block_shape), \
                       GetTensorShape(op_context.paddings),            \
                       GetTensorData<int32_t>(op_context.paddings),    \
                       GetTensorShape(op_context.output),              \
                       GetTensorData<scalar>(op_context.output))
  switch (op_context.input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, float, 0);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, float, 0);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, uint8_t,
                                  op_context.output->params.zero_point);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, uint8_t,
                                  op_context.output->params.zero_point);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, int8_t,
                                  op_context.output->params.zero_point);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, int8_t,
                                  op_context.output->params.zero_point);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, int32_t, 0);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, int32_t, 0);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, int64_t, 0);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, int64_t, 0);
      }
      break;
    default:
      context->ReportError(
          context, "Type %d is currently not supported by SpaceToBatch.",
          op_context.input->type);
      return kTfLiteError;
  }
#undef TF_LITE_SPACE_TO_BATCH_ND
  return kTfLiteOk;
}

}  // namespace space_to_batch_nd

TfLiteRegistration* Register_SPACE_TO_BATCH_ND_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, space_to_batch_nd::Prepare,
      space_to_batch_nd::Eval<space_to_batch_nd::kReference>};
  return &r;
}

TfLiteRegistration* Register_SPACE_TO_BATCH_ND_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, space_to_batch_nd::Prepare,
      space_to_batch_nd::Eval<space_to_batch_nd::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_SPACE_TO_BATCH_ND() {
  // return Register_SPACE_TO_BATCH_ND_REF();
  return Register_SPACE_TO_BATCH_ND_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

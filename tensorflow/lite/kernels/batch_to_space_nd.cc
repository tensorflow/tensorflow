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

const int kBlockSizeDimensionNum = 1;
const int kSpatialDimensionNum = 2;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                BatchToSpaceNDContext* op_context) {
  TfLiteIntArray* input_size = op_context->input->dims;
  const int* block_shape = GetTensorData<int32>(op_context->block_shape);
  const int* crops = GetTensorData<int32>(op_context->crops);

  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->block_shape),
                    kBlockSizeDimensionNum);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->crops),
                    kSpatialDimensionNum);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context->block_shape, 0),
                    SizeOfDimension(op_context->crops, 0));

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);

  int prod_block_shape = 1;
  for (int dim = 0; dim < SizeOfDimension(op_context->block_shape, 0); ++dim) {
    TF_LITE_ENSURE(context, crops[dim * 2] >= 0);
    TF_LITE_ENSURE(context, crops[dim * 2 + 1] >= 0);
    TF_LITE_ENSURE(context, block_shape[dim] >= 1);
    int final_dim_size = (input_size->data[dim + 1] * block_shape[dim] -
                          crops[dim * 2] - crops[dim * 2 + 1]);
    // Should be non-negative
    TF_LITE_ENSURE(context, final_dim_size >= 0);
    output_size->data[dim + 1] = final_dim_size;
    prod_block_shape *= block_shape[dim];
  }

  // Number of batch must be multiple of
  // (block_shape[0] * ...  * block_shape[M]).
  TF_LITE_ENSURE_EQ(context, input_size->data[0] % prod_block_shape, 0);

  // output_batch_size =  [batch / prod(block_shape)]
  output_size->data[0] = input_size->data[0] / prod_block_shape;

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  BatchToSpaceNDContext op_context(context, node);

  // Input dimension should be minimum [batch] + spatial_shape
  TF_LITE_ENSURE(context, NumDimensions(op_context.input) >
                              NumElements(op_context.block_shape) + 1);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Currently only int32 is supported for block_shape &  crops
  if (op_context.block_shape->type != kTfLiteInt32 ||
      op_context.crops->type != kTfLiteInt32) {
    context->ReportError(
        context, "Batch_to_space only supports int32 for block_shape & crops.");
    return kTfLiteError;
  }

  // Allocate temporary tensors which are required in kernel operation
  int temp_input_indices_tensor_id = 0;
  int temp_output_indices_tensor_id = 0;
  context->AddTensors(context, 1, &temp_input_indices_tensor_id);
  context->AddTensors(context, 1, &temp_output_indices_tensor_id);

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = temp_input_indices_tensor_id;
  node->temporaries->data[1] = temp_output_indices_tensor_id;

  // As input data is allocated including Paddings, so we can capture
  // coordinates, based on  - [block_shape[0], ..., block_shape[M-1],
  // batch / prod(block_shape), input_shape[1], ..., input_shape[M]]
  // As we need to capture indices along with co-efficient, so multiply by 2
  const int input_num_indices =
      (2 * SizeOfDimension(op_context.block_shape, 0) + 1) * 2;
  // Output data removes padding from input, so we have to capture
  // actual coordinates based on  - [batch / prod(block_shape)] +
  // [spatial_shape]
  const int output_num_indices =
      (SizeOfDimension(op_context.block_shape, 0) + 1) * 2;

  TfLiteTensor* temp_input_indices_tensor =
      GetTemporary(context, node, /*index=*/0);
  temp_input_indices_tensor->type = op_context.block_shape->type;
  temp_input_indices_tensor->allocation_type = kTfLiteArenaRw;

  TfLiteIntArray* temp_input_indices_tensor_size = TfLiteIntArrayCreate(1);
  temp_input_indices_tensor_size->data[0] = input_num_indices;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, temp_input_indices_tensor,
                                          temp_input_indices_tensor_size));

  TfLiteTensor* temp_output_indices_tensor =
      GetTemporary(context, node, /*index=*/1);
  temp_output_indices_tensor->type = op_context.block_shape->type;
  temp_output_indices_tensor->allocation_type = kTfLiteArenaRw;

  TfLiteIntArray* temp_output_indices_tensor_size = TfLiteIntArrayCreate(1);
  temp_output_indices_tensor_size->data[0] = output_num_indices;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, temp_output_indices_tensor,
                                          temp_output_indices_tensor_size));

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
  TfLiteTensor* input_indices_tensor = GetTemporary(context, node, /*index=*/0);
  TfLiteTensor* output_indices_tensor =
      GetTemporary(context, node, /*index=*/1);

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
                       GetTensorData<int32_t>(input_indices_tensor),   \
                       GetTensorData<int32_t>(output_indices_tensor),  \
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
      context->ReportError(
          context, "Type %d is currently not supported by BatchToSpace.",
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

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
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace space_to_batch_nd {

// This file has two implementations of SpaceToBatchND.
enum KernelType {
  kReference,
  kGenericOptimized,
};

// Inputs specified in the 2nd tensor (block_shape) and 3rd tensor (paddings)
// are ignored. Only use the `block_shape` and `paddings` specified in params.
// TODO(nupurgarg): Support inputs as tensors in SpaceToBatchND.
struct SpaceToBatchNDContext {
  SpaceToBatchNDContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteSpaceToBatchNDParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  TfLiteSpaceToBatchNDParams* params;
  TfLiteTensor* input;
  TfLiteTensor* output;
};

// Currently, only 4D NHWC input/output op_context are supported.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(nupurgarg): Support arbitrary dimension in SpaceToBatchND.
const int kInputDimensionNum = 4;
const int kOutputDimensionNum = 4;
const int kSpatialDimensionNum = 2;
const int kPaddingDimensionNum = 4;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) >= 1 && NumInputs(node) <= 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  SpaceToBatchNDContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.input),
                    kInputDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.params->num_spatial_dimensions,
                    kSpatialDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  const TfLiteIntArray* input_size = op_context.input->dims;
  const int* block_shape = op_context.params->block_shape;

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(kOutputDimensionNum);

  // Ensures the input height and width (with padding) is a multiple of block
  // shape height and width.
  for (int dim = 0; dim < kSpatialDimensionNum; ++dim) {
    int final_dim_size =
        (input_size->data[dim + 1] + op_context.params->before_paddings[dim] +
         op_context.params->after_paddings[dim]);
    TF_LITE_ENSURE_EQ(context, final_dim_size % block_shape[dim], 0);
    output_size->data[dim + 1] = final_dim_size / block_shape[dim];
  }

  const int output_batch_size =
      input_size->data[0] * block_shape[0] * block_shape[1];
  const int output_channel_size = input_size->data[3];

  output_size->data[0] = output_batch_size;
  output_size->data[3] = output_channel_size;

  return context->ResizeTensor(context, op_context.output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  SpaceToBatchNDContext op_context(context, node);

  int block_shape_dims_array[1] = {kSpatialDimensionNum};
  Dims<4> block_shape_dims = GetTensorDims(block_shape_dims_array, 1);

  // Initialize padding array in the format accepted by the kernel code.
  // TODO(nupurgarg): Make kernel code accept padding array format that is
  // consistent with Pad operation (i.e. before_paddings and after_paddings).
  TfLiteIntArray* padding_data = TfLiteIntArrayCreate(kPaddingDimensionNum);
  padding_data->data[0] = op_context.params->before_paddings[0];
  padding_data->data[1] = op_context.params->after_paddings[0];
  padding_data->data[2] = op_context.params->before_paddings[1];
  padding_data->data[3] = op_context.params->after_paddings[1];
  int padding_dims_array[1] = {kPaddingDimensionNum};
  Dims<4> padding_dims = GetTensorDims(padding_dims_array, 1);

#define TF_LITE_SPACE_TO_BATCH_ND(type, scalar)                          \
  type::SpaceToBatchND(GetTensorData<scalar>(op_context.input),          \
                       GetTensorDims(op_context.input),                  \
                       op_context.params->block_shape, block_shape_dims, \
                       padding_data->data, padding_dims,                 \
                       GetTensorData<scalar>(op_context.output),         \
                       GetTensorDims(op_context.output))
  switch (op_context.input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, float);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, uint8_t);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, int32_t);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_SPACE_TO_BATCH_ND(reference_ops, int64_t);
      } else {
        TF_LITE_SPACE_TO_BATCH_ND(optimized_ops, int64_t);
      }
      break;
    default:
      context->ReportError(context,
                           "Type is currently not supported by SpaceToBatch.");
      return kTfLiteError;
  }
#undef TF_LITE_SPACE_TO_BATCH_ND

  TfLiteIntArrayFree(padding_data);
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

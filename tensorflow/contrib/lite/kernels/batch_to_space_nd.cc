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
namespace batch_to_space_nd {

// This file has two implementations of BatchToSpaceND.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct BatchToSpaceNDContext {
  BatchToSpaceNDContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteBatchToSpaceNDParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  TfLiteBatchToSpaceNDParams* params;
  TfLiteTensor* input;
  TfLiteTensor* output;
};

// Currently, only 4D NHWC input/output op_context are supported.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(ycling): Support arbitrary dimension in BatchToSpaceND.
const int kInputDimensionNum = 4;
const int kOutputDimensionNum = 4;
const int kSpatialDimensionNum = 2;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // The 2nd tensor (block_shape) and the 3rd tensor (crops) are ignored now.
  TF_LITE_ENSURE(context, NumInputs(node) >= 1 && NumInputs(node) <= 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  BatchToSpaceNDContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.input),
                    kInputDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.params->num_spatial_dimensions,
                    kSpatialDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  const TfLiteIntArray* input_size = op_context.input->dims;
  const int* block_shape = op_context.params->block_shape;

  // Number of batch must be multiple of (block_shape[0] * block_shape[1]).
  TF_LITE_ENSURE_EQ(context,
                    input_size->data[0] % (block_shape[0] * block_shape[1]), 0);

  const int output_batch_size =
      input_size->data[0] / (block_shape[0] * block_shape[1]);
  const int output_height = input_size->data[1] * block_shape[0];
  const int output_width = input_size->data[2] * block_shape[1];
  const int output_channel_size = input_size->data[3];

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(kOutputDimensionNum);
  output_size->data[0] = output_batch_size;
  output_size->data[1] = output_height;
  output_size->data[2] = output_width;
  output_size->data[3] = output_channel_size;

  return context->ResizeTensor(context, op_context.output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  BatchToSpaceNDContext op_context(context, node);

  int block_shape_dims_array[1] = {kSpatialDimensionNum};
  Dims<4> block_shape_dims = GetTensorDims(block_shape_dims_array, 1);

#define TF_LITE_BATCH_TO_SPACE_ND(type, scalar)                          \
  type::BatchToSpaceND(GetTensorData<scalar>(op_context.input),          \
                       GetTensorDims(op_context.input),                  \
                       op_context.params->block_shape, block_shape_dims, \
                       GetTensorData<scalar>(op_context.output),         \
                       GetTensorDims(op_context.output))
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
      context->ReportError(context,
                           "Type is currently not supported by BatchToSpace.");
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

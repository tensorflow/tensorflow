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
namespace transpose {

// This file has two implementations of Transpose.
enum KernelType {
  kReference,
};

// TODO(nupurgarg): Permutation arrays represented as a tensor are ignored. Only
// use the `perm` specified in `params`.
struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteTransposeParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  TfLiteTransposeParams* params;
  TfLiteTensor* input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TransposeContext op_context(context, node);
  int dims = NumDimensions(op_context.input);

  // Ensure validity of input tensor and permutation array.
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);
  TF_LITE_ENSURE_EQ(context, dims, op_context.params->num_dimensions);
  TF_LITE_ENSURE_MSG(context, dims <= 4,
                     "Transpose op only supports 1D-4D input arrays.");
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context,
                       op_context.params->perm[idx] >= 0 &&
                           op_context.params->perm[idx] < dims,
                       "Transpose op permutations array is out of bounds.");
  }

  // Determine size of output tensor.
  const TfLiteIntArray* input_size = op_context.input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(dims);
  for (int idx = 0; idx < dims; ++idx) {
    output_size->data[idx] = input_size->data[op_context.params->perm[idx]];
  }

  return context->ResizeTensor(context, op_context.output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

  // Reverse the permuted axes and convert to 4D due to the way Dims are
  // constructed in GetTensorDims.
  const int kOutputDimensionNum = 4;
  int reversed_perm[kOutputDimensionNum];
  int size = op_context.params->num_dimensions;
  for (int output_k = 0, input_k = size - 1; output_k < size;
       ++output_k, --input_k) {
    reversed_perm[output_k] = size - op_context.params->perm[input_k] - 1;
  }
  for (int k = size; k < kOutputDimensionNum; ++k) {
    reversed_perm[k] = k;
  }

#define TF_LITE_TRANSPOSE(type, scalar)                     \
  type::Transpose(GetTensorData<scalar>(op_context.input),  \
                  GetTensorDims(op_context.input),          \
                  GetTensorData<scalar>(op_context.output), \
                  GetTensorDims(op_context.output), reversed_perm)

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_TRANSPOSE(reference_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_TRANSPOSE(reference_ops, uint8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_TRANSPOSE(reference_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_TRANSPOSE(reference_ops, int64_t);
      }
      break;
    default:
      context->ReportError(context,
                           "Type is currently not supported by Transpose.");
      return kTfLiteError;
  }
#undef TF_LITE_TRANSPOSE

  return kTfLiteOk;
}

}  // namespace transpose

TfLiteRegistration* Register_TRANSPOSE_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, transpose::Prepare,
                                 transpose::Eval<transpose::kReference>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE() { return Register_TRANSPOSE_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

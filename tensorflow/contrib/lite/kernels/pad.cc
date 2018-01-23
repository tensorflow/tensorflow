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
namespace pad {

// This file has two implementations of Pad.
enum KernelType {
  kReference,
  kGenericOptimized,
};

// TODO(nupurgarg): Padding represented as a tensor is ignored. Only use the
// `left_padding` and `right_padding` specified in `params`.
struct PadContext {
  PadContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLitePadParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  TfLitePadParams* params;
  TfLiteTensor* input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Determines size of output tensor.
  PadContext op_context(context, node);
  int dims = NumDimensions(op_context.input);
  TF_LITE_ENSURE_EQ(context, dims, op_context.params->num_dimensions);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // TODO(nupurgarg): Our current implementations rely on the inputs being 4D.
  TF_LITE_ENSURE_EQ(context, dims, 4);

  const TfLiteIntArray* input_size = op_context.input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(dims);
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context,
                       (op_context.params->before_padding[idx] >= 0 &&
                        op_context.params->after_padding[idx] >= 0),
                       "Pad value has to be greater than equal to 0.");
    output_size->data[idx] =
        (input_size->data[idx] + op_context.params->before_padding[idx] +
         op_context.params->after_padding[idx]);
  }

  return context->ResizeTensor(context, op_context.output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);

  std::vector<int> before_padding(
      op_context.params->before_padding,
      op_context.params->before_padding + op_context.params->num_dimensions);
  std::vector<int> after_padding(
      op_context.params->after_padding,
      op_context.params->after_padding + op_context.params->num_dimensions);

  // TODO(nupurgarg): Change TOCO's implementation to use padding arrays
  // in forward order (depth, width, height, batch).
  // Converts from int[] = {depth, width, height, batch} to int[] = {batch,
  // height, width, depth} to match TOCO's implementation of pad in
  // referenced_ops.h and optimized_ops.h.
  std::reverse(before_padding.begin(), before_padding.end());
  std::reverse(after_padding.begin(), after_padding.end());

#define TF_LITE_PAD(type, scalar)                                           \
  type::Pad(GetTensorData<scalar>(op_context.input),                        \
            GetTensorDims(op_context.input), before_padding, after_padding, \
            GetTensorData<scalar>(op_context.output),                       \
            GetTensorDims(op_context.output))

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, float);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, uint8_t);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, int32_t);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, int64_t);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, int64_t);
      }
      break;
    default:
      context->ReportError(context, "Type is currently not supported by Pad.");
      return kTfLiteError;
  }
#undef TF_LITE_PAD
  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration* Register_PAD_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kReference>};
  return &r;
}

TfLiteRegistration* Register_PAD_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PAD() {
  return Register_PAD_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

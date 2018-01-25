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

struct PadContext {
  PadContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    paddings = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
    dims = NumDimensions(input);
  }
  TfLiteTensor* input;
  TfLiteTensor* paddings;
  TfLiteTensor* output;
  int dims;
};

// Resizes output array based on the input size and padding size. This function
// is callable from both Prepare() and Eval() as long as the caller ensures the
// paddings data is present.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                PadContext* op_context) {
  // TODO(nupurgarg): Our current implementations rely on the inputs being 4D.
  TF_LITE_ENSURE_EQ(context, op_context->dims, 4);

  // Ensures the paddings array is dims x 2.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context->paddings, 0),
                    op_context->dims);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context->paddings, 1), 2);

  // Determines the size of the output tensor.
  const TfLiteIntArray* input_size = op_context->input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(op_context->dims);
  const int32* paddings_data = GetTensorData<int32>(op_context->paddings);

  for (int idx = 0; idx < op_context->dims; ++idx) {
    int before_padding = *paddings_data++;
    int after_padding = *paddings_data++;

    TF_LITE_ENSURE_MSG(context, (before_padding >= 0 && after_padding >= 0),
                       "Pad value has to be greater than equal to 0.");

    output_size->data[idx] =
        (input_size->data[idx] + before_padding + after_padding);
  }

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PadContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // TODO(nupurgarg): Create wrapper functions for dynamic tensor logic.
  // Exit early if paddings is a non-const tensor. Set output tensor to
  // dynamic so output size can be determined in Eval.
  if (op_context.paddings->allocation_type != kTfLiteMmapRo) {
    op_context.output->allocation_type = kTfLiteDynamic;
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (op_context.output->allocation_type == kTfLiteDynamic) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
    TfLiteTensorRealloc(op_context.output->bytes, op_context.output);
  }

  // TODO(nupurgarg): Change kernel implementation to take in int* instead of
  // vector<int> to remove malloc from Eval().
  // Create before and after padding arrays that are accepted by the kernel.
  std::vector<int> before_padding;
  std::vector<int> after_padding;
  const int32* paddings_data = GetTensorData<int32>(op_context.paddings);

  // TODO(nupurgarg): Change kernel implementation to use padding arrays in
  // forward order (depth, width, height, batch).
  // Build paddings in order of int[] = {batch, height, width, depth} to match
  // kernel implementation of Pad in referenced_ops.h and optimized_ops.h.
  for (int idx = op_context.dims - 1; idx >= 0; --idx) {
    before_padding.push_back(paddings_data[idx * 2]);
    after_padding.push_back(paddings_data[idx * 2 + 1]);
  }

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

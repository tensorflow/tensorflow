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
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// clang-format off: Clang-format thinks this header is paired.
#include "tensorflow/lite/kernels/internal/optimized/resize_bilinear.h"
// clang-format on
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace resize_bilinear {

// This file has three implementation of RESIZE_BILINEAR.
enum KernelType {
  kReference,
  kOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* input,
                                const TfLiteTensor* size,
                                TfLiteTensor* output) {
  const int32* size_data = GetTensorData<int32>(size);
  // Sanity check, the up/down sampling size should always be positive.
  TF_LITE_ENSURE(context, size_data[0] > 0);
  TF_LITE_ENSURE(context, size_data[1] > 0);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = input->dims->data[0];
  output_size->data[1] = size_data[0];
  output_size->data[2] = size_data[1];
  output_size->data[3] = input->dims->data[3];
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // TODO(ahentz): Our current implementations rely on the inputs being 4D.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);

  TF_LITE_ENSURE_EQ(context, size->type, kTfLiteInt32);
  // ResizeBilinear creates a float tensor even when the input is made of
  // integers.
  output->type = input->type;

  if (!IsConstantTensor(size)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  // Ensure params are valid.
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);
  if (params->half_pixel_centers && params->align_corners) {
    context->ReportError(
        context, "If half_pixel_centers is True, align_corners must be False.");
    return kTfLiteError;
  }

  return ResizeOutputTensor(context, input, size, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, input, size, output));
  }

  if (output->type == kTfLiteFloat32) {
#define TF_LITE_RESIZE_BILINEAR(type, opname, datatype)              \
  tflite::ResizeBilinearParams op_params;                            \
  op_params.align_corners = params->align_corners;                   \
  op_params.half_pixel_centers = params->half_pixel_centers;         \
  type::opname(op_params, GetTensorShape(input),                     \
               GetTensorData<datatype>(input), GetTensorShape(size), \
               GetTensorData<int32>(size), GetTensorShape(output),   \
               GetTensorData<datatype>(output))

    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, ResizeBilinear, float);
    } else if (kernel_type == kOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops, ResizeBilinear, float);
    }
  } else if (output->type == kTfLiteUInt8) {
    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, ResizeBilinear, uint8_t);
    } else if (kernel_type == kOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops, ResizeBilinear, uint8_t);
    }
  } else if (output->type == kTfLiteInt8) {
    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, ResizeBilinearInteger, int8_t);
    } else if (kernel_type == kOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops, ResizeBilinear, int8_t);
    }
  } else if (output->type == kTfLiteInt16) {
    TF_LITE_RESIZE_BILINEAR(reference_ops, ResizeBilinearInteger, int16_t);
#undef TF_LITE_RESIZE_BILINEAR
  } else {
    context->ReportError(context, "Output type is %d, requires float.",
                         output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace resize_bilinear

TfLiteRegistration* Register_RESIZE_BILINEAR_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, resize_bilinear::Prepare,
      resize_bilinear::Eval<resize_bilinear::kReference>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_BILINEAR() {
  static TfLiteRegistration r = {
      nullptr, nullptr, resize_bilinear::Prepare,
      resize_bilinear::Eval<resize_bilinear::kOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

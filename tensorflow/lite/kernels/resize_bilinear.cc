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
namespace resize_bilinear {

// This file has three implementation of RESIZE_BILINEAR.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
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

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* size = GetInput(context, node, kSizeTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

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
  return ResizeOutputTensor(context, input, size, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* size = GetInput(context, node, kSizeTensor);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, input, size, output));
  }

  if (output->type == kTfLiteFloat32) {
#define TF_LITE_RESIZE_BILINEAR(type, datatype)                              \
  tflite::ResizeBilinearParams op_params;                                    \
  op_params.align_corners = params->align_corners;                           \
  op_params.half_pixel_centers = false;                                      \
  type::ResizeBilinear(op_params, GetTensorShape(input),                     \
                       GetTensorData<datatype>(input), GetTensorShape(size), \
                       GetTensorData<int32>(size), GetTensorShape(output),   \
                       GetTensorData<datatype>(output))

    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, float);
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops, float);
    }
  } else if (output->type == kTfLiteUInt8) {
    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, uint8_t);
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops, uint8_t);
    }
  } else if (output->type == kTfLiteInt8) {
    TF_LITE_RESIZE_BILINEAR(reference_ops, int8_t);
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

TfLiteRegistration* Register_RESIZE_BILINEAR_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, resize_bilinear::Prepare,
      resize_bilinear::Eval<resize_bilinear::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_BILINEAR_NEON_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, resize_bilinear::Prepare,
      resize_bilinear::Eval<resize_bilinear::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_BILINEAR() {
#ifdef USE_NEON
  return Register_RESIZE_BILINEAR_NEON_OPT();
#else
  return Register_RESIZE_BILINEAR_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

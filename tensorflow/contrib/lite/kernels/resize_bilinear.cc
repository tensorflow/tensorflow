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
namespace resize_bilinear {

// This file has three implementation of RESIZE_BILINEAR.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // TODO(ahentz): Our current implementations rely on the inputs being 4D.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  // TODO(ahentz): Our current implementations only support float32.
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = input->dims->data[0];
  output_size->data[1] = params->new_height;
  output_size->data[2] = params->new_width;
  output_size->data[3] = input->dims->data[3];

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);

  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // We have to fake a tensor here, to satisfy ResizeBilinear().
  int32 output_size_data[2] = {params->new_height, params->new_width};

  if (output->type == kTfLiteFloat32) {
#define TF_LITE_RESIZE_BILINEAR(type)                                     \
  type::ResizeBilinear(GetTensorData<float>(input), GetTensorDims(input), \
                       output_size_data, GetTensorDims({1, 1, 1, 2}),     \
                       GetTensorData<float>(output), GetTensorDims(output))

    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops);
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      TF_LITE_RESIZE_BILINEAR(optimized_ops);
    }
#undef TF_LITE_RESIZE_BILINEAR
  } else {
    context->ReportError(context, "Inputs and outputs not all float types.");
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

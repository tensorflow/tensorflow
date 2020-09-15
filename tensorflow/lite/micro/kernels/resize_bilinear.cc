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
#include "tensorflow/lite/kernels/internal/reference/resize_bilinear.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace resize_bilinear {

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

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

  TF_LITE_ENSURE_MSG(context, IsConstantTensor(size),
                     "Non constant size tensor not supported");

  // Ensure params are valid.
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);
  if (params->half_pixel_centers && params->align_corners) {
    context->ReportError(
        context, "If half_pixel_centers is True, align_corners must be False.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32) {
#define TF_LITE_RESIZE_BILINEAR(type, datatype)                         \
  tflite::ResizeBilinearParams op_params;                               \
  op_params.align_corners = params->align_corners;                      \
  op_params.half_pixel_centers = params->half_pixel_centers;            \
  type::ResizeBilinear(op_params, tflite::micro::GetTensorShape(input), \
                       tflite::micro::GetTensorData<datatype>(input),   \
                       tflite::micro::GetTensorShape(size),             \
                       tflite::micro::GetTensorData<int32_t>(size),     \
                       tflite::micro::GetTensorShape(output),           \
                       tflite::micro::GetTensorData<datatype>(output))

    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, float);
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, float);
    }
  } else if (output->type == kTfLiteUInt8) {
    if (kernel_type == kReference) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, uint8_t);
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      TF_LITE_RESIZE_BILINEAR(reference_ops, uint8_t);
    }
  } else if (output->type == kTfLiteInt8) {
    TF_LITE_RESIZE_BILINEAR(reference_ops, int8_t);
#undef TF_LITE_RESIZE_BILINEAR
  } else {
    context->ReportError(context,
                         "Output type is %d, requires float, int8 or uint8.",
                         output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace resize_bilinear

TfLiteRegistration Register_RESIZE_BILINEAR_REF() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/resize_bilinear::Prepare,
          /*invoke=*/resize_bilinear::Eval<resize_bilinear::kReference>,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RESIZE_BILINEAR_GENERIC_OPT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/resize_bilinear::Prepare,
          /*invoke=*/resize_bilinear::Eval<resize_bilinear::kGenericOptimized>,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RESIZE_BILINEAR_NEON_OPT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/resize_bilinear::Prepare,
          /*invoke=*/resize_bilinear::Eval<resize_bilinear::kNeonOptimized>,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RESIZE_BILINEAR() {
#ifdef USE_NEON
  return Register_RESIZE_BILINEAR_NEON_OPT();
#else
  return Register_RESIZE_BILINEAR_GENERIC_OPT();
#endif
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

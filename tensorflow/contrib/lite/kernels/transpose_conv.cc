/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose_conv {

constexpr int kOutputShapeTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kDataInputTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutputShape(TfLiteContext* context,
                               const TfLiteTensor* output_shape,
                               TfLiteTensor* output) {
  // Currently only support int32 for output shape.
  if (output_shape->type != kTfLiteInt32) {
    context->ReportError(context, "Output shape is %d, not int32.",
                         output_shape->type);
    return kTfLiteError;
  }
  const int output_dimensions = NumElements(output_shape);
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(output_dimensions);
  for (int i = 0; i < output_dimensions; ++i) {
    output_shape_array->data[i] = GetTensorData<int32_t>(output_shape)[i];
  }

  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 4);

  // Currently only supports float32.
  const TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context, data_type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, weights->type, data_type);

  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 3),
                    SizeOfDimension(weights, 3));

  if (!IsConstantTensor(output_shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputShape(context, output_shape, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const auto* params =
      reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputShape(context, output_shape, output));
  }

  // Get height and width of the output image.
  const int width = SizeOfDimension(output, 2);
  const int height = SizeOfDimension(output, 1);
  const int filter_width = SizeOfDimension(weights, 1);
  const int filter_height = SizeOfDimension(weights, 2);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;

  const TfLitePaddingValues& padding_size =
      ComputePaddingHeightWidth(stride_height, stride_width, 1, height, width,
                                filter_height, filter_width, params->padding);

  // Currently only support float32.
  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::ConvParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = padding_size.width;
      op_params.padding_values.height = padding_size.height;
      op_params.stride_width = stride_width;
      op_params.stride_height = stride_height;

      reference_ops::TransposeConv(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(weights), GetTensorData<float>(weights),
          GetTensorShape(output), GetTensorData<float>(output),
          // Last two args specify im2col which reference_ops ignores.
          // (Note this does not lead to a performance regression, as the
          // previous optimized version was just a copy of the reference code.)
          // TODO(b/110208176): Allocate im2col tensors and switch to
          // optimized_ops.
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    }
    default:
      context->ReportError(context, "Type %d, not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace transpose_conv

TfLiteRegistration* Register_TRANSPOSE_CONV() {
  static TfLiteRegistration r = {nullptr, nullptr, transpose_conv::Prepare,
                                 transpose_conv::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

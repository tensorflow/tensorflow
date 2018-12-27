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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/eigen_support.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose_conv {

// This file has 2 implementation of TransposeConv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
};

constexpr int kOutputShapeTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kDataInputTensor = 2;
constexpr int kOutputTensor = 0;

const int kTensorNotAllocated = -1;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int im2col_id = kTensorNotAllocated;

  // im2col is the only temporary currently tracked, therefore always index 0.
  // If more temporaries are added, they should be properly tracked.
  int32_t im2col_index = 0;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to use as scratch space for im2col, and
  // to carry information from Prepare() to Eval().
  auto* data = new OpData;
  eigen_support::IncrementUsageCounter(context);
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  eigen_support::DecrementUsageCounter(context);
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* shape_tensor,
                                TfLiteTensor* output) {
  // Currently only support int32 for output shape.
  if (shape_tensor->type != kTfLiteInt32) {
    context->ReportError(context, "Output shape is %d, not int32.",
                         shape_tensor->type);
    return kTfLiteError;
  }

  TfLiteIntArray* shape = TfLiteIntArrayCreate(NumElements(shape_tensor));
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = GetTensorData<int32_t>(shape_tensor)[i];
  }

  return context->ResizeTensor(context, output, shape);
}

static TfLiteStatus AllocateIm2colTensorIfRequired(TfLiteContext* context,
                                                   TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (data->im2col_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->im2col_id);
    context->tensors[data->im2col_id].type = kTfLiteFloat32;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[data->im2col_index] = data->im2col_id;

  return kTfLiteOk;
}

TfLiteStatus ResizeIm2ColTensor(TfLiteContext* context,
                                const TfLiteTensor* output_shape,
                                const TfLiteTensor* weights,
                                const TfLiteTensor* input,
                                TfLiteTensor* im2col) {
  if (output_shape->type != kTfLiteInt32) {
    context->ReportError(context, "im2col shape is %d, not int32.",
                         output_shape->type);
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 4);
  TfLiteIntArray* im2col_shape_array = TfLiteIntArrayCreate(4);
  im2col_shape_array->data[0] = output_shape->data.i32[0];
  im2col_shape_array->data[1] = output_shape->data.i32[1];
  im2col_shape_array->data[2] = output_shape->data.i32[2];
  const int input_depth = SizeOfDimension(input, 3);
  const int filter_width = SizeOfDimension(weights, 1);
  const int filter_height = SizeOfDimension(weights, 2);
  im2col_shape_array->data[3] = input_depth * filter_height * filter_width;

  im2col->type = input->type;
  im2col->allocation_type = kTfLiteDynamic;
  return context->ResizeTensor(context, im2col, im2col_shape_array);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Sanity checks on op
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Allocate Im2col Tensor
  TF_LITE_ENSURE_STATUS(AllocateIm2colTensorIfRequired(context, node));

  // Retrieve tensors
  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  OpData* user_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteTensor* im2col =
      &context->tensors[node->temporaries->data[user_data->im2col_index]];

  // Tensor sanity checks
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, weights->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 3),
                    SizeOfDimension(weights, 3));

  if (!IsConstantTensor(output_shape)) {
    // Defer resizing until Eval().
    SetTensorToDynamic(output);
    SetTensorToDynamic(im2col);
  } else {
    TF_LITE_ENSURE_STATUS(ResizeOutputTensor(context, output_shape, output));
    TF_LITE_ENSURE_STATUS(
        ResizeIm2ColTensor(context, output_shape, weights, input, im2col));
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Retrieve tensors (All should be allocated by now)
  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  OpData* user_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteTensor* im2col =
      &context->tensors[node->temporaries->data[user_data->im2col_index]];
  const auto* params =
      reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);

  // Resize any deferred dynamic tensors
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, output_shape, output));
  }
  if (IsDynamicTensor(im2col)) {
    TF_LITE_ENSURE_OK(context, ResizeIm2ColTensor(context, output_shape,
                                                  weights, input, im2col));
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
      switch (kernel_type) {
        case kReference: {
          reference_ops::TransposeConv(
              op_params, GetTensorShape(input), GetTensorData<float>(input),
              GetTensorShape(weights), GetTensorData<float>(weights),
              GetTensorShape(output), GetTensorData<float>(output),
              GetTensorShape(im2col), GetTensorData<float>(im2col));
          break;
        }
        case kGenericOptimized: {
          optimized_ops::TransposeConv(
              op_params, GetTensorShape(input), GetTensorData<float>(input),
              GetTensorShape(weights), GetTensorData<float>(weights),
              GetTensorShape(output), GetTensorData<float>(output),
              GetTensorShape(im2col), GetTensorData<float>(im2col));
          break;
        }
      }
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

TfLiteRegistration* Register_TRANSPOSECONV_REF() {
  static TfLiteRegistration r = {
      transpose_conv::Init, transpose_conv::Free, transpose_conv::Prepare,
      transpose_conv::Eval<transpose_conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSECONV_GENERIC_OPT() {
  static TfLiteRegistration r = {
      transpose_conv::Init, transpose_conv::Free, transpose_conv::Prepare,
      transpose_conv::Eval<transpose_conv::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE_CONV() {
  return Register_TRANSPOSECONV_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

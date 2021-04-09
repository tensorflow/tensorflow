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
#include "tensorflow/lite/kernels/internal/reference/arg_min_max.h"

#include <stdint.h>

#include <functional>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace arg_min_max {

constexpr int kInputTensor = 0;
constexpr int kAxis = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* input,
                          const TfLiteTensor* axis, TfLiteTensor* output) {
  int axis_value;
  // Retrive all 8 bytes when axis type is kTfLiteInt64 to avoid data loss.
  if (axis->type == kTfLiteInt64) {
    axis_value = static_cast<int>(*GetTensorData<int64_t>(axis));
  } else {
    axis_value = *GetTensorData<int>(axis);
  }
  if (axis_value < 0) {
    axis_value += NumDimensions(input);
  }

  // Copy the input dimensions to output except the axis dimension.
  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(NumDimensions(input) - 1);
  int j = 0;
  for (int i = 0; i < NumDimensions(input); ++i) {
    if (i != axis_value) {
      output_dims->data[j] = SizeOfDimension(input, i);
      ++j;
    }
  }
  return context->ResizeTensor(context, output, output_dims);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxis, &axis));
  // Make sure the axis is only 1 dimension.
  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);
  // Make sure the axis is only either int32 or int64.
  TF_LITE_ENSURE(context,
                 axis->type == kTfLiteInt32 || axis->type == kTfLiteInt64);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  auto* params = reinterpret_cast<TfLiteArgMaxParams*>(node->builtin_data);
  switch (params->output_type) {
    case kTfLiteInt32:
      output->type = kTfLiteInt32;
      break;
    case kTfLiteInt64:
      output->type = kTfLiteInt64;
      break;
    default:
      context->ReportError(context, "Unknown index output data type: %d",
                           params->output_type);
      return kTfLiteError;
  }

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt32:
      break;

    default:
      context->ReportError(
          context,
          "Unknown input type: %d, only float32 and int types are supported",
          input->type);
      return kTfLiteError;
  }

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  if (IsConstantTensor(axis)) {
    TF_LITE_ENSURE_STATUS(ResizeOutput(context, input, axis, output));
  } else {
    SetTensorToDynamic(output);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node, bool is_arg_max) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxis, &axis));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_STATUS(ResizeOutput(context, input, axis, output));
  }

#define TF_LITE_ARG_MIN_MAX(data_type, axis_type, output_type) \
  optimized_ops::ArgMinMax(                                    \
      GetTensorShape(input), GetTensorData<data_type>(input),  \
      GetTensorData<axis_type>(axis), GetTensorShape(output),  \
      GetTensorData<output_type>(output), is_arg_max)
  if (axis->type == kTfLiteInt32) {
    switch (output->type) {
      case kTfLiteInt32: {
        switch (input->type) {
          case kTfLiteFloat32:
            TF_LITE_ARG_MIN_MAX(float, int32_t, int32_t);
            break;
          case kTfLiteUInt8:
            TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int32_t);
            break;
          case kTfLiteInt8:
            TF_LITE_ARG_MIN_MAX(int8_t, int32_t, int32_t);
            break;
          case kTfLiteInt32:
            TF_LITE_ARG_MIN_MAX(int32_t, int32_t, int32_t);
            break;
          default:
            context->ReportError(context,
                                 "Only float32, uint8, int8 and int32 are "
                                 "supported currently, got %s.",
                                 TfLiteTypeGetName(input->type));
            return kTfLiteError;
        }
      } break;
      case kTfLiteInt64: {
        switch (input->type) {
          case kTfLiteFloat32:
            TF_LITE_ARG_MIN_MAX(float, int32_t, int64_t);
            break;
          case kTfLiteUInt8:
            TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int64_t);
            break;
          case kTfLiteInt8:
            TF_LITE_ARG_MIN_MAX(int8_t, int32_t, int64_t);
            break;
          case kTfLiteInt32:
            TF_LITE_ARG_MIN_MAX(int32_t, int32_t, int64_t);
            break;
          default:
            context->ReportError(context,
                                 "Only float32, uint8, int8 and int32 are "
                                 "supported currently, got %s.",
                                 TfLiteTypeGetName(input->type));
            return kTfLiteError;
        }
      } break;
      default:
        context->ReportError(
            context, "Only int32 and int64 are supported currently, got %s.",
            TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    switch (output->type) {
      case kTfLiteInt32: {
        switch (input->type) {
          case kTfLiteFloat32:
            TF_LITE_ARG_MIN_MAX(float, int64_t, int32_t);
            break;
          case kTfLiteUInt8:
            TF_LITE_ARG_MIN_MAX(uint8_t, int64_t, int32_t);
            break;
          case kTfLiteInt8:
            TF_LITE_ARG_MIN_MAX(int8_t, int64_t, int32_t);
            break;
          case kTfLiteInt32:
            TF_LITE_ARG_MIN_MAX(int32_t, int64_t, int32_t);
            break;
          default:
            context->ReportError(context,
                                 "Only float32, uint8, int8 and int32 are "
                                 "supported currently, got %s.",
                                 TfLiteTypeGetName(input->type));
            return kTfLiteError;
        }
      } break;
      case kTfLiteInt64: {
        switch (input->type) {
          case kTfLiteFloat32:
            TF_LITE_ARG_MIN_MAX(float, int64_t, int64_t);
            break;
          case kTfLiteUInt8:
            TF_LITE_ARG_MIN_MAX(uint8_t, int64_t, int64_t);
            break;
          case kTfLiteInt8:
            TF_LITE_ARG_MIN_MAX(int8_t, int64_t, int64_t);
            break;
          case kTfLiteInt32:
            TF_LITE_ARG_MIN_MAX(int32_t, int64_t, int64_t);
            break;
          default:
            context->ReportError(context,
                                 "Only float32, uint8, int8 and int32 are "
                                 "supported currently, got %s.",
                                 TfLiteTypeGetName(input->type));
            return kTfLiteError;
        }
      } break;
      default:
        context->ReportError(
            context, "Only int32 and int64 are supported currently, got %s.",
            TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  }
#undef TF_LITE_ARG_MIN_MAX

  return kTfLiteOk;
}

TfLiteStatus ArgMinEval(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, false);
}

TfLiteStatus ArgMaxEval(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, true);
}

}  // namespace arg_min_max

TfLiteRegistration* Register_ARG_MAX() {
  static TfLiteRegistration r = {nullptr, nullptr, arg_min_max::Prepare,
                                 arg_min_max::ArgMaxEval};
  return &r;
}

TfLiteRegistration* Register_ARG_MIN() {
  static TfLiteRegistration r = {nullptr, nullptr, arg_min_max::Prepare,
                                 arg_min_max::ArgMinEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

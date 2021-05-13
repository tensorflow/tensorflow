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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <functional>
#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace range {
namespace {

constexpr int kStartTensor = 0;
constexpr int kLimitTensor = 1;
constexpr int kDeltaTensor = 2;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus GetSize(TfLiteContext* context, T start, T limit, T delta,
                     int* size) {
  TF_LITE_ENSURE(context, !std::equal_to<T>()(delta, 0));
  TF_LITE_ENSURE(
      context, (start >= limit && delta < 0) || (start <= limit && delta > 0));
  *size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));
  return kTfLiteOk;
}

TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* start,
                          const TfLiteTensor* limit, const TfLiteTensor* delta,
                          TfLiteTensor* output) {
  // The output will always be a 1-d array.
  int size = 0;
  switch (start->type) {
    case kTfLiteInt32: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<int32_t>(start),
                                *GetTensorData<int32_t>(limit),
                                *GetTensorData<int32_t>(delta), &size));
      break;
    }
    case kTfLiteFloat32: {
      TF_LITE_ENSURE_OK(context, GetSize(context, *GetTensorData<float>(start),
                                         *GetTensorData<float>(limit),
                                         *GetTensorData<float>(delta), &size));
      break;
    }
    default: {
      context->ReportError(context, "Unknown data type: %d", start->type);
      return kTfLiteError;
    }
  }
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(1);
  output_shape_array->data[0] = size;
  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));
  // Make sure all the inputs are scalars.
  TF_LITE_ENSURE_EQ(context, NumDimensions(start), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(limit), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(delta), 0);

  // Currently only supports int32 and float.
  // TODO(b/117912892): Support quantization as well.
  const auto dtype = start->type;
  if (dtype != kTfLiteFloat32 && dtype != kTfLiteInt32) {
    context->ReportError(context, "Unknown index output data type: %s",
                         TfLiteTypeGetName(dtype));
    return kTfLiteError;
  }

  TF_LITE_ENSURE_TYPES_EQ(context, limit->type, dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, delta->type, dtype);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = dtype;

  if (IsConstantTensor(start) && IsConstantTensor(limit) &&
      IsConstantTensor(delta)) {
    return ResizeOutput(context, start, limit, delta, output);
  }

  SetTensorToDynamic(output);
  return kTfLiteOk;
}

template <typename T>
void EvalImpl(const TfLiteTensor* start, const TfLiteTensor* delta,
              TfLiteTensor* output) {
  const T start_value = *GetTensorData<T>(start);
  const T delta_value = *GetTensorData<T>(delta);
  T* output_data = GetTensorData<T>(output);
  const int num_elements = NumElements(output);
  T value = start_value;
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = value;
    value += delta_value;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutput(context, start, limit, delta, output));
  }

  switch (output->type) {
    case kTfLiteInt32: {
      EvalImpl<int32_t>(start, delta, output);
      break;
    }
    case kTfLiteFloat32: {
      EvalImpl<float>(start, delta, output);
      break;
    }
    default: {
      context->ReportError(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace range

TfLiteRegistration* Register_RANGE() {
  static TfLiteRegistration r = {nullptr, nullptr, range::Prepare, range::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

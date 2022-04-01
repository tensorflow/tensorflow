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

#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace slice {

enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kSizeTensor = 2;
constexpr int kOutputTensor = 0;

// This Op only supports 1-5D cases and since we use the optimized ops 5D
// implementation, the 1-4D tensors are mapped to 5D.
const int kMaxDim = 5;

template <typename T>
TfLiteStatus CalculateOutputShapeVector(TfLiteContext* context,
                                        const TfLiteTensor* input,
                                        const TfLiteTensor* begin,
                                        const TfLiteTensor* size,
                                        std::vector<int>* output_shape_vector) {
  for (int idx = 0; idx < NumDimensions(input); ++idx) {
    T size_value = GetTensorData<T>(size)[idx];
    if (size_value < 0) {
      if (size_value != -1) {
        context->ReportError(context, "Invalid size.");
        return kTfLiteError;
      }
      size_value = SizeOfDimension(input, idx) - GetTensorData<T>(begin)[idx];
    } else {
      if (SizeOfDimension(input, idx) <
          GetTensorData<T>(begin)[idx] + size_value) {
        context->ReportError(context, "Invalid begin and size.");
        return kTfLiteError;
      }
    }
    output_shape_vector->push_back(static_cast<int>(size_value));
  }
  return kTfLiteOk;
}

template <typename T>
void GetBeginAndSizeVectors(int dimensions, const TfLiteTensor* begin,
                            const TfLiteTensor* size, std::vector<int>* begins,
                            std::vector<int>* sizes) {
  for (int idx = 0; idx < dimensions; ++idx) {
    begins->push_back(GetTensorData<T>(begin)[idx]);
    sizes->push_back(GetTensorData<T>(size)[idx]);
  }
}

TfLiteStatus ResizeOutputShape(TfLiteContext* context,
                               const TfLiteTensor* input,
                               const TfLiteTensor* begin,
                               const TfLiteTensor* size, TfLiteTensor* output) {
  std::vector<int> output_shape_vector;

  if (begin->type == kTfLiteInt32) {
    TF_LITE_ENSURE_STATUS(CalculateOutputShapeVector<int32_t>(
        context, input, begin, size, &output_shape_vector));
  } else if (begin->type == kTfLiteInt64) {
    TF_LITE_ENSURE_STATUS(CalculateOutputShapeVector<int64_t>(
        context, input, begin, size, &output_shape_vector));
  } else {
    context->ReportError(
        context, "Type %d is currently not supported by Slice.", begin->type);
    return kTfLiteError;
  }

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(output_shape_vector.size());
  std::copy(output_shape_vector.begin(), output_shape_vector.end(),
            output_shape->data);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* begin;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBeginTensor, &begin));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Ensure validity of input tensor and its dimension.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context,
                 begin->type == kTfLiteInt32 || begin->type == kTfLiteInt64);
  TF_LITE_ENSURE(context,
                 size->type == kTfLiteInt32 || size->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumElements(size));
  TF_LITE_ENSURE_MSG(context, NumDimensions(input) <= kMaxDim,
                     "Slice op only supports 1D-5D input arrays.");

  // Postpone allocation of output if any of the indexing tensors is not
  // constant, or the input tensor has dynamic dimension.
  if (!(IsConstantTensor(begin) && IsConstantTensor(size)) ||
      HasUnspecifiedDimension(input)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  return ResizeOutputShape(context, input, begin, size, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* begin;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBeginTensor, &begin));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputShape(context, input, begin, size, output));
  }

  std::vector<int> begins;
  begins.reserve(kMaxDim);
  std::vector<int> sizes;
  sizes.reserve(kMaxDim);

  for (int i = NumDimensions(input); i < kMaxDim; ++i) {
    begins.push_back(0);
    sizes.push_back(1);
  }

  if (begin->type == kTfLiteInt32) {
    GetBeginAndSizeVectors<int32_t>(NumDimensions(input), begin, size, &begins,
                                    &sizes);
  } else if (begin->type == kTfLiteInt64) {
    GetBeginAndSizeVectors<int64_t>(NumDimensions(input), begin, size, &begins,
                                    &sizes);
  } else {
    context->ReportError(
        context, "Type %d is currently not supported by Slice.", begin->type);
    return kTfLiteError;
  }

  // The Slice op implementation only accepts 5-D sizes. That constraint is, for
  // the present, maintained here.
  //
  // The dimensions in the kernel used to be in reverse-order, and TFLite
  // arranged the begins and sizes vectors accordingly. This macro incorporates
  // the needed reversing.
#define TF_LITE_SLICE(data_type)                                               \
  {                                                                            \
    TF_LITE_ENSURE_EQ(context, begins.size(), kMaxDim);                        \
    TF_LITE_ENSURE_EQ(context, sizes.size(), kMaxDim);                         \
    tflite::SliceParams op_params;                                             \
    op_params.begin_count = kMaxDim;                                           \
    op_params.size_count = kMaxDim;                                            \
    for (int i = 0; i < kMaxDim; ++i) {                                        \
      op_params.begin[i] = begins[i];                                          \
      op_params.size[i] = sizes[i];                                            \
    }                                                                          \
                                                                               \
    if (kernel_type == kGenericOptimized) {                                    \
      optimized_ops::Slice<data_type>(op_params, GetTensorShape(input), input, \
                                      GetTensorShape(output), output);         \
    } else {                                                                   \
      reference_ops::Slice<data_type>(op_params, GetTensorShape(input), input, \
                                      GetTensorShape(output), output);         \
    }                                                                          \
  }

  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_SLICE(float);
      break;
    case kTfLiteInt32:
      TF_LITE_SLICE(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_SLICE(int64_t);
      break;
    case kTfLiteInt8:
      TF_LITE_SLICE(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_SLICE(int16_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_SLICE(uint8_t);
      break;
    case kTfLiteBool:
      TF_LITE_SLICE(bool);
      break;
    case kTfLiteString:
      TF_LITE_SLICE(string);
      break;
    default:
      context->ReportError(
          context, "Type %d is currently not supported by Slice.", input->type);
      return kTfLiteError;
  }
#undef TF_LITE_SLICE
  return kTfLiteOk;
}

}  // namespace slice

TfLiteRegistration* Register_SLICE_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, slice::Prepare,
                                 slice::Eval<slice::kReference>};
  return &r;
}

TfLiteRegistration* Register_SLICE() {
  static TfLiteRegistration r = {nullptr, nullptr, slice::Prepare,
                                 slice::Eval<slice::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cumsum {

static const int kInputTensor = 0;
static const int kAxisTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus ValidateCumsumFlattenedShape(TfLiteContext* context,
                                          const RuntimeShape& input_shape,
                                          int axis) {
  // optimized_ops::CumSum flattens the input shape into [outer, depth, inner].
  // Check those products using RuntimeShape's checked range-product helper
  // instead of duplicating shape-product logic in this kernel.
  constexpr size_t kMaxFlattenedDim =
      static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const auto product_fits_index = [&](int start, int end) {
    size_t checked_product = 0;
    return input_shape.CheckedNumElementsInRange(start, end, checked_product) &&
           checked_product <= kMaxFlattenedDim;
  };
  TF_LITE_ENSURE_MSG(
      context,
      product_fits_index(0, axis) && product_fits_index(axis, axis + 1) &&
          product_fits_index(axis + 1, input_shape.DimensionsCount()),
      "Cumsum input shape is too large.");
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteInt32 ||
                              input->type == kTfLiteFloat32 ||
                              input->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis_tensor = GetInput(context, node, kAxisTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  auto* params = reinterpret_cast<TfLiteCumsumParams*>(node->builtin_data);

  int axis = 0;
  TF_LITE_ENSURE_OK(
      context, kernel_utils::ReadAndNormalizeAxis(context, *axis_tensor,
                                                  NumDimensions(input), axis));

  const RuntimeShape input_shape = GetTensorShape(input);
  if (input_shape.HasZeroDimension()) {
    return kTfLiteOk;
  }

  TF_LITE_ENSURE_OK(context,
                    ValidateCumsumFlattenedShape(context, input_shape, axis));

  switch (input->type) {
    case kTfLiteInt32: {
      optimized_ops::CumSum(GetTensorData<int>(input), input_shape, axis,
                            params->exclusive, params->reverse,
                            GetTensorData<int>(output));
      break;
    }
    case kTfLiteInt64: {
      optimized_ops::CumSum(GetTensorData<int64_t>(input), input_shape, axis,
                            params->exclusive, params->reverse,
                            GetTensorData<int64_t>(output));
      break;
    }
    case kTfLiteFloat32: {
      optimized_ops::CumSum(GetTensorData<float>(input), input_shape, axis,
                            params->exclusive, params->reverse,
                            GetTensorData<float>(output));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context,
          "Unsupported input type, cumsum only supports int32 & float32.");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace cumsum

TfLiteRegistration* Register_CUMSUM() {
  static TfLiteRegistration r = {nullptr, nullptr, cumsum::Prepare,
                                 cumsum::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

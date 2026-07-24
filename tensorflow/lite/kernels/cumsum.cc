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

#include "tensorflow/lite/kernels/internal/reference/cumsum.h"

#include <cstdint>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cumsum {

static const int kInputTensor = 0;
static const int kAxisTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus CheckedDimensionProduct(TfLiteContext* context,
                                     const TfLiteTensor* tensor, int begin,
                                     int end, int* product) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, product != nullptr);
  TF_LITE_ENSURE(context, begin >= 0);
  TF_LITE_ENSURE(context, end >= begin);
  TF_LITE_ENSURE(context, end <= NumDimensions(tensor));

  CheckedInt<int> checked_product = 1;
  for (int i = begin; i < end; ++i) {
    const int dim = SizeOfDimension(tensor, i);
    TF_LITE_ENSURE(context, dim >= 0);
    checked_product *= dim;
    TF_LITE_ENSURE_MSG(context, !checked_product.Overflow(),
                       "Dimension product overflows int.");
  }
  *product = checked_product.Value();
  return kTfLiteOk;
}

TfLiteStatus ReadAndNormalizeAxis(TfLiteContext* context,
                                  const TfLiteTensor* input,
                                  const TfLiteTensor* axis_tensor, int* axis) {
  TF_LITE_ENSURE(context, axis != nullptr);
  int axis_value = *GetTensorData<int>(axis_tensor);
  if (axis_value < 0) axis_value += NumDimensions(input);

  TF_LITE_ENSURE_MSG(context,
                     axis_value >= 0 && axis_value < NumDimensions(input),
                     "Invalid axis: %d", axis_value);
  *axis = axis_value;
  return kTfLiteOk;
}

TfLiteStatus ValidateCumsumTensorData(TfLiteContext* context,
                                      const TfLiteTensor* input,
                                      const TfLiteTensor* output, int axis) {
  int input_elements = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(input, input_elements) == kTfLiteOk,
                     "Input element count overflows int.");
  int output_elements = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(output, output_elements) == kTfLiteOk,
                     "Output element count overflows int.");

  int inner = 0;
  TF_LITE_ENSURE_OK(context,
                    CheckedDimensionProduct(context, input, 0, axis, &inner));
  int outer = 0;
  TF_LITE_ENSURE_OK(context,
                    CheckedDimensionProduct(context, input, axis + 1,
                                            NumDimensions(input), &outer));
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
  TF_LITE_ENSURE_OK(context,
                    ReadAndNormalizeAxis(context, input, axis_tensor, &axis));
  TF_LITE_ENSURE_OK(context,
                    ValidateCumsumTensorData(context, input, output, axis));
  int input_elements = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(input, input_elements) == kTfLiteOk,
                     "Input element count overflows int.");
  if (input_elements == 0) {
    return kTfLiteOk;
  }

  switch (input->type) {
    case kTfLiteInt32: {
      reference_ops::CumSum(GetTensorData<int>(input), GetTensorShape(input),
                            axis, params->exclusive, params->reverse,
                            GetTensorData<int>(output));
      break;
    }
    case kTfLiteInt64: {
      reference_ops::CumSum(GetTensorData<int64_t>(input),
                            GetTensorShape(input), axis, params->exclusive,
                            params->reverse, GetTensorData<int64_t>(output));
      break;
    }
    case kTfLiteFloat32: {
      optimized_ops::CumSum(GetTensorData<float>(input), GetTensorShape(input),
                            axis, params->exclusive, params->reverse,
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

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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  int32_t input_zero_point;
  int32_t input_range_radius;
  int32_t input_multiplier;
  int input_left_shift;
};

TfLiteStatus CalculateArithmeticOpData(TfLiteContext* context, TfLiteNode* node,
                                       OpData* data) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    // The number if input integer bits is set to be consistent with the
    // required value in reference_integer_ops::Tanh
    static constexpr int kInputIntegerBits = 4;
    const double input_real_multiplier =
        static_cast<double>(input->params.scale) *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    const double q = std::frexp(input_real_multiplier, &data->input_left_shift);
    data->input_multiplier = static_cast<int32_t>(TfLiteRound(q * (1ll << 31)));

    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 31);
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  OpData data;
  CalculateArithmeticOpData(context, node, &data);

  if (input->type == kTfLiteFloat32) {
    switch (output->type) {
      case kTfLiteFloat32: {
        reference_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
        return kTfLiteOk;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                           TfLiteTypeGetName(input->type),
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt8) {
    switch (output->type) {
      case kTfLiteInt8: {
        reference_integer_ops::Tanh(
            input->params.zero_point, data.input_range_radius,
            data.input_multiplier, data.input_left_shift,
            NumElements(input->dims), GetTensorData<int8_t>(input),
            GetTensorData<int8_t>(output));
        return kTfLiteOk;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                           TfLiteTypeGetName(input->type),
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                       TfLiteTypeGetName(input->type),
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_TANH() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/activations::TanhEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite

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

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace ceil {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE(
      context,
      input->type == kTfLiteFloat32 || input->type == kTfLiteBFloat16 ||
          input->type == kTfLiteFloat16 || input->type == kTfLiteInt32 ||
          input->type == kTfLiteInt16 || input->type == kTfLiteInt8);

  bool is_quantized = input->quantization.type != kTfLiteNoQuantization;
  if ((input->type == kTfLiteInt8 && is_quantized) ||
      (input->type == kTfLiteInt16 && is_quantized)) {
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);

    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);

    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);

    if (input->type == kTfLiteInt16) {
      // In case of int16, quantization is symmetic and
      // zero point should be zero.
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }

  output->type = input->type;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
TfLiteStatus EvalCeil(const TfLiteTensor* input, TfLiteTensor* output) {
  optimized_ops::Ceil<T>(GetTensorShape(input), GetTensorData<T>(input),
                         GetTensorShape(output), GetTensorData<T>(output));
  return TfLiteStatus::kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalCeilQuantized(const TfLiteTensor* input,
                               TfLiteTensor* output) {
  const int32_t input_zero_point = input->params.zero_point;
  const float input_scale = input->params.scale;
  const int32_t output_zero_point = output->params.zero_point;
  const float output_scale = output->params.scale;

  const int num_elements = NumElements(input);
  const T* input_data = GetTensorData<T>(input);
  T* output_data = GetTensorData<T>(output);

  for (int i = 0; i < num_elements; ++i) {
    const float input_value = (input_data[i] - input_zero_point) * input_scale;
    const float ceiled_value = std::ceil(input_value);
    int32_t quantized_value =
        static_cast<int32_t>(std::round(ceiled_value / output_scale)) +
        output_zero_point;
    // Clamp the value to fit in the range of the output type
    quantized_value =
        std::min(std::max(quantized_value,
                          static_cast<int32_t>(std::numeric_limits<T>::min())),
                 static_cast<int32_t>(std::numeric_limits<T>::max()));

    output_data[i] = static_cast<T>(quantized_value);
  }
  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(context, EvalCeil<float>(input, output));
      break;
    case kTfLiteFloat16:
      TF_LITE_ENSURE_OK(context, EvalCeil<Eigen::half>(input, output));
      break;
    case kTfLiteBFloat16:
      TF_LITE_ENSURE_OK(context, EvalCeil<Eigen::bfloat16>(input, output));
      break;
    case kTfLiteInt32:
      TF_LITE_ENSURE_OK(context, EvalCeil<int32_t>(input, output));
      break;
    case kTfLiteInt8: {
      if (output->quantization.type == kTfLiteNoQuantization)
        TF_LITE_ENSURE_OK(context, EvalCeil<int8_t>(input, output));
      else  
        TF_LITE_ENSURE_OK(context, EvalCeilQuantized<int8_t>(input, output));
      } break;
    case kTfLiteInt16: {
      if (output->quantization.type == kTfLiteNoQuantization)
        TF_LITE_ENSURE_OK(context, EvalCeil<int16_t>(input, output));
      else
        TF_LITE_ENSURE_OK(context, EvalCeilQuantized<int16_t>(input, output));
    } break;
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported datatype for ceil output: %s",
                         TfLiteTypeGetName(output->type));
      return TfLiteStatus::kTfLiteError;
    }
  }
  return kTfLiteOk;
}
}  // namespace ceil

TfLiteRegistration* Register_CEIL() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr, ceil::Prepare, ceil::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

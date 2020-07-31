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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace arg_min_max {

constexpr int kInputTensor = 0;
constexpr int kAxis = 1;
constexpr int kOutputTensor = 0;

template <typename T1, typename T2, typename T3>
inline void ArgMinMaxHelper(const RuntimeShape& input1_shape,
                            const T1* input1_data, const T3* input2_data,
                            const RuntimeShape& output_shape, T2* output_data,
                            bool is_arg_max) {
  if (is_arg_max) {
    reference_ops::ArgMinMax(input1_shape, input1_data, input2_data,
                             output_shape, output_data, micro::Greater());
  } else {
    reference_ops::ArgMinMax(input1_shape, input1_data, input2_data,
                             output_shape, output_data, micro::Less());
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node, bool is_arg_max) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      tflite::micro::GetEvalInput(context, node, kAxis);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

#define TF_LITE_ARG_MIN_MAX(data_type, axis_type, output_type)       \
  ArgMinMaxHelper(tflite::micro::GetTensorShape(input),              \
                  tflite::micro::GetTensorData<data_type>(input),    \
                  tflite::micro::GetTensorData<axis_type>(axis),     \
                  tflite::micro::GetTensorShape(output),             \
                  tflite::micro::GetTensorData<output_type>(output), \
                  is_arg_max)
  if (axis->type == kTfLiteInt32) {
    if (output->type == kTfLiteInt32) {
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
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Only float32, uint8_t and int8_t are "
                             "supported currently, got %s.",
                             TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
    } else {
      TF_LITE_KERNEL_LOG(context,
                         "Only int32_t are supported currently, got %s.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context, "Only int32_t are supported currently, got %s.",
                       TfLiteTypeGetName(axis->type));
    return kTfLiteError;
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

TfLiteRegistration Register_ARG_MAX() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/arg_min_max::ArgMaxEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_ARG_MIN() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/arg_min_max::ArgMinEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

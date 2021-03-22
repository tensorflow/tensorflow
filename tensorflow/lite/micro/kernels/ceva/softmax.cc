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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_common.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h"
#endif

namespace tflite {
namespace {

// Takes a tensor and performs softmax along the last dimension.
void SoftmaxFloatCEVA(const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      const SoftmaxParams& op_data) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  float* output_data = tflite::micro::GetTensorData<float>(output);

  const float beta = static_cast<float>(op_data.beta);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  int outer_size_mcps = outer_size;
  int depth_mcps = depth;

#ifdef MCPS_MEASUREMENT
  MCPS_START_ONE;
#endif
  for (int i = 0; i < outer_size; ++i) {
    CEVA_TFLM_Softmax_Float32(&input_data[i * depth], &output_data[i * depth],
                              beta, depth);
  }
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_Softmax_Float32 %d times, inetrnal loop = %d",
      outer_size_mcps, depth_mcps);
#endif
}

TfLiteStatus SoftmaxQuantizedCEVA(TfLiteContext* context,
                                  const TfLiteEvalTensor* input,
                                  TfLiteEvalTensor* output,
                                  const SoftmaxParams& op_data) {
  if (input->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    } else {
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);

      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

      const int32_t input_beta_multiplier =
          static_cast<int32_t>(op_data.input_multiplier);
      const int32_t input_beta_left_shift =
          static_cast<int32_t>(op_data.input_left_shift);
      const int trailing_dim = input_shape.DimensionsCount() - 1;
      const int outer_size =
          MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
      const int depth =
          MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
      int outer_size_mcps = outer_size;
      int depth_mcps = depth;

      if (depth > CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL) {
        TF_LITE_KERNEL_LOG(context, "Scratch size (%d) less that required (%d)",
                           CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL, depth);
        return kTfLiteError;
      }

#ifdef MCPS_MEASUREMENT
      MCPS_START_ONE;
#endif
      for (int i = 0; i < outer_size; ++i) {
        CEVA_TFLM_Softmax_Int8(&input_data[i * depth], &output_data[i * depth],
                               input_beta_multiplier, input_beta_left_shift,
                               depth, CEVA_TFLM_KERNELS_SCRATCH);
      }
#ifdef MCPS_MEASUREMENT
      MCPS_STOP_ONE(
          "Test params:Call CEVA_TFLM_Softmax_Int8 %d times, inetrnal loop = "
          "%d",
          outer_size_mcps, depth_mcps);
#endif
    }
  } else {
    tflite::reference_ops::SoftmaxInt16(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalCEVA(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      SoftmaxFloatCEVA(input, output, op_data);
      return kTfLiteOk;
    }
    case kTfLiteInt8:
    case kTfLiteInt16: {
      return SoftmaxQuantizedCEVA(context, input, output, op_data);
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return SoftmaxEvalCEVA(context, node);
#else
  return SoftmaxEval(context, node);  // reference fallback
#endif
}
}  // namespace

TfLiteRegistration Register_SOFTMAX() {
  return {/*init=*/SoftmaxInit,
          /*free=*/nullptr,
          /*prepare=*/SoftmaxPrepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

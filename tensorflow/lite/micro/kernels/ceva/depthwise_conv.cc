/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_common.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h"
#endif

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, const OpDataConv& data,
               const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
               const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params = DepthwiseConvParamsFloat(*params, data);

  const float *input_data, *filter_data, *bias_data;
  float* output_data;
  input_data = tflite::micro::GetTensorData<float>(input);
  filter_data = tflite::micro::GetTensorData<float>(filter);
  bias_data = tflite::micro::GetTensorData<float>(bias);
  output_data = tflite::micro::GetTensorData<float>(output);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int pad_width = data.padding.width;
  const int pad_height = data.padding.height;
  const int depth_multiplier = params->depth_multiplier;

  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;

#ifdef MCPS_MEASUREMENT
  MCPS_START_ONE;
#endif
  for (int k = 0; k < batches; k++) {
    CEVA_TFLM_DepthwiseConv_Float32(
        // 1,
        stride_width, stride_height, pad_width, pad_height, depth_multiplier,
        input_height, input_width, input_depth,
        &input_data[k * input_height * input_width * input_depth],
        filter_height, filter_width, filter_depth, filter_data, bias_data,
        output_height, output_width, output_depth,
        &output_data[k * output_height * output_width * output_depth],
        dilation_width_factor, dilation_height_factor, output_activation_min,
        output_activation_max

    );
  }
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_DepthwiseConv_Float32 %d times, inetrnal "
      "loop = %dx%dx%dx%dx%dx%d",
      batches, output_height, output_width, filter_height, filter_width,
      output_depth, input_depth);
#endif
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params,
                             const OpDataConv& data,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output) {
  DepthwiseParams op_params = DepthwiseConvParamsQuantized(*params, data);

  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  const int8_t* input_data;
  const int8_t* filter_data;
  const int32_t* bias_data;
  int8_t* output_data;
  const int32_t input_offset = op_params.input_offset;
  const int32_t output_offset = op_params.output_offset;

  input_data = tflite::micro::GetTensorData<int8_t>(input);
  filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  output_data = tflite::micro::GetTensorData<int8_t>(output);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int pad_width = data.padding.width;
  const int pad_height = data.padding.height;
  const int depth_multiplier = params->depth_multiplier;

  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;

  if ((input_depth * 4) > CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL) {
    TF_LITE_KERNEL_LOG(context, "Scratch size (%d) less that required (%d)",
                       CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL, (input_depth * 4));
  }

#ifdef MCPS_MEASUREMENT
  MCPS_START_ONE;
#endif
  for (int k = 0; k < batches; k++) {
    CEVA_TFLM_DepthwiseConvPerChannel_int8(
        // 1,
        stride_width, stride_height, pad_width, pad_height, depth_multiplier,
        input_offset, output_offset, data.per_channel_output_multiplier,
        data.per_channel_output_shift, input_height, input_width, input_depth,
        &input_data[k * input_height * input_width * input_depth],
        filter_height, filter_width, filter_depth, filter_data, bias_data,
        output_height, output_width, output_depth,
        &output_data[k * output_height * output_width * output_depth],
        CEVA_TFLM_KERNELS_SCRATCH, dilation_width_factor,
        dilation_height_factor, op_params.quantized_activation_min,
        op_params.quantized_activation_max

    );
  }
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_DepthwiseConvPerChannel_int8 %d times, "
      "inetrnal loop = %dx%dx%dx%dx%dx%d",
      batches, output_height, output_width, filter_height, filter_width,
      output_depth, input_depth);
#endif
}

TfLiteStatus EvalCEVA(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, data, input, filter, bias,
                              output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus DepthWiseConvEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return EvalCEVA(context, node);
#else
  return Eval(context, node);  // reference fallback
#endif
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/DepthwiseConvPrepare,
          /*invoke=*/DepthWiseConvEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

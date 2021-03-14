/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h"
#endif

#if defined(CEVA_BX1) || defined(CEVA_SP500)
extern int32_t* CEVA_TFLM_KERNELS_SCRATCH;
extern int32_t CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL;
#endif

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, const OpDataConv& data,
                   const TfLiteEvalTensor* input,
                   const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
                   TfLiteEvalTensor* im2col, TfLiteEvalTensor* hwcn_weights,
                   TfLiteEvalTensor* output) {
  const int32_t input_offset = -data.input_zero_point;
  const int32_t filter_offset = -data.filter_zero_point;
  const int32_t output_offset = data.output_zero_point;

  // TODO(b/154032858): Investigate removing extra copies.
  ConvParams op_params = ConvParamsQuantized(*params, data);

  reference_ops::Conv(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<uint8_t>(input),
                      tflite::micro::GetTensorShape(filter),
                      tflite::micro::GetTensorData<uint8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetTensorData<int32_t>(bias),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<uint8_t>(output),
                      tflite::micro::GetTensorShape(im2col),
                      tflite::micro::GetTensorData<uint8_t>(im2col), nullptr);
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, const OpDataConv& data,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output,
                             TfLiteEvalTensor* im2col) {
  // TODO(b/154032858): Investigate removing extra copies.

  ConvParams op_params = ConvParamsQuantized(*params, data);
  const int32_t input_offset = op_params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = op_params.output_offset;

  const int8_t *input_data, *filter_data;
  const int32_t* bias_data;
  int8_t* output_data;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const RuntimeShape& im2col_shape = tflite::micro::GetTensorShape(im2col);

  const int stride_width = op_params.stride_width;
  const int stride_height = op_params.stride_height;
  const int dilation_width_factor = op_params.dilation_width_factor;
  const int dilation_height_factor = op_params.dilation_height_factor;
  const int pad_width = op_params.padding_values.width;
  const int pad_height = op_params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth_Dims3 = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth_Dims3 = output_shape.Dims(3);

  input_data = tflite::micro::GetTensorData<int8_t>(input);
  filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  output_data = tflite::micro::GetTensorData<int8_t>(output);

  int sizeof_scr = filter_depth;
  if (sizeof_scr < output_depth_Dims3) sizeof_scr = output_depth_Dims3;

  if (sizeof_scr > CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL) {
    TF_LITE_KERNEL_LOG(context, "Scratch size (%d) less that required (%d)",
                       CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL, sizeof_scr);
  }

#ifdef MCPS_MEASUREMENT
  MCPS_START_ONE;
#endif
  for (int k = 0; k < batches; k++) {
    CEVA_TFLM_ConvPerChannel_Int8(

        stride_width, stride_height, pad_width,
        pad_height,  // const int depth_multiplier,
        input_offset, output_offset, data.per_channel_output_multiplier,
        data.per_channel_output_shift, input_height, input_width,
        input_depth_Dims3, input_depth,

        &input_data[k * input_height * input_width * input_depth_Dims3],
        filter_height, filter_width, filter_depth, filter_data, bias_data,
        output_height, output_width, output_depth_Dims3, output_depth,

        &output_data[k * output_height * output_width * output_depth_Dims3],
        CEVA_TFLM_KERNELS_SCRATCH, dilation_width_factor,
        dilation_height_factor, data.output_activation_min,
        data.output_activation_max);
  }
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_ConvPerChannel_Int8 %d times, inetrnal loop "
      "= %dx%dx%dx%dx%dx%d",
      batches, output_height, output_width, filter_height, filter_width,
      output_depth, input_depth);
#endif
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, const OpDataConv& data,
               const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
               const TfLiteEvalTensor* bias, TfLiteEvalTensor* im2col,
               TfLiteEvalTensor* hwcn_weights, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  // TODO(b/154032858): Investigate removing extra copies.
  ConvParams op_params = ConvParamsFloat(*params, data);

  const float *input_data, *filter_data, *bias_data, *im2col_data;
  float* output_data;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const RuntimeShape& im2col_shape = tflite::micro::GetTensorShape(im2col);

  const int stride_width = op_params.stride_width;
  const int stride_height = op_params.stride_height;
  const int dilation_width_factor = op_params.dilation_width_factor;
  const int dilation_height_factor = op_params.dilation_height_factor;
  const int pad_width = op_params.padding_values.width;
  const int pad_height = op_params.padding_values.height;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth_Dims3 = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth_Dims3 = output_shape.Dims(3);

  input_data = tflite::micro::GetTensorData<float>(input);
  filter_data = tflite::micro::GetTensorData<float>(filter);
  bias_data = tflite::micro::GetTensorData<float>(bias);
  output_data = tflite::micro::GetTensorData<float>(output);
  im2col_data = tflite::micro::GetTensorData<float>(im2col);

#ifdef MCPS_MEASUREMENT
  MCPS_START_ONE;
#endif
  for (int k = 0; k < batches; k++) {
    CEVA_TFLM_Conv_Float32(
        stride_width, stride_height, pad_width, pad_height, input_height,
        input_width, input_depth_Dims3, input_depth,
        &input_data[k * input_height * input_width * input_depth_Dims3],
        filter_height, filter_width, filter_depth, filter_data, bias_data,
        output_height, output_width, output_depth_Dims3, output_depth,
        &output_data[k * output_height * output_width * output_depth_Dims3],
        dilation_width_factor, dilation_height_factor, output_activation_min,
        output_activation_max

    );
  }
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_Conv_Float32 %d times, inetrnal loop = "
      "%dx%dx%dx%dx%dx%d",
      batches, output_height, output_width, filter_height, filter_width,
      output_depth, input_depth);
#endif
}

TfLiteStatus EvalCEVA(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, data, input, filter, bias, nullptr,
                nullptr, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, data, input, filter, bias,
                              output, nullptr);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, data, input, filter, bias, nullptr,
                    nullptr, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return EvalCEVA(context, node);
#else
  return Eval(context, node);  // reference fallback
#endif
}
}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ConvPrepare,
          /*invoke=*/ConvEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "CMSIS/NN/Include/arm_nn_types.h"
#include "CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  int32_t buf_size = 0;
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kConvInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kConvWeightsTensor);
  const TfLiteTensor* output = GetOutput(context, node, kConvOutputTensor);

  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);

  // Initialize cmsis_nn input dimensions
  cmsis_nn_dims input_dims;
  input_dims.n = MatchingDim(input_shape, 0, output_shape, 0);
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input_shape.Dims(3);

  // Initialize cmsis_nn filter dimensions
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_shape.Dims(3);
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = input_dims.c;

  // Initialize cmsis_nn output dimensions
  cmsis_nn_dims output_dims;
  output_dims.n = input_dims.n;
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output_shape.Dims(3);

  // Dynamically allocate per-channel quantization parameters.
  // TODO(#42883): This allocation is done even for non-int8 cases to get around
  // a bug in kernel_util.cc which incorrectly uses per_channel_output_shift in
  // non-int8 cases. Protect this section with a if (input->type == kTfLiteInt8)
  // when the issue is fixed.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->reference_op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->reference_op_data.per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_dims.w, input_dims.h, filter_dims.w,
      filter_dims.h, output_dims.w, output_dims.h, input->type,
      &data->reference_op_data));

  if (input->type == kTfLiteInt8) {
    // Initialize cmsis_nn convolution parameters
    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = -input->params.zero_point;
    conv_params.output_offset = output->params.zero_point;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.dilation.h = params.dilation_height_factor;
    conv_params.dilation.w = params.dilation_width_factor;
    conv_params.padding.h = data->reference_op_data.padding.height;
    conv_params.padding.w = data->reference_op_data.padding.width;
    conv_params.activation.min = data->reference_op_data.output_activation_min;
    conv_params.activation.max = data->reference_op_data.output_activation_max;

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(
        &conv_params, &input_dims, &filter_dims, &output_dims);
  }

  if (buf_size > 0) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, buf_size, &data->buffer_idx));
  } else {
    data->buffer_idx = -1;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, const TfLiteConvParams& params,
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output, TfLiteEvalTensor* im2col) {
  cmsis_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;
  // TODO(#43557) Remove checks for dilation and call to reference
  // implementation when dilation is supported in the optimized implementation
  // by CMSIS-NN.
  if (conv_params.dilation.h == 1 && conv_params.dilation.w == 1) {
    // Initialize cmsis_nn convolution parameters
    conv_params.input_offset = -data.reference_op_data.input_zero_point;
    conv_params.output_offset = data.reference_op_data.output_zero_point;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.padding.h = data.reference_op_data.padding.height;
    conv_params.padding.w = data.reference_op_data.padding.width;
    conv_params.activation.min = data.reference_op_data.output_activation_min;
    conv_params.activation.max = data.reference_op_data.output_activation_max;

    // Initialize cmsis_nn per channel quantization parameters
    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = const_cast<int32_t*>(
        data.reference_op_data.per_channel_output_multiplier);
    quant_params.shift =
        const_cast<int32_t*>(data.reference_op_data.per_channel_output_shift);

    RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

    // Consistency check.
    TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (tflite::micro::GetTensorData<int8_t>(bias)) {
      TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    // Initialize cmsis_nn dimensions
    // Input
    cmsis_nn_dims input_dims;
    input_dims.n = batch_size;
    input_dims.h = input_shape.Dims(1);
    input_dims.w = input_shape.Dims(2);
    input_dims.c = input_depth;

    // Filter
    cmsis_nn_dims filter_dims;
    filter_dims.n = output_depth;
    filter_dims.h = filter_shape.Dims(1);
    filter_dims.w = filter_shape.Dims(2);
    filter_dims.c = input_depth;

    // Bias
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_depth;

    // Output
    cmsis_nn_dims output_dims;
    output_dims.n = batch_size;
    output_dims.h = output_shape.Dims(1);
    output_dims.w = output_shape.Dims(2);
    output_dims.c = output_depth;

    // Initialize cmsis_nn context
    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    ctx.size = 0;

    if (data.buffer_idx > -1) {
      ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
      // Note: ctx.size is currently not used in cmsis_nn.
      // The buffer should be allocated in the Prepare function through
      // arm_convolve_wrapper_s8_get_buffer_size
    }

    // arm_convolve_wrapper_s8 dispatches the optimized kernel accordingly with
    // the parameters passed
    TFLITE_DCHECK_EQ(
        arm_convolve_wrapper_s8(
            &ctx, &conv_params, &quant_params, &input_dims,
            tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
            tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
            tflite::micro::GetTensorData<int32_t>(bias), &output_dims,
            tflite::micro::GetTensorData<int8_t>(output)),
        ARM_MATH_SUCCESS);
  } else {
    reference_integer_ops::ConvPerChannel(
        ConvParamsQuantized(params, data.reference_op_data),
        data.reference_op_data.per_channel_output_multiplier,
        data.reference_op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

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
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8:
      return EvalQuantizedPerChannel(context, node, params, data, input, filter,
                                     bias, output, nullptr);
      break;
    case kTfLiteUInt8: {
      reference_ops::Conv(ConvParamsQuantized(params, data.reference_op_data),
                          tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<uint8_t>(input),
                          tflite::micro::GetTensorShape(filter),
                          tflite::micro::GetTensorData<uint8_t>(filter),
                          tflite::micro::GetTensorShape(bias),
                          tflite::micro::GetTensorData<int32_t>(bias),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<uint8_t>(output),
                          tflite::micro::GetTensorShape(nullptr), nullptr,
                          nullptr);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

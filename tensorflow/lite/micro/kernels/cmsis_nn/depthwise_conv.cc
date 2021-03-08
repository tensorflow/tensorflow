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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
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

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));

  const TfLiteTensor* input =
      GetInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteTensor* filter =
      GetInput(context, node, kDepthwiseConvWeightsTensor);
  TfLiteTensor* output = GetOutput(context, node, kDepthwiseConvOutputTensor);

  const TfLiteType data_type = input->type;
  int input_width = SizeOfDimension(input, 2);
  int input_height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);
  int output_width = SizeOfDimension(output, 2);
  int output_height = SizeOfDimension(output, 1);

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    // All per-channel quantized tensors need valid zero point and scale arrays.
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  // Allocate memory for per-channel quantization parameters
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

  data->reference_op_data.per_channel_output_multiplier =
      reinterpret_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->reference_op_data.per_channel_output_shift =
      reinterpret_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  TF_LITE_ENSURE_STATUS(CalculateOpDataDepthwiseConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, data_type,
      &data->reference_op_data));

  if (input->type == kTfLiteInt8) {
    RuntimeShape input_shape = GetTensorShape(input);
    RuntimeShape output_shape = GetTensorShape(output);
    RuntimeShape filter_shape = GetTensorShape(filter);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(output_shape, 3, filter_shape, 3);
    TFLITE_DCHECK_EQ(batch_size, 1); /* Only batch = 1 is supported */

    cmsis_nn_dims input_dims;
    input_dims.n = batch_size;
    input_dims.h = input_height;
    input_dims.w = input_width;
    input_dims.c = input_shape.Dims(3);

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = filter_height;
    filter_dims.w = filter_width;
    filter_dims.c = output_depth;

    cmsis_nn_dims output_dims;
    output_dims.n = batch_size;
    output_dims.h = output_height;
    output_dims.w = output_width;
    output_dims.c = output_depth;

    cmsis_nn_dw_conv_params dw_conv_params;
    dw_conv_params.padding.h = data->reference_op_data.padding.height;
    dw_conv_params.padding.w = data->reference_op_data.padding.width;

    const int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
        &dw_conv_params, &input_dims, &filter_dims, &output_dims);

    if (buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, buf_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }
  return kTfLiteOk;
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteDepthwiseConvParams& params,
                             const OpData& data, const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output) {
  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.dilation.h = params.dilation_height_factor;
  dw_conv_params.dilation.w = params.dilation_width_factor;
  // Call to reference implementation can be removed when dilation is supported
  // in the optimized implementations.
  if (1 == dw_conv_params.dilation.h && 1 == dw_conv_params.dilation.w) {
    dw_conv_params.input_offset = -data.reference_op_data.input_zero_point;
    dw_conv_params.output_offset = data.reference_op_data.output_zero_point;
    dw_conv_params.stride.h = params.stride_height;
    dw_conv_params.stride.w = params.stride_width;
    dw_conv_params.padding.h = data.reference_op_data.padding.height;
    dw_conv_params.padding.w = data.reference_op_data.padding.width;
    // TODO(b/130439627): Use calculated value for clamping.
    dw_conv_params.activation.min = std::numeric_limits<int8_t>::min();
    dw_conv_params.activation.max = std::numeric_limits<int8_t>::max();
    dw_conv_params.ch_mult = params.depth_multiplier;

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier =
        data.reference_op_data.per_channel_output_multiplier;
    quant_params.shift = data.reference_op_data.per_channel_output_shift;

    RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

    TFLITE_DCHECK_LE(dw_conv_params.activation.min,
                     dw_conv_params.activation.max);

    const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

    if (tflite::micro::GetTensorData<int8_t>(bias)) {
      TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    cmsis_nn_dims input_dims;
    input_dims.n = batch_size;
    input_dims.h = input_shape.Dims(1);
    input_dims.w = input_shape.Dims(2);
    input_dims.c = input_shape.Dims(3);

    cmsis_nn_dims filter_dims;
    filter_dims.n = filter_shape.Dims(0);
    filter_dims.h = filter_shape.Dims(1);
    filter_dims.w = filter_shape.Dims(2);
    filter_dims.c = output_depth;

    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_depth;

    cmsis_nn_dims output_dims;
    output_dims.n = batch_size;
    output_dims.h = output_shape.Dims(1);
    output_dims.w = output_shape.Dims(2);
    output_dims.c = output_depth;

    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    /* 'size' is unused */
    ctx.size = 0;

    if (data.buffer_idx > -1) {
      ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    }

    TFLITE_DCHECK_EQ(
        arm_depthwise_conv_wrapper_s8(
            &ctx, &dw_conv_params, &quant_params, &input_dims,
            tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
            tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
            tflite::micro::GetTensorData<int32_t>(bias), &output_dims,
            tflite::micro::GetTensorData<int8_t>(output)),
        ARM_MATH_SUCCESS);
  } else {
    reference_integer_ops::DepthwiseConvPerChannel(
        DepthwiseConvParamsQuantized(params, data.reference_op_data),
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
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   const TfLiteDepthwiseConvParams& params, const OpData& data,
                   const TfLiteEvalTensor* input,
                   const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
                   TfLiteEvalTensor* output) {
  tflite::DepthwiseParams op_params =
      DepthwiseConvParamsQuantized(params, data.reference_op_data);

  if (1 == op_params.dilation_width_factor &&
      1 == op_params.dilation_height_factor) {
    RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    arm_depthwise_conv_u8_basic_ver1(
        tflite::micro::GetTensorData<uint8_t>(input), input_width, input_height,
        input_depth, tflite::micro::GetTensorData<uint8_t>(filter),
        filter_width, filter_height, op_params.depth_multiplier,
        op_params.padding_values.width, op_params.padding_values.height,
        op_params.stride_width, op_params.stride_height,
        op_params.dilation_width_factor, op_params.dilation_height_factor,
        tflite::micro::GetTensorData<int32_t>(bias), op_params.input_offset,
        op_params.weights_offset, op_params.output_offset,
        tflite::micro::GetTensorData<uint8_t>(output), output_width,
        output_height, op_params.quantized_activation_min,
        op_params.quantized_activation_max, op_params.output_shift,
        op_params.output_multiplier);
  } else {
    tflite::reference_ops::DepthwiseConv(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<uint8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<uint8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<uint8_t>(output));
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpData& data = *(static_cast<OpData*>(node->user_data));

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
    case kTfLiteFloat32: {
      tflite::reference_ops::DepthwiseConv(
          DepthwiseConvParamsFloat(params, data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, data, input, filter, bias,
                              output);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, data, input, filter, bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
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

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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  // Set buffer index to a reset value
  data->buffer_idx = -1;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  if (input->type == kTfLiteInt8) {
    // Allocate memory for per-channel quantization parameters
    const int num_channels =
        filter->dims->data[kDepthwiseConvQuantizedDimension];
    // Dynamically allocate per-channel quantization parameters.
    TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
        context, num_channels * sizeof(int32_t),
        reinterpret_cast<void**>(&data->per_channel_output_multiplier)));
    TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
        context, num_channels * sizeof(int32_t),
        reinterpret_cast<void**>(&data->per_channel_output_shift)));
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

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  if (input->type == kTfLiteInt8) {
    const TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
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
    input_dims.h = height;
    input_dims.w = width;
    input_dims.c = input_shape.Dims(3);

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = filter_height;
    filter_dims.w = filter_width;
    filter_dims.c = output_depth;

    cmsis_nn_dims output_dims;
    output_dims.n = batch_size;
    output_dims.h = output_shape.Dims(1);
    output_dims.w = output_shape.Dims(2);
    output_dims.c = output_depth;

    cmsis_nn_dw_conv_params dw_conv_params;
    dw_conv_params.padding.h = data->padding.height;
    dw_conv_params.padding.w = data->padding.width;

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

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, const OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params,
                             const OpData* data, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.dilation.h = params->dilation_height_factor;
  dw_conv_params.dilation.w = params->dilation_width_factor;
  // Call to reference implementation can be removed when dilation is supported
  // in the optimized implementations.
  if (1 == dw_conv_params.dilation.h && 1 == dw_conv_params.dilation.w) {
    dw_conv_params.input_offset = -input->params.zero_point;
    dw_conv_params.output_offset = output->params.zero_point;
    dw_conv_params.stride.h = params->stride_height;
    dw_conv_params.stride.w = params->stride_width;
    dw_conv_params.padding.h = data->padding.height;
    dw_conv_params.padding.w = data->padding.width;
    // TODO(b/130439627): Use calculated value for clamping.
    dw_conv_params.activation.min = std::numeric_limits<int8_t>::min();
    dw_conv_params.activation.max = std::numeric_limits<int8_t>::max();
    dw_conv_params.ch_mult = params->depth_multiplier;

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = data->per_channel_output_multiplier;
    quant_params.shift = data->per_channel_output_shift;

    RuntimeShape filter_shape = GetTensorShape(filter);
    RuntimeShape input_shape = GetTensorShape(input);
    RuntimeShape output_shape = GetTensorShape(output);
    RuntimeShape bias_shape = GetTensorShape(bias);

    TFLITE_DCHECK_LE(dw_conv_params.activation.min,
                     dw_conv_params.activation.max);

    const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

    if (GetTensorData<int8_t>(bias)) {
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

    if (data->buffer_idx > -1) {
      ctx.buf = context->GetScratchBuffer(context, data->buffer_idx);
    }

    TFLITE_DCHECK_EQ(arm_depthwise_conv_wrapper_s8(
                         &ctx, &dw_conv_params, &quant_params, &input_dims,
                         GetTensorData<int8_t>(input), &filter_dims,
                         GetTensorData<int8_t>(filter), &bias_dims,
                         GetTensorData<int32>(bias), &output_dims,
                         GetTensorData<int8_t>(output)),
                     ARM_MATH_SUCCESS);
  } else {
    DepthwiseParams op_params;
    op_params.padding_type = PaddingType::kSame;
    op_params.padding_values.width = data->padding.width;
    op_params.padding_values.height = data->padding.height;
    op_params.stride_width = params->stride_width;
    op_params.stride_height = params->stride_height;
    op_params.dilation_width_factor = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.depth_multiplier = params->depth_multiplier;
    op_params.input_offset = -input->params.zero_point;
    op_params.weights_offset = 0;
    op_params.output_offset = output->params.zero_point;
    // TODO(b/130439627): Use calculated value for clamping.
    op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

    reference_integer_ops::DepthwiseConvPerChannel(
        op_params, data->per_channel_output_multiplier,
        data->per_channel_output_shift, GetTensorShape(input),
        GetTensorData<int8>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<int32>(bias), GetTensorShape(output),
        GetTensorData<int8>(output));
  }
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, const OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;

  if (1 == op_params.dilation_width_factor &&
      1 == op_params.dilation_height_factor) {
    RuntimeShape filter_shape = GetTensorShape(filter);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    RuntimeShape input_shape = GetTensorShape(input);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    RuntimeShape output_shape = GetTensorShape(output);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    arm_depthwise_conv_u8_basic_ver1(
        GetTensorData<uint8_t>(input), input_width, input_height, input_depth,
        GetTensorData<uint8_t>(filter), filter_width, filter_height,
        op_params.depth_multiplier, op_params.padding_values.width,
        op_params.padding_values.height, op_params.stride_width,
        op_params.stride_height, op_params.dilation_width_factor,
        op_params.dilation_height_factor, GetTensorData<int32_t>(bias),
        op_params.input_offset, op_params.weights_offset,
        op_params.output_offset, GetTensorData<uint8_t>(output), output_width,
        output_height, op_params.quantized_activation_min,
        op_params.quantized_activation_max, op_params.output_shift,
        op_params.output_multiplier);
  } else {
    tflite::reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, &data, input, filter, bias, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, &data, input, filter, bias,
                              output);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, &data, input, filter, bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/depthwise_conv::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/depthwise_conv::Prepare,
                                 /*invoke=*/depthwise_conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

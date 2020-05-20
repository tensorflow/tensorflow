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

#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  // TODO(b/141139247): Allocate these dynamically when possible.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw;
  context->AllocatePersistentBuffer(context, sizeof(int), &raw);
  return raw;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(__ARM_FEATURE_DSP)
  OpData data;
  int32_t buf_size = 0;

  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);

  // Initialize cmsis-nn input dimensions
  cmsis_nn_dims input_dims;
  input_dims.n = MatchingDim(input_shape, 0, output_shape, 0);
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input_shape.Dims(3);

  // Initialize cmsis-nn filter dimensions
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_shape.Dims(3);
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = input_dims.c;

  // Initialize cmsis-nn output dimensions
  cmsis_nn_dims output_dims;
  output_dims.n = input_dims.n;
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output_shape.Dims(3);

  int* buffer_idx = reinterpret_cast<int*>(node->user_data);

  TF_LITE_ENSURE_STATUS(CalculateOpData(
      context, node, params, input_dims.w, input_dims.h, filter_dims.w,
      filter_dims.h, output_dims.w, output_dims.h, input->type, &data));

  if(input->type == kTfLiteInt8) {
    // Initialize cmsis-nn convolution parameters
    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = -input->params.zero_point;
    conv_params.output_offset = output->params.zero_point;
    conv_params.stride.h = params->stride_height;
    conv_params.stride.w = params->stride_width;
    conv_params.dilation.h = params->dilation_height_factor;
    conv_params.dilation.w = params->dilation_width_factor;
    conv_params.padding.h = data.padding.height;
    conv_params.padding.w = data.padding.width;
    conv_params.activation.min = data.output_activation_min;
    conv_params.activation.max = data.output_activation_max;

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params,
                                                       &input_dims,
                                                       &filter_dims,
                                                       &output_dims);
  }

  node->user_data = buffer_idx;
  if (buf_size > 0) {
    TF_LITE_ENSURE_STATUS(
        context->RequestScratchBufferInArena(context, buf_size, buffer_idx));
  } else {
    *buffer_idx = -1;
  }
#endif
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteConvParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* im2col, TfLiteTensor* hwcn_weights,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<uint8_t>(input), GetTensorShape(filter),
                      GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                      GetTensorData<int32_t>(bias), GetTensorShape(output),
                      GetTensorData<uint8_t>(output), GetTensorShape(im2col),
                      GetTensorData<uint8_t>(im2col), nullptr);
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, TfLiteConvParams* params,
    OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output, TfLiteTensor* im2col) {

  // Initialize cmsis-nn convolution parameters
  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = -input->params.zero_point;
  conv_params.output_offset = output->params.zero_point;
  conv_params.stride.h = params->stride_height;
  conv_params.stride.w = params->stride_width;
  conv_params.dilation.h = params->dilation_height_factor;
  conv_params.dilation.w = params->dilation_width_factor;
  conv_params.padding.h = data->padding.height;
  conv_params.padding.w = data->padding.width;
  conv_params.activation.min = data->output_activation_min;
  conv_params.activation.max = data->output_activation_max;

  // Initialize cmsis-nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = data->per_channel_output_multiplier;
  quant_params.shift = data->per_channel_output_shift;

#if defined(__ARM_FEATURE_DSP)
  RuntimeShape filter_shape = GetTensorShape(filter);
  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);
  RuntimeShape bias_shape = GetTensorShape(bias);

  // Sanity check.
  TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (GetTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Initialize cmsis-nn dimensions
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

  // Initialize cmsis-nn context
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  auto* buffer_idx = reinterpret_cast<int*>(node->user_data);
  if (*buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, *buffer_idx);
    // Note: ctx.size is currently not used in cmsis-nn.
    // The buffer should be allocated in the Prepare function through arm_convolve_wrapper_s8_get_buffer_size
  }

  // arm_convolve_wrapper_s8 dispatches the optimized kernel accordingly with the parameters passed
  arm_status status = arm_convolve_wrapper_s8(&ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              GetTensorData<int8_t>(input),
                                              &filter_dims,
                                              GetTensorData<int8_t>(filter),
                                              &bias_dims,
                                              GetTensorData<int32>(bias),
                                              &output_dims,
                                              GetTensorData<int8_t>(output));

  if(status == ARM_MATH_SUCCESS) {
      return kTfLiteOk;
  } else {
      return kTfLiteError;
  }

#else
#pragma message( \
    "CMSIS-NN optimization for conv not available for this target. Using reference kernel.")

  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  reference_integer_ops::ConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));

#endif
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* im2col,
                       TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<float>(input), GetTensorShape(filter),
                      GetTensorData<float>(filter), GetTensorShape(bias),
                      GetTensorData<float>(bias), GetTensorShape(output),
                      GetTensorData<float>(output), GetTensorShape(im2col),
                      GetTensorData<float>(im2col));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  OpData data;

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, &data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, &data, input, filter, bias,
                       nullptr, nullptr, output);
      break;
    case kTfLiteInt8:
      return EvalQuantizedPerChannel(context, node, params, &data, input,
                                     filter, bias, output, nullptr);
      break;
    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, &data, input, filter, bias,
                           nullptr, nullptr, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/conv::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/conv::Prepare,
                                 /*invoke=*/conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite


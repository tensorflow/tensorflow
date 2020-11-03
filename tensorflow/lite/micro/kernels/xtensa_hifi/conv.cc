/*******************************************************************************
* Copyright (c) 2019-2020 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
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
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

// This file has 2 implementation of Conv.

struct OpData {
  TfLitePaddingValues padding;

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

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
                             const TfLiteConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             int out_width, int out_height,
                             const TfLiteType data_type, OpData* data) {
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
    TF_LITE_ENSURE(context, input != nullptr);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    TF_LITE_ENSURE(context, filter != nullptr);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels));
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params = static_cast<const TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  // Dynimically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
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
      filter_height, output_width, output_height, input->type, data));

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  return kTfLiteOk;
}  // namespace conv

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteConvParams* params, const OpData& data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* im2col,
                           TfLiteEvalTensor* hwcn_weights,
                           TfLiteEvalTensor* output) {
  const int32_t input_offset = -data.input_zero_point;
  const int32_t filter_offset = -data.filter_zero_point;
  const int32_t output_offset = data.output_zero_point;

  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const uint8_t *input_data, *filter_data;
    const int32_t* bias_data;
    uint8_t* output_data;
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);

    input_data = tflite::micro::GetTensorData<uint8_t>(input);
    filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
    bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    output_data = tflite::micro::GetTensorData<uint8_t>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int pad_width = data.padding.width;
    const int pad_height = data.padding.height;
    const int32_t output_activation_min = data.output_activation_min;
    const int32_t output_activation_max = data.output_activation_max;
    const int32_t output_multiplier = data.output_multiplier;
    const int output_shift = -data.output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
      TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int filter_depth = filter_shape.Dims(3);

    int err, output_data_format = 0;
    uint8_t* p_scratch;
    uint8_t* p_filter;
    // Calculate filter_depth_padded as next near multiple of 4
    int filter_depth_padded = (filter_depth + 3) & (~3);
    int out_length = output_height * output_width * output_depth;
    int filter_size_padded = filter_height * filter_width * filter_depth_padded;
    int required_scratch, input_precision = PREC_ASYM8;
    int h, c;

    required_scratch = xa_nn_conv2d_std_getsize(
        input_height, input_depth, filter_height, filter_width, stride_height,
        pad_height, output_height, input_precision);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
                         "conv2d_std_asym8: xa_nn_conv2d_std_getsize failed");
      return kTfLiteError;
    }

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = xtensa_nnlib_scratch_buf;

    p_filter = p_scratch;
    required_scratch +=
        ALIGNED_SIZE((sizeof(uint8_t) * filter_size_padded * output_depth), 8);
    p_scratch +=
        ALIGNED_SIZE(sizeof(uint8_t) * filter_size_padded * output_depth, 8);

    if (required_scratch > static_cast<int>(XTENSA_NNLIB_MAX_SCRATCH_SIZE)) {
      TF_LITE_KERNEL_LOG(context,
                         "conv2d_std_asym8: insufficient scratch memory");
      return kTfLiteError;
    }

    // Padding filter coefficients depthwise
    for (h = 0; h < filter_height * filter_width * output_depth; h++) {
      for (c = 0; c < filter_depth; c++) {
        p_filter[h * filter_depth_padded + c] =
            filter_data[h * filter_depth + c];
      }
      for (c = input_depth; c < filter_depth_padded; c++) {
        p_filter[h * filter_depth_padded + c] =
            -filter_offset;  // filter_depth[h*input_depth + c];
      }
    }

    for (int batch = 0; batch < batches; ++batch) {
      uint8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      err = xa_nn_conv2d_std_asym8xasym8(
          p_out_temp,
          &input_data[batch * input_height * input_width * input_depth],
          p_filter,  // filter_data,
          bias_data, input_height, input_width, input_depth, filter_height,
          filter_width, output_depth, stride_width, stride_height, pad_width,
          pad_height, output_height, output_width, input_offset, filter_offset,
          output_multiplier, output_shift, output_offset, output_data_format,
          static_cast<void*>(p_scratch));

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "conv2d_std_asym8: xa_nn_conv2d_std_asym8xasym8 failed");

      err = xa_nn_vec_activation_min_max_asym8_asym8(
          p_out_temp, p_out_temp, output_activation_min, output_activation_max,
          out_length);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "xa_nn_vec_activation_min_max_asym8_asym8 failed");
    }
  } else {
    // TODO(b/154032858): Investigate removing extra copies.
    ConvParams op_params;
    op_params.padding_type = RuntimePaddingType(params->padding);
    op_params.padding_values.width = data.padding.width;
    op_params.padding_values.height = data.padding.height;
    op_params.stride_width = params->stride_width;
    op_params.stride_height = params->stride_height;
    op_params.dilation_width_factor = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.input_offset = input_offset;
    op_params.weights_offset = filter_offset;
    op_params.output_offset = output_offset;
    op_params.output_multiplier = data.output_multiplier;
    op_params.output_shift = -data.output_shift;
    op_params.quantized_activation_min = data.output_activation_min;
    op_params.quantized_activation_max = data.output_activation_max;
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
  return kTfLiteOk;
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, const OpData& data,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output,
                             TfLiteEvalTensor* im2col) {
  // TODO(b/154032858): Investigate removing extra copies.
  ConvParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data.padding.height;
  op_params.padding_values.width = data.padding.width;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

  reference_integer_ops::ConvPerChannel(
      op_params, data.per_channel_output_multiplier,
      data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteConvParams* params, const OpData& data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* im2col,
                       TfLiteEvalTensor* hwcn_weights,
                       TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

#if HIFI_VFPU
  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const float *input_data, *filter_data;
    const float* bias_data;
    float* output_data;
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);

    input_data = tflite::micro::GetTensorData<float>(input);
    filter_data = tflite::micro::GetTensorData<float>(filter);
    bias_data = tflite::micro::GetTensorData<float>(bias);
    output_data = tflite::micro::GetTensorData<float>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int pad_width = data.padding.width;
    const int pad_height = data.padding.height;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
      TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int filter_depth = filter_shape.Dims(3);
    int err, output_data_format = 0;
    uint8_t* p_scratch;
    float* p_filter;
    // Calculate filter_depth_padded as next near multiple of 2
    int filter_depth_padded = (filter_depth + 1) & (~1);
    int out_length = output_height * output_width * output_depth;
    int filter_size_padded = filter_height * filter_width * filter_depth_padded;
    int required_scratch, input_precision = PREC_F32;
    int h, c;

    required_scratch = xa_nn_conv2d_std_getsize(
        input_height, input_depth, filter_height, filter_width, stride_height,
        pad_height, output_height, input_precision);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
                         "conv2d_std_f32: xa_nn_conv2d_std_getsize failed");
      return kTfLiteError;
    }

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = xtensa_nnlib_scratch_buf;

    p_filter = reinterpret_cast<float*>(p_scratch);
    p_scratch +=
        ALIGNED_SIZE((sizeof(float) * filter_size_padded * output_depth), 8);
    required_scratch +=
        ALIGNED_SIZE((sizeof(float) * filter_size_padded * output_depth), 8);

    if (required_scratch > static_cast<int>(XTENSA_NNLIB_MAX_SCRATCH_SIZE)) {
      TF_LITE_KERNEL_LOG(context,
                         "conv2d_std_f32: insufficient scratch memory");
      return kTfLiteError;
    }

    // Padding filter coefficients depthwise
    for (h = 0; h < filter_height * filter_width * output_depth; h++) {
      for (c = 0; c < filter_depth; c++) {
        p_filter[h * filter_depth_padded + c] =
            filter_data[h * filter_depth + c];
      }
      for (c = input_depth; c < filter_depth_padded; c++) {
        p_filter[h * filter_depth_padded + c] = 0;
      }
    }

    for (int batch = 0; batch < batches; ++batch) {
      float* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      err = xa_nn_conv2d_std_f32(
          p_out_temp,
          &input_data[batch * input_height * input_width * input_depth],
          p_filter, bias_data, input_height, input_width, input_depth,
          filter_height, filter_width, output_depth, stride_width,
          stride_height, pad_width, pad_height, output_height, output_width,
          output_data_format, static_cast<void*>(p_scratch));

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "conv2d_std_f32: xa_nn_conv2d_std_f32xf32 failed");

      err = xa_nn_vec_activation_min_max_f32_f32(
          p_out_temp, p_out_temp, output_activation_min, output_activation_max,
          out_length);

      CHECK_ERR_HIFI_NNLIB_KER(err,
                               "xa_nn_vec_activation_min_max_f32_f32 failed");
    }
    return kTfLiteOk;
  }
#endif /* HIFI_VFPU */

  // TODO(b/154032858): Investigate removing extra copies.
  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::Conv(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<float>(input),
                      tflite::micro::GetTensorShape(filter),
                      tflite::micro::GetTensorData<float>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetTensorData<float>(bias),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<float>(output),
                      tflite::micro::GetTensorShape(im2col),
                      tflite::micro::GetTensorData<float>(im2col));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

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

/******************************************************************************
 * Copyright (C) 2019 Cadence Design Systems, Inc.
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
#include "xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
// Per channel quantization is not needed for any model on xtensa.
constexpr int kMaxChannels = 256;

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
  // (b/141139247): Allocate these dynamically when possible.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
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

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteDepthwiseConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const float *input_data, *filter_data, *bias_data;
    float* output_data;
    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& filter_shape = GetTensorShape(filter);
    const RuntimeShape& output_shape = GetTensorShape(output);
    const RuntimeShape& bias_shape = GetTensorShape(bias);

    input_data = GetTensorData<float>(input);
    filter_data = GetTensorData<float>(filter);
    bias_data = GetTensorData<float>(bias);
    output_data = GetTensorData<float>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int dilation_width_factor = 1;
    const int dilation_height_factor = 1;
    // const int dilation_width_factor = params->dilation_width_factor;;
    // const int dilation_height_factor = params->dilation_height_factor;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int depth_multiplier = params->depth_multiplier;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int filter_depth = filter_shape.Dims(3);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    int32_t err, input_data_format = 0, output_data_format = 0;
    void* p_scratch;
    float* p_filter;
    int filter_depth_padded, filter_size_padded, required_scratch;
    int input_precision = PREC_F32;
    int h, c, i;

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = xtensa_nnlib_scratch_buf;

    filter_depth_padded = (filter_depth + 1) & (~1);
    filter_size_padded = filter_height * filter_width * filter_depth_padded;

    required_scratch = xa_nn_conv2d_depthwise_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        depth_multiplier, stride_width, stride_height, pad_width, pad_height,
        output_height, output_width, input_precision, input_data_format);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(
          context, "DepthwiseConvFloat: xa_nn_conv2d_depthwise_getsize failed");
      return kTfLiteError;
    }

    required_scratch += ALIGNED_SIZE(sizeof(float) * filter_size_padded, 8);
    if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context,
                         "DepthwiseConvFloat: insufficient scratch memory");
      return kTfLiteError;
    }

    p_filter = (float*)p_scratch;
    p_scratch = (void*)((uint8_t*)p_filter +
                        ALIGNED_SIZE(sizeof(float) * filter_size_padded, 8));

    for (h = 0; h < filter_height * filter_width; h++) {
      for (c = 0; c < filter_depth; c++) {
        p_filter[h * filter_depth_padded + c] =
            filter_data[h * filter_depth + c];
      }
      for (c = filter_depth; c < filter_depth_padded; c++) {
        p_filter[h * filter_depth_padded + c] = 0;
      }
    }

    for (i = 0; i < batches; i++) {
      err = xa_nn_conv2d_depthwise_f32(
          &output_data[i * output_height * output_width * output_depth],
          p_filter,  // filter_data,
          &input_data[i * input_height * input_width * input_depth], bias_data,
          input_height, input_width, input_depth, filter_height, filter_width,
          depth_multiplier, stride_width, stride_height, pad_width, pad_height,
          output_height, output_width, input_data_format, output_data_format,
          p_scratch);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "DepthwiseConvFloat: xa_nn_conv2d_depthwise_f32 failed");
    }

    // pre loop for activation_min_max to handle alignment
    int out_length = batches * output_height * output_width * output_depth;
    uint32 p_unalign_val = (uint32)output_data, p_align_val;
    p_align_val = (p_unalign_val + 7) & (~7);

    int pre_loop_count = p_align_val - p_unalign_val;
    pre_loop_count = MIN(pre_loop_count, out_length);

    for (i = 0; i < pre_loop_count; i++) {
      ACTIVATION_MIN_MAX(float, output_data[i], output_data[i],
                         output_activation_min, output_activation_max)
    }

    out_length = out_length - pre_loop_count;

    if (out_length) {
      err = xa_nn_vec_activation_min_max_f32_f32(
          &output_data[i], &output_data[i], output_activation_min,
          output_activation_max, out_length);

      CHECK_ERR_HIFI_NNLIB_KER(
          err,
          "DepthwiseConvFloat: xa_nn_vec_activation_min_max_f32_f32 failed");
    }
  } else {
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
        GetTensorShape(bias), GetTensorData<float>(bias),
        GetTensorShape(output), GetTensorData<float>(output));
  }
  return kTfLiteOk;
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, OpData* data,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
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
  // (b/130439627): Use calculated value for clamping.
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

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDepthwiseConvParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const uint8 *input_data, *filter_data;
    const int32_t* bias_data;
    uint8* output_data;
    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& filter_shape = GetTensorShape(filter);
    const RuntimeShape& output_shape = GetTensorShape(output);
    const RuntimeShape& bias_shape = GetTensorShape(bias);

    input_data = GetTensorData<uint8_t>(input);
    filter_data = GetTensorData<uint8_t>(filter);
    bias_data = GetTensorData<int32_t>(bias);
    output_data = GetTensorData<uint8_t>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int dilation_width_factor = 1;
    const int dilation_height_factor = 1;
    // const int dilation_width_factor = params->dilation_width_factor;
    // const int dilation_height_factor = params->dilation_height_factor;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int depth_multiplier = params->depth_multiplier;
    const int32 output_activation_min = data->output_activation_min;
    const int32 output_activation_max = data->output_activation_max;
    const int32 output_multiplier = data->output_multiplier;
    // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
    const int output_shift = -data->output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int filter_depth = filter_shape.Dims(3);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    int32_t err, i, input_data_format = 0, output_data_format = 0;
    void* p_scratch;
    uint8* p_filter;
    int filter_depth_padded, filter_size_padded, required_scratch;
    int input_precision = PREC_ASYM8;
    int h, c;

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = xtensa_nnlib_scratch_buf;

    required_scratch = xa_nn_conv2d_depthwise_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        depth_multiplier, stride_width, stride_height, pad_width, pad_height,
        output_height, output_width, input_precision, input_data_format);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(
          context, "DepthwiseConvAsym8: xa_nn_conv2d_depthwise_getsize failed");
      return kTfLiteError;
    }

    filter_depth_padded = (filter_depth + 3) & (~3);
    filter_size_padded = filter_height * filter_width * filter_depth_padded;
    required_scratch += ALIGNED_SIZE(sizeof(uint8_t) * filter_size_padded, 8);

    if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context,
                         "DepthwiseConvAsym8: insufficient scratch memory");
      return kTfLiteError;
    }

    p_filter = (uint8*)p_scratch;
    p_scratch = (void*)(p_filter +
                        ALIGNED_SIZE(sizeof(uint8_t) * filter_size_padded, 8));

    for (h = 0; h < filter_height * filter_width; h++) {
      for (c = 0; c < filter_depth; c++) {
        p_filter[h * filter_depth_padded + c] =
            filter_data[h * filter_depth + c];
      }
      for (c = filter_depth; c < filter_depth_padded; c++) {
        p_filter[h * filter_depth_padded + c] = -filter_offset;
      }
    }

    for (i = 0; i < batches; i++) {
      err = xa_nn_conv2d_depthwise_asym8xasym8(
          &output_data[i * output_height * output_width * output_depth],
          p_filter,  // filter_data,
          &input_data[i * input_height * input_width * input_depth], bias_data,
          input_height, input_width, input_depth, filter_height, filter_width,
          depth_multiplier, stride_width, stride_height, pad_width, pad_height,
          output_height, output_width, input_offset, filter_offset,
          output_multiplier, output_shift, output_offset, input_data_format,
          output_data_format, p_scratch);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "DepthwiseConvAsym8: xa_nn_conv2d_depthwise_asym8xasym8 failed");
    }

    // pre loop for activation_min_max to handle alignment
    int out_length = batches * output_height * output_width * output_depth;
    uint32 p_unalign_val = (uint32)output_data, p_align_val;
    p_align_val = (p_unalign_val + 7) & (~7);

    int pre_loop_count = p_align_val - p_unalign_val;
    pre_loop_count = MIN(pre_loop_count, out_length);

    for (i = 0; i < pre_loop_count; i++) {
      ACTIVATION_MIN_MAX_ASYM8(output_data[i], output_data[i],
                               output_activation_min, output_activation_max)
    }

    out_length = out_length - pre_loop_count;

    if (out_length > 0) {
      err = xa_nn_vec_activation_min_max_asym8_asym8(
          &output_data[i], &output_data[i], output_activation_min,
          output_activation_max, out_length);

      CHECK_ERR_HIFI_NNLIB_KER(
          err,
          "DepthwiseConvAsym8: xa_nn_vec_activation_min_max_asym8_asym8 "
          "failed");
    }
  } else {
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

    tflite::reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

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
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        &data));

  // (aselle): Consider whether float conv and quantized conv should be
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
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
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

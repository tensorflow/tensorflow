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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

TfLiteStatus AverageEvalFloat(TfLiteContext* context, const TfLiteNode* node,
                              const TfLitePoolParams* params,
                              const OpData* data, const TfLiteTensor* input,
                              TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int kernel_height = params->filter_height;
  const int kernel_width = params->filter_width;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const float* inp_data_ptr;
  float* out_data_ptr;
  int inp_data_format = 0, out_data_format = 0, out_length;
  int inp_precision = PREC_F32, out_precision = PREC_F32;
  void* p_scratch;
  int err, required_scratch = 0;

  ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
  p_scratch = (void*)xtensa_nnlib_scratch_buf;

  required_scratch = xa_nn_avgpool_getsize(
      depth, inp_precision, out_precision, input_height, input_width,
      kernel_height, kernel_width,
      stride_width,   // x_stride,
      stride_height,  // y_stride,
      pad_width,      // x_padding,
      pad_height,     // y_padding,
      output_height, output_width, inp_data_format, out_data_format);

  if (required_scratch <= 0) {
    TF_LITE_KERNEL_LOG(context,
                       "AveragepoolFloat: xa_nn_avgpool_getsize failed");
    return kTfLiteError;
  }

  if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
    TF_LITE_KERNEL_LOG(context,
                       "AveragepoolFloat: insufficient scratch memory");
    return kTfLiteError;
  }

  inp_data_ptr = GetTensorData<float>(input);
  out_data_ptr = GetTensorData<float>(output);

  for (int batch = 0; batch < batches; ++batch) {
    err = xa_nn_avgpool_f32(
        &out_data_ptr[output_height * output_width * depth * batch],
        &inp_data_ptr[output_height * output_width * depth * batch],
        input_height, input_width, depth, kernel_height, kernel_width,
        stride_width, stride_height, pad_width, pad_height, output_height,
        output_width, inp_data_format, out_data_format, p_scratch);

    CHECK_ERR_HIFI_NNLIB_KER(err, "AveragepoolFloat: xa_nn_avgpool_f32 failed");
  }

  out_length = batches * output_height * output_width * depth;
  uint32 p_unalign_val = (uint32)out_data_ptr, p_align_val;
  p_align_val = (p_unalign_val + 7) & (~7);

  // pre loop for activation_min_max
  int pre_loop_count = p_align_val - p_unalign_val;
  pre_loop_count = MIN(pre_loop_count, out_length);

  for (int i = 0; i < pre_loop_count; i++) {
    ACTIVATION_MIN_MAX(float, out_data_ptr[i], out_data_ptr[i], activation_min,
                       activation_max)
  }

  out_length = out_length - pre_loop_count;

  if (out_length) {
    err = xa_nn_vec_activation_min_max_f32_f32(
        out_data_ptr, out_data_ptr, activation_min, activation_max, out_length);

    CHECK_ERR_HIFI_NNLIB_KER(
        err, "AveragepoolFloat: xa_nn_vec_activation_min_max_f32_f32 failed");
  }
  return kTfLiteOk;
}

TfLiteStatus AverageEvalQuantized(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLitePoolParams* params,
                                  const OpData* data, const TfLiteTensor* input,
                                  TfLiteTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);

  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  if (input->type == kTfLiteUInt8) {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& output_shape = GetTensorShape(output);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const uint8* inp_data_ptr;
    uint8* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    int inp_precision = PREC_ASYM8, out_precision = PREC_ASYM8;
    void* p_scratch;
    int err, required_scratch = 0;

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = (void*)xtensa_nnlib_scratch_buf;

    required_scratch = xa_nn_avgpool_getsize(
        depth, inp_precision, out_precision, input_height, input_width,
        kernel_height, kernel_width,
        stride_width,   // x_stride,
        stride_height,  // y_stride,
        pad_width,      // x_padding,
        pad_height,     // y_padding,
        output_height, output_width, inp_data_format, out_data_format);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
                         "AveragepoolAsym8: xa_nn_avgpool_getsize failed");
      return kTfLiteError;
    }

    if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context,
                         "AveragepoolAsym8: insufficient scratch memory");
      return kTfLiteError;
    }

    inp_data_ptr = GetTensorData<uint8_t>(input);
    out_data_ptr = GetTensorData<uint8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_avgpool_asym8(
          &out_data_ptr[output_height * output_width * depth * batch],
          &inp_data_ptr[output_height * output_width * depth * batch],
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      CHECK_ERR_HIFI_NNLIB_KER(err,
                               "AveragepoolAsym8: xa_nn_avgpool_asym8 failed");
    }

    out_length = batches * output_height * output_width * depth;
    uint32 p_unalign_val = (uint32)out_data_ptr, p_align_val;
    p_align_val = (p_unalign_val + 7) & (~7);

    // pre loop for activation_min_max
    int pre_loop_count = p_align_val - p_unalign_val;
    pre_loop_count = MIN(pre_loop_count, out_length);

    for (int i = 0; i < pre_loop_count; i++) {
      ACTIVATION_MIN_MAX_ASYM8(out_data_ptr[i], out_data_ptr[i], activation_min,
                               activation_max)
    }

    out_length = out_length - pre_loop_count;

    if (out_length > 0) {
      err = xa_nn_vec_activation_min_max_asym8_asym8(
          out_data_ptr, out_data_ptr, activation_min, activation_max,
          out_length);

      CHECK_ERR_HIFI_NNLIB_KER(
          err,
          "AveragepoolAsym8: xa_nn_vec_activation_min_max_asym8_asym8 failed");
    }
  } else {
    PoolParams op_params;
    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_values.height = data->padding.height;
    op_params.padding_values.width = data->padding.width;
    op_params.quantized_activation_min = activation_min;
    op_params.quantized_activation_max = activation_max;
    reference_integer_ops::AveragePool(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                          TfLitePoolParams* params, OpData* data,
                          const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int kernel_height = params->filter_height;
  const int kernel_width = params->filter_width;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const float* inp_data_ptr;
  float* out_data_ptr;
  int inp_data_format = 0, out_data_format = 0, out_length;
  int inp_precision = PREC_F32, out_precision = PREC_F32;
  void* p_scratch;
  int err, required_scratch = 0;

  ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
  p_scratch = (void*)xtensa_nnlib_scratch_buf;

  required_scratch = xa_nn_maxpool_getsize(
      depth, inp_precision, out_precision, input_height, input_width,
      kernel_height, kernel_width,
      stride_width,   // x_stride,
      stride_height,  // y_stride,
      pad_width,      // x_padding,
      pad_height,     // y_padding,
      output_height, output_width, inp_data_format, out_data_format);

  if (required_scratch <= 0) {
    TF_LITE_KERNEL_LOG(context, "MaxpoolFloat: xa_nn_maxpool_getsize failed");
    return kTfLiteError;
  }

  if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
    TF_LITE_KERNEL_LOG(context, "MaxpoolFloat: insufficient scratch memory");
    return kTfLiteError;
  }

  inp_data_ptr = GetTensorData<float>(input);
  out_data_ptr = GetTensorData<float>(output);

  for (int batch = 0; batch < batches; ++batch) {
    err = xa_nn_maxpool_f32(
        &out_data_ptr[output_height * output_width * depth * batch],
        &inp_data_ptr[output_height * output_width * depth * batch],
        input_height, input_width, depth, kernel_height, kernel_width,
        stride_width, stride_height, pad_width, pad_height, output_height,
        output_width, inp_data_format, out_data_format, p_scratch);

    CHECK_ERR_HIFI_NNLIB_KER(err, "MaxpoolFloat: xa_nn_maxpool_f32 failed");
  }

  out_length = batches * output_height * output_width * depth;
  uint32 p_unalign_val = (uint32)out_data_ptr, p_align_val;
  p_align_val = (p_unalign_val + 7) & (~7);

  // pre loop for activation_min_max
  int pre_loop_count = p_align_val - p_unalign_val;
  pre_loop_count = MIN(pre_loop_count, out_length);

  for (int i = 0; i < pre_loop_count; i++) {
    ACTIVATION_MIN_MAX(float, out_data_ptr[i], out_data_ptr[i], activation_min,
                       activation_max)
  }

  out_length = out_length - pre_loop_count;

  if (out_length > 0) {
    err = xa_nn_vec_activation_min_max_f32_f32(
        out_data_ptr, out_data_ptr, activation_min, activation_max, out_length);

    CHECK_ERR_HIFI_NNLIB_KER(
        err, "MaxpoolFloat: xa_nn_vec_activation_min_max_f32_f32 failed");
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLitePoolParams* params, OpData* data,
                              const TfLiteTensor* input, TfLiteTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);

  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  if (input->type == kTfLiteUInt8) {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& output_shape = GetTensorShape(output);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const uint8* inp_data_ptr;
    uint8* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    int inp_precision = PREC_ASYM8, out_precision = PREC_ASYM8;
    void* p_scratch;
    int err, required_scratch = 0;

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    p_scratch = (void*)xtensa_nnlib_scratch_buf;

    required_scratch = xa_nn_maxpool_getsize(
        depth, inp_precision, out_precision, input_height, input_width,
        kernel_height, kernel_width,
        stride_width,   // x_stride,
        stride_height,  // y_stride,
        pad_width,      // x_padding,
        pad_height,     // y_padding,
        output_height, output_width, inp_data_format, out_data_format);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context, "MaxpoolAsym8: xa_nn_maxpool_getsize failed");
      return kTfLiteError;
    }

    if (required_scratch > (int)XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context, "MaxpoolAsym8: insufficient scratch memory");
      return kTfLiteError;
    }

    inp_data_ptr = GetTensorData<uint8_t>(input);
    out_data_ptr = GetTensorData<uint8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_maxpool_asym8(
          &out_data_ptr[output_height * output_width * depth * batch],
          &inp_data_ptr[output_height * output_width * depth * batch],
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      CHECK_ERR_HIFI_NNLIB_KER(err, "MaxpoolAsym8: xa_nn_maxpool_asym8 failed");
    }

    out_length = batches * output_height * output_width * depth;
    uint32 p_unalign_val = (uint32)out_data_ptr, p_align_val;
    p_align_val = (p_unalign_val + 7) & (~7);

    // pre loop for activation_min_max
    int pre_loop_count = p_align_val - p_unalign_val;
    pre_loop_count = MIN(pre_loop_count, out_length);

    for (int i = 0; i < pre_loop_count; i++) {
      ACTIVATION_MIN_MAX_ASYM8(out_data_ptr[i], out_data_ptr[i], activation_min,
                               activation_max)
    }

    out_length = out_length - pre_loop_count;

    if (out_length > 0) {
      err = xa_nn_vec_activation_min_max_asym8_asym8(
          out_data_ptr, out_data_ptr, activation_min, activation_max,
          out_length);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "MaxpoolAsym8: xa_nn_vec_activation_min_max_asym8_asym8 failed");
    }
  } else {
    tflite::PoolParams op_params;
    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_values.height = data->padding.height;
    op_params.padding_values.width = data->padding.width;
    op_params.quantized_activation_min = activation_min;
    op_params.quantized_activation_max = activation_max;
    reference_integer_ops::MaxPool(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

}  // namespace


TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  // Inputs and outputs share the same type, guarenteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      AverageEvalQuantized(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      MaxEvalQuantized(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::AverageEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::MaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

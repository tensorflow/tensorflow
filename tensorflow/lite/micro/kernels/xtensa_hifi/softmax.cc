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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "xtensa_tf_micro_common.h"
namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

TfLiteStatus CalculateSoftmaxParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt8);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
        // NOTE: Current int16 softmax output does not require symmetric scaling
        // - so no need to verify scale here.
      } else {
        TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
        TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
      }
    }

    static const int kScaledDiffIntegerBits = 5;

    int input_left_shift;
    tflite::PreprocessSoftmaxScaling(
        static_cast<double>(params->beta),
        static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
        &op_data->input_multiplier, &input_left_shift);
    op_data->input_left_shift = input_left_shift;
    op_data->diff_min =
        -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                            op_data->input_left_shift);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    op_data->beta = static_cast<double>(params->beta);
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  return kTfLiteOk;
}

// Takes a tensor and performs softmax along the last dimension.
TfLiteStatus SoftmaxFloat(TfLiteContext* context, const TfLiteTensor* input,
                          TfLiteTensor* output, const SoftmaxParams& op_data) {
  const RuntimeShape& input_shape = GetTensorShape(input);
  const float* input_data = GetTensorData<float>(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  float* output_data = GetTensorData<float>(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
  float* p_scratch = (float*)xtensa_nnlib_scratch_buf;

  if (depth * sizeof(float) > XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
    TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
    return kTfLiteError;
  }

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      p_scratch[c] =
          input_data[i * depth + c] * static_cast<float>(op_data.beta);
    }

    int err =
        xa_nn_vec_softmax_f32_f32(&output_data[i * depth], p_scratch, depth);
    CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_f32_f32 failed");
  }
  return kTfLiteOk;
}

TfLiteStatus SoftmaxQuantized(TfLiteContext* context, const TfLiteTensor* input,
                              TfLiteTensor* output,
                              const SoftmaxParams& op_data) {
  if (input->type == kTfLiteUInt8) {
    const RuntimeShape& input_shape = GetTensorShape(input);
    const uint8_t* input_data = GetTensorData<uint8_t>(input);
    const RuntimeShape& output_shape = GetTensorShape(output);
    uint8_t* output_data = GetTensorData<uint8_t>(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    void* p_scratch = (void*)xtensa_nnlib_scratch_buf;

    if (get_softmax_scratch_size(PREC_ASYM8, PREC_ASYM8, depth) >
        XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
      return kTfLiteError;
    }

    for (int i = 0; i < outer_size; ++i) {
      int err = xa_nn_vec_softmax_asym8_asym8(
          &output_data[i * depth], &input_data[i * depth], op_data.diff_min,
          op_data.input_left_shift, op_data.input_multiplier, depth, p_scratch);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_asym8_asym8 failed");
    }
  } else {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(
          op_data, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(
          op_data, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  SoftmaxParams op_data;
  TF_LITE_ENSURE_STATUS(
      CalculateSoftmaxParams(context, input, output, params, &op_data));

  switch (input->type) {
    case kTfLiteFloat32: {
      return SoftmaxFloat(context, input, output, op_data);
    }
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      return SoftmaxQuantized(context, input, output, op_data);
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
}
}  // namespace activations

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::SoftmaxPrepare,
                                 /*invoke=*/activations::SoftmaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

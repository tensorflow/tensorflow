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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"
namespace tflite {
namespace {

// Softmax parameter data that persists in user_data
static constexpr int kInt16LUTArraySize = 513;

TfLiteStatus CalculateSoftmaxParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt8);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
      TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 32768,
                          (0.001f * 1.f / 32768));
    } else {  // input->type == kTfLiteInt8
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
        TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 65536,
                            (0.001f * 1.f / 65536));
      } else {  // output->type == kTfLiteint8
        TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
        TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
      }
    }

    static const int kScaledDiffIntegerBits = 5;

    // Calculate input_multiplier and input_left_shift
    if (input->type == kTfLiteInt16) {
      int input_left_shift;
      double input_scale_beta_rescale =
          static_cast<double>(input->params.scale) *
          static_cast<double>(params->beta) /
          (10.0 / 65535.0);  // scale the input_diff such that [-65535, 0]
                             // correspond to [-10.0, 0.0]
      QuantizeMultiplier(input_scale_beta_rescale, &op_data->input_multiplier,
                         &input_left_shift);
      op_data->input_left_shift = input_left_shift;
    } else {
      int input_left_shift;
      tflite::PreprocessSoftmaxScaling(
          static_cast<double>(params->beta),
          static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
          &op_data->input_multiplier, &input_left_shift);
      op_data->input_left_shift = input_left_shift;
      op_data->diff_min =
          -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                              op_data->input_left_shift);
    }
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    op_data->beta = static_cast<double>(params->beta);
  }
  return kTfLiteOk;
}

// Takes a tensor and performs softmax along the last dimension.
TfLiteStatus SoftmaxFloat(TfLiteContext* context, const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output,
                          const SoftmaxParams& op_data) {
#if HIFI_VFPU
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  float* output_data = tflite::micro::GetTensorData<float>(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
  float* p_scratch = reinterpret_cast<float*>(xtensa_nnlib_scratch_buf);

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
#else
  tflite::reference_ops::Softmax(op_data, tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<float>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<float>(output));
#endif /* HIFI_VFPU */
  return kTfLiteOk;
}

TfLiteStatus SoftmaxQuantized(TfLiteContext* context,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output,
                              const SoftmaxParams& op_data) {
  if (input->type == kTfLiteUInt8) {
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    void* p_scratch = reinterpret_cast<void*>(xtensa_nnlib_scratch_buf);

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
  } else if (input->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt16) {
#ifndef NNLIB_HIFI5
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
#else
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
      const int trailing_dim = input_shape.DimensionsCount() - 1;
      const int outer_size =
          MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
      const int depth =
          MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

      ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
      void* p_scratch = xtensa_nnlib_scratch_buf;

      if (get_softmax_scratch_size(-4, -4, depth) >
          XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
        TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
        return kTfLiteError;
      }

      for (int i = 0; i < outer_size; ++i) {
        int err = xa_nn_vec_softmax_asym8s_16(
            &output_data[i * depth], &input_data[i * depth], op_data.diff_min,
            op_data.input_left_shift, op_data.input_multiplier, depth,
            p_scratch);
        CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_asym8s_16 failed");
      }
      return kTfLiteOk;
#endif
    } else {
#ifndef NNLIB_HIFI5
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#else
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
      const int trailing_dim = input_shape.DimensionsCount() - 1;
      const int outer_size =
          MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
      const int depth =
          MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

      ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
      void* p_scratch = xtensa_nnlib_scratch_buf;

      if (get_softmax_scratch_size(-4, -4, depth) >
          XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
        TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
        return kTfLiteError;
      }

      for (int i = 0; i < outer_size; ++i) {
        int err = xa_nn_vec_softmax_asym8s_asym8s(
            &output_data[i * depth], &input_data[i * depth], op_data.diff_min,
            op_data.input_left_shift, op_data.input_multiplier, depth,
            p_scratch);
        CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_asym8s_asym8s failed");
      }
#endif
    }
  } else {
    tflite::reference_ops::SoftmaxInt16(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }
  return kTfLiteOk;
}

void* SoftmaxInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(SoftmaxParams));
}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, node->user_data != nullptr);
  SoftmaxParams* op_data = static_cast<SoftmaxParams*>(node->user_data);
  // Only allocate LUTs for KTfLiteInt16 data type
  if (input->type == kTfLiteInt16) {
    void* raw_exp_lut = context->AllocatePersistentBuffer(
        context, sizeof(int16_t) * kInt16LUTArraySize);
    TF_LITE_ENSURE(context, raw_exp_lut != nullptr);
    op_data->exp_lut = reinterpret_cast<int16_t*>(raw_exp_lut);
    void* one_over_one_plus_x_lut = context->AllocatePersistentBuffer(
        context, sizeof(int16_t) * kInt16LUTArraySize);
    TF_LITE_ENSURE(context, one_over_one_plus_x_lut != nullptr);
    op_data->one_over_one_plus_x_lut =
        reinterpret_cast<int16_t*>(one_over_one_plus_x_lut);
  }

  if (output->type == kTfLiteInt16) {
    TF_LITE_ENSURE(context, input->type == kTfLiteInt8 ||
                                input->type == kTfLiteUInt8 ||
                                input->type == kTfLiteInt16);
  } else {
    TF_LITE_ENSURE_EQ(context, input->type, output->type);
  }

  // Populate LUT if required
  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    // exp LUT only used on negative values
    // we consider exp(-10.0) is insignificant to accumulation
    gen_lut([](float value) { return std::exp(value); }, -10.0f, 0.0f,
            op_data->exp_lut, kInt16LUTArraySize);
    gen_lut([](float value) { return 1.0f / (1.0f + value); }, 0.0f, 1.0f,
            op_data->one_over_one_plus_x_lut, kInt16LUTArraySize);
    op_data->zero_point = output->params.zero_point;
    op_data->scale = output->params.scale;
  }

  auto* params = static_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  return CalculateSoftmaxParams(context, input, output, params, op_data);
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      return SoftmaxFloat(context, input, output, op_data);
    }
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteInt16: {
      return SoftmaxQuantized(context, input, output, op_data);
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
}
}  // namespace

TfLiteRegistration Register_SOFTMAX() {
  return {/*init=*/SoftmaxInit,
          /*free=*/nullptr,
          /*prepare=*/SoftmaxPrepare,
          /*invoke=*/SoftmaxEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

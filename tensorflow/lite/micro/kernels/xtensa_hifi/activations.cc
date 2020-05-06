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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename Q>
inline void ReluQuantized(int32_t lower, const RuntimeShape& input_shape,
                          const Q* input_data, const RuntimeShape& output_shape,
                          Q* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const Q val = input_data[i];
    const Q clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void ReluFloat(const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float lower = 0.0f;
    const float clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu6Float(const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6.0f;
    const float lower = 0.0f;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename Q>
inline void Relu6Quantized(Q lower, Q upper, const RuntimeShape& input_shape,
                           const Q* input_data,
                           const RuntimeShape& output_shape, Q* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const Q val = input_data[i];
    const Q clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = GetTensorShape(input);
      const RuntimeShape& output_shape = GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = GetTensorData<float>(input);
      out_data_ptr = GetTensorData<float>(output);

      const float f32_pos_inf = 0x7F800000;
      err = xa_nn_vec_relu_f32_f32(out_data_ptr, inp_data_ptr, f32_pos_inf,
                                   flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_relu1_f32_f32 failed");
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      ReluQuantized<int8_t>(input->params.zero_point, GetTensorShape(input),
                            GetTensorData<int8_t>(input),
                            GetTensorShape(output),
                            GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      int err;
      const uint8_t* inp_data_ptr;
      uint8_t* out_data_ptr;
      const RuntimeShape& input_shape = GetTensorShape(input);
      const RuntimeShape& output_shape = GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = GetTensorData<uint8_t>(input);
      out_data_ptr = GetTensorData<uint8_t>(output);

      err = xa_nn_vec_activation_min_max_asym8_asym8(
          out_data_ptr, inp_data_ptr, 0, 255, flat_size);  // Is 255 right?

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_activation_min_max_8_8 failed");
      return kTfLiteOk;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Only float32 is supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

TfLiteStatus Relu6Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = GetTensorShape(input);
      const RuntimeShape& output_shape = GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = GetTensorData<float>(input);
      out_data_ptr = GetTensorData<float>(output);

      err = xa_nn_vec_relu6_f32_f32(out_data_ptr, inp_data_ptr, flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_relu1_f32_f32 failed");
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const int8_t six = FloatToAsymmetricQuantizedInt8(
          6.0f, input->params.scale, input->params.zero_point);
      const int8_t zero = input->params.zero_point;
      Relu6Quantized<int8_t>(
          zero, six, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      const uint8_t six = FloatToAsymmetricQuantizedUInt8(
          6.0f, input->params.scale, input->params.zero_point);
      const uint8_t zero = input->params.zero_point;
      int err;
      const uint8_t* inp_data_ptr;
      uint8_t* out_data_ptr;
      const RuntimeShape& input_shape = GetTensorShape(input);
      const RuntimeShape& output_shape = GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = GetTensorData<uint8_t>(input);
      out_data_ptr = GetTensorData<uint8_t>(output);

      err = xa_nn_vec_activation_min_max_asym8_asym8(out_data_ptr, inp_data_ptr,
                                                     zero, six, flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_activation_min_max_8_8 failed");
      return kTfLiteOk;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Only float32 is supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace activations

TfLiteRegistration* Register_RELU() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::ReluPrepare,
                                 /*invoke=*/activations::ReluEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_RELU6() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::Relu6Prepare,
                                 /*invoke=*/activations::Relu6Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

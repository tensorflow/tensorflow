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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

struct ReluOpData {
  ReluParams params;
  uint8_t zero_uint8;
};

struct Relu6OpData {
  int8_t six_int8;
  int8_t zero_int8;
  uint8_t six_uint8;
  uint8_t zero_uint8;
};

}  // namespace

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename T>
inline void ReluQuantized(const ReluOpData& data,
                          const RuntimeShape& input_shape,
                          const RuntimeShape& output_shape, const T* input_data,
                          T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const int32_t val = static_cast<int32_t>(input_data[i]);
    int32_t clamped =
        data.params.output_offset +
        MultiplyByQuantizedMultiplier(val - data.params.input_offset,
                                      data.params.output_multiplier,
                                      data.params.output_shift);
    clamped = std::max(data.params.quantized_activation_min, clamped);
    clamped = std::min(data.params.quantized_activation_max, clamped);
    output_data[i] = static_cast<T>(clamped);
  }
}

template <typename T>
inline void CalculateReluOpData(const TfLiteTensor* input, TfLiteTensor* output,
                                ReluOpData* data) {
  float act_min = 0.0;
  float act_max = std::numeric_limits<float>::infinity();
  double real_multiplier =
      static_cast<double>(input->params.scale / output->params.scale);

  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);

  QuantizeMultiplier(real_multiplier, &data->params.output_multiplier,
                     &data->params.output_shift);

  data->params.quantized_activation_min = std::max(
      static_cast<int32_t>(std::numeric_limits<T>::min()),
      output->params.zero_point +
          static_cast<int32_t>(roundf(act_min / output->params.scale)));
  data->params.quantized_activation_max =
      act_max == std::numeric_limits<float>::infinity()
          ? static_cast<int32_t>(std::numeric_limits<T>::max())
          : std::min(static_cast<int32_t>(std::numeric_limits<T>::max()),
                     output->params.zero_point +
                         static_cast<int32_t>(
                             roundf(act_max / output->params.scale)));
  data->params.input_offset = input->params.zero_point;
  data->params.output_offset = output->params.zero_point;
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

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(ReluOpData));
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  ReluOpData* data = static_cast<ReluOpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  if (input->type == kTfLiteInt8) {
    CalculateReluOpData<int8_t>(input, output, data);
  } else if (input->type == kTfLiteUInt8) {
    CalculateReluOpData<uint8_t>(input, output, data);
    data->zero_uint8 = input->params.zero_point;
  }

  return kTfLiteOk;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const ReluOpData& data = *(static_cast<const ReluOpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if HIFI_VFPU
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<float>(input);
      out_data_ptr = tflite::micro::GetTensorData<float>(output);

      err = xa_nn_vec_relu_std_f32_f32(out_data_ptr, inp_data_ptr, flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_relu_std_f32_f32 failed");
#else
      ReluFloat(tflite::micro::GetTensorShape(input),
                tflite::micro::GetTensorData<float>(input),
                tflite::micro::GetTensorShape(output),
                tflite::micro::GetTensorData<float>(output));
#endif /* HIFI_VFPU */
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      ReluQuantized<int8_t>(data, tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(input),
                            tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      int err;
      const uint8_t* inp_data_ptr;
      uint8_t* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<uint8_t>(input);
      out_data_ptr = tflite::micro::GetTensorData<uint8_t>(output);

      err = xa_nn_vec_activation_min_max_asym8_asym8(
          out_data_ptr, inp_data_ptr, data.zero_uint8,
          std::numeric_limits<uint8_t>::max(), flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "xa_nn_vec_activation_min_max_asym8_asym8 failed");
      return kTfLiteOk;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Only float32 is supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

void* Relu6Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(Relu6OpData));
}

TfLiteStatus Relu6Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  Relu6OpData* data = static_cast<Relu6OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

  if (input->type == kTfLiteInt8) {
    data->six_int8 = FloatToAsymmetricQuantizedInt8(6.0f, input->params.scale,
                                                    input->params.zero_point);
    data->zero_int8 = input->params.zero_point;
  } else if (input->type == kTfLiteUInt8) {
    data->six_uint8 = FloatToAsymmetricQuantizedUInt8(6.0f, input->params.scale,
                                                      input->params.zero_point);
    data->zero_uint8 = input->params.zero_point;
  }

  return kTfLiteOk;
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const Relu6OpData& data = *(static_cast<const Relu6OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if HIFI_VFPU
      int err;
      const float* inp_data_ptr;
      float* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<float>(input);
      out_data_ptr = tflite::micro::GetTensorData<float>(output);

      err = xa_nn_vec_relu6_f32_f32(out_data_ptr, inp_data_ptr, flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_relu6_f32_f32 failed");
#else
      Relu6Float(tflite::micro::GetTensorShape(input),
                 tflite::micro::GetTensorData<float>(input),
                 tflite::micro::GetTensorShape(output),
                 tflite::micro::GetTensorData<float>(output));
#endif /* HIFI_VFPU */
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      Relu6Quantized<int8_t>(data.zero_int8, data.six_int8,
                             tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<int8_t>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      int err;
      const uint8_t* inp_data_ptr;
      uint8_t* out_data_ptr;
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      inp_data_ptr = tflite::micro::GetTensorData<uint8_t>(input);
      out_data_ptr = tflite::micro::GetTensorData<uint8_t>(output);

      err = xa_nn_vec_activation_min_max_asym8_asym8(out_data_ptr, inp_data_ptr,
                                                     data.zero_uint8,
                                                     data.six_uint8, flat_size);

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "xa_nn_vec_activation_min_max_asym8_asym8 failed");
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

TfLiteRegistration Register_RELU() {
  return {/*init=*/activations::ReluInit,
          /*free=*/nullptr,
          /*prepare=*/activations::ReluPrepare,
          /*invoke=*/activations::ReluEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RELU6() {
  return {/*init=*/activations::Relu6Init,
          /*free=*/nullptr,
          /*prepare=*/activations::Relu6Prepare,
          /*invoke=*/activations::Relu6Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

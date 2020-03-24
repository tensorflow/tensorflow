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

#include "tensorflow/lite/kernels/internal/reference/sub.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace sub {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32 output_activation_min;
  int32 output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int output_shift;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteSubParams* params,
                             const TfLiteTensor* input1,
                             const TfLiteTensor* input2, TfLiteTensor* output,
                             OpData* data) {
  data->requires_broadcast = !HaveSameShapes(input1, input2);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    // 8bit -> 8bit general quantized path, with general rescalings
    data->input1_offset = -input1->params.zero_point;
    data->input2_offset = -input2->params.zero_point;
    data->output_offset = output->params.zero_point;
    data->left_shift = 20;
    const float twice_max_input_scale =
        2 * std::max(input1->params.scale, input2->params.scale);
    const double real_input1_multiplier =
        static_cast<double>(input1->params.scale / twice_max_input_scale);
    const double real_input2_multiplier =
        static_cast<double>(input2->params.scale / twice_max_input_scale);
    const double real_output_multiplier =
        static_cast<double>(twice_max_input_scale /
                            ((1 << data->left_shift) * output->params.scale));

    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->input1_multiplier, &data->input1_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->input2_multiplier, &data->input2_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  return kTfLiteOk;
}

void EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpData* data, const TfLiteTensor* input1,
             const TfLiteTensor* input2, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
#define TF_LITE_SUB(opname)                                               \
  opname(op_params, GetTensorShape(input1), GetTensorData<float>(input1), \
         GetTensorShape(input2), GetTensorData<float>(input2),            \
         GetTensorShape(output), GetTensorData<float>(output))
  if (data->requires_broadcast) {
    TF_LITE_SUB(tflite::reference_ops::BroadcastSubSlow);
  } else {
    TF_LITE_SUB(tflite::reference_ops::SubWithActivation);
  }
#undef TF_LITE_SUB
}

TfLiteStatus EvalSubQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpData* data,
                              const TfLiteTensor* input1,
                              const TfLiteTensor* input2,
                              TfLiteTensor* output) {
  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    tflite::ArithmeticParams op_params;
    op_params.left_shift = data->left_shift;
    op_params.input1_offset = data->input1_offset;
    op_params.input1_multiplier = data->input1_multiplier;
    op_params.input1_shift = data->input1_shift;
    op_params.input2_offset = data->input2_offset;
    op_params.input2_multiplier = data->input2_multiplier;
    op_params.input2_shift = data->input2_shift;
    op_params.output_offset = data->output_offset;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    bool need_broadcast = reference_ops::ProcessBroadcastShapes(
        GetTensorShape(input1), GetTensorShape(input2), &op_params);
#define TF_LITE_SUB(opname, dtype)                                        \
  opname(op_params, GetTensorShape(input1), GetTensorData<dtype>(input1), \
         GetTensorShape(input2), GetTensorData<dtype>(input2),            \
         GetTensorShape(output), GetTensorData<dtype>(output));
    if (output->type == kTfLiteInt8) {
      if (need_broadcast) {
        TF_LITE_SUB(tflite::reference_ops::BroadcastSubSlow, int8_t);
      } else {
        TF_LITE_SUB(tflite::reference_ops::Sub, int8_t);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_SUB(tflite::reference_ops::BroadcastSubSlow, uint8_t);
      } else {
        TF_LITE_SUB(tflite::reference_ops::Sub, uint8_t);
      }
    }
#undef TF_LITE_SUB
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  OpData data;
  TF_LITE_ENSURE_STATUS(
      CalculateOpData(context, params, input1, input2, output, &data));

  if (output->type == kTfLiteFloat32) {
    EvalSub(context, node, params, &data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, EvalSubQuantized(context, node, params, &data,
                                                input1, input2, output));
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Inputs and outputs not all float|uint8|int8 types.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace sub

TfLiteRegistration* Register_SUB() {
  static TfLiteRegistration r = {};
  r.init = sub::Init;
  r.free = sub::Free;
  r.prepare = sub::Prepare;
  r.invoke = sub::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

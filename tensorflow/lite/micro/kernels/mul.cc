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

#include "tensorflow/lite/kernels/internal/reference/mul.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace mul {

constexpr int kInput1Tensor = 0;
constexpr int kInput2Tensor = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t output_multiplier;
  int output_shift;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteMulParams* params, OpData* data) {
  const TfLiteTensor* input1 = GetInput(context, node, kInput1Tensor);
  const TfLiteTensor* input2 = GetInput(context, node, kInput2Tensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));

    double real_multiplier = static_cast<double>(input1->params.scale) *
                             static_cast<double>(input2->params.scale) /
                             static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  return kTfLiteOk;
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteMulParams* params, OpData* data,
                   const TfLiteTensor* input1, const TfLiteTensor* input2,
                   TfLiteTensor* output) {
  if (output->type == kTfLiteInt8 || output->type == kTfLiteUInt8) {
    tflite::ArithmeticParams op_params;
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    op_params.input1_offset = -input1->params.zero_point;
    op_params.input2_offset = -input2->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    bool need_broadcast = reference_ops::ProcessBroadcastShapes(
        GetTensorShape(input1), GetTensorShape(input2), &op_params);

#define TF_LITE_MUL(type, opname, dtype)                             \
  type::opname(op_params, GetTensorShape(input1),                    \
               GetTensorData<dtype>(input1), GetTensorShape(input2), \
               GetTensorData<dtype>(input2), GetTensorShape(output), \
               GetTensorData<dtype>(output));

    if (output->type == kTfLiteInt8) {
      if (need_broadcast) {
        TF_LITE_MUL(reference_integer_ops, BroadcastMul4DSlow, int8_t);
      } else {
        TF_LITE_MUL(reference_integer_ops, Mul, int8_t);
      }
    } else if (output->type == kTfLiteUInt8) {
      if (need_broadcast) {
        TF_LITE_MUL(reference_ops, BroadcastMul4DSlow, uint8_t);
      } else {
        TF_LITE_MUL(reference_ops, Mul, uint8_t);
      }
    }
#undef TF_LITE_MUL
  }
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteMulParams* params, OpData* data,
               const TfLiteTensor* input1, const TfLiteTensor* input2,
               TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      GetTensorShape(input1), GetTensorShape(input2), &op_params);
#define TF_LITE_MUL(opname)                                                   \
  reference_ops::opname(op_params, GetTensorShape(input1),                    \
                        GetTensorData<float>(input1), GetTensorShape(input2), \
                        GetTensorData<float>(input2), GetTensorShape(output), \
                        GetTensorData<float>(output));

  if (need_broadcast) {
    TF_LITE_MUL(BroadcastMul4DSlow);
  } else {
    TF_LITE_MUL(Mul);
  }
#undef TF_LITE_MUL
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input1 = GetInput(context, node, kInput1Tensor);
  const TfLiteTensor* input2 = GetInput(context, node, kInput2Tensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, &data));

  switch (input1->type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      EvalQuantized(context, node, params, &data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalFloat(context, node, params, &data, input1, input2, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace mul

TfLiteRegistration* Register_MUL() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/mul::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

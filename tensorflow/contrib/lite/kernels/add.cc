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
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace add {

// This file has three implementation of Add.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void EvalAddFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLiteAddParams* params, const OpData* data,
                  TfLiteTensor* input1, TfLiteTensor* input2,
                  TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);
#define TF_LITE_ADD(type, opname)                                   \
  type::opname(GetTensorData<float>(input1), GetTensorDims(input1), \
               GetTensorData<float>(input2), GetTensorDims(input2), \
               output_activation_min, output_activation_max,        \
               GetTensorData<float>(output), GetTensorDims(output))
  if (kernel_type == kReference) {
    if (data->requires_broadcast) {
      TF_LITE_ADD(reference_ops, BroadcastAdd);
    } else {
      TF_LITE_ADD(reference_ops, Add);
    }
  } else {
    if (data->requires_broadcast) {
      TF_LITE_ADD(optimized_ops, BroadcastAdd);
    } else {
      TF_LITE_ADD(optimized_ops, Add);
    }
  }
#undef TF_LITE_ADD
}

template <KernelType kernel_type>
void EvalAddQuantized(TfLiteContext* context, TfLiteNode* node,
                      TfLiteAddParams* params, const OpData* data,
                      TfLiteTensor* input1, TfLiteTensor* input2,
                      TfLiteTensor* output) {
  auto input1_offset = -input1->params.zero_point;
  auto input2_offset = -input2->params.zero_point;
  auto output_offset = output->params.zero_point;
  const int left_shift = 20;
  const double twice_max_input_scale =
      2 * std::max(input1->params.scale, input2->params.scale);
  const double real_input1_multiplier =
      input1->params.scale / twice_max_input_scale;
  const double real_input2_multiplier =
      input2->params.scale / twice_max_input_scale;
  const double real_output_multiplier =
      twice_max_input_scale / ((1 << left_shift) * output->params.scale);

  int32 input1_multiplier;
  int input1_shift;
  QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &input1_multiplier,
                                   &input1_shift);
  int32 input2_multiplier;
  int input2_shift;
  QuantizeMultiplierSmallerThanOne(real_input2_multiplier, &input2_multiplier,
                                   &input2_shift);
  int32 output_multiplier;
  int output_shift;
  QuantizeMultiplierSmallerThanOne(real_output_multiplier, &output_multiplier,
                                   &output_shift);

  int32 output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(params->activation, output,
                                &output_activation_min, &output_activation_max);

#define TF_LITE_ADD(type, opname)                                            \
  type::opname(left_shift, GetTensorData<uint8_t>(input1),                   \
               GetTensorDims(input1), input1_offset, input1_multiplier,      \
               input1_shift, GetTensorData<uint8_t>(input2),                 \
               GetTensorDims(input2), input2_offset, input2_multiplier,      \
               input2_shift, output_offset, output_multiplier, output_shift, \
               output_activation_min, output_activation_max,                 \
               GetTensorData<uint8_t>(output), GetTensorDims(output));
  // The quantized version of Add doesn't support activations, so we
  // always use BroadcastAdd.
  if (kernel_type == kReference) {
    TF_LITE_ADD(reference_ops, BroadcastAdd);
  } else {
    TF_LITE_ADD(optimized_ops, BroadcastAdd);
  }
#undef TF_LITE_ADD
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32) {
    EvalAddFloat<kernel_type>(context, node, params, data, input1, input2,
                              output);
  } else if (output->type == kTfLiteUInt8) {
    EvalAddQuantized<kernel_type>(context, node, params, data, input1, input2,
                                  output);
  } else {
    context->ReportError(context,
                         "Inputs and outputs not all float|unit8 types.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace add

TfLiteRegistration* Register_ADD_REF() {
  static TfLiteRegistration r = {add::Init, add::Free, add::Prepare,
                                 add::Eval<add::kReference>};
  return &r;
}

TfLiteRegistration* Register_ADD_GENERIC_OPT() {
  static TfLiteRegistration r = {add::Init, add::Free, add::Prepare,
                                 add::Eval<add::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_ADD_NEON_OPT() {
  static TfLiteRegistration r = {add::Init, add::Free, add::Prepare,
                                 add::Eval<add::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_ADD() {
#ifdef USE_NEON
  return Register_ADD_NEON_OPT();
#else
  return Register_ADD_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

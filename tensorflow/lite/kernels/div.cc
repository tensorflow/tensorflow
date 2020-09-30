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
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace div {

// This file has three implementation of Div.
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

  // Parameters used in the quantized paths where the output is 8bit
  int32 output_activation_min;
  int32 output_activation_max;

  // Parameters used in all quantized paths
  int32_t output_multiplier;
  int output_shift;
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
  auto* params = reinterpret_cast<TfLiteDivParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
    const double real_multiplier =
        input1->params.scale / (input2->params.scale * output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void EvalDiv(TfLiteContext* context, TfLiteNode* node, TfLiteDivParams* params,
             const OpData* data, const TfLiteTensor* input1,
             const TfLiteTensor* input2, TfLiteTensor* output) {
#define TF_LITE_DIV(type, opname, data_type)                             \
  tflite::ArithmeticParams op_params;                                    \
  data_type output_activation_min, output_activation_max;                \
  CalculateActivationRange(params->activation, &output_activation_min,   \
                           &output_activation_max);                      \
  SetActivationParams(output_activation_min, output_activation_max,      \
                      &op_params);                                       \
  type::opname(op_params, GetTensorShape(input1),                        \
               GetTensorData<data_type>(input1), GetTensorShape(input2), \
               GetTensorData<data_type>(input2), GetTensorShape(output), \
               GetTensorData<data_type>(output))
  if (output->type == kTfLiteInt32) {
    if (kernel_type == kReference) {
      if (data->requires_broadcast) {
        TF_LITE_DIV(reference_ops, BroadcastDivSlow, int32_t);
      } else {
        TF_LITE_DIV(reference_ops, Div, int32_t);
      }
    } else {
      if (data->requires_broadcast) {
        TF_LITE_DIV(optimized_ops, BroadcastDivSlow, int32_t);
      } else {
        TF_LITE_DIV(optimized_ops, Div, int32_t);
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    if (kernel_type == kReference) {
      if (data->requires_broadcast) {
        TF_LITE_DIV(reference_ops, BroadcastDivSlow, float);
      } else {
        TF_LITE_DIV(reference_ops, Div, float);
      }
    } else {
      if (data->requires_broadcast) {
        TF_LITE_DIV(optimized_ops, BroadcastDivSlow, float);
      } else {
        TF_LITE_DIV(optimized_ops, Div, float);
      }
    }
  }
#undef TF_LITE_DIV
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDivParams* params, const OpData* data,
                           const TfLiteTensor* input1,
                           const TfLiteTensor* input2, TfLiteTensor* output) {
  if (input1->type == kTfLiteUInt8 && input2->type == kTfLiteUInt8 &&
      output->type == kTfLiteUInt8) {
    tflite::ArithmeticParams op_params;
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    op_params.input1_offset = -input1->params.zero_point;
    op_params.input2_offset = -input2->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
        GetTensorShape(input1), GetTensorShape(input2), &op_params);
#define TF_LITE_DIV(type, opname, dtype)                             \
  type::opname(op_params, GetTensorShape(input1),                    \
               GetTensorData<dtype>(input1), GetTensorShape(input2), \
               GetTensorData<dtype>(input2), GetTensorShape(output), \
               GetTensorData<dtype>(output))
    if (kernel_type == kReference) {
      if (need_broadcast) {
        TF_LITE_DIV(reference_ops, BroadcastDivSlow, uint8_t);
      } else {
        TF_LITE_DIV(reference_ops, Div, uint8_t);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_DIV(optimized_ops, BroadcastDivSlow, uint8_t);
      } else {
        TF_LITE_DIV(optimized_ops, Div, uint8_t);
      }
    }
#undef TF_LITE_DIV
  } else {
    context->ReportError(
        context, "Unsupported combination of input and output types in Div.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteDivParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt32) {
    EvalDiv<kernel_type>(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_OK(
        context, EvalQuantized<kernel_type>(context, node, params, data, input1,
                                            input2, output));
  } else {
    context->ReportError(
        context,
        "Div only supports FLOAT32, INT32 and quantized UINT8 now, got %d.",
        output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace div

TfLiteRegistration* Register_DIV_REF() {
  static TfLiteRegistration r = {div::Init, div::Free, div::Prepare,
                                 div::Eval<div::kReference>};
  return &r;
}

TfLiteRegistration* Register_DIV_GENERIC_OPT() {
  static TfLiteRegistration r = {div::Init, div::Free, div::Prepare,
                                 div::Eval<div::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_DIV_NEON_OPT() {
  static TfLiteRegistration r = {div::Init, div::Free, div::Prepare,
                                 div::Eval<div::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_DIV() {
#ifdef USE_NEON
  return Register_DIV_NEON_OPT();
#else
  return Register_DIV_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

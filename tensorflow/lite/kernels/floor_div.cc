/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/floor_div.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace floor_div {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for floor_div op.
struct OpData {
  bool requires_broadcast;
  // Parameters used in the quantized paths where the output is 8bit
  int32 output_activation_min;
  int32 output_activation_max;

  int q_one;
  int pot;
  // These fields are used only in the general 8-bit -> 8bit quantized path
  int input1_shift;
  int input2_shift;
  int output_shift;
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;
};

template <typename T>
T FloorDiv(T input1, T input2) {
  return std::floor(std::divides<double>()(static_cast<double>(input1),
                                           static_cast<double>(input2)));
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Reinterprete the opaque data provided by user.
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      // 8bit -> 8bit general quantized path, with general rescalings
      data->input1_offset = -input1->params.zero_point;
      data->input2_offset = -input2->params.zero_point;
      data->output_offset = output->params.zero_point;
      data->left_shift = 20;

      // To reuse the existing paramters, use input1 multiplier, to quantize
      // reciprocal(input2).
      double real_input1_multiplier =
          1 / ((1 << (data->left_shift)) * input2->params.scale);
      double real_input2_multiplier = input2->params.scale;
      double real_output_multiplier =
          input1->params.scale * input2->params.scale / output->params.scale;

      QuantizeMultiplier(real_input1_multiplier, &data->input1_multiplier,
                         &data->input1_shift);

      QuantizeMultiplier(real_input2_multiplier, &data->input2_multiplier,
                         &data->input2_shift);

      QuantizeMultiplier(real_output_multiplier, &data->output_multiplier,
                         &data->output_shift);

      // Get the Quantized value of One
      if (type == kTfLiteUInt8) {
        data->q_one = static_cast<uint8_t>(std::max<float>(
            std::numeric_limits<uint8_t>::min(),
            std::min<float>(std::numeric_limits<uint8_t>::max(),
                            std::round(input1->params.zero_point +
                                       (1.0 / input1->params.scale)))));
      } else {
        data->q_one = static_cast<int8_t>(std::max<float>(
            std::numeric_limits<int8_t>::min(),
            std::min<float>(std::numeric_limits<int8_t>::max(),
                            std::round(input1->params.zero_point +
                                       (1.0 / input1->params.scale)))));
      }

      data->q_one = data->q_one - input1->params.zero_point;

      // Get the power of two for the Quantized value
      data->pot = std::ceil(std::log2(data->q_one));

    } break;

    default:
      context->ReportError(context, "Type '%s' is not supported by floor_div.",
                           TfLiteTypeGetName(type));
      return kTfLiteError;
  }
  output->type = type;

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

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, bool requires_broadcast,
                      const TfLiteTensor* input1, const TfLiteTensor* input2,
                      TfLiteTensor* output) {
  const T* denominator_data = GetTensorData<T>(input2);

  // Validate the denominator.
  for (int i = 0; i < NumElements(input2); ++i) {
    if (std::equal_to<T>()(denominator_data[i], 0)) {
      context->ReportError(context, "Division by 0");
      return kTfLiteError;
    }
  }
  if (requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), denominator_data, GetTensorShape(output),
        GetTensorData<T>(output), FloorDiv<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), FloorDiv<T>);
  }

  return kTfLiteOk;
}

void EvalFloorDivQuantized(TfLiteContext* context, bool need_broadcast,
                           TfLiteNode* node, TfLiteDivParams* params,
                           const OpData* data, const TfLiteTensor* input1,
                           const TfLiteTensor* input2, TfLiteTensor* output) {
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

#define TF_LITE_FLOOR_DIV(type, opname, dtype)                       \
  type::opname(op_params, GetTensorShape(input1),                    \
               GetTensorData<dtype>(input1), GetTensorShape(input2), \
               GetTensorData<dtype>(input2), GetTensorShape(output), \
               GetTensorData<dtype>(output), data->q_one, data->pot)
  if (input1->type == kTfLiteInt8) {
    if (need_broadcast) {
      TF_LITE_FLOOR_DIV(reference_integer_ops, BroadcastFloorDiv4DSlow, int8_t);
    } else {
      TF_LITE_FLOOR_DIV(reference_integer_ops, FloorDiv, int8_t);
    }
  } else {
    // type == kTfLiteUInt8
    if (need_broadcast) {
      TF_LITE_FLOOR_DIV(reference_integer_ops, BroadcastFloorDiv4DSlow,
                        uint8_t);
    } else {
      TF_LITE_FLOOR_DIV(reference_integer_ops, FloorDiv, uint8_t);
    }
  }
#undef TF_LITE_FLOOR_DIV
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input1->type) {
    case kTfLiteInt32: {
      return EvalImpl<int32_t>(context, data->requires_broadcast, input1,
                               input2, output);
    }
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, data->requires_broadcast, input1, input2,
                             output);
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      auto* params = reinterpret_cast<TfLiteDivParams*>(node->builtin_data);
      EvalFloorDivQuantized(context, data->requires_broadcast, node, params,
                            data, input1, input2, output);
      return kTfLiteOk;
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by floor_div.",
                           TfLiteTypeGetName(input1->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace floor_div

TfLiteRegistration* Register_FLOOR_DIV() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  static TfLiteRegistration r = {floor_div::Init, floor_div::Free,
                                 floor_div::Prepare, floor_div::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

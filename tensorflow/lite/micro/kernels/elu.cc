/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/elu.h"

#include <algorithm>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

// Input/output tensor index.
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// OLD-TODO(b/142762739): We should figure out a multi-threading plan for most
// of the activation ops below.

struct OpData {
  int8_t table[256];
};

using TransformFunc = float (*)(float);

template <typename T>
void PopulateLookupTable(const TfLiteTensor* input, const TfLiteTensor* output,
                         const TransformFunc transform, OpData* data) {
  if (sizeof(T) != 1) TF_LITE_FATAL("Lookup table valid only for 8bit");

  const float inverse_scale = 1 / output->params.scale;
  int32_t maxval = std::numeric_limits<T>::max();
  int32_t minval = std::numeric_limits<T>::min();
  for (int32_t val = minval; val <= maxval; ++val) {
    const float dequantized =
        input->params.scale * (val - input->params.zero_point);
    const float transformed = transform(dequantized);
    const float rescaled = TfLiteRound(transformed * inverse_scale);
    const int32_t quantized =
        static_cast<int32_t>(rescaled + output->params.zero_point);
    data->table[static_cast<uint8_t>(static_cast<T>(val))] =
        static_cast<T>(std::max(std::min(maxval, quantized), minval));
  }
}

// OLD-TODO(b/143696793): move this to optimized_ops.
void EvalUsingLookupTable(const OpData* data, const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output) {
  const int size = MatchingFlatSize(tflite::micro::GetTensorShape(input),
                                    tflite::micro::GetTensorShape(output));
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);

  for (int i = 0; i < size; ++i) {
    output_data[i] = data->table[static_cast<uint8_t>(input_data[i])];
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  // Use LUT to handle quantized elu path.
  if (input->type == kTfLiteInt8) {
    OpData* data = static_cast<OpData*>(node->user_data);
    TransformFunc transform = [](float value) {
      return value < 0.0f ? std::exp(value) - 1.0f : value;
    };
    PopulateLookupTable<int8_t>(input, output, transform, data);
  }

  return kTfLiteOk;
}

void* EluInit(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus EluPrepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Elu(tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<float>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const OpData* data = static_cast<OpData*>(node->user_data);
      EvalUsingLookupTable(data, input, output);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context, "ELU only supports float32 and int8 currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace

TfLiteRegistration Register_ELU() {
  return {/*init=*/EluInit,
          /*free=*/nullptr,
          /*prepare=*/EluPrepare,
          /*invoke=*/EluEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite

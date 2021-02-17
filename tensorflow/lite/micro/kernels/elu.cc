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
#include <cmath>
#include <functional>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

// OLD-TODO(b/142762739): We should figure out a multi-threading plan for most
// of the activation ops below.

struct OpData {
  uint8_t table[256] = {0};
};

template <typename T>
void PopulateLookupTable(struct OpData* data, const TfLiteTensor* input,
                         TfLiteTensor* output,
                         const std::function<float(float)>& transform) {
  static_assert(sizeof(T) == 1, "Lookup table valid only for 8bit");
  const float inverse_scale = 1 / output->params.scale;
  int32_t maxval = std::numeric_limits<T>::max();
  int32_t minval = std::numeric_limits<T>::min();
  for (int32_t val = minval; val <= maxval; ++val) {
    const float dequantized =
        input->params.scale * (val - input->params.zero_point);
    const float transformed = transform(dequantized);
    const float rescaled = std::round(transformed * inverse_scale);
    const int32_t quantized =
        static_cast<int32_t>(rescaled + output->params.zero_point);
    data->table[static_cast<uint8_t>(static_cast<T>(val))] =
        static_cast<uint8_t>(
            static_cast<T>(std::max(std::min(maxval, quantized), minval)));
  }
}

// OLD-TODO(b/143696793): move this to optimized_ops.
void EvalUsingLookupTable(struct OpData* data, const TfLiteTensor* input,
                          TfLiteTensor* output) {
  const int size =
      MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));
  uint8_t* output_data = GetTensorData<uint8_t>(output);
  const uint8_t* input_data = GetTensorData<uint8_t>(input);
  int i = 0;

  for (; i < size; ++i) {
    output_data[i] = data->table[input_data[i]];
  }
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return nullptr;
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  return kTfLiteError;
}

TfLiteStatus EluPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Use LUT to handle quantized elu path.
  if (input->type == kTfLiteInt8) {
    PopulateLookupTable<int8_t>(data, input, output, [](float value) {
      return value < 0.0 ? std::exp(value) - 1.0f : value;
    });
  }
  return GenericPrepare(context, node);
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Elu(GetTensorShape(input), GetTensorData<float>(input),
                         GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      OpData* data = reinterpret_cast<OpData*>(node->user_data);
      EvalUsingLookupTable(data, input, output);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context, "Only float32 and int8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_ELU() { return nullptr; }

}  // namespace micro
}  // namespace ops
}  // namespace tflite

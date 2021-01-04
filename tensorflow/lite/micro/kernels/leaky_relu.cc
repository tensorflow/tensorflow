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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace activations {
namespace {

// OLD-TODO(b/142762739): We should figure out a multi-threading plan for most
// of the activation ops below.

enum KernelType {
  kReference,
  kGenericOptimized,
  kFixedPointOptimized,
};

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
  uint8_t table[256] = {0};
};

struct LeakyReluOpData : public OpData {
  int32_t output_multiplier_alpha = 0;
  int32_t output_shift_alpha = 0;
  int32_t output_multiplier_identity = 0;
  int32_t output_shift_identity = 0;
};

template <typename T>
void QuantizeLeakyRelu(const TfLiteTensor* input, TfLiteTensor* output,
                       const LeakyReluOpData* data) {
  LeakyReluParams op_params;

  op_params.input_offset = input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier_alpha = data->output_multiplier_alpha;
  op_params.output_shift_alpha = data->output_shift_alpha;
  op_params.output_multiplier_identity = data->output_multiplier_identity;
  op_params.output_shift_identity = data->output_shift_identity;
  reference_ops::QuantizeLeakyRelu(
      op_params, GetTensorShape(input), GetTensorData<T>(input),
      GetTensorShape(output), GetTensorData<T>(output));
}

}  // namespace

void* LeakyReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new LeakyReluOpData;
}

void LeakyReluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<LeakyReluOpData*>(buffer);
}

TfLiteStatus LeakyReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  LeakyReluOpData* data = reinterpret_cast<LeakyReluOpData*>(node->user_data);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
      output->type == kTfLiteInt16) {
    const auto* params =
        reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);

    double alpha_multiplier =
        input->params.scale * params->alpha / output->params.scale;
    QuantizeMultiplier(alpha_multiplier, &data->output_multiplier_alpha,
                       &data->output_shift_alpha);
    double identity_multiplier = input->params.scale / output->params.scale;
    QuantizeMultiplier(identity_multiplier, &data->output_multiplier_identity,
                       &data->output_shift_identity);
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const auto* params =
      reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);
  const LeakyReluOpData* data =
      reinterpret_cast<LeakyReluOpData*>(node->user_data);

  LeakyReluParams op_params;
  switch (input->type) {
    case kTfLiteFloat32: {
      op_params.alpha = params->alpha;
      optimized_ops::LeakyRelu(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      QuantizeLeakyRelu<uint8_t>(input, output, data);
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      QuantizeLeakyRelu<int8_t>(input, output, data);
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      QuantizeLeakyRelu<int16_t>(input, output, data);
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Only float32, int8, int16 and uint8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_LEAKY_RELU() {
  static TfLiteRegistration r = {
      activations::LeakyReluInit, activations::LeakyReluFree,
      activations::LeakyReluPrepare, activations::LeakyReluEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

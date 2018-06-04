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
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

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
namespace activations {

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8) {
    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
        input->params.scale *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    QuantizeMultiplierGreaterThanOne(input_real_multiplier,
                                     &data->input_multiplier,
                                     &data->input_left_shift);
    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SigmoidPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    TF_LITE_ENSURE(context, output->params.scale == 1. / 256);

    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
        input->params.scale *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    QuantizeMultiplierGreaterThanOne(input_real_multiplier,
                                     &data->input_multiplier,
                                     &data->input_left_shift);
    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  TF_LITE_ENSURE(context,
                 NumDimensions(input) == 2 || NumDimensions(input) == 4);

  if (input->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    TF_LITE_ENSURE(context, output->params.scale == 1. / 256);

    static const int kScaledDiffIntegerBits = 5;

    tflite::PreprocessSoftmaxScaling(
        params->beta, input->params.scale, kScaledDiffIntegerBits,
        &data->input_multiplier, &data->input_left_shift);
    data->diff_min = -1.0 * tflite::CalculateInputRadius(
                                kScaledDiffIntegerBits, data->input_left_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* alpha = GetInput(context, node, 1);

  output->type = input->type;

  // Currently only Float32 is supported
  // TODO(ycling): Support other data types.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, alpha->type, kTfLiteFloat32);

  // Currently, only support 4D `input` and 3D `alpha` with shape
  // (1, 1, channels).
  // TODO(impjdi): Support other cases where `alpha` is broadcastable
  // to `input`.
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, alpha->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[1], 1);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[2], input->dims->data[3]);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* out = output->data.f;
      for (; in < in_end; in++, out++) *out = std::max(0.f, *in);
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context, "Only float32 supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* out = output->data.f;
      for (; in < in_end; in++, out++) {
        *out = std::min(std::max(-1.f, *in), 1.f);
      }
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context, "Only float32 supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* out = output->data.f;
      for (; in < in_end; in++, out++) *out = std::min(std::max(0.f, *in), 6.f);
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context, "Only float32 supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
}

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* out = output->data.f;
      for (; in < in_end; in++, out++) *out = std::tanh(*in);
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      optimized_ops::Tanh(GetTensorData<uint8_t>(input), GetTensorDims(input),
                          input->params.zero_point, data->input_range_radius,
                          data->input_multiplier, data->input_left_shift,
                          GetTensorData<uint8_t>(output),
                          GetTensorDims(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context, "Only float32 supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
}

// Sigmoid is also know as "Logistic".
TfLiteStatus SigmoidEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* out = output->data.f;
      for (; in < in_end; in++, out++) *out = 1.f / (1.f + std::exp(-*in));
      break;
    }
    case kTfLiteUInt8: {
      optimized_ops::Logistic(
          GetTensorData<uint8_t>(input), GetTensorDims(input),
          input->params.zero_point, data->input_range_radius,
          data->input_multiplier, data->input_left_shift,
          GetTensorData<uint8_t>(output), GetTensorDims(output));
      break;
    }
    default:
      context->ReportError(context, "Only float32 supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Takes a 2D tensor and perform softmax along the second dimension.
void Softmax2DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  float* in = input->data.f;
  float* out = output->data.f;
  TF_LITE_ASSERT(input_size > 0);

  // For each batch
  for (int b = 0; b < batch_size; b++) {
    // Find the max coeff.
    float max_coeff = in[0];
    for (int i = 1; i < input_size; i++) {
      if (in[i] > max_coeff) max_coeff = in[i];
    }

    // Compute the normalized sum of exps.
    float exp_sum = 0.0;
    for (int i = 0; i < input_size; i++) {
      out[i] = std::exp((in[i] - max_coeff) * params->beta);
      exp_sum += out[i];
    }

    // Divide by the sum of exps.
    float reciprocal_sum_exp = 1.f / exp_sum;
    for (int i = 0; i < input_size; i++) {
      out[i] *= reciprocal_sum_exp;
    }

    // Advance in and out pointers for the next batch.
    in += input_size;
    out += input_size;
  }
}

void Softmax2DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  // TODO(ahentz): this is arguably a dirty trick. Since the implementation
  // always traverses the last dimension of a 4D tensor, we will pretend our 2D
  // tensor is 4D in a special way. We will convert a (X, Y) shape into a (X,
  // 1, 1, Y) shape.
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  optimized_ops::Softmax(GetTensorData<uint8_t>(input),
                         GetTensorDims({batch_size, 1, 1, input_size}),
                         data->input_multiplier, data->input_left_shift,
                         data->diff_min, GetTensorData<uint8_t>(output),
                         GetTensorDims({batch_size, 1, 1, input_size}));
}

// Takes a 4D tensor and perform softmax along the forth dimension.
void Softmax4DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  optimized_ops::Softmax(GetTensorData<float>(input), GetTensorDims(input),
                         params->beta, GetTensorData<float>(output),
                         GetTensorDims(output));
}

void Softmax4DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  optimized_ops::Softmax(GetTensorData<uint8_t>(input), GetTensorDims(input),
                         data->input_multiplier, data->input_left_shift,
                         data->diff_min, GetTensorData<uint8_t>(output),
                         GetTensorDims(output));
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // TODO(ahentz): consider an implementation that works for many (all?)
  // dimensions.
  switch (input->type) {
    case kTfLiteFloat32: {
      if (NumDimensions(input) == 2) {
        Softmax2DFloat(input, output, params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 4) {
        Softmax4DFloat(input, output, params);
        return kTfLiteOk;
      }
      context->ReportError(
          context, "Only 2D and 4D tensors supported currently, got %dD.",
          NumDimensions(input));
      return kTfLiteError;
    }
    case kTfLiteUInt8: {
      if (NumDimensions(input) == 2) {
        Softmax2DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 4) {
        Softmax4DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      context->ReportError(
          context, "Only 2D and 4D tensors supported currently, got %dD.",
          NumDimensions(input));
      return kTfLiteError;
    }
    default:
      context->ReportError(
          context, "Only float32 and uint8_t supported currently, got %d.",
          input->type);
      return kTfLiteError;
  }
}

TfLiteStatus LogSoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32:
      optimized_ops::LogSoftmax(
          GetTensorData<float>(input), GetTensorDims(input),
          GetTensorData<float>(output), GetTensorDims(output));
      return kTfLiteOk;
    default:
      context->ReportError(context, "Only float32 supported currently., got %d",
                           input->type);
      return kTfLiteError;
  }
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* alpha = GetInput(context, node, 1);
  const TfLiteTensor* output = GetOutput(context, node, 0);

  if (input->type != kTfLiteFloat32) {
    context->ReportError(context, "Only float32 supported currently, got %d.",
                         input->type);
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  const int batches = input->dims->data[0];
  const int height = input->dims->data[1];
  const int width = input->dims->data[2];
  const int channels = input->dims->data[3];

  TF_LITE_ENSURE_EQ(context, alpha->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[1], 1);
  TF_LITE_ENSURE_EQ(context, alpha->dims->data[2], channels);

  const int n = batches * height * width * channels;
  for (int i = 0; i < n; ++i) {
    const float x = input->data.f[i];
    output->data.f[i] = x >= 0.0f ? x : alpha->data.f[i % channels] * x;
  }

  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_RELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 activations::GenericPrepare,
                                 activations::ReluEval};
  return &r;
}

TfLiteRegistration* Register_RELU_N1_TO_1() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 activations::GenericPrepare,
                                 activations::Relu1Eval};
  return &r;
}

TfLiteRegistration* Register_RELU6() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 activations::GenericPrepare,
                                 activations::Relu6Eval};
  return &r;
}

TfLiteRegistration* Register_TANH() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::TanhPrepare,
                                 activations::TanhEval};
  return &r;
}

TfLiteRegistration* Register_LOGISTIC() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::SigmoidPrepare,
                                 activations::SigmoidEval};
  return &r;
}

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::SoftmaxPrepare,
                                 activations::SoftmaxEval};
  return &r;
}

TfLiteRegistration* Register_LOG_SOFTMAX() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::GenericPrepare,
                                 activations::LogSoftmaxEval};
  return &r;
}

TfLiteRegistration* Register_PRELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 activations::PreluPrepare,
                                 activations::PreluEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

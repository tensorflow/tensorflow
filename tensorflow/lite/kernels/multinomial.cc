/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace multinomial {

struct MultinomialParams {
  std::default_random_engine rng;
};

// Draws a sample from a categorical distribution.
template <typename FloatType, typename IntegralType>
TfLiteStatus MultinomialSample(std::default_random_engine& rng,
                               const FloatType* logits, int logits_size,
                               IntegralType* outputs, int output_size) {
  // Computes arg_max(cumsum(exp(logits)) > rand()).
  // TODO(b/169166131): Remove hard-coded double for constrained use-cases.
  std::vector<double> cumulative_odds;
  cumulative_odds.reserve(logits_size);
  double last_odds = 0.0;

  // Compute max logit for numerical stability.
  FloatType max_logit = std::numeric_limits<FloatType>::lowest();
  for (int i = 0; i < logits_size; i++) {
    max_logit = std::max(max_logit, logits[i]);
  }

  for (int i = 0; i < logits_size; i++) {
    FloatType odds = std::exp(logits[i] - max_logit) + last_odds;
    cumulative_odds.push_back(odds);
    last_odds = odds;
  }

  std::uniform_real_distribution<double> distribution{0.0,
                                                      cumulative_odds.back()};

  for (int i = 0; i < output_size; i++) {
    double sample = distribution(rng);
    auto it = std::lower_bound(cumulative_odds.begin(), cumulative_odds.end(),
                               sample);
    if (it == cumulative_odds.end()) {
      // This should be impossible by construction.
      return kTfLiteError;
    }
    *outputs++ = static_cast<IntegralType>(it - cumulative_odds.begin());
  }
  return kTfLiteOk;
}

template <typename FloatType>
TfLiteStatus MultinomialSample(TfLiteContext* context,
                               std::default_random_engine& rng,
                               const FloatType* logits, int logits_size,
                               TfLiteTensor* output, int outputs_offset,
                               int output_size) {
  switch (output->type) {
    case kTfLiteInt32:
      return MultinomialSample<FloatType, int32_t>(
          rng, logits, logits_size,
          GetTensorData<int32_t>(output) + outputs_offset, output_size);
      break;
    case kTfLiteInt64:
      return MultinomialSample<FloatType, int64_t>(
          rng, logits, logits_size,
          GetTensorData<int64_t>(output) + outputs_offset, output_size);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported datatype for multinomial output: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
}

TfLiteStatus MultinomialSample(TfLiteContext* context,
                               std::default_random_engine& rng,
                               const TfLiteTensor* logits, int logits_offset,
                               int logits_size, TfLiteTensor* output,
                               int outputs_offset, int output_size) {
  switch (logits->type) {
    case kTfLiteFloat16:
      TF_LITE_KERNEL_LOG(context, "TfLiteFloat16 is currently not supported.");
      return kTfLiteError;
      break;
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(
          context,
          MultinomialSample<float>(
              context, rng, GetTensorData<float>(logits) + logits_offset,
              logits_size, output, outputs_offset, output_size));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(
          context,
          MultinomialSample<double>(
              context, rng, GetTensorData<double>(logits) + logits_offset,
              logits_size, output, outputs_offset, output_size));
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported datatype for multinomial logit input: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new MultinomialParams();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<MultinomialParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // TODO(b/169166131): Handle optional seed input.
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  // 'logits' is a float matrix [batch_size, num_categories]
  const TfLiteTensor* logits_input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(logits_input), 2);
  int batch_size = tflite::SizeOfDimension(logits_input, 0);

  // 'num_samples' is an int scalar.
  const TfLiteTensor* num_samples_input = tflite::GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(num_samples_input), 0);
  // TODO(b/169166131): Allow different integer input types.
  TF_LITE_ENSURE_EQ(context, num_samples_input->type, kTfLiteInt32);
  // TODO(b/169166131): Support dynamic output tensors.
  TF_LITE_ENSURE(context, IsConstantTensor(num_samples_input));

  int num_samples = *num_samples_input->data.i32;

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
  output_shape->data[0] = batch_size;
  output_shape->data[1] = num_samples;

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  // ResizeTensor takes ownership of output_shape.
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // TODO(b/169166131): Handle optional seed input.
  MultinomialParams* params =
      reinterpret_cast<MultinomialParams*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  const TfLiteTensor* logits = tflite::GetInput(context, node, 0);
  int batch_size = tflite::SizeOfDimension(logits, 0);
  int logits_size = tflite::SizeOfDimension(logits, 1);

  const TfLiteTensor* num_samples_input = tflite::GetInput(context, node, 1);
  int output_size = *num_samples_input->data.i32;

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  for (int batch = 0; batch < batch_size; ++batch) {
    int logits_offset = logits_size * batch;
    int output_offset = output_size * batch;

    TF_LITE_ENSURE_OK(
        context,
        MultinomialSample(context, params->rng, logits, logits_offset,
                          logits_size, output, output_offset, output_size));
  }

  return kTfLiteOk;
}

}  // namespace multinomial

TfLiteRegistration* Register_MULTINOMIAL() {
  static TfLiteRegistration r = {multinomial::Init, multinomial::Free,
                                 multinomial::Prepare, multinomial::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite

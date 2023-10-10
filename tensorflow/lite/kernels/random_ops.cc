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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tsl/lib/random/philox_random.h"
#include "tsl/lib/random/random_distributions_utils.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace random {

namespace {

using Generator = ::tsl::random::PhiloxRandom;

enum RandomType { kRandomUniform, kRandomStandardNormal, kMultinomial };

struct OpData {
  Generator rng;
};

// Initialize the OpData based on the seed and seed2 values.
void InitializeOpData(TfLiteNode* node) {
  static std::mt19937_64* seed_generator = []() {
    std::random_device device("/dev/urandom");
    return new std::mt19937_64(device());
  }();
  auto* params = static_cast<TfLiteRandomParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int64_t seed = params->seed;
  int64_t seed2 = params->seed2;
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, generate non-deterministic random numbers.
    seed = (*seed_generator)();
    seed2 = (*seed_generator)();
  }
  Generator rng(seed, seed2);
  data->rng = rng;
}

// Generates random numbers following a uniform distribution.
// Source: third_party/tensorflow/core/kernels/random_op.cc
void GenerateRandomUniformNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i++) {
      buffer[current_size + i] = tsl::random::Uint32ToFloat(samples[i]);
    }
    current_size += rng_net_size;
  }
}

// Generates random numbers following a standard normal distribution.
// Source: third_party/tensorflow/core/kernels/random_op.cc
void GenerateRandomStandardNormalNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i += 2) {
      tsl::random::BoxMullerFloat(samples[i], samples[i + 1],
                                  &buffer[current_size + i],
                                  &buffer[current_size + i + 1]);
    }
    current_size += rng_net_size;
  }
}

// Generates random numbers following a multinomial distribution.
// Source: third_party/tensorflow/core/kernels/multinomial_op.cc
template <typename IntType>
void GenerateMultinomialNumbers(Generator& rng, int batch_size,
                                const float* logits, size_t logits_size,
                                IntType* output, size_t num_samples) {
  // Skip a large fixed number of samples in the rng (random number generator)
  // for each op invoke to ensure that the output is always unique. (Make a copy
  // of the rng before skipping samples to use it in the current op invoke)
  // Context: This feature (to skip fixed samples) was added in TF as some
  // versions of the Multinomial op draw an unknown number of samples from the
  // rng. Though the TFLite version below only draws a fixed number of samples,
  // we still need to keep this feature to maintain parity with the TF op.
  Generator rng_copy = rng;
  rng.Skip(batch_size * ((num_samples + 3) / 4 * 4) * 2 *
           256);  // Round to a multiple of 4, 2x is for CPU and 256 is a
                  // conservative multiplier

  // Variables to store intermediate results between batches.
  typename Generator::ResultType rng_results;
  int used_rng_results_index = Generator::kResultElementCount;
  typename Generator::ResultElementType x0, x1;

  // Iterate over all batches to compute the outputs.
  for (int batch = 0; batch < batch_size; ++batch) {
    const float* logits_row = logits + batch * logits_size;
    IntType* output_row = output + batch * num_samples;

    // Compute the maximum logit.
    float max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < logits_size; i++) {
      if (std::isfinite(logits_row[i])) {
        max = std::max(max, logits_row[i]);
      }
    }
    const double max_logit = static_cast<double>(max);

    // Compute the (unnormalized) cumulative probability distribution.
    // For numerical stability (as the exponential function grows very fast),
    // subtract the maximum logit. Though you can subtract any value without
    // changing the output, we use the maximum logit for convenience.
    std::vector<double> cdf(logits_size);
    double cumulative_total = 0.0f;
    for (size_t i = 0; i < logits_size; i++) {
      if (std::isfinite(logits_row[i])) {
        cumulative_total += exp(logits_row[i] - max_logit);
      }
      cdf[i] = cumulative_total;
    }

    // Generate random categorical numbers and populate the output.
    for (int64_t j = 0; j < num_samples; ++j) {
      if (used_rng_results_index == Generator::kResultElementCount) {
        rng_results = rng_copy();
        used_rng_results_index = 0;
      }
      x0 = rng_results[used_rng_results_index];
      x1 = rng_results[used_rng_results_index + 1];
      used_rng_results_index += 2;
      const double to_find =
          (tsl::random::Uint64ToDouble(x0, x1) * cumulative_total);
      auto found_iter = std::upper_bound(cdf.begin(), cdf.end(), to_find);
      output_row[j] = std::distance(cdf.begin(), found_iter);
    }
  }
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // 'shape' is a 1-D int array
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &shape));
  TF_LITE_ENSURE_EQ(context, shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shape), 1);

  // Initialize the random number generator
  InitializeOpData(node);

  TfLiteTensor* output = GetOutput(context, node, 0);
  if (!IsConstantOrPersistentTensor(shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TfLiteIntArray* output_shape;
  TF_LITE_ENSURE_OK(context,
                    GetOutputShapeFromInput(context, shape, &output_shape));
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus PrepareMultinomial(TfLiteContext* context, TfLiteNode* node) {
  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // 'logits' is a 2-D input float matrix with shape [batch_size, num_classes]
  const TfLiteTensor* logits;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &logits));
  TF_LITE_ENSURE(context, logits->type == kTfLiteFloat32);

  // 'num_samples' is a 0-D input int scalar
  const TfLiteTensor* num_samples;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &num_samples));
  TF_LITE_ENSURE_EQ(context, num_samples->type, kTfLiteInt32);

  // Initialize the random number generator
  InitializeOpData(node);

  TfLiteTensor* output = GetOutput(context, node, 0);
  if (!IsConstantOrPersistentTensor(logits) ||
      !IsConstantOrPersistentTensor(num_samples)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  // 'output' is a 2-D int64 matrix with shape [batch_size, num_samples]
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
  output_shape->data[0] = SizeOfDimension(logits, 0);  // batch_size
  output_shape->data[1] = *num_samples->data.i32;      // num_samples
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus EvalRandomType(
    TfLiteContext* context, TfLiteNode* node, RandomType random_type) {
  TfLiteTensor* output = GetOutput(context, node, 0);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const size_t output_size = NumElements(output);
  switch (random_type) {
    case kRandomUniform:
      GenerateRandomUniformNumbers(
        data->rng, GetTensorData<float>(output), output_size);
      break;
    case kRandomStandardNormal:
      GenerateRandomStandardNormalNumbers(
        data->rng, GetTensorData<float>(output), output_size);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <RandomType rtype>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, 0);

  if (IsDynamicTensor(output)) {
    const TfLiteTensor* shape = GetInput(context, node, 0);
    TfLiteIntArray* output_shape;
    TF_LITE_ENSURE_OK(context,
                      GetOutputShapeFromInput(context, shape, &output_shape));
    context->ResizeTensor(context, output, output_shape);
  }

  switch (output->type) {
    case kTfLiteFloat32:
        EvalRandomType(context, node, rtype);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Unsupported output datatype for %s op: %s",
          rtype == kRandomUniform? "RandomUniform": "RandomStandardNormal",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMultinomial(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // 'logits' is a 2-D float matrix with shape [batch_size, num_classes]
  const TfLiteTensor* logits_tensor = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(logits_tensor), 2);
  const float* logits = GetTensorData<float>(logits_tensor);
  const int batch_size = SizeOfDimension(logits_tensor, 0);
  const int num_classes = SizeOfDimension(logits_tensor, 1);
  TF_LITE_ENSURE(context, num_classes > 0);

  // 'num_samples' is an int scalar
  const TfLiteTensor* num_samples_tensor = GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(num_samples_tensor), 0);
  const int num_samples = *num_samples_tensor->data.i32;
  TF_LITE_ENSURE(context, num_samples >= 0);

  TfLiteTensor* output_tensor = GetOutput(context, node, 0);
  if (IsDynamicTensor(output_tensor)) {
    // 'output' is a 2-D int64 matrix with shape [batch_size, num_samples]
    TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
    output_shape->data[0] = batch_size;
    output_shape->data[1] = num_samples;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output_tensor, output_shape));
  }

  switch (output_tensor->type) {
    case kTfLiteInt64:
      GenerateMultinomialNumbers<int64_t>(
          data->rng, batch_size, logits, num_classes,
          GetTensorData<int64_t>(output_tensor), num_samples);
      break;
    case kTfLiteInt32:
      GenerateMultinomialNumbers<int32_t>(
          data->rng, batch_size, logits, num_classes,
          GetTensorData<int32_t>(output_tensor), num_samples);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for Multinomial op: %s",
                         TfLiteTypeGetName(output_tensor->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace random

TfLiteRegistration* Register_RANDOM_UNIFORM() {
  static TfLiteRegistration r = {random::Init, random::Free, random::Prepare,
                                 random::Eval<random::kRandomUniform>};
  return &r;
}

TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() {
  static TfLiteRegistration r = {random::Init, random::Free, random::Prepare,
                                 random::Eval<random::kRandomStandardNormal>};
  return &r;
}

TfLiteRegistration* Register_MULTINOMIAL() {
  static TfLiteRegistration r = {random::Init, random::Free,
                                 random::PrepareMultinomial,
                                 random::EvalMultinomial};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

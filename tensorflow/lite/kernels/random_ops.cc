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
#include <random>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions_utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace random {

namespace {

constexpr int kShapeTensor = 0;
constexpr int kOutputTensor = 0;

using Generator = ::tensorflow::random::PhiloxRandom;

enum RandomType {
  kRandomUniform,
  kRandomStandardNormal,
};

struct OpData {
  Generator rng;
};

// Generates non-deterministic seed.
inline int64_t GetNonDeterministicSeed() {
  static std::mt19937_64* seed_generator = []() {
    std::random_device device("/dev/urandom");
    return new std::mt19937_64(device());
  }();
  return (*seed_generator)();
}


// Generates random numbers following a uniform distribution from the underlying
// random integer generator, which returns a uint32 array of size
// kResultElementCount on each invocation.
void GenerateRandomUniformNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i++) {
      buffer[current_size + i] = tensorflow::random::Uint32ToFloat(samples[i]);
    }
    current_size += rng_net_size;
  }
}

// Generates random numbers following a standard normal distribution from the
// underlying random integer generator, which returns a uint32 array of size
// kResultElementCount on each invocation.
void GenerateRandomStandardNormalNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i += 2) {
      tensorflow::random::BoxMullerFloat(samples[i], samples[i + 1],
                                         &buffer[current_size + i],
                                         &buffer[current_size + i + 1]);
    }
    current_size += rng_net_size;
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
  TF_LITE_ENSURE(context, NumInputs(node) == 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Validate input
  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  TF_LITE_ENSURE_EQ(context, shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shape), 1);

  // Initialize the random number generation with seeds
  auto* params = static_cast<TfLiteRandomParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int64_t seed = params->seed;
  int64_t seed2 = params->seed2;
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, generate non-deterministic random numbers.
    seed = random::GetNonDeterministicSeed();
    seed2 = random::GetNonDeterministicSeed();
  }
  Generator rng(seed, seed2);
  data->rng = rng;

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (!IsConstantTensor(shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TfLiteIntArray* output_shape;
  TF_LITE_ENSURE_OK(context,
                    GetOutputShapeFromInput(context, shape, &output_shape));
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus EvalRandomType(
    TfLiteContext* context, TfLiteNode* node, RandomType random_type) {
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
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
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (IsDynamicTensor(output)) {
    const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
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

}  // namespace random

TfLiteRegistration* Register_RANDOM_UNIFORM() {
  static TfLiteRegistration r = {random::Init, random::Free,
                                 random::Prepare,
                                 random::Eval<random::kRandomUniform>};
  return &r;
}

TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() {
  static TfLiteRegistration r = {random::Init, random::Free,
                                 random::Prepare,
                                 random::Eval<random::kRandomStandardNormal>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

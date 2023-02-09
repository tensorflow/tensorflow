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
#include <limits>
#include <random>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace random_uniform {

struct OpData {
  // This implementation uses a random generator from the standard C++ library
  // on the platform where TFLite is build. This is different from the TF
  // version of the kernel that uses custom implementations of random
  // generator, different for different hardware.
  std::default_random_engine rng;
};

namespace {

template <typename T, typename dist_type>
void RandomUniformSample(std::default_random_engine& rng, T* buffer,
                         size_t buffer_size, T min_value, T max_value) {
  dist_type dist(min_value, max_value);
  std::generate(buffer, buffer + buffer_size, [&]() { return dist(rng); });
}

TfLiteIntArray* CreateDimensionsFromTensor(const TfLiteTensor* tensor) {
  const int output_dims = tflite::SizeOfDimension(tensor, 0);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_dims);
  for (int i = 0; i < output_dims; i++) {
    if (tensor->type == kTfLiteInt32) {
      output_shape->data[i] = tensor->data.i32[i];
    } else {
      output_shape->data[i] = tensor->data.i64[i];
    }
  }
  return output_shape;
}
}  // namespace
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // TODO(b/169611265): Handle optional seed input.
  TF_LITE_ENSURE(context, tflite::NumInputs(node) >= 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  // Input is a shape tensor.
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt32 || input->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (!IsConstantTensor(input)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return context->ResizeTensor(context, output,
                               CreateDimensionsFromTensor(input));
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node) {
  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output,
                                            CreateDimensionsFromTensor(input)));
  }
  const size_t output_size = tflite::NumElements(output);
  switch (output->type) {
    case kTfLiteFloat32:
      RandomUniformSample<float, std::uniform_real_distribution<float>>(
          params->rng, GetTensorData<float>(output), output_size, 0.f, 1.f);
      break;
    case kTfLiteFloat64:
      RandomUniformSample<double, std::uniform_real_distribution<double>>(
          params->rng, GetTensorData<double>(output), output_size, 0.f, 1.f);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for RandomUniform: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

int64_t IntValueFromTensor(const TfLiteTensor* tensor) {
  switch (tensor->type) {
    case kTfLiteInt8:
      return *GetTensorData<int8_t>(tensor);
    case kTfLiteInt32:
      return *GetTensorData<int32_t>(tensor);
    case kTfLiteInt64:
      return *GetTensorData<int64_t>(tensor);
    default:
      return -1;
  }
}

TfLiteStatus EvalInt(TfLiteContext* context, TfLiteNode* node) {
  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TF_LITE_ENSURE(context, tflite::NumInputs(node) >= 3);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output,
                                            CreateDimensionsFromTensor(input)));
  }
  int64_t min_value = IntValueFromTensor(tflite::GetInput(context, node, 1));
  int64_t max_value = IntValueFromTensor(tflite::GetInput(context, node, 2));
  TF_LITE_ENSURE(context, min_value < max_value);
  size_t output_size = tflite::NumElements(output);
  switch (output->type) {
    case kTfLiteInt8:
      RandomUniformSample<int8_t, std::uniform_int_distribution<int32_t>>(
          params->rng, GetTensorData<int8_t>(output), output_size, min_value,
          max_value);
      break;
    case kTfLiteInt32:
      RandomUniformSample<int32_t, std::uniform_int_distribution<int32_t>>(
          params->rng, GetTensorData<int32_t>(output), output_size, min_value,
          max_value);
      break;
    case kTfLiteInt64:
      RandomUniformSample<int64_t, std::uniform_int_distribution<int64_t>>(
          params->rng, GetTensorData<int64_t>(output), output_size, min_value,
          max_value);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for RandomUniformInt: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace random_uniform

TfLiteRegistration* Register_RANDOM_UNIFORM() {
  static TfLiteRegistration r = {random_uniform::Init, random_uniform::Free,
                                 random_uniform::Prepare,
                                 random_uniform::EvalFloat};
  return &r;
}

TfLiteRegistration* Register_RANDOM_UNIFORM_INT() {
  static TfLiteRegistration r = {random_uniform::Init, random_uniform::Free,
                                 random_uniform::Prepare,
                                 random_uniform::EvalInt};
  return &r;
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite

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

#include <algorithm>
#include <cstddef>
#include <random>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace random_standard_normal {

struct OpData {
  std::default_random_engine rng;
};

namespace {

constexpr int kShapeTensor = 0;
constexpr int kOutputTensor = 0;

// Draws a sample from standard normal distribution.
template <typename T>
TfLiteStatus RandomStandardNormalSample(std::default_random_engine& rng,
                                        T* output, size_t output_size) {
  std::normal_distribution<T> dist;
  std::generate(output, output + output_size, [&]() { return dist(rng); });

  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // The seed/seed2 attributes are not handled in this custom op implementation.
  TF_LITE_ENSURE(context, NumInputs(node) == 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  TF_LITE_ENSURE_EQ(context, shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shape), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (!IsConstantOrPersistentTensor(shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TfLiteIntArray* output_shape;
  TF_LITE_ENSURE_OK(context,
                    GetOutputShapeFromInput(context, shape, &output_shape));
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
    TfLiteIntArray* output_shape;
    TF_LITE_ENSURE_OK(context,
                      GetOutputShapeFromInput(context, shape, &output_shape));
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_shape));
  }
  const size_t output_size = NumElements(output);
  switch (output->type) {
    case kTfLiteFloat32:
      RandomStandardNormalSample<float>(
          params->rng, GetTensorData<float>(output), output_size);
      break;
    case kTfLiteFloat64:
      RandomStandardNormalSample<double>(
          params->rng, GetTensorData<double>(output), output_size);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Unsupported output datatype for RandomStandardNormal: %s",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace random_standard_normal

TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() {
  static TfLiteRegistration r = {
      random_standard_normal::Init, random_standard_normal::Free,
      random_standard_normal::Prepare, random_standard_normal::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite

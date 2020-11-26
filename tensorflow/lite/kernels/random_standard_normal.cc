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
namespace random_standard_normal {

struct OpData {
  std::default_random_engine rng;
};

// Draws a sample from standard normal distribution.
template <typename Float>
TfLiteStatus RandomStandardNormalSample(std::default_random_engine& rng,
                                        Float* output, size_t output_size) {
  std::normal_distribution<Float> dist;
  for (Float* it = output; it != output + output_size; ++it) {
    *it = dist(rng);
  }
  return kTfLiteOk;
}

TfLiteStatus RandomStandardNormalSample(TfLiteContext* context,
                                        std::default_random_engine& rng,
                                        TfLiteTensor* output,
                                        size_t output_size) {
  switch (output->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(context,
                        RandomStandardNormalSample<float>(
                            rng, GetTensorData<float>(output), output_size));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(context,
                        RandomStandardNormalSample<double>(
                            rng, GetTensorData<double>(output), output_size));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Unsupported output datatype for RandomStandardNormal: %s",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // TODO(b/169611265): Handle optional seed input.
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  // Input is a shape tensor.
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 1);
  // TODO(b/169611265): Support dynamic output tensors.
  TF_LITE_ENSURE(context, IsConstantTensor(input));

  // TODO(b/169611265): Handle other input data types.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);

  int output_dims = tflite::SizeOfDimension(input, 0);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_dims);
  for (int i = 0; i < output_dims; i++) {
    output_shape->data[i] = input->data.i32[i];
  }

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  // ResizeTensor takes ownership of output_shape.
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // TODO(b/169611265): Handle optional seed input.
  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  size_t output_size = tflite::NumElements(output);

  TF_LITE_ENSURE_OK(context, RandomStandardNormalSample(context, params->rng,
                                                        output, output_size));

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

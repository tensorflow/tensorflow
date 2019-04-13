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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reluvariants {

enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};

struct PreluOpData : public OpData {
  int32_t output_multiplier = 0;
  int output_shift = 0;
};

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new PreluOpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

void PreluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<PreluOpData*>(buffer);
}

template <typename T>
inline void Relu(const RuntimeShape& input_shape, const T* input_data,
                 const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T lower = 0;
    const T clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}
template <typename T>
inline void Relu1(const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T upper = 1;
    const T lower = -1;
    const T clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Elu(const RuntimeShape& input_shape, const float* input_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val < 0.0 ? std::exp(val) - 1 : val;
  }
}

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that this implementation matches that of TensorFlow, and corresponds
    // to the traditional LeakyRelu equation only for alpha <= 1.
    output_data[i] = std::max(val, val * params.alpha);
  }
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      Relu(GetTensorShape(input), GetTensorData<float>(input),
           GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32 is supported currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      Elu(GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32 is supported currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      Relu1(GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32 is supported currently, got %s.",
                           TfLiteTypeGetName(input->type));
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
      context->ReportError(
          context,
          "Only float32, uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const auto* params =
      reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);

  LeakyReluParams op_params;
  op_params.alpha = params->alpha;
  switch (input->type) {
    case kTfLiteFloat32: {
      LeakyRelu(op_params, GetTensorShape(input), GetTensorData<float>(input),
                GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32 is supported currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace reluvariants

TfLiteRegistration* Register_RELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 reluvariants::GenericPrepare,
                                 reluvariants::ReluEval};
  return &r;
}

TfLiteRegistration* Register_RELU6() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 reluvariants::GenericPrepare,
                                 reluvariants::Relu6Eval};
  return &r;
}

TfLiteRegistration* Register_RELU_N1_TO_1() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 reluvariants::GenericPrepare,
                                 reluvariants::Relu1Eval};
  return &r;
}

TfLiteRegistration* Register_ELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 reluvariants::GenericPrepare,
                                 reluvariants::EluEval};
  return &r;
}

TfLiteRegistration* Register_LEAKY_RELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 reluvariants::GenericPrepare,
                                 reluvariants::LeakyReluEval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

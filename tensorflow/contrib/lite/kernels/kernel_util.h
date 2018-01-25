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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_KERNEL_UTIL_H_

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"

namespace tflite {

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
  return t->dims->data[dim];
}
inline TfLiteTensor* GetInput(TfLiteContext* context, TfLiteNode* node,
                              int index) {
  return &context->tensors[node->inputs->data[index]];
}
inline TfLiteTensor* GetOutput(TfLiteContext* context, TfLiteNode* node,
                               int index) {
  return &context->tensors[node->outputs->data[index]];
}
inline int NumInputs(const TfLiteNode* node) { return node->inputs->size; }
inline int NumOutputs(const TfLiteNode* node) { return node->outputs->size; }

inline TfLiteTensor* GetOptionalInputTensor(TfLiteContext* context,
                                            const TfLiteNode* node, int index) {
  const bool use_tensor = node->inputs->data[index] != kOptionalTensor;
  if (use_tensor) {
    return &context->tensors[node->inputs->data[index]];
  }
  return nullptr;
}

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(
    TfLiteContext* context, TfLiteTensor* input, TfLiteTensor* filter,
    TfLiteTensor* bias, TfLiteTensor* output, double* multiplier);

// Calculates the useful range of an activation layer given its activation
// tensor.
void CalculateActivationRangeUint8(TfLiteFusedActivation activation,
                                   TfLiteTensor* output, int32_t* act_min,
                                   int32_t* act_max);
void CalculateActivationRangeFloat(TfLiteFusedActivation activation,
                                   float* activation_min,
                                   float* activation_max);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_KERNEL_UTIL_H_

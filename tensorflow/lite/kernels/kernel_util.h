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
#ifndef TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

#include <algorithm>
#include <limits>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
  return t->dims->data[dim];
}
inline const TfLiteTensor* GetInput(TfLiteContext* context,
                                    const TfLiteNode* node, int index) {
  return &context
              ->tensors[flatbuffers::EndianScalar(node->inputs->data[index])];
}
inline TfLiteTensor* GetVariableInput(TfLiteContext* context,
                                      const TfLiteNode* node, int index) {
  TfLiteTensor* tensor =
      &context->tensors[flatbuffers::EndianScalar(node->inputs->data[index])];
  return (tensor->is_variable) ? tensor : nullptr;
}
inline TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                               int index) {
  return &context
              ->tensors[flatbuffers::EndianScalar(node->outputs->data[index])];
}
inline TfLiteTensor* GetTemporary(TfLiteContext* context,
                                  const TfLiteNode* node, int index) {
  return &context->tensors[flatbuffers::EndianScalar(
      node->temporaries->data[index])];
}
inline const TfLiteTensor* GetIntermediates(TfLiteContext* context,
                                            const TfLiteNode* node, int index) {
  return &context->tensors[node->intermediates->data[index]];
}
inline int NumInputs(const TfLiteNode* node) { return node->inputs->size; }
inline int NumOutputs(const TfLiteNode* node) { return node->outputs->size; }
inline int NumIntermediates(const TfLiteNode* node) {
  return node->intermediates->size;
}

inline int64_t NumElements(const TfLiteIntArray* dims) {
  int64_t count = 1;
  for (int i = 0; i < dims->size; ++i) {
    count *= dims->data[i];
  }
  return count;
}

inline int64_t NumElements(const TfLiteTensor* t) {
  return NumElements(t->dims);
}

inline const TfLiteTensor* GetOptionalInputTensor(TfLiteContext* context,
                                                  const TfLiteNode* node,
                                                  int index) {
  const bool use_tensor = node->inputs->data[index] != kTfLiteOptionalTensor;
  if (use_tensor) {
    return &context
                ->tensors[flatbuffers::EndianScalar(node->inputs->data[index])];
  }
  return nullptr;
}

// Determines whether tensor is constant.
inline bool IsConstantTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteMmapRo;
}

// Determines whether tensor is dynamic. Note that a tensor can be non-const and
// not dynamic. This function specifically checks for a dynamic tensor.
inline bool IsDynamicTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteDynamic;
}

// Sets tensor to dynamic.
inline void SetTensorToDynamic(TfLiteTensor* tensor) {
  if (tensor->allocation_type != kTfLiteDynamic) {
    tensor->allocation_type = kTfLiteDynamic;
    tensor->data.raw = nullptr;
  }
}

// Determines whether it is a hybrid op - one that has float inputs and
// quantized weights.
inline bool IsHybridOp(const TfLiteTensor* input, const TfLiteTensor* weight) {
  return ((weight->type == kTfLiteUInt8 || weight->type == kTfLiteInt8) &&
          input->type == kTfLiteFloat32);
}

// Check dimensionality match and populate OpData for Conv and DepthwiseConv.
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift);

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier);

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation tensor.
TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max);
void CalculateActivationRangeUint8(TfLiteFusedActivation activation,
                                   TfLiteTensor* output, int32_t* act_min,
                                   int32_t* act_max);
void CalculateActivationRangeInt8(TfLiteFusedActivation activation,
                                  TfLiteTensor* output, int32_t* act_min,
                                  int32_t* act_max);
// Calculates the useful range of an activation layer given its activation
// tensor.a
template <typename T>
void CalculateActivationRange(TfLiteFusedActivation activation,
                              T* activation_min, T* activation_max) {
  if (activation == kTfLiteActRelu) {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0;
    *activation_max = 6;
  } else if (activation == kTfLiteActRelu1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2);

// Calculate the output_shape that is necessary for element-wise operations
// with broadcasting involving the two input tensors.
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

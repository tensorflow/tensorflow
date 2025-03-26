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

#include <stdint.h>

#include <limits>
#ifndef TF_LITE_STATIC_MEMORY
#include <string>
#endif  // TF_LITE_STATIC_MEMORY

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#ifndef NDEBUG
#include "tensorflow/lite/kernels/op_macros.h"
#endif

namespace tflite {

// A fair number of functions in this header have historically been inline.
// It is ok to change functions to not be inline if the latency with
// benchmark_model for MobileNet + MobileBERT is unaffected. If such a change is
// made, move the newly non-inlined function declarations to the top of this
// header file.

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
const TfLiteTensor* GetInput(const TfLiteContext* context,
                             const TfLiteNode* node, int index);

// Same as `GetInput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetInputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetInputSafe(const TfLiteContext* context, const TfLiteNode* node,
                          int index, const TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetVariableInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOutput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                        int index);

// Same as `GetOutput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetOutputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetOutputSafe(const TfLiteContext* context, const TfLiteNode* node,
                           int index, TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOptionalInputTensor(context, node, kIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
//
// Deprecated. GetInput has the same functionality.
const TfLiteTensor* GetOptionalInputTensor(const TfLiteContext* context,
                                           const TfLiteNode* node, int index);

#ifndef TF_LITE_STATIC_MEMORY
// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetTemporary(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetTemporary(TfLiteContext* context, const TfLiteNode* node,
                           int index);

// Same as `GetTemporary` but returns boolean and uses output argument for
// tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetTemporarySafe(context, node, kMyTensorIdx,
//                     &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetTemporarySafe(const TfLiteContext* context,
                              const TfLiteNode* node, int index,
                              TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetIntermediates(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
const TfLiteTensor* GetIntermediates(TfLiteContext* context,
                                     const TfLiteNode* node, int index);

// Same as `GetIntermediates` but returns boolean and uses output argument for
// tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetIntermediatesSafe(context, node, kMyTensorIdx,
//                     &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetIntermediatesSafe(const TfLiteContext* context,
                                  const TfLiteNode* node, int index,
                                  TfLiteTensor** tensor);
#endif  // TF_LITE_STATIC_MEMORY

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
  return t->dims->data[dim];
}

inline int NumInputs(const TfLiteNode* node) {
  return node->inputs == nullptr ? 0 : node->inputs->size;
}
inline int NumOutputs(const TfLiteNode* node) {
  return node->outputs == nullptr ? 0 : node->outputs->size;
}

#ifndef TF_LITE_STATIC_MEMORY
inline int NumIntermediates(const TfLiteNode* node) {
  return node->intermediates->size;
}
#endif  // TF_LITE_STATIC_MEMORY

inline int64_t NumElements(const int* dims, int num_dims) {
  int64_t count = 1;
  for (int i = 0; i < num_dims; ++i) {
#ifndef NDEBUG
    if (count <= 0) {
      break;
    }
    // Check that number of elements can fit in 32 bit int. Most of tflite
    // assumes the result of `NumElements` is < MAX_INT and static or implicit
    // casts to `int32_t` without any checks. It is more meaningful to check
    // that the result fits into 32 bits than for standard overflow on 64 bit
    // type.
    TF_LITE_ASSERT(dims[i] < std::numeric_limits<int>::max() / count);
#endif
    count *= dims[i];
  }
  return count;
}

inline int64_t NumElements(const TfLiteIntArray* dims) {
  return NumElements(dims->data, dims->size);
}

inline int64_t NumElements(const TfLiteTensor* t) {
  return NumElements(t->dims);
}

// Determines whether tensor is constant.
// TODO(b/138199592): Introduce new query which checks for constant OR
// persistent-read-only, which would be useful for most tensor kernels that
// are potentially dynamic based on the input tensor value availability at the
// time of prepare.
inline bool IsConstantTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteMmapRo;
}

inline bool IsConstantOrPersistentTensor(const TfLiteTensor* tensor) {
  return IsConstantTensor(tensor) ||
         (tensor->allocation_type == kTfLitePersistentRo);
}

// Determines whether tensor is dynamic. Note that a tensor can be non-const and
// not dynamic. This function specifically checks for a dynamic tensor.
inline bool IsDynamicTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteDynamic;
}
#ifndef TF_LITE_STATIC_MEMORY
// Sets tensor to dynamic.
inline void SetTensorToDynamic(TfLiteTensor* tensor) {
  if (tensor->allocation_type != kTfLiteDynamic) {
    TfLiteTensorDataFree(tensor);
    tensor->allocation_type = kTfLiteDynamic;
  }
}

// Sets tensor to persistent and read-only.
inline void SetTensorToPersistentRo(TfLiteTensor* tensor) {
  if (tensor->allocation_type != kTfLitePersistentRo) {
    TfLiteTensorDataFree(tensor);
    tensor->allocation_type = kTfLitePersistentRo;
  }
}
#endif  // TF_LITE_STATIC_MEMORY

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
    int32_t* per_channel_multiplier, int32_t* per_channel_shift);

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels);

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
  } else if (activation == kTfLiteActReluN1To1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2);

#if !defined(TF_LITE_STATIC_MEMORY)
// Gets the output shape from the input tensor.
TfLiteStatus GetOutputShapeFromInput(TfLiteContext* context,
                                     const TfLiteTensor* input,
                                     TfLiteIntArray** output_shape);

std::string GetShapeDebugString(const TfLiteIntArray* shape);

std::string GetTensorDebugString(const TfLiteTensor* tensor);

#endif  // !defined(TF_LITE_STATIC_MEMORY)

// Calculates the output_shape that is necessary for element-wise operations
// with broadcasting involving the two input tensors.
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape);

// Calculates the output_shape that is necessary for element-wise operations
// with broadcasting involving the three input tensors.
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        const TfLiteTensor* input3,
                                        TfLiteIntArray** output_shape);

// Return the size of given type in bytes. Return 0 in case of string.
int TfLiteTypeGetSize(TfLiteType type);

// Whether the current platform is mobile (Android or iOS).
bool IsMobilePlatform();

// Returns whether there is unspecified dimension in the tensor's dim signature.
bool HasUnspecifiedDimension(const TfLiteTensor* tensor);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

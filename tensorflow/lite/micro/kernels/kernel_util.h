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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace micro {

// Returns a mutable tensor for a given input index. is_variable must be checked
// during prepare when the full TfLiteTensor is available.
inline TfLiteEvalTensor* GetMutableEvalInput(const TfLiteContext* context,
                                             const TfLiteNode* node,
                                             int index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  return context->GetEvalTensor(context, node->inputs->data[index]);
}

// Returns the TfLiteEvalTensor struct for a given input index in a node.
inline const TfLiteEvalTensor* GetEvalInput(const TfLiteContext* context,
                                            const TfLiteNode* node, int index) {
  return GetMutableEvalInput(context, node, index);
}

// Returns the TfLiteEvalTensor struct for a given output index in a node.
inline TfLiteEvalTensor* GetEvalOutput(const TfLiteContext* context,
                                       const TfLiteNode* node, int index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  return context->GetEvalTensor(context, node->outputs->data[index]);
}

// Returns data for a TfLiteEvalTensor struct.
template <typename T>
T* GetTensorData(TfLiteEvalTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<T*>(tensor->data.raw) : nullptr;
}

// Returns const data for a TfLiteEvalTensor struct.
template <typename T>
const T* GetTensorData(const TfLiteEvalTensor* tensor) {
  TFLITE_DCHECK(tensor != nullptr);
  return reinterpret_cast<const T*>(tensor->data.raw);
}

// Returns the shape of a TfLiteEvalTensor struct.
const RuntimeShape GetTensorShape(const TfLiteEvalTensor* tensor);

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2);

PaddingType RuntimePaddingType(TfLitePadding padding);

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_

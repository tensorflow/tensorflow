/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/buffer_map.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace flex {

BufferMap::BufferMap() {}

BufferMap::~BufferMap() {}

bool BufferMap::HasTensor(int tensor_index) const {
  return id_to_tensor_.count(tensor_index) != 0;
}

bool BufferMap::IsTensorFlowTensor(int tensor_index) const {
  return HasTensor(tensor_index) && owned_by_tf_.count(tensor_index) > 0;
}

tensorflow::Tensor BufferMap::GetTensor(int tensor_index) const {
  return id_to_tensor_.at(tensor_index);
}

const tensorflow::Tensor* BufferMap::GetTensorPtr(int tensor_index) const {
  auto& tensor = id_to_tensor_.at(tensor_index);
  return &tensor;
}

void BufferMap::SetFromTfLite(int tensor_index, const TfLiteTensor* tensor) {
  TFLITE_CHECK(
      SetTfTensorFromTfLite(tensor, &id_to_tensor_[tensor_index]).ok());
  if (tensor->type == kTfLiteResource || tensor->type == kTfLiteVariant) {
    owned_by_tf_.insert(tensor_index);
    return;
  }
  owned_by_tf_.erase(tensor_index);
}

void BufferMap::SetFromTensorFlow(int tensor_index, tensorflow::Tensor tensor) {
  id_to_tensor_[tensor_index] = std::move(tensor);
  owned_by_tf_.insert(tensor_index);
}

}  // namespace flex
}  // namespace tflite

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/variants/tensor_array.h"

#include <cstring>

namespace tflite {
namespace variants {

TensorArray::TensorArray(const TensorArray& other) {
  TfLiteIntArray* copied_shape = TfLiteIntArrayCopy(other.element_shape_.get());
  element_shape_ = IntArrayUniquePtr(copied_shape);
  element_type_ = other.element_type_;
  num_elements_ = other.num_elements_;
  elements_ =
      (RefCountedTensor*)malloc(sizeof(RefCountedTensor) * other.num_elements_);
  other.AssignBuffer(elements_);
}

TensorArray& TensorArray::operator=(const TensorArray& other) {
  TfLiteIntArray* copied_shape = TfLiteIntArrayCopy(other.element_shape_.get());
  element_shape_ = IntArrayUniquePtr(copied_shape);
  Resize(other.num_elements_);
  Clear();
  other.AssignBuffer(elements_);
  return *this;
}

void TensorArray::Resize(int num_elements) {
  if (num_elements == NumElements() || num_elements < 0) return;
  if (num_elements > NumElements()) {
    // The length of the array is being increased. Reallocate the buffer
    // to the appropriate size and setup the new `RefCountedTensors`.
    elements_ = (RefCountedTensor*)realloc(
        elements_, num_elements * sizeof(RefCountedTensor));
    for (int i = NumElements(); i < num_elements; ++i) {
      elements_[i].count = nullptr;
      elements_[i].tensor = nullptr;
    }
  } else {
    // The length of the array is being decreased, update each of the
    // references that we will no longer keep before reallocating.
    for (int i = num_elements; i < NumElements(); ++i) {
      Drop(i);
    }
    elements_ = (RefCountedTensor*)realloc(
        elements_, num_elements * sizeof(RefCountedTensor));
  }
  num_elements_ = num_elements;
}

const TfLiteTensor* TensorArray::At(int index) const {
  if (index < 0 || index >= NumElements()) {
    return nullptr;
  }
  return elements_[index].tensor;
}

bool TensorArray::Set(int index, TensorUniquePtr tensor) {
  if (index < 0 || index >= NumElements()) {
    return false;
  }
  // Drop element if it exists.
  Drop(index);
  // Setup the `RefCountedTensor` at given index to wrap the given tensor.
  int* c = (int*)malloc(sizeof(int));
  *c = 1;
  elements_[index].tensor = tensor.release();
  elements_[index].count = c;
  return true;
}

TensorArray::~TensorArray() {
  Clear();
  free(elements_);
  elements_ = nullptr;
}

void TensorArray::Drop(int i) {
  RefCountedTensor* t = elements_ + i;
  int* count = t->count;
  if (count == nullptr) {
    return;
  }
  if (*count == 1) {
    TfLiteTensorFree(t->tensor);
    free(t->tensor);
    free(t->count);
    t->tensor = nullptr;
    t->count = nullptr;
    return;
  }
  (*count)--;
}

// `Drop`s each element in the list.
void TensorArray::Clear() {
  for (int i = 0; i < num_elements_; ++i) {
    Drop(i);
  }
}

void TensorArray::AssignBuffer(RefCountedTensor* dst) const {
  // Copy `this` underlying buffer.
  std::memcpy(dst, elements_, sizeof(RefCountedTensor) * num_elements_);
  // Increment the reference count for each copied tensor.
  for (int i = 0; i < num_elements_; ++i) {
    if (dst[i].count == nullptr) {
      continue;
    }
    (*dst[i].count)++;
  }
}

}  // namespace variants
}  // namespace tflite

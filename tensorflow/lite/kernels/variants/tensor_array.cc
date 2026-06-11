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

#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

TensorArray::TensorArray(const TensorArray& other) {
  TfLiteIntArray* copied_shape = TfLiteIntArrayCopy(other.element_shape_.get());
  if (copied_shape == nullptr && other.element_shape_ != nullptr) {
    element_shape_ = nullptr;
    num_elements_ = 0;
    elements_ = nullptr;
    return;
  }
  element_shape_ = IntArrayUniquePtr(copied_shape);
  element_type_ = other.element_type_;
  if (other.num_elements_ == 0) {
    num_elements_ = 0;
    elements_ = nullptr;
    return;
  }
  size_t bytes;
  if (__builtin_mul_overflow(other.num_elements_, sizeof(RefCountedTensor),
                             &bytes)) {
    num_elements_ = 0;
    elements_ = nullptr;
    return;
  }
  elements_ = static_cast<RefCountedTensor*>(malloc(bytes));
  if (elements_ == nullptr) {
    num_elements_ = 0;
    return;
  }
  num_elements_ = other.num_elements_;
  other.AssignBuffer(elements_);
}

TensorArray::TensorArray(TensorArray&& other) noexcept
    : elements_(other.elements_),
      num_elements_(other.num_elements_),
      element_shape_(std::move(other.element_shape_)),
      element_type_(other.element_type_) {
  other.elements_ = nullptr;
  other.num_elements_ = 0;
}

TensorArray& TensorArray::operator=(const TensorArray& other) {
  if (this == &other) return *this;
  TensorArray temp(other);
  if (other.num_elements_ > 0 && temp.elements_ == nullptr) {
    return *this;
  }
  std::swap(element_shape_, temp.element_shape_);
  std::swap(element_type_, temp.element_type_);
  std::swap(elements_, temp.elements_);
  std::swap(num_elements_, temp.num_elements_);
  return *this;
}

TensorArray& TensorArray::operator=(TensorArray&& other) noexcept {
  if (this == &other) return *this;
  Clear();
  free(elements_);
  elements_ = other.elements_;
  num_elements_ = other.num_elements_;
  element_shape_ = std::move(other.element_shape_);
  element_type_ = other.element_type_;
  other.elements_ = nullptr;
  other.num_elements_ = 0;
  return *this;
}

bool TensorArray::Resize(int num_elements) {
  if (num_elements == NumElements()) return true;
  if (num_elements < 0) return false;
  size_t bytes;
  if (__builtin_mul_overflow(num_elements, sizeof(RefCountedTensor), &bytes)) {
    return false;
  }
  if (num_elements > NumElements()) {
    // The length of the array is being increased. Reallocate the buffer
    // to the appropriate size and setup the new `RefCountedTensors`.
    RefCountedTensor* new_elements =
        static_cast<RefCountedTensor*>(realloc(elements_, bytes));
    if (new_elements == nullptr && num_elements > 0) {
      return false;
    }
    elements_ = new_elements;
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
    if (num_elements == 0) {
      free(elements_);
      elements_ = nullptr;
    } else {
      RefCountedTensor* new_elements =
          static_cast<RefCountedTensor*>(realloc(elements_, bytes));
      if (new_elements != nullptr) {
        elements_ = new_elements;
      }
    }
  }
  num_elements_ = num_elements;
  return true;
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
  int* c = static_cast<int*>(malloc(sizeof(int)));
  if (c == nullptr) {
    return false;
  }
  // Drop element if it exists.
  Drop(index);
  // Setup the `RefCountedTensor` at given index to wrap the given tensor.
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
    if (t->tensor != nullptr) {
      TfLiteTensorFree(t->tensor);
      free(t->tensor);
    }
    free(t->count);
  } else {
    (*count)--;
  }
  t->tensor = nullptr;
  t->count = nullptr;
}

// `Drop`s each element in the list.
void TensorArray::Clear() {
  for (int i = 0; i < num_elements_; ++i) {
    Drop(i);
  }
}

void TensorArray::AssignBuffer(RefCountedTensor* dst) const {
  if (num_elements_ == 0) return;
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

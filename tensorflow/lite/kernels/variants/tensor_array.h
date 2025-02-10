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
#ifndef TENSORFLOW_LITE_KERNELS_VARIANTS_TENSOR_ARRAY_H_
#define TENSORFLOW_LITE_KERNELS_VARIANTS_TENSOR_ARRAY_H_

#include <utility>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

// `VariantData` implementation for a dynamically sized array of `TfLiteTensor`.
// Each element of the array is a lightweight `RefCountedTensor`.
// --- WARNING ---
// This is intended to be used in a single-threaded manner
// and users must take care when calling non-const methods, even on different
// instances. Different instances may share underlying control structures (when
// using the copy constructor to initialize them), and in such cases function
// calls across all affected instances must be properly synchronized. Calling
// non-const functions on any of the linked objects requires exclusive access to
// all of them.
//
// TODO(b/288302706) Implement standard container methods.
class TensorArray : public AbstractVariantData<TensorArray> {
 public:
  // Takes ownership of `element_shape` input.
  TensorArray(TfLiteType element_type, IntArrayUniquePtr element_shape)
      : element_shape_(std::move(element_shape)), element_type_(element_type) {}

  // Copying a `TensorArray` copies the sources underlying array of
  // `RefCountedTensor` in one `memcpy` and increments each of the ref counts.
  TensorArray(const TensorArray& other);

  // Drops the references of `this` and assigns members in the same way
  // as the copy constructor.
  TensorArray& operator=(const TensorArray& other);

  const TfLiteIntArray* ElementShape() const { return element_shape_.get(); }

  int NumElements() const { return num_elements_; }

  // Resizes the array for given number of elements. If the length of the array
  // is being decreased, `Drop` the reference to the elements that will no
  // longer be in the array. If index is out of bounds, no effect.
  void Resize(int num_elements);

  // Retrieve the tensor at the given index.
  const TfLiteTensor* At(int index) const;

  TfLiteType ElementType() const { return element_type_; }

  // Set the item at the given index with the given tensor. Takes ownership
  // of the given tensor. If there exists an element at the given index,
  // `Drop` this array's reference to it.
  bool Set(int index, TensorUniquePtr tensor);

  // `Drop`s each reference that exists in the array.
  ~TensorArray() override;

 private:
  // Simple structure to hold tensor pointer and ref count. Only to be used
  // as elements within `TensorArray`.
  struct RefCountedTensor {
    TfLiteTensor* tensor = nullptr;
    int* count = nullptr;
  };

  // "Drops" the reference at the given index because it will no longer be held
  // in this array. Decrements the reference count, if this array holds the only
  // reference than free the underlying tensor.
  void Drop(int i);

  // `Drop`s each element in the list.
  void Clear();

  // Assigns this `elements_` buffer to `dst`. Requires that the size
  // of this elements buffer be the same as `dst` and that `dst` has
  // been `Clear`ed. Like the rest of this class, copying `this` buffer
  // needs to increment references of `const this`, so beware.
  void AssignBuffer(RefCountedTensor* dst) const;

  // elements_ is nullptr iff num_elements is 0.
  RefCountedTensor* elements_ = nullptr;
  int num_elements_ = 0;

  IntArrayUniquePtr element_shape_;
  TfLiteType element_type_;
};

}  // namespace variants
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_TENSOR_ARRAY_H_

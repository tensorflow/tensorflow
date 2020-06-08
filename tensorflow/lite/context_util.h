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
// This provides a few C++ helpers that are useful for manipulating C structures
// in C++.
#ifndef TENSORFLOW_LITE_CONTEXT_UTIL_H_
#define TENSORFLOW_LITE_CONTEXT_UTIL_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Provide a range iterable wrapper for TfLiteIntArray* (C lists that TfLite
// C api uses. Can't use the google array_view, since we can't depend on even
// absl for embedded device reasons.
class TfLiteIntArrayView {
 public:
  // Construct a view of a TfLiteIntArray*. Note, `int_array` should be non-null
  // and this view does not take ownership of it.
  explicit TfLiteIntArrayView(const TfLiteIntArray* int_array)
      : int_array_(int_array) {}

  TfLiteIntArrayView(const TfLiteIntArrayView&) = default;
  TfLiteIntArrayView& operator=(const TfLiteIntArrayView& rhs) = default;

  typedef const int* const_iterator;
  const_iterator begin() const { return int_array_->data; }
  const_iterator end() const { return &int_array_->data[int_array_->size]; }
  size_t size() const { return end() - begin(); }
  int operator[](size_t pos) const { return int_array_->data[pos]; }

 private:
  const TfLiteIntArray* int_array_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CONTEXT_UTIL_H_

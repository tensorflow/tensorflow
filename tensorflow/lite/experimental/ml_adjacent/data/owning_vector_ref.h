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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_DATA_OWNING_VECTOR_REF_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_DATA_OWNING_VECTOR_REF_H_
#include <vector>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace data {

// A MutableDataRef implemenation that manages its buffer through underlying
// vector which it owns.
class OwningVectorRef : public MutableDataRef {
 public:
  explicit OwningVectorRef(etype_t type) : MutableDataRef(type) {}

  OwningVectorRef(const OwningVectorRef&) = delete;
  OwningVectorRef(OwningVectorRef&&) = delete;
  OwningVectorRef& operator=(const OwningVectorRef&) = delete;
  OwningVectorRef& operator=(OwningVectorRef&&) = delete;

  // Resizes the underlying vector to prod(dims) * type width.
  void Resize(dims_t&& dims) override;

  // Gets read-only pointer to vector's buffer.
  const void* Data() const override;

  // Gets a write-read pointer to vector's buffer.
  void* Data() override;

  ind_t NumElements() const override;

  size_t Bytes() const override;

  ~OwningVectorRef() override = default;

 private:
  std::vector<char> raw_data_buffer_;
  ind_t num_elements_ = 0;
};

}  // namespace data
}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_DATA_OWNING_VECTOR_REF_H_

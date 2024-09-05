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
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"

#include <cstddef>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace data {

void OwningVectorRef::Resize(dims_t&& dims) {
  dims_ = dims;

  num_elements_ = 0;
  // TODO(b/292143456) Add a helper for this.
  for (dim_t d : dims_) {
    if (d <= 0) {
      break;
    }
    if (num_elements_ == 0) {
      num_elements_ = d;
    } else {
      num_elements_ *= d;
    }
  }

  raw_data_buffer_.resize(num_elements_ * TypeWidth(Type()));
}

const void* OwningVectorRef::Data() const { return raw_data_buffer_.data(); }

void* OwningVectorRef::Data() { return raw_data_buffer_.data(); }

ind_t OwningVectorRef::NumElements() const { return num_elements_; }

size_t OwningVectorRef::Bytes() const {
  return NumElements() * TypeWidth(Type());
}

}  // namespace data
}  // namespace ml_adj

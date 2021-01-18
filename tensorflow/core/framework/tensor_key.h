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
#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_KEY_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_KEY_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class TensorKey : public Tensor {
 public:
  using Tensor::Tensor;

  TensorKey(const Tensor& t) : Tensor(t) {}

  // Equality operator. Needed for absl hashing.
  friend bool operator==(const TensorKey& t1, const TensorKey& t2) {
    if (t1.dtype() != t2.dtype() || t1.shape() != t2.shape()) {
      return false;
    }
    if (DataTypeCanUseMemcpy(t1.dtype())) {
      return t1.tensor_data() == t2.tensor_data();
    }
    if (t1.dtype() == DT_STRING) {
      const auto s1 = t1.unaligned_flat<tstring>();
      const auto s2 = t2.unaligned_flat<tstring>();
      for (int64 i = 0, n = t1.NumElements(); i < n; ++i) {
        if (TF_PREDICT_FALSE(s1(i) != s2(i))) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  friend bool operator!=(const TensorKey& t1, const TensorKey& t2) {
    return !(t1 == t2);
  }

  // Needed for absl hash function.
  template <typename H>
  friend H AbslHashValue(H h, const TensorKey& k) {
    const uint8* d = static_cast<uint8*>(k.data());
    size_t s = k.AllocatedBytes();
    std::vector<uint8> vec;
    for (int i = 0; i < s; i++) {
      vec.push_back(d[i]);
    }
    return H::combine(std::move(h), s);
  }
};

}  // namespace tensorflow

#endif

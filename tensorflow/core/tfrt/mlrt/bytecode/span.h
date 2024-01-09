/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_SPAN_H_
#define TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_SPAN_H_

#include <cstdint>
#include <vector>

#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace mlrt {
namespace bc {

// Span is a range view of contiguous byte region like bc::Vector. It reads the
// array size and start pointer eagerly, so that the range can be adapted.
template <typename T>
class Span {
 public:
  using value_type = T;
  using iterator = ReadIterator<T>;
  using const_iterator = iterator;

  Span() = default;
  Span(const char* data, size_t size) : data_(data), size_(size) {}

  template <typename SizeType>
  Span(const Vector<T, SizeType>& vec)  // NOLINT(google-explicit-constructor)
      : Span(vec.data(), vec.size()) {}
  Span(const String& vec)  // NOLINT(google-explicit-constructor)
      : Span(vec.data(), vec.size()) {}
  Span(const std::vector<T>& vec)  // NOLINT(google-explicit-constructor)
      : Span(reinterpret_cast<const char*>(vec.data()), vec.size()) {}

  const char* data() const { return data_; }
  const char* data(size_t index) const { return data_ + index * sizeof(T); }

  iterator begin() const { return iterator(data_); }
  iterator end() const { return iterator(data_ + size_ * sizeof(T)); }
  T back() const {
    DCHECK_GT(size_, 0);
    return *iterator(data_ + (size_ - 1) * sizeof(T));
  }

  T operator[](size_t index) const {
    DCHECK_LT(index, size());
    auto iter = begin();
    iter += index;
    return *iter;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  Span drop_front(size_t num = 1) const {
    auto beg = begin();
    beg += num;
    DCHECK_GE(size(), num);
    return Span(beg.data(), size() - num);
  }

  Span drop_back(size_t num = 1) const {
    DCHECK_GE(size(), num);
    return Span(data(), size() - num);
  }

 private:
  const char* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace bc
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_SPAN_H_

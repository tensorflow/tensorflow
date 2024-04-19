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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ITERATOR_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ITERATOR_H_

#include <iterator>

#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace mlrt {
namespace iterator_internal {

template <typename Iter, typename ValueType, typename ValueRangeType>
class IteratorBase {
  const Iter& self() const { return static_cast<const Iter&>(*this); }
  Iter& self() { return static_cast<Iter&>(*this); }

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = ValueType;
  using pointer = ValueType*;
  using reference = ValueType&;
  using iterator_category = std::random_access_iterator_tag;

  explicit IteratorBase(bc::ReadIterator<uint32_t> index_iter,
                        ValueRangeType values)
      : index_iter_(index_iter), values_(values) {}

  reference operator*() const { return values_[*index_iter_]; }

  pointer operator->() const { return &values_[*index_iter_]; }

  reference operator[](difference_type i) const {
    return values_[*(index_iter_ + i)];
  }

  Iter& operator+=(difference_type d) {
    index_iter_ += d;
    return self();
  }

  Iter& operator-=(difference_type d) {
    index_iter_ -= d;
    return self();
  }

  Iter& operator++() {
    ++index_iter_;
    return self();
  }

  Iter operator++(int) {
    Iter r = self();
    ++index_iter_;
    return r;
  }

  Iter& operator--() {
    --index_iter_;
    return self();
  }

  Iter operator--(int) {
    Iter r = self();
    --index_iter_;
    return r;
  }

  Iter operator+(difference_type d) const {
    Iter r = self();
    r += d;
    return r;
  }

  friend Iter operator+(difference_type d, const Iter& i) { return i + d; }

  Iter operator-(difference_type d) const {
    Iter r = self();
    r -= d;
    return r;
  }

  difference_type operator-(const Iter& other) const {
    return index_iter_ - other.index_iter_;
  }

  friend bool operator==(const Iter& a, const Iter& b) {
    return a.index_iter_ == b.index_iter_;
  }

  friend bool operator!=(const Iter& a, const Iter& b) {
    return a.index_iter_ != b.index_iter_;
  }

  friend bool operator<(const Iter& a, const Iter& b) {
    return a.index_iter_ < b.index_iter_;
  }

  friend bool operator<=(const Iter& a, const Iter& b) {
    return a.index_iter_ <= b.index_iter_;
  }

  friend bool operator>(const Iter& a, const Iter& b) {
    return a.index_iter_ > b.index_iter_;
  }

  friend bool operator>=(const Iter& a, const Iter& b) {
    return a.index_iter_ >= b.index_iter_;
  }

 private:
  bc::ReadIterator<uint32_t> index_iter_;
  ValueRangeType values_;
};

}  // namespace iterator_internal
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ITERATOR_H_

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_REGISTER_SPAN_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_REGISTER_SPAN_H_

#include <iterator>

#include "absl/types/span.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/iterator.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"

namespace mlrt {

class RegisterIterator
    : public iterator_internal::IteratorBase<RegisterIterator, Value,
                                             absl::Span<Value>> {
 public:
  using IteratorBase<RegisterIterator, Value, absl::Span<Value>>::IteratorBase;
};

class ConstRegisterIterator
    : public iterator_internal::IteratorBase<ConstRegisterIterator, const Value,
                                             absl::Span<const Value>> {
  using IteratorBase<ConstRegisterIterator, const Value,
                     absl::Span<const Value>>::IteratorBase;
};

class RegisterSpan {
 public:
  using value_type = Value;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = Value&;
  using const_reference = const Value&;
  using pointer = Value*;
  using const_pointer = const Value*;
  using iterator = RegisterIterator;
  using const_iterator = ConstRegisterIterator;

  RegisterSpan() = default;
  RegisterSpan(bc::Span<uint32_t> reg_indices, absl::Span<Value> regs)
      : reg_indices_(reg_indices), regs_(regs) {}

  Value& operator[](size_t idx) { return regs_[reg_indices_[idx]]; }
  const Value& operator[](size_t idx) const { return regs_[reg_indices_[idx]]; }
  Value& back() const { return regs_[reg_indices_.back()]; }

  size_t size() const { return reg_indices_.size(); }

  iterator begin() const { return iterator(reg_indices_.begin(), regs_); }
  iterator end() const { return iterator(reg_indices_.end(), regs_); }

  RegisterSpan drop_front(int num = 1) {
    return RegisterSpan(reg_indices_.drop_front(num), regs_);
  }

  RegisterSpan drop_back(int num = 1) {
    return RegisterSpan(reg_indices_.drop_back(num), regs_);
  }

 private:
  bc::Span<uint32_t> reg_indices_;
  absl::Span<Value> regs_;
};

template <typename T>
class RegisterValueIterator {
  using Iter = RegisterValueIterator;

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  explicit RegisterValueIterator(RegisterIterator reg_iter)
      : reg_iter_(reg_iter) {}

  reference operator*() const { return (*reg_iter_).Get<T>(); }

  pointer operator->() const { return &(*reg_iter_).Get<T>(); }

  reference operator[](difference_type i) const {
    return (*(reg_iter_ + i)).Get<T>();
  }

  Iter& operator+=(difference_type d) {
    reg_iter_ += d;
    return *this;
  }

  Iter& operator-=(difference_type d) {
    reg_iter_ -= d;
    return *this;
  }

  Iter& operator++() {
    ++reg_iter_;
    return *this;
  }

  Iter operator++(int) {
    Iter r = *this;
    ++reg_iter_;
    return r;
  }

  Iter& operator--() {
    --reg_iter_;
    return *this;
  }

  Iter operator--(int) {
    Iter r = *this;
    --reg_iter_;
    return r;
  }

  Iter operator+(difference_type d) const {
    Iter r = *this;
    r += d;
    return r;
  }

  friend Iter operator+(difference_type d, const Iter& i) { return i + d; }

  Iter operator-(difference_type d) const {
    Iter r = *this;
    r -= d;
    return r;
  }

  difference_type operator-(const Iter& other) const {
    return reg_iter_ - other.reg_iter_;
  }

  friend bool operator==(const Iter& a, const Iter& b) {
    return a.reg_iter_ == b.reg_iter_;
  }

  friend bool operator!=(const Iter& a, const Iter& b) {
    return a.reg_iter_ != b.reg_iter_;
  }

  friend bool operator<(const Iter& a, const Iter& b) {
    return a.reg_iter_ < b.reg_iter_;
  }

  friend bool operator<=(const Iter& a, const Iter& b) {
    return a.reg_iter_ <= b.reg_iter_;
  }

  friend bool operator>(const Iter& a, const Iter& b) {
    return a.reg_iter_ > b.reg_iter_;
  }

  friend bool operator>=(const Iter& a, const Iter& b) {
    return a.reg_iter_ >= b.reg_iter_;
  }

 private:
  RegisterIterator reg_iter_;
};

template <typename T>
class RegisterValueSpan {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = RegisterValueIterator<T>;
  using const_iterator = RegisterValueIterator<const T>;

  RegisterValueSpan(bc::Span<uint32_t> reg_indices, absl::Span<Value> regs)
      : reg_span_(reg_indices, regs) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  RegisterValueSpan(RegisterSpan reg_span) : reg_span_(reg_span) {}

  T& operator[](size_t idx) { return reg_span_[idx].Get<T>(); }
  const T& operator[](size_t idx) const { return reg_span_[idx].Get<T>(); }

  void Destroy(size_t idx) { reg_span_[idx].Destroy<T>(); }

  size_t size() const { return reg_span_.size(); }

  iterator begin() const { return iterator(reg_span_.begin()); }
  iterator end() const { return iterator(reg_span_.end()); }

  bool empty() const { return size() == 0; }

  RegisterValueSpan drop_front(int num = 1) {
    return reg_span_.drop_front(num);
  }

  RegisterValueSpan drop_back(int num = 1) { return reg_span_.drop_back(num); }

  RegisterSpan reg_span() const { return reg_span_; }

 private:
  RegisterSpan reg_span_;
};

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_REGISTER_SPAN_H_

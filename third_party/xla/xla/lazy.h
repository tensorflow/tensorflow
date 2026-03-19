/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_LAZY_H_
#define XLA_LAZY_H_

#include <memory>

#include "absl/functional/any_invocable.h"

namespace xla {

template <typename T>
class Lazy {
  using Value = std::unique_ptr<T>;

 public:
  using Initializer = absl::AnyInvocable<T() &&>;

  explicit Lazy(Initializer init)
      : initializer_(std::move(init)), initialized_(false) {}

  Lazy(const Lazy& other) = delete;
  Lazy& operator=(const Lazy& other) = delete;

  Lazy(Lazy&& other) : initialized_(other.initialized_) {
    if (other.initialized_) {
      new (&value_) Value(std::move(other.value_));
    } else {
      new (&initializer_) Initializer(std::move(other.initializer_));
    }
  }

  Lazy& operator=(Lazy&& other) {
    if (this != &other) {
      this->~Lazy();
      new (this) Lazy(std::move(other));
    }
    return *this;
  }

  ~Lazy() {
    if (initialized_) {
      value_.~unique_ptr();
    } else {
      initializer_.~Initializer();
    }
  }

  bool has_value() const { return initialized_; }

  const T& get() const {
    if (!has_value()) {
      // Using `make_unique` here since `Value` is a `unique_ptr`. If this
      // changes, we'll need to update this.
      auto value_ptr = std::make_unique<T>(std::move(initializer_)());
      initializer_.~Initializer();

      new (&value_) Value(std::move(value_ptr));
      initialized_ = true;
    }
    return *value_;
  }

 private:
  union {
    mutable Initializer initializer_;
    mutable Value value_;
  };
  mutable bool initialized_;
};

}  // namespace xla

#endif  // XLA_LAZY_H_

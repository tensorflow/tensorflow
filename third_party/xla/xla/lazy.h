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
#include <variant>

#include "absl/functional/any_invocable.h"

namespace xla {

template <typename T>
class Lazy {
  using Value = std::unique_ptr<T>;

 public:
  using Initializer = absl::AnyInvocable<T() &&>;

  explicit Lazy(Initializer init) : data_(std::move(init)) {}

  bool has_value() const { return std::holds_alternative<Value>(data_); }

  const T& get() const {
    if (!has_value()) {
      data_ = std::make_unique<T>(std::move(std::get<Initializer>(data_))());
    }
    return *std::get<Value>(data_);
  }

 private:
  mutable std::variant<Initializer, Value> data_;
};

}  // namespace xla

#endif  // XLA_LAZY_H_

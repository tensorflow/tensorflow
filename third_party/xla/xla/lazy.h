/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <variant>

#include "absl/functional/any_invocable.h"

namespace xla {

template <typename T>
class Lazy {
 public:
  explicit Lazy(absl::AnyInvocable<T() &&> func)
      : maybe_value_(std::move(func)) {}

  const T& get() const {
    if (!std::holds_alternative<T>(maybe_value_)) {
      maybe_value_ =
          std::move(std::get<absl::AnyInvocable<T() &&>>(maybe_value_))();
    }
    return std::get<T>(maybe_value_);
  }

 private:
  mutable std::variant<absl::AnyInvocable<T() &&>, T> maybe_value_;
};

}  // namespace xla

#endif  // XLA_LAZY_H_

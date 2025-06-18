/* Copyright 2025 The OpenXLA Authors.

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

// This file defines a free function ValueOrDie that can be used to safely
// dereference absl::StatusOr, std::optional and pointer values (or in fact any
// value that provides a compatible 'operator*' and is contextually convertible
// to bool).
//
// Example usage:
//
// int UnwrapNestedValueOrDie(absl::StatusOr<std::optional<int>> value) {
//   return tsl::gtl::ValueOrDie(tsl::gtl::ValueOrDie(value));
// }

#ifndef XLA_TSL_LIB_GTL_VALUE_OR_DIE_H_
#define XLA_TSL_LIB_GTL_VALUE_OR_DIE_H_

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tsl {
namespace gtl {
namespace internal_value_or_die {

// LOG(FATAL), with a source location and an optional 'status' for details.
ABSL_ATTRIBUTE_NORETURN
void DieBecauseEmptyValue(const char* file, int line,
                          const absl::Status* status = nullptr);

// SFINAE helper to detect instances of StatusOr<T>.
template <int&... kDoNotSpecify, typename T>
void IsStatusOr(const absl::StatusOr<T>&);

}  // namespace internal_value_or_die

template <
    int&... kDoNotSpecify, typename T,
    typename = decltype(internal_value_or_die::IsStatusOr(std::declval<T>()))>
decltype(auto) ValueOrDie(T&& value ABSL_ATTRIBUTE_LIFETIME_BOUND,
                          const char* file = __builtin_FILE(),
                          int line = __builtin_LINE()) {
  if (ABSL_PREDICT_FALSE(!value.ok())) {
    internal_value_or_die::DieBecauseEmptyValue(file, line, &value.status());
  }
  return *std::forward<T>(value);
}

template <int&... kDoNotSpecify, typename T,
          typename = decltype(*std::declval<T>()),
          decltype(static_cast<bool>(std::declval<T>())) = true>
decltype(auto) ValueOrDie(T&& value ABSL_ATTRIBUTE_LIFETIME_BOUND,
                          const char* file = __builtin_FILE(),
                          int line = __builtin_LINE()) {
  if (ABSL_PREDICT_FALSE(!value)) {
    internal_value_or_die::DieBecauseEmptyValue(file, line);
  }
  return *std::forward<T>(value);
}

template <int&... kDoNotSpecify, typename T>
decltype(auto) ValueOrDie(T* value ABSL_ATTRIBUTE_LIFETIME_BOUND,
                          const char* file = __builtin_FILE(),
                          int line = __builtin_LINE()) {
  if (ABSL_PREDICT_FALSE(!value)) {
    internal_value_or_die::DieBecauseEmptyValue(file, line);
  }
  return *value;
}

}  // namespace gtl
}  // namespace tsl

#endif  // XLA_TSL_LIB_GTL_VALUE_OR_DIE_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_STATUS_MACROS_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_STATUS_MACROS_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

// A few macros to help with propagating status.
// They are mainly adapted from the ones in  tensorflow/core/platform/errors.h
// and tensorflow/core/platform/statusor.h but these files come with many
// transitive deps which can be too much for TFLite use cases. If these type of
// macros end up in `absl` we can replace them with those.

// The macros are prefixed with SH_ to avoid name collision.

namespace tflite {
namespace shim {
template <typename... Args>
void AppendToMessage(::absl::Status* status, Args... args) {
  *status = ::absl::Status(status->code(),
                           ::absl::StrCat(status->message(), "\n\t", args...));
}
}  // namespace shim
}  // namespace tflite

// Propagates error up the stack and appends to the error message.
#define SH_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)            \
  do {                                                        \
    ::absl::Status _status = (expr);                          \
    if (!_status.ok()) {                                      \
      ::tflite::shim::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                         \
    }                                                         \
  } while (0)

// Propages the error up the stack.
// This can't be merged with the SH_RETURN_WITH_CONTEXT_IF_ERROR unless some
// overly clever/unreadable macro magic is used.
#define SH_RETURN_IF_ERROR(...)             \
  do {                                      \
    ::absl::Status _status = (__VA_ARGS__); \
    if (!_status.ok()) return _status;      \
  } while (0)

// Internal helper for concatenating macro values.
#define SH_STATUS_MACROS_CONCAT_NAME_INNER(x, y) x##y
#define SH_STATUS_MACROS_CONCAT_NAME(x, y) \
  SH_STATUS_MACROS_CONCAT_NAME_INNER(x, y)

// Assigns an expression to lhs or propagates the error up.
#define SH_ASSIGN_OR_RETURN(lhs, rexpr) \
  SH_ASSIGN_OR_RETURN_IMPL(             \
      SH_STATUS_MACROS_CONCAT_NAME(statusor, __COUNTER__), lhs, rexpr)

#define SH_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                             \
  if (!statusor.ok()) return statusor.status();        \
  lhs = std::move(statusor.value())

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_STATUS_MACROS_H_

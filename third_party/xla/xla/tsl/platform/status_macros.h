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

#ifndef XLA_TSL_PLATFORM_STATUS_MACROS_H_
#define XLA_TSL_PLATFORM_STATUS_MACROS_H_

#include "xla/tsl/platform/statusor.h"

#ifndef ASSIGN_OR_RETURN
#define ASSIGN_OR_RETURN(lhs, rexpr) \
  ASSIGN_OR_RETURN_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                          \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {         \
    return statusor.status();                       \
  }                                                 \
  lhs = std::move(statusor).value()
#endif  // ASSIGN_OR_RETURN

#ifndef RETURN_IF_ERROR
// For propagating errors when calling a function.
#define RETURN_IF_ERROR(...)                 \
  do {                                       \
    absl::Status _status = (__VA_ARGS__);    \
    if (ABSL_PREDICT_FALSE(!_status.ok())) { \
      MAYBE_ADD_SOURCE_LOCATION(_status)     \
      return _status;                        \
    }                                        \
  } while (0)
#endif  // RETURN_IF_ERROR

#endif  // XLA_TSL_PLATFORM_STATUS_MACROS_H_

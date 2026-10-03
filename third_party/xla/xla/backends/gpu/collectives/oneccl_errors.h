/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_

#include <atomic>

#include "oneapi/ccl.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"  // IWYU pragma: keep

#define XLA_ONECCL_STATUS(expr)                                 \
  [](onecclResult_t s, absl::string_view str) -> absl::Status { \
    if (s == onecclSuccess) return absl::OkStatus();            \
    return xla::Internal("OneCCL operation %s failed", str);    \
  }(expr, #expr)

#define XLA_ONECCL_RETURN_IF_ERROR(expr)      \
  do {                                        \
    absl::Status s = XLA_ONECCL_STATUS(expr); \
    if (!s.ok()) {                            \
      return s;                               \
    }                                         \
  } while (0)

#define XLA_ONECCL_LOG_IF_ERROR(expr)         \
  do {                                        \
    absl::Status s = XLA_ONECCL_STATUS(expr); \
    if (!s.ok()) {                            \
      LOG(ERROR) << s.ToString();             \
    }                                         \
  } while (0)

#define XLA_ONECCL_CHECK(expr) CHECK(XLA_ONECCL_STATUS(expr).ok())

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_

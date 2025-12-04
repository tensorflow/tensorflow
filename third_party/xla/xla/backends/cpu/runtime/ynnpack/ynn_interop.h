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

#ifndef XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_INTEROP_H_
#define XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_INTEROP_H_

#include <memory>

#include "ynnpack/include/ynnpack.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// YNNPACK status to ABSL status conversion macros.
//===----------------------------------------------------------------------===//

#define YNN_RETURN_IF_ERROR(expr)             \
  do {                                        \
    absl::Status s = YnnStatusToStatus(expr); \
    if (!s.ok()) {                            \
      return s;                               \
    }                                         \
  } while (0)

#define YNN_LOG_IF_ERROR(expr)                         \
  do {                                                 \
    absl::Status s = YnnStatusToStatus(expr);          \
    if (!s.ok()) {                                     \
      LOG(ERROR) << "YNNPACK operation failed: " << s; \
    }                                                  \
  } while (0)

// Converts YNNPACK status to absl::Status.
inline absl::Status YnnStatusToStatus(ynn_status status) {
  if (ABSL_PREDICT_TRUE(status == ynn_status_success)) {
    return absl::OkStatus();
  }

  auto error_message = [](ynn_status status) {
    switch (status) {
      case ynn_status_success:
        return "";
      case ynn_status_deprecated:
        return "deprecated";
      case ynn_status_error:
        return "error";
      case ynn_status_invalid_parameter:
        return "invalid parameter";
      case ynn_status_unsupported_parameter:
        return "unsupported parameter";
    }
  };

  return Internal("YNNPACK operation failed: %s", error_message(status));
}

//===----------------------------------------------------------------------===//
// XLA to YNNPACK type conversions.
//===----------------------------------------------------------------------===//

absl::StatusOr<ynn_type> YnnType(const PrimitiveType& type);

//===----------------------------------------------------------------------===//
// RAII wrappers for YNNPACK types.
//===----------------------------------------------------------------------===//

namespace internal {

struct YnnDeleter {
  void operator()(ynn_subgraph* subgraph) { ynn_delete_subgraph(subgraph); }
  void operator()(ynn_runtime* runtime) { ynn_delete_runtime(runtime); }
  void operator()(ynn_threadpool* threadpool) {
    ynn_delete_threadpool(threadpool);
  }
};

}  // namespace internal

using YnnSubgraph = std::unique_ptr<ynn_subgraph, internal::YnnDeleter>;
using YnnRuntime = std::unique_ptr<ynn_runtime, internal::YnnDeleter>;
using YnnThreadpool = std::unique_ptr<ynn_threadpool, internal::YnnDeleter>;

absl::StatusOr<YnnSubgraph> CreateYnnSubgraph(
    absl::FunctionRef<ynn_status(ynn_subgraph_t*)> builder);

absl::StatusOr<YnnRuntime> CreateYnnRuntime(
    absl::FunctionRef<ynn_status(ynn_runtime_t*)> builder);

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    absl::FunctionRef<ynn_status(ynn_threadpool_t*)> builder);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_INTEROP_H_

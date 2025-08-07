/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_INTEROP_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_INTEROP_H_

#include <memory>

#include "experimental.h"  // xnnpack
#include "xnnpack.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// XNNPACK status to ABSL status conversion macros.
//===----------------------------------------------------------------------===//

#define XNN_RETURN_IF_ERROR(expr)             \
  do {                                        \
    absl::Status s = XnnStatusToStatus(expr); \
    if (!s.ok()) {                            \
      return s;                               \
    }                                         \
  } while (0)

#define XNN_LOG_IF_ERROR(expr)                         \
  do {                                                 \
    absl::Status s = XnnStatusToStatus(expr);          \
    if (!s.ok()) {                                     \
      LOG(ERROR) << "XNNPACK operation failed: " << s; \
    }                                                  \
  } while (0)

// Statically initializes XNNPACK for the current process.
absl::Status InitializeXnnPack();

// Converts XNNPACK status to absl::Status.
inline absl::Status XnnStatusToStatus(xnn_status status) {
  if (ABSL_PREDICT_TRUE(status == xnn_status_success)) {
    return absl::OkStatus();
  }

  auto error_message = [](xnn_status status) {
    switch (status) {
      case xnn_status_success:
        return "";
      case xnn_status_uninitialized:
        return "uninitialized";
      case xnn_status_invalid_parameter:
        return "invalid parameter";
      case xnn_status_invalid_state:
        return "invalid state";
      case xnn_status_unsupported_parameter:
        return "unsupported parameter";
      case xnn_status_unsupported_hardware:
        return "unsupported hardware";
      case xnn_status_out_of_memory:
        return "out of memory";
      case xnn_status_reallocation_required:
        return "reallocation required";
      case xnn_status_deprecated:
        return "deprecated";
    }
  };

  return Internal("XNNPACK operation failed: %s", error_message(status));
}

//===----------------------------------------------------------------------===//
// RAII wrappers for XNNPACK types.
//===----------------------------------------------------------------------===//

namespace internal {
struct XnnDeleter {
  void operator()(xnn_subgraph* subgraph) {
    XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
  }
  void operator()(xnn_runtime* runtime) {
    XNN_LOG_IF_ERROR(xnn_delete_runtime(runtime));
  }
  void operator()(xnn_threadpool* threadpool) {
    XNN_LOG_IF_ERROR(xnn_delete_threadpool(threadpool));
  }
};
}  // namespace internal

using XnnSubgraph = std::unique_ptr<xnn_subgraph, internal::XnnDeleter>;
using XnnRuntime = std::unique_ptr<xnn_runtime, internal::XnnDeleter>;
using XnnThreadpool = std::unique_ptr<xnn_threadpool, internal::XnnDeleter>;

absl::StatusOr<XnnSubgraph> CreateXnnSubgraph(
    absl::FunctionRef<xnn_status(xnn_subgraph_t*)> builder);

absl::StatusOr<XnnRuntime> CreateXnnRuntime(
    absl::FunctionRef<xnn_status(xnn_runtime_t*)> builder);

absl::StatusOr<XnnThreadpool> CreateXnnThreadpool(
    absl::FunctionRef<xnn_status(xnn_threadpool_t*)> builder);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_INTEROP_H_

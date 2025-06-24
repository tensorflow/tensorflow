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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_INTEROP_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_INTEROP_H_

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::cpu {

#define ONEDNN_RETURN_IF_ERROR(expr)             \
  do {                                           \
    absl::Status s = OneDnnStatusToStatus(expr); \
    if (!s.ok()) {                               \
      return s;                                  \
    }                                            \
  } while (0)

#define ONEDNN_LOG_IF_ERROR(expr)                   \
  do {                                              \
    absl::Status s = OneDnnStatusToStatus(expr);    \
    if (!s.ok()) {                                  \
      LOG(ERROR) << "DNNL operation failed: " << s; \
    }                                               \
  } while (0)

// Statically initializes XNNPACK for the current process.
absl::Status InitializeXnnPack();

// Converts oneDNN status to absl::Status.
inline absl::Status OneDnnStatusToStatus(dnnl::graph::status status) {
  if (ABSL_PREDICT_TRUE(status == dnnl::graph::status::success)) {
    return absl::OkStatus();
  }

  auto error_message = [](dnnl::graph::status status) {
    switch (status) {
      case dnnl::graph::status::success:
        return "";
      case dnnl::graph::status::out_of_memory:
        return "out of memory";
      case dnnl::graph::status::invalid_arguments:
        return "invalid arguments";
      case dnnl::graph::status::unimplemented:
        return "unimplemented";
      case dnnl::graph::status::last_impl_reached:
        return "last implementation reached";
      case dnnl::graph::status::runtime_error:
        return "runtime error";
      case dnnl::graph::status::not_required:
        return "not required";
      case dnnl::graph::status::invalid_graph:
        return "invalid graph";
      case dnnl::graph::status::invalid_graph_op:
        return "invalid graph op";
      case dnnl::graph::status::invalid_shape:
        return "invalid shape";
      case dnnl::graph::status::invalid_data_type:
        return "invalid data type";
    }
  };

  return Internal("DNNL operation failed: %s", error_message(status));
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_INTEROP_H_

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

#include "xla/tsl/platform/logging.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tsl {
namespace internal {

void LogString(absl::string_view fname, int line, absl::LogSeverity severity,
               absl::string_view message) {
  LOG(LEVEL(severity)).AtLocation(fname, line) << message;
}

void LogLines(absl::LogSeverity sev, absl::string_view text, const char* fname,
              int lineno) {
  const absl::LogSeverity orig_sev = sev;
  if (sev == absl::LogSeverity::kFatal) {
    sev = absl::LogSeverity::kError;
  }

  // Protect calls with a mutex so we don't interleave calls to LogLines from
  // multiple threads.
  static absl::Mutex log_lines_mu(absl::kConstInit);
  absl::MutexLock lock(log_lines_mu);

  size_t cur = 0;
  while (cur < text.size()) {
    size_t eol = text.find('\n', cur);
    if (eol == absl::string_view::npos) {
      eol = text.size();
    }
    auto msg = text.substr(cur, eol - cur);
    LogString(fname, lineno, sev, msg);
    cur = eol + 1;
  }

  if (orig_sev == absl::LogSeverity::kFatal) {
    LogString(fname, lineno, orig_sev, "Aborting due to errors.");
  }
}

}  // namespace internal

void UpdateLogVerbosityIfDefined(absl::string_view env_var) {
  if (env_var.empty()) {
    return;
  }
  int vlog_level = 0;
  std::vector<std::string> bridge_files = {
      "bridge",
      "bridge_logger",
      "cluster_tf",
      "compile_mlir_util",
      "graph_analysis",
      "import_model",
      "legalize_tf",
      "mlir_bridge_pass",
      "mlir_graph_optimization_pass",
      "optimization_registry",
      "tf_dialect_to_executor",
      "tpu_compile",
      "tpu_compile_op_impl",
      "xla_compiler",
  };

  const char* env_var_val = getenv(env_var.data());
  if (!env_var_val) {
    return;
  }
  int log_verbosity = std::stoi(env_var_val);
  switch (log_verbosity) {
    case 1:
      // This level prints minimum logs with statements only.
      vlog_level = 1;
      bridge_files = {"bridge_logger", "graph_analysis", "tpu_compile",
                      "tpu_compile_op_impl"};
      break;
    case 2:
      // This level enables logging before and after the entire pass pipeline.
      vlog_level = 1;
      break;
    case 3:
      // This level enables logging before and after each of the passes in the
      // pass pipeline as well as other important points during compilation
      // outside of the pipeline.
      vlog_level = 5;
      break;
    default:
      LOG(INFO) << "Unknown value for bridge log verbosity: " << log_verbosity;
      return;
  }
  // NOLINTBEGIN(abseil-no-internal-dependencies)
  for (const auto& file : bridge_files) {
    int curr_vlog_level = absl::log_internal::VLogLevel(file);
    if (vlog_level > curr_vlog_level) {
      LOG(INFO) << "Updating vlog level of " << file << ".cc from "
                << curr_vlog_level << " to " << vlog_level;
      absl::SetVLogLevel(file, vlog_level);
    } else {
      LOG(INFO) << "Current vlog level of " << file << ".cc ("
                << curr_vlog_level << ") is greater than or equal to "
                << vlog_level;
    }
  }
  // NOLINTEND(abseil-no-internal-dependencies)
}

}  // namespace tsl

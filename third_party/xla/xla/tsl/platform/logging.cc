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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/platform.h"

namespace tsl {
namespace {

std::optional<int64_t> LogLevelStrToInt(const char* str) {
  if (str == nullptr || strlen(str) == 0) {
    return std::nullopt;
  }
  if (int level; absl::SimpleAtoi(str, &level)) {
    return level;
  }
  return std::nullopt;
}

void UpdateVlogLevels(const char* spec) {
  if (spec == nullptr || strlen(spec) == 0) {
    return;
  }
  for (absl::string_view entry : absl::StrSplit(spec, ',')) {
    std::vector<absl::string_view> parts = absl::StrSplit(entry, '=');
    if (parts.size() != 2) {
      continue;
    }
    absl::string_view module = parts[0];
    if (int level; absl::SimpleAtoi(parts[1], &level)) {
      absl::SetVLogLevel(module, level);
    }
  }
}

// Initializes logging and configures it based on environment variables.
// This class is intended to be used as a global instance, ensuring that
// logging is initialized before any other code that might use it.
class LoggingInitializer {
 public:
  LoggingInitializer() {
    LOG(INFO) << "Init logging!";
    if (auto severity = LogLevelStrToInt(std::getenv("TF_CPP_MIN_LOG_LEVEL"))) {
      absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(*severity));
    }
    if (auto threshold =
            LogLevelStrToInt(std::getenv("TF_CPP_MAX_VLOG_LEVEL"))) {
      absl::SetGlobalVLogLevel(*threshold);
    }
    UpdateVlogLevels(std::getenv("TF_CPP_VMODULE"));
  }
};

#ifndef PLATFORM_GOOGLE
// Global instance. Its constructor is called before main().
LoggingInitializer g_initializer;
#endif

}  // namespace

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

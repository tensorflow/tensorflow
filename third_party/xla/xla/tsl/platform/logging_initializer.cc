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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

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
    LOG(ERROR) << "Initializing TSL logging";
    // We log everything to stderr for backwards compatibility with TSL
    // logging.
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    if (auto severity = LogLevelStrToInt(std::getenv("TF_CPP_MIN_LOG_LEVEL"))) {
      LOG(ERROR) << "Setting min log level to "
                 << std::getenv("TF_CPP_MIN_LOG_LEVEL") << " or "
                 << static_cast<absl::LogSeverityAtLeast>(*severity);
      absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(*severity));
    }
    if (auto threshold =
            LogLevelStrToInt(std::getenv("TF_CPP_MAX_VLOG_LEVEL"))) {
      absl::SetGlobalVLogLevel(*threshold);
    }
    UpdateVlogLevels(std::getenv("TF_CPP_VMODULE"));
  }
};

// Global instance. Its constructor is called before main().
LoggingInitializer g_initializer;

}  // namespace
}  // namespace tsl

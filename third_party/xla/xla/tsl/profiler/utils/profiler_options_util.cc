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

#include "xla/tsl/profiler/utils/profiler_options_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace tsl {
namespace profiler {

std::optional<std::variant<std::string, bool, int64_t>> GetConfigValue(
    const tensorflow::ProfileOptions& options, const std::string& key) {
  auto config = options.advanced_configuration().find(key);

  if (config != options.advanced_configuration().end()) {
    const tensorflow::ProfileOptions::AdvancedConfigValue& config_value =
        config->second;
    if (config_value.has_string_value()) {
      return config_value.string_value();
    } else if (config_value.has_bool_value()) {
      return config_value.bool_value();
    } else if (config_value.has_int64_value()) {
      return config_value.int64_value();
    }
  }

  return std::nullopt;
}
}  // namespace profiler
}  // namespace tsl

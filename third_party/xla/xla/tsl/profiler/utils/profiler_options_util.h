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

#ifndef XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_
#define XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
// Get config value from the profiler options, if the key is not found, return
// std::nullopt.
std::optional<std::variant<std::string, bool, int64_t>> GetConfigValue(
    const tensorflow::ProfileOptions& options, const std::string& key);
}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TOOL_OPTIONS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TOOL_OPTIONS_H_

#include <optional>
#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"

namespace tensorflow {
namespace profiler {

// Tool options for HloProtoToToolData conversion.
struct HloToolOptions {
  std::optional<std::string> module_name;
  std::optional<std::string> type;
  std::optional<std::string> node_name;
  std::optional<std::string> format;
  int graph_width;
  bool show_metadata;
  bool merge_fusion;
};

using ToolOptions =
    absl::flat_hash_map<std::string, std::variant<int, std::string>>;

// Helper function to get parameter from tool options.
template <typename T>
std::optional<T> GetParam(const ToolOptions& options, const std::string& key) {
  const auto iter = options.find(key);
  if (iter == options.end()) {
    return std::nullopt;
  }

  const T* result = std::get_if<T>(&iter->second);
  if (!result) {
    return std::nullopt;
  }
  return *result;
}

// Helper function to get parameter from tool options with default value.
template <typename T>
T GetParamWithDefault(const ToolOptions& options, const std::string& key,
                      const T& default_param) {
  if (auto param = GetParam<T>(options, key)) {
    return *param;
  }
  return default_param;
}

inline HloToolOptions ToolOptionsToHloToolOptions(const ToolOptions& options) {
  HloToolOptions hlo_options;
  hlo_options.module_name = GetParam<std::string>(options, "module_name");
  hlo_options.type = GetParam<std::string>(options, "type");
  hlo_options.node_name = GetParam<std::string>(options, "node_name");
  hlo_options.format = GetParam<std::string>(options, "format");
  hlo_options.graph_width = GetParamWithDefault<int>(options, "graph_width", 3);
  hlo_options.show_metadata =
      GetParamWithDefault<int>(options, "show_metadata", 0);
  hlo_options.merge_fusion =
      GetParamWithDefault<int>(options, "merge_fusion", 0);
  return hlo_options;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TOOL_OPTIONS_H_

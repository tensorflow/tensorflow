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

#include "xla/tools/benchmarks/utils/generate_benchmark_matrices.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"
#include "xla/tsl/platform/file_system_helper.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace tools {
namespace benchmarks {

absl::StatusOr<BenchmarkSuite> ParseRegistry(const std::string& registry_path) {
  BenchmarkSuite suite;
  std::ifstream file_stream(registry_path);

  if (!file_stream.is_open()) {
    // Use tsl::errors for consistency
    return absl::NotFoundError(
        absl::StrCat("Registry file not found at ", registry_path));
  }

  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  std::string content = buffer.str();
  file_stream.close();  // Close the stream after reading

  // Use tsl::protobuf::TextFormat for consistency with existing includes
  if (!tsl::protobuf::TextFormat::ParseFromString(content, &suite)) {
    return absl::InternalError(
        absl::StrCat("Error parsing TextProto registry file ", registry_path));
  }

  return suite;
}

absl::StatusOr<Json::Value> GenerateMatrix(const BenchmarkSuite& suite) {
  return absl::UnimplementedError("Not implemented yet");
}

absl::StatusOr<std::string> ResolveRegistryPath(
    const std::string& registry_path_str) {
  // 1. Check if absolute using tsl::io::IsAbsolutePath
  if (tsl::io::IsAbsolutePath(registry_path_str)) {
    std::cerr << "Registry path is absolute: " << registry_path_str << "\n";
    auto file_exists =
        tsl::internal::FileExists(tsl::Env::Default(), registry_path_str);
    if (file_exists.ok() && *file_exists) {
      std::cerr << "Absolute path exists.\n";
      return registry_path_str;
    }
    return absl::FailedPreconditionError(absl::StrCat(
        "Absolute registry path specified but not found: ", registry_path_str));
  }

  // --- Path is Relative ---
  std::cerr << "Registry file path '" << registry_path_str
            << "' is relative. Attempting resolution...\n";

  // 2. Check relative to BUILD_WORKSPACE_DIRECTORY (if set)
  const char* build_workspace_dir_cstr =
      std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (build_workspace_dir_cstr != nullptr &&
      build_workspace_dir_cstr[0] != '\0') {
    std::string build_workspace_dir(build_workspace_dir_cstr);
    std::string workspace_path =
        tsl::io::JoinPath(build_workspace_dir, registry_path_str);

    std::cerr << "Checking workspace path: " << workspace_path << "\n";
    auto file_exists =
        tsl::internal::FileExists(tsl::Env::Default(), workspace_path);
    if (file_exists.ok()) {
      std::cerr << "Found registry file in workspace: " << workspace_path
                << "\n";
      return workspace_path;
    }
    std::cerr << "Registry file not found relative to workspace.\n";
  } else {
    std::cerr << "BUILD_WORKSPACE_DIRECTORY not set or empty, skipping "
                 "workspace check.\n";
  }

  // --- If we reach here, the file was not found via absolute or workspace ---
  std::cerr << "Registry file '" << registry_path_str
            << "' not found. Tried absolute and relative to workspace.\n";
  return absl::FailedPreconditionError(
      absl::StrCat("Registry file '", registry_path_str,
                   "' not found. Tried absolute and relative to workspace (if "
                   "BUILD_WORKSPACE_DIRECTORY was set)."));
}

}  // namespace benchmarks
}  // namespace tools
}  // namespace xla

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
#include <vector>

#include "absl/container/flat_hash_map.h"
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

namespace {

// --- Mapping Logic ---
const absl::flat_hash_map<std::string, std::string>&
GetHardwareToRunnerLabel() {
  static const auto* kHardwareToRunnerLabel =
      new absl::flat_hash_map<std::string, std::string>{
          {"CPU_X86", "linux-x86-n2-128"},
          {"CPU_ARM64", "linux-arm64-c4a-64"},
          {"GPU_L4", "linux-x86-g2-16-l4-1gpu"},
          {"GPU_B200", "linux-x86-a4-224-b200-1gpu"},
          {"GPU_L4_1H_4D", "linux-x86-g2-48-l4-4gpu"},
      };
  return *kHardwareToRunnerLabel;
}

const absl::flat_hash_map<std::string, std::string>&
GetHardwareToContainerImage() {
  static const auto* kHardwareToContainerImage =
      new absl::flat_hash_map<std::string, std::string>{
          {"CPU_X86",
           "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
           "ml-build:latest"},
          {"CPU_ARM64",
           "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
           "ml-build-arm64:latest"},
          {"GPU_L4",
           "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
          {"GPU_B200",
           "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
          {"GPU_L4_1H_4D",
           "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
      };
  return *kHardwareToContainerImage;
}

// Mapping from hardware category name to target metric enums
const absl::flat_hash_map<std::string, std::vector<TargetMetric>>&
GetHardwareToTargetMetrics() {
  static const auto* kHardwareToTargetMetrics =
      new absl::flat_hash_map<std::string, std::vector<TargetMetric>>{
          {"CPU_X86", {CPU_TIME}},
          {"CPU_ARM64", {CPU_TIME}},
          {"GPU_L4", {GPU_DEVICE_TIME, GPU_DEVICE_MEMCPY_TIME}},
          {"GPU_B200", {GPU_DEVICE_TIME, GPU_DEVICE_MEMCPY_TIME}},
          {"GPU_L4_1H_4D", {GPU_DEVICE_TIME, GPU_DEVICE_MEMCPY_TIME}},
      };
  return *kHardwareToTargetMetrics;
}

// Helper function to get string names of enums
std::string GetHardwareCategoryName(HardwareCategory category) {
  const std::string& name = HardwareCategory_Name(category);
  if (!name.empty()) {
    return name;
  }
  return "UNKNOWN_HARDWARE";
}

std::string GetRunFrequencyName(RunFrequency frequency) {
  return RunFrequency_Name(frequency);
}

std::string GetTargetMetricName(TargetMetric metric) {
  const std::string& name = TargetMetric_Name(metric);
  if (!name.empty()) {
    return name;
  }
  return "UNKNOWN_METRIC";
}

// Helper function to create a JSON object from a BenchmarkConfig's topology.
absl::StatusOr<Json::Value> CreateTopologyJson(const BenchmarkConfig& config) {
  Json::Value topology_json = Json::Value(Json::objectValue);
  topology_json["multi_host"] = config.topology().multi_host();
  topology_json["multi_device"] = config.topology().multi_device();
  topology_json["num_hosts"] = config.topology().num_hosts();
  topology_json["num_devices_per_host"] =
      config.topology().num_devices_per_host();
  return topology_json;
}

}  // namespace

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
  // TODO(b/408495113): Implement the logic to generate the matrix using helper
  // functions.
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

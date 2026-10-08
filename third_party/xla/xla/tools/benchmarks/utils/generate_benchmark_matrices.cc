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

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
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

// Helper to get string name safely
std::string GetHardwareCategoryName(HardwareCategory category) {
  const std::string& name = HardwareCategory_Name(category);
  if (!name.empty()) {
    return name;
  }
  return "UNKNOWN_HARDWARE";  // Or handle error differently
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

// Generates a unique, filesystem-friendly ID for the benchmark configuration.
std::string GenerateConfigId(const BenchmarkConfig& config) {
  std::string hw_enum_name =
      GetHardwareCategoryName(config.hardware_category());

  // Use lowercase and remove prefix for brevity
  std::string hw_short = hw_enum_name;
  absl::StrReplaceAll({{"GPU_", ""}, {"CPU_", ""}}, &hw_short);
  for (char& c : hw_short) {
    c = std::tolower(c);
  }

  const auto& topo = config.topology();
  std::string topo_str =
      absl::StrCat(topo.num_hosts(), "h_", topo.num_devices_per_host(), "d");

  // Sanitize name slightly
  std::string sanitized_name = config.name();
  absl::StrReplaceAll({{" ", "_"}, {"/", "_"}}, &sanitized_name);

  return absl::StrCat(sanitized_name, "_", hw_short, "_", topo_str);
}

// Maps BenchmarkConfig hardware/topology to GHA runner label and container.
absl::StatusOr<std::pair<std::string, std::string>> GetRunnerInfo(
    const BenchmarkConfig& config) {
  std::string hw_enum_name =
      GetHardwareCategoryName(config.hardware_category());
  const auto& topology = config.topology();
  std::string mapping_key = hw_enum_name;  // Default

  std::string runner_label;
  if (auto it = GetHardwareToRunnerLabel().find(mapping_key);
      it != GetHardwareToRunnerLabel().end()) {
    runner_label = it->second;
  } else {
    return absl::NotFoundError(absl::StrCat(
        "No runner mapping for config '", config.name(), "' with hardware '",
        hw_enum_name, "' and topology (", topology.num_hosts(), "h/",
        topology.num_devices_per_host(), "d)."));
  }

  std::string container_image;
  if (auto it = GetHardwareToContainerImage().find(mapping_key);
      it != GetHardwareToContainerImage().end()) {
    container_image = it->second;
  } else {
    return absl::NotFoundError(absl::StrCat(
        "No container image mapping for config '", config.name(),
        "' with hardware '", hw_enum_name, "' and topology (",
        topology.num_hosts(), "h/", topology.num_devices_per_host(), "d)."));
  }

  return std::make_pair(runner_label, container_image);
}

absl::StatusOr<Json::Value> CreateTopologyJson(const BenchmarkConfig& config) {
  Json::Value topology_json = Json::Value(Json::objectValue);
  topology_json["multi_host"] = config.topology().multi_host();
  topology_json["multi_device"] = config.topology().multi_device();
  topology_json["num_hosts"] = config.topology().num_hosts();
  topology_json["num_devices_per_host"] =
      config.topology().num_devices_per_host();
  return topology_json;
}

absl::StatusOr<Json::Value> CreateMatrixEntry(
    const BenchmarkConfig& config, const std::string& config_id,
    const std::string& runner_label, const std::string& container_image,
    const std::string& hlo_location, bool is_gcs_hlo,
    const std::string& run_frequency_name,
    const std::string& hardware_category_name,
    const std::vector<std::string>& target_metrics_str) {
  Json::Value xla_flags_json = Json::Value(Json::arrayValue);
  for (const auto& flag : config.xla_compilation_flags()) {
    xla_flags_json.append(flag);
  }

  Json::Value runtime_flags_json = Json::Value(Json::arrayValue);
  for (const auto& flag : config.runtime_flags()) {
    runtime_flags_json.append(flag);
  }

  Json::Value github_labels_json = Json::Value(Json::arrayValue);
  for (const auto& label : config.github_labels()) {
    github_labels_json.append(label);
  }

  Json::Value matrix_entry = Json::Value(Json::objectValue);
  matrix_entry["config_id"] = config_id;
  matrix_entry["benchmark_name"] = config.name();
  matrix_entry["run_frequency"] = run_frequency_name;
  matrix_entry["runner_label"] = runner_label;
  matrix_entry["container_image"] = container_image;
  matrix_entry["hlo_location"] = hlo_location;
  matrix_entry["is_gcs_hlo"] = is_gcs_hlo;
  matrix_entry["target_metrics"] = Json::Value(Json::arrayValue);
  for (const auto& metric : target_metrics_str) {
    matrix_entry["target_metrics"].append(metric);
  }
  matrix_entry["xla_compilation_flags"] = xla_flags_json;
  matrix_entry["runtime_flags"] = runtime_flags_json;
  matrix_entry["required_hardware_category"] = hardware_category_name;
  auto topology_json = CreateTopologyJson(config);
  if (!topology_json.ok()) {
    return topology_json.status();
  }
  matrix_entry["topology"] = *topology_json;
  matrix_entry["github_labels"] = github_labels_json;

  return matrix_entry;
}

void AppendMatrixEntryForRunFrequency(
    const BenchmarkConfig& config, Json::Value& matrix,
    RunFrequency run_frequency_enum, const std::string& config_id,
    const std::string& runner_label, const std::string& container_image,
    const std::string& hlo_location, bool is_gcs_hlo,
    const std::string& run_frequency_name,
    const std::string& hardware_category_name) {
  // Get target metrics
  std::vector<std::string> target_metrics_str;
  if (config.target_metrics_size() > 0) {
    // Use metrics explicitly defined in the proto
    std::cerr << "Using target_metrics defined in proto for config '"
              << config.name() << "'\n";
    for (const auto& metric_enum : config.target_metrics()) {
      target_metrics_str.push_back(
          GetTargetMetricName(static_cast<TargetMetric>(metric_enum)));
    }
  } else {
    // Fallback to map lookup if proto field is empty
    std::string hardware_category_name_local =
        GetHardwareCategoryName(config.hardware_category());
    if (auto it =
            GetHardwareToTargetMetrics().find(hardware_category_name_local);
        it != GetHardwareToTargetMetrics().end()) {
      for (const auto& metric_enum : it->second) {
        target_metrics_str.push_back(GetTargetMetricName(metric_enum));
      }
    } else {
      std::cerr << "Warning: No target metrics defined in proto or map for "
                   "hardware '"
                << hardware_category_name_local << "' in config '"
                << config.name() << "'\n";
    }
  }

  auto matrix_entry =
      CreateMatrixEntry(config, config_id, runner_label, container_image,
                        hlo_location, is_gcs_hlo, run_frequency_name,
                        hardware_category_name, target_metrics_str);
  if (!matrix_entry.ok()) {
    std::cerr << "Skipping config '" << config.name()
              << "': " << matrix_entry.status().message() << "\n";
    return;
  }
  matrix["include"].append(*matrix_entry);
}

void AppendMatrixEntriesForConfig(const BenchmarkConfig& config,
                                  Json::Value& matrix) {
  for (int i = 0; i < config.run_frequencies_size(); ++i) {
    RunFrequency run_frequency_enum = config.run_frequencies(i);
    std::string config_id = GenerateConfigId(config);
    auto runner_info = GetRunnerInfo(config);

    if (!runner_info.ok()) {
      std::cerr << "Skipping config '" << config.name()
                << "': " << runner_info.status().message() << "\n";
      continue;
    }
    auto [runner_label, container_image] = *runner_info;

    std::string hlo_location = config.hlo_path().empty()
                                   ? config.hlo_gcs_bucket_path()
                                   : config.hlo_path();
    bool is_gcs_hlo = !config.hlo_gcs_bucket_path().empty();

    std::string run_frequency_name = GetRunFrequencyName(run_frequency_enum);
    std::string hardware_category_name =
        GetHardwareCategoryName(config.hardware_category());

    AppendMatrixEntryForRunFrequency(
        config, matrix, run_frequency_enum, config_id, runner_label,
        container_image, hlo_location, is_gcs_hlo, run_frequency_name,
        hardware_category_name);
  }
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

Json::Value GenerateMatrix(const BenchmarkSuite& suite) {
  Json::Value matrix = Json::Value(Json::objectValue);
  matrix["include"] = Json::Value(Json::arrayValue);

  for (const auto& config : suite.configs()) {
    AppendMatrixEntriesForConfig(config, matrix);
  }
  return matrix;
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
    // Note: JoinPath doesn't fully normalize like std::filesystem::absolute
    // might. Consider tsl::io::CleanPath if needed, although often JoinPath is
    // sufficient.
    // workspace_path = tsl::io::CleanPath(workspace_path);

    std::cerr << "Checking workspace path: " << workspace_path << "\n";
    auto file_exists =
        tsl::internal::FileExists(tsl::Env::Default(), workspace_path);
    if (file_exists.ok() && *file_exists) {
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

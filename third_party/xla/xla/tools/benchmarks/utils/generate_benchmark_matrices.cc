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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace tools {
namespace benchmarks {

namespace {
// --- String Conversion Helpers ---

std::string HardwareCategoryToString(HardwareCategory category) {
  const std::string name = HardwareCategory_Name(category);
  return name.empty() ? "UNKNOWN_HARDWARE" : name;
}

std::string TargetMetricToString(TargetMetric metric) {
  const std::string name = TargetMetric_Name(metric);
  return name.empty() ? "UNKNOWN_METRIC" : name;
}

// --- Mapping Definitions ---

// Creates a standardized key (e.g., "GPU_L4_1h_4d") for looking up runner/image
// info. This key incorporates topology for multi-host/device cases.
std::string MakeHardwareTargetMappingKey(const HardwareTarget& target) {
  std::string hw_name = HardwareCategoryToString(target.hardware_category());
  const auto& topology = target.topology();
  // Only add topology suffix if it's multi-host or multi-device
  if (topology.num_hosts() > 1 || topology.num_devices_per_host() > 1) {
    // Use uppercase H and D as in the example map keys
    return absl::StrCat(hw_name, "_", topology.num_hosts(), "H_",
                        topology.num_devices_per_host(), "D");
  }
  // Otherwise, just use the hardware category name (e.g., "CPU_X86", "GPU_L4")
  return hw_name;
}

const absl::flat_hash_map<std::string, std::string>&
GetHardwareToRunnerLabel() {
  static const auto* kHardwareToRunnerLabel =
      new absl::flat_hash_map<std::string, std::string>{
          {"CPU_X86", "linux-x86-n2-128"},
          {"CPU_ARM64", "linux-arm64-c4a-64"},
          {"GPU_L4", "linux-x86-g2-16-l4-1gpu"},
          {"GPU_B200", "linux-x86-a4-224-b200-1gpu"},
          // Key uses H_D suffix for multi-device
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

// Default mapping from hardware category name to target metric enums.
// Note: TargetMetric can be overridden in the BenchmarkConfig proto.
const absl::flat_hash_map<std::string, std::vector<TargetMetric>>&
GetHardwareToDefaultTargetMetrics() {
  static const auto* kHardwareToDefaultTargetMetrics =
      new absl::flat_hash_map<std::string, std::vector<TargetMetric>>{
          // Key is just the hardware category name for this map
          {"CPU_X86", {TargetMetric::CPU_TIME}},
          {"CPU_ARM64", {TargetMetric::CPU_TIME}},
          {"GPU_L4",
           {TargetMetric::GPU_DEVICE_TIME,
            TargetMetric::GPU_DEVICE_MEMCPY_TIME}},
          {"GPU_B200",
           {TargetMetric::GPU_DEVICE_TIME,
            TargetMetric::GPU_DEVICE_MEMCPY_TIME}},
      };
  return *kHardwareToDefaultTargetMetrics;
}

// --- Mapping Retrieval ---

// Retrieves the GHA runner label using the provided global map.
// e.g., CPU_X86 -> "linux-x86-n2-128",
//       GPU_L4_1H_4D -> "linux-x86-g2-48-l4-4gpu"
absl::StatusOr<std::string> GetRunnerLabelForTarget(
    const HardwareTarget& target) {
  std::string mapping_key = MakeHardwareTargetMappingKey(target);
  const auto& map = GetHardwareToRunnerLabel();
  auto it = map.find(mapping_key);
  if (it == map.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No GHA runner label mapping found for key: ", mapping_key));
  }
  return it->second;
}

// Retrieves the container image name using the provided global map.
// e.g., CPU_X86 ->
// "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest",
//       GPU_L4_1H_4D ->
//       "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest"
absl::StatusOr<std::string> GetContainerImageForTarget(
    const HardwareTarget& target) {
  std::string mapping_key = MakeHardwareTargetMappingKey(target);
  const auto& map =
      GetHardwareToContainerImage();  // Use the global map function
  auto it = map.find(mapping_key);
  if (it == map.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No container image mapping found for key: ", mapping_key));
  }
  return it->second;
}

// --- JSON Generation Helpers ---

Json::Value JsonStringArrayFromProto(
    const tsl::protobuf::RepeatedPtrField<std::string>& proto_array) {
  Json::Value json_array(Json::arrayValue);
  for (const auto& item : proto_array) {
    json_array.append(item);
  }
  return json_array;
}

// Creates the JSON array of target metric strings.
// It prioritizes metrics defined *directly* in the HardwareTarget proto, over
// the default map defined in GetHardwareToDefaultTargetMetrics().
Json::Value JsonTargetMetricsArrayFromProto(const HardwareTarget& target) {
  Json::Value metrics_json(Json::arrayValue);

  if (target.target_metrics_size() > 0) {
    // Use metrics defined in the proto
    for (const auto& metric_enum_val : target.target_metrics()) {
      TargetMetric metric_enum = static_cast<TargetMetric>(metric_enum_val);
      metrics_json.append(TargetMetricToString(metric_enum));
    }
  } else {
    LOG(INFO)
        << "No target_metrics specified in the proto for hardware target '"
        << HardwareCategoryToString(target.hardware_category())
        << "' with topology (" << target.topology().num_hosts() << "h/"
        << target.topology().num_devices_per_host() << "d). "
        << "Benchmark runner uses the default metrics.";
    const auto& map = GetHardwareToDefaultTargetMetrics();
    std::string hw_key = HardwareCategoryToString(target.hardware_category());
    auto it = map.find(hw_key);
    if (it != map.end()) {
      for (const auto& metric_enum : it->second) {
        metrics_json.append(TargetMetricToString(metric_enum));
      }
    } else {
      LOG(WARNING) << "No target metrics found for hardware category '"
                   << hw_key << "'.";
    }
  }
  return metrics_json;
}

// --- Matrix Entry Creation ---

// Struct to hold artifact details extracted from InputArtifact message
struct ArtifactInfo {
  std::string location;
  bool is_gcs_path = false;
  InputFormat format = InputFormat::INPUT_FORMAT_UNSPECIFIED;
};

// Extracts artifact location and format info from the InputArtifact message.
absl::StatusOr<ArtifactInfo> GetArtifactInfo(const BenchmarkConfig& benchmark) {
  if (!benchmark.has_input_artifact()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Benchmark '", benchmark.name(),
                     "' is missing required field 'input_artifact'."));
  }
  const InputArtifact& artifact = benchmark.input_artifact();
  ArtifactInfo info;
  info.format = artifact.input_format();

  // Validate format.
  if (info.format == InputFormat::INPUT_FORMAT_UNSPECIFIED) {
    LOG(WARNING)
        << "Benchmark '" << benchmark.name()
        << "' has INPUT_FORMAT_UNSPECIFIED for input_artifact.input_format.";
    return absl::InvalidArgumentError(absl::StrCat(
        "Benchmark '", benchmark.name(), "' has unspecified input_format."));
  }

  if (!artifact.artifact_path().empty()) {
    info.location = artifact.artifact_path();
    info.is_gcs_path = false;
  } else if (!artifact.artifact_gcs_bucket_path().empty()) {
    info.location = artifact.artifact_gcs_bucket_path();
    info.is_gcs_path = true;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Benchmark '", benchmark.name(),
        "' input_artifact requires artifact_path or artifact_gcs_bucket_path"));
  }
  return info;
}

// Builds a single JSON object representing one row in the GHA `include` input
// matrix to trigger a benchmark run.
absl::StatusOr<Json::Value> BuildGitHubActionMatrixEntry(
    const BenchmarkConfig& benchmark, const HardwareTarget& target,
    RunFrequency frequency) {
  // 1. Get Runner and Image using the specific target info
  absl::StatusOr<std::string> runner_label = GetRunnerLabelForTarget(target);
  if (!runner_label.ok()) {
    return absl::Status(runner_label.status().code(),
                        absl::StrCat("Benchmark '", benchmark.name(),
                                     "', Hardware Target Key '",
                                     MakeHardwareTargetMappingKey(target),
                                     "': ", runner_label.status().message()));
  }
  absl::StatusOr<std::string> container_image =
      GetContainerImageForTarget(target);
  if (!container_image.ok()) {
    return absl::Status(
        container_image.status().code(),
        absl::StrCat("Benchmark '", benchmark.name(),
                     "', Hardware Target Key '",
                     MakeHardwareTargetMappingKey(target),
                     "': ", container_image.status().message()));
  }
  // 2. Get Artifact Info (Location and Format)
  absl::StatusOr<ArtifactInfo> artifact_info = GetArtifactInfo(benchmark);
  if (!artifact_info.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Benchmark '", benchmark.name(),
        "' has invalid input_artifact: ", artifact_info.status().message()));
  }

  // 3. Build JSON Entry
  Json::Value entry(Json::objectValue);
  entry["benchmark_name"] = benchmark.name();
  // Use config_id if it exists in the final proto, otherwise rely on name +
  // target + freq
  if (!benchmark.config_id().empty()) {  // Check if field exists and is set
    entry["config_id"] = benchmark.config_id();
  }
  entry["run_frequency"] = RunFrequency_Name(frequency);
  entry["runner_label"] = *runner_label;
  entry["container_image"] = *container_image;
  entry["artifact_location"] =
      artifact_info->location;  // Use extracted location
  entry["is_gcs_artifact"] = artifact_info->is_gcs_path;  // Use extracted flag
  entry["input_format"] =
      InputFormat_Name(artifact_info->format);  // Use extracted format

  entry["target_metrics"] = JsonTargetMetricsArrayFromProto(target);

  entry["xla_compilation_flags"] =
      JsonStringArrayFromProto(benchmark.xla_compilation_flags());
  entry["runtime_flags"] = JsonStringArrayFromProto(benchmark.runtime_flags());
  entry["hardware_category"] =
      HardwareCategory_Name(target.hardware_category());

  Json::Value topo_json(Json::objectValue);
  topo_json["num_hosts"] = target.topology().num_hosts();
  topo_json["num_devices_per_host"] = target.topology().num_devices_per_host();
  topo_json["multi_host"] = target.topology().multi_host();
  topo_json["multi_device"] = target.topology().multi_device();
  entry["topology"] = std::move(topo_json);

  entry["github_labels"] = JsonStringArrayFromProto(benchmark.github_labels());

  return entry;
}

}  // namespace

// --- Public API Function Implementations ---

absl::StatusOr<BenchmarkSuite> LoadBenchmarkSuiteFromFile(
    const std::string& registry_path) {
  BenchmarkSuite suite;
  std::string content;
  absl::Status read_status =
      tsl::ReadFileToString(tsl::Env::Default(), registry_path, &content);
  if (!read_status.ok()) {
    return absl::Status(
        static_cast<absl::StatusCode>(read_status.code()),
        absl::StrCat("Failed to read registry file: ", registry_path, " - ",
                     read_status.message()));
  }

  if (!tsl::protobuf::TextFormat::ParseFromString(content, &suite)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Error parsing TextProto registry file: ", registry_path));
  }
  return suite;
}

absl::StatusOr<Json::Value> BuildGitHubActionsMatrix(
    const BenchmarkSuite& suite) {
  Json::Value github_actions_matrix(Json::objectValue);
  Json::Value matrix_entries(Json::arrayValue);

  for (const auto& benchmark : suite.benchmarks()) {
    if (benchmark.name().empty()) {
      LOG(WARNING)
          << "Skipping BenchmarkConfig entry because 'name' field is missing.";
      continue;
    }
    // Build a matrix entry for each hardware target and run frequency.
    for (const auto& target : benchmark.hardware_targets()) {
      for (const auto& freq_enum_val : benchmark.run_frequencies()) {
        RunFrequency frequency = static_cast<RunFrequency>(freq_enum_val);

        absl::StatusOr<Json::Value> matrix_entry =
            BuildGitHubActionMatrixEntry(benchmark, target, frequency);

        if (matrix_entry.ok()) {
          matrix_entries.append(std::move(*matrix_entry));
        } else {
          LOG(WARNING) << "Skipping matrix entry generation. Reason: "
                       << matrix_entry.status().message();
        }
      }
    }
  }
  github_actions_matrix["include"] = matrix_entries;
  return github_actions_matrix;
}

absl::StatusOr<std::string> FindRegistryFile(
    const std::string& registry_path_or_name) {
  tsl::Env* env = tsl::Env::Default();
  const std::string& path_str = registry_path_or_name;

  if (tsl::io::IsAbsolutePath(path_str) && env->FileExists(path_str).ok()) {
    VLOG(1) << "Absolute path exists.";
    return path_str;
  }
  if (!env->FileExists(path_str).ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Registry path specified but not found: ", path_str));
  }

  VLOG(1) << "Registry file path '" << path_str
          << "' is relative. Attempting resolution...";
  const char* build_workspace_dir_cstr =
      std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (build_workspace_dir_cstr != nullptr &&
      build_workspace_dir_cstr[0] != '\0') {
    std::string workspace_path =
        tsl::io::JoinPath(build_workspace_dir_cstr, path_str);
    VLOG(1) << "Checking workspace path: " << workspace_path;
    if (env->FileExists(workspace_path).ok()) {
      VLOG(1) << "Found registry file in workspace: " << workspace_path;
      char* full_path_cstr = realpath(workspace_path.c_str(), nullptr);
      if (full_path_cstr) {
        std::string full_path(full_path_cstr);
        free(full_path_cstr);
        return full_path;
      }
      LOG(WARNING) << "Could not resolve workspace path '" << workspace_path
                   << "' to absolute path.";
      return workspace_path;
    }
    VLOG(1) << "Registry file not found relative to workspace.";
  } else {
    VLOG(1) << "BUILD_WORKSPACE_DIRECTORY not set or empty, skipping workspace "
               "check.";
  }

  VLOG(1) << "Checking relative to current directory: " << path_str;
  if (env->FileExists(path_str).ok()) {
    VLOG(1) << "Found registry file relative to current directory: "
            << path_str;
    char* full_path_cstr = realpath(path_str.c_str(), nullptr);
    if (full_path_cstr) {
      std::string full_path(full_path_cstr);
      free(full_path_cstr);
      VLOG(1) << "Resolved CWD path to absolute: " << full_path;
      return full_path;
    }
    LOG(WARNING) << "Could not get absolute path for CWD match of " << path_str
                 << ", returning relative path.";
    return path_str;
  }
  VLOG(1) << "Registry file not found relative to current directory.";

  return absl::FailedPreconditionError(
      absl::StrCat("Registry file '", path_str,
                   "' not found. Tried absolute, relative to "
                   "workspace (if set), and relative to CWD."));
}

}  // namespace benchmarks
}  // namespace tools
}  // namespace xla

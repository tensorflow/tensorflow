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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
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

// Converts HardwareCategory enum to a short, lowercase string
// (e.g., "gpu_l4" -> "l4")
std::string HardwareCategoryToShortString(HardwareCategory category) {
  std::string str = HardwareCategory_Name(category);
  absl::AsciiStrToLower(&str);
  // Remove common prefixes
  if (str.rfind("gpu_", 0) == 0) {
    return str.substr(4);
  }
  if (str.rfind("cpu_", 0) == 0) {
    return str.substr(4);
  }
  if (str.rfind("hardware_category_", 0) == 0) {
    return str.substr(18);
  }
  return str;  // Return as is if no known prefix
}

// Converts ExecutionTopology to a short string (e.g., "1h1d")
std::string TopologyToShortString(const ExecutionTopology& topology) {
  return absl::StrCat(topology.num_hosts(), "h",
                      topology.num_devices_per_host(), "d");
}

// Converts enums to their string names, returning "UNKNOWN_..." on failure.
std::string HardwareCategoryToString(HardwareCategory category) {
  const std::string name = HardwareCategory_Name(category);
  return name.empty() ? "UNKNOWN_HARDWARE" : name;
}

std::string TargetMetricToString(TargetMetric metric) {
  const std::string name = TargetMetric_Name(metric);
  return name.empty() ? "UNKNOWN_METRIC" : name;
}

std::string WorkflowTypeToString(WorkflowType workflow_type) {
  const std::string name = WorkflowType_Name(workflow_type);
  return name.empty() ? "UNKNOWN_WORKFLOW_TYPE" : name;
}

std::string InputFormatToString(InputFormat format) {
  const std::string name = InputFormat_Name(format);
  return name.empty() ? "UNKNOWN_INPUT_FORMAT" : name;
}

// --- Mapping Definitions ---
// Key for maps: HardwareCategory as string (e.g., "GPU_L4", "CPU_X86")
// These maps might need to become more complex if topology also affects
// runner/image choice.
const absl::flat_hash_map<std::string, std::string>&
GetHardwareToRunnerLabelMap() {
  // Currently, we support the following runners:
  // - CPU_X86: linux-x86-n2-128
  // - GPU_L4: linux-x86-g2-16-l4-1gpu
  // - GPU_B200: linux-x86-a4-224-b200-1gpu
  static const auto* kMap = new absl::flat_hash_map<std::string, std::string>{
      {"CPU_X86", "linux-x86-n2-128"},
      {"GPU_L4", "linux-x86-g2-16-l4-1gpu"},
      {"GPU_B200", "linux-x86-a4-224-b200-1gpu"},
      // Add more mappings
  };
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>&
GetHardwareToContainerImage() {
  static const auto* kHardwareToContainerImage =
      new absl::flat_hash_map<std::string, std::string>{
          {"CPU_X86",
           "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/"
           "ml-build:latest"},
          {"CPU_ARM64",
           "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/"
           "ml-build-arm64:latest"},
          {"GPU_L4",
           "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
          {"GPU_B200",
           "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
          {"GPU_L4_1H_4D",
           "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/"
           "ml-build-cuda12.8-cudnn9.8:latest"},
      };
  return *kHardwareToContainerImage;
}

// --- JSON Generation Helpers ---
Json::Value RepeatedStringFieldToJsonArray(
    const tsl::protobuf::RepeatedPtrField<std::string>& field) {
  Json::Value array(Json::arrayValue);
  for (const auto& item : field) {
    array.append(item);
  }
  return array;
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

// Creates the JSON array of target metric strings.
// It prioritizes metrics defined *directly* in the HardwareTarget proto, over
// the default map defined in GetHardwareToDefaultTargetMetrics().
Json::Value JsonTargetMetricsArrayFromProto(
    const HardwareExecutionConfig& hw_exec_config) {
  Json::Value metrics_json(Json::arrayValue);

  if (hw_exec_config.target_metrics_size() > 0) {
    // Use metrics defined in the proto
    for (const auto& metric_enum_val : hw_exec_config.target_metrics()) {
      TargetMetric metric_enum = static_cast<TargetMetric>(metric_enum_val);
      metrics_json.append(TargetMetricToString(metric_enum));
    }
  } else {
    LOG(INFO)
        << "No target_metrics specified in the proto for hardware target '"
        << HardwareCategoryToString(hw_exec_config.hardware_category())
        << "' with topology (" << hw_exec_config.topology().num_hosts() << "h/"
        << hw_exec_config.topology().num_devices_per_host() << "d). "
        << "Benchmark runner uses the default metrics.";
    const auto& map = GetHardwareToDefaultTargetMetrics();
    std::string hw_key =
        HardwareCategoryToString(hw_exec_config.hardware_category());
    const auto it = map.find(hw_key);
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
struct ArtifactDetails {
  std::string location;
  bool is_gcs = false;
  InputFormat format = INPUT_FORMAT_UNSPECIFIED;
};

absl::StatusOr<ArtifactDetails> GetArtifactDetails(
    const InputArtifact& input_artifact) {
  ArtifactDetails details;
  details.format = input_artifact.input_format();
  if (details.format == INPUT_FORMAT_UNSPECIFIED) {
    return absl::InvalidArgumentError("Input artifact has unspecified format.");
  }

  if (!input_artifact.artifact_gcs_bucket_path().empty()) {
    details.location = input_artifact.artifact_gcs_bucket_path();
    details.is_gcs = true;
  } else if (!input_artifact.artifact_path().empty()) {
    details.location = input_artifact.artifact_path();
    details.is_gcs = false;
  } else {
    return absl::InvalidArgumentError(
        "Input artifact must specify either 'artifact_gcs_bucket_path' or "
        "'artifact_path'.");
  }
  return details;
}

// Builds a single JSON object for the GHA `include` matrix.
absl::StatusOr<Json::Value> BuildMatrixEntry(
    const BenchmarkConfig& benchmark_def,
    const HardwareExecutionConfig& hw_exec_config,
    WorkflowType current_workflow_type_enum) {
  Json::Value entry(Json::objectValue);

  // 1. Basic benchmark info from parent
  entry["benchmark_name"] = benchmark_def.name();
  entry["description"] = benchmark_def.description();
  entry["owner"] = benchmark_def.owner();
  entry["model_source_info"] =
      RepeatedStringFieldToJsonArray(benchmark_def.model_source_info());
  entry["github_labels"] =
      RepeatedStringFieldToJsonArray(benchmark_def.github_labels());

  // 2. Artifact info from parent
  if (!benchmark_def.has_input_artifact()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Benchmark '", benchmark_def.name(), "' is missing 'input_artifact'."));
  }
  absl::StatusOr<ArtifactDetails> artifact_details =
      GetArtifactDetails(benchmark_def.input_artifact());

  entry["artifact_location"] = artifact_details->location;
  entry["is_gcs_artifact"] = artifact_details->is_gcs;
  entry["input_format"] = InputFormatToString(artifact_details->format);

  // 3. Info from the specific HardwareExecutionConfig
  std::string hw_category_str =
      HardwareCategoryToString(hw_exec_config.hardware_category());
  entry["hardware_category"] = hw_category_str;

  Json::Value topology_json(Json::objectValue);
  const auto& topology_proto = hw_exec_config.topology();
  topology_json["num_hosts"] = topology_proto.num_hosts();
  topology_json["num_devices_per_host"] = topology_proto.num_devices_per_host();
  topology_json["multi_host"] = topology_proto.multi_host();
  topology_json["multi_device"] = topology_proto.multi_device();
  entry["topology"] = topology_json;

  entry["target_metrics"] = JsonTargetMetricsArrayFromProto(hw_exec_config);
  entry["runtime_flags"] =
      RepeatedStringFieldToJsonArray(hw_exec_config.runtime_flags());

  // XLA flags: use specific if defined, else parent, else empty
  if (hw_exec_config.xla_compilation_flags_size() > 0) {
    entry["xla_compilation_flags"] =
        RepeatedStringFieldToJsonArray(hw_exec_config.xla_compilation_flags());
  } else {
    entry["xla_compilation_flags"] = Json::Value(Json::arrayValue);
  }

  // 4. Mapped runner info
  const auto& runner_map = GetHardwareToRunnerLabelMap();
  const auto runner_it = runner_map.find(hw_category_str);
  if (runner_it == runner_map.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No GHA runner label mapping for hardware_category: ", hw_category_str,
        " in benchmark '", benchmark_def.name(), "'"));
  }
  entry["runner_label"] = runner_it->second;

  const auto& container_map = GetHardwareToContainerImage();
  const auto container_it = container_map.find(hw_category_str);
  if (container_it == container_map.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No container image mapping for hardware_category: ", hw_category_str,
        " in benchmark '", benchmark_def.name(), "'"));
  }
  entry["container_image"] = container_it->second;

  // 5. Generate the comprehensive config_id for this specific matrix entry
  // This config_id will be used as the key in the baseline file.
  std::string workflow_type_short_str =
      absl::AsciiStrToLower(WorkflowType_Name(current_workflow_type_enum));
  //  Remove "run_frequency_" prefix if it exists for brevity
  if (workflow_type_short_str.rfind("run_frequency_", 0) == 0) {
    workflow_type_short_str = workflow_type_short_str.substr(14);
  }

  // Example config_id: "gemma3_1b_flax_call_l4_1h1d_presubmit"
  entry["config_id"] = absl::StrCat(
      benchmark_def.name(), "_",
      HardwareCategoryToShortString(hw_exec_config.hardware_category()), "_",
      TopologyToShortString(hw_exec_config.topology()), "_",
      workflow_type_short_str  // Add workflow type to make it unique if
                               // baselines differ by workflow
  );
  // Also add the specific workflow type this matrix entry is for
  entry["workflow_type"] = WorkflowType_Name(current_workflow_type_enum);

  return entry;
}

bool ShouldIncludeHwExecConfig(const HardwareExecutionConfig& hw_exec_config,
                               WorkflowType target_workflow_type) {
  for (int wf_type_int : hw_exec_config.workflow_type()) {
    if (static_cast<WorkflowType>(wf_type_int) == target_workflow_type) {
      return true;
    }
  }
  return false;
}

// FindRegistryFile implementation (remains the same as your provided version)

absl::StatusOr<std::string> ResolvePath(const std::string& path_str,
                                        tsl::Env* env) {
  char* real_path_cstr = realpath(path_str.c_str(), nullptr);
  if (real_path_cstr) {
    std::string resolved_path = real_path_cstr;
    free(real_path_cstr);
    VLOG(1) << "Path '" << path_str << "' resolved to '" << resolved_path
            << "'. Checking existence...";
    if (env->FileExists(resolved_path).ok()) {
      VLOG(1) << "Found registry file (resolved): " << resolved_path;
      return resolved_path;
    }
  } else {
    // realpath can fail if intermediate dirs don't exist, but file might still
    // be accessible
    if (env->FileExists(path_str).ok()) {
      VLOG(1) << "Found registry file (unresolved, realpath failed but file "
                 "exists): "
              << path_str;
      return path_str;
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "Registry file not found at resolved/input path: ", path_str));
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
    return absl::FailedPreconditionError(
        absl::StrCat("Error reading registry file: ", registry_path));
  }
  // Using TextFormat::ParseFromString for TextProto
  if (!tsl::protobuf::TextFormat::ParseFromString(content, &suite)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Error parsing TextProto registry file: ", registry_path));
  }

  // Basic validation: each benchmark must have a name.
  // config_id is not checked here as it will be generated per matrix entry.
  for (const auto& benchmark_def : suite.benchmarks()) {
    if (benchmark_def.name().empty()) {
      return absl::InvalidArgumentError(
          "Benchmark definition found with an empty 'name'. 'name' is "
          "required.");
    }
    if (benchmark_def.hardware_execution_configs_size() == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Benchmark '", benchmark_def.name(),
                       "' has no 'hardware_execution_configs'. At least one is "
                       "required."));
    }
  }
  return suite;
}

absl::StatusOr<Json::Value> BuildGitHubActionsMatrix(
    const BenchmarkSuite& suite, WorkflowType current_workflow_type) {
  Json::Value matrix_entries(Json::arrayValue);

  for (const auto& benchmark_def : suite.benchmarks()) {
    // Basic validation after loading
    if (benchmark_def.name().empty() ||
        benchmark_def.hardware_execution_configs_size() == 0) {
      LOG(WARNING)
          << "Skipping benchmark definition (name: '" << benchmark_def.name()
          << "') due to missing name or no hardware_execution_configs.";
      continue;
    }

    for (const auto& hw_exec_config :
         benchmark_def.hardware_execution_configs()) {
      if (!ShouldIncludeHwExecConfig(hw_exec_config, current_workflow_type)) {
        VLOG(2) << "Skipping benchmark '" << benchmark_def.name()
                << "' with hardware '"
                << HardwareCategoryToString(hw_exec_config.hardware_category())
                << "' because its workflow_type list does not include '"
                << WorkflowTypeToString(current_workflow_type) << "'.";
        continue;
      }

      absl::StatusOr<Json::Value> entry_status = BuildMatrixEntry(
          benchmark_def, hw_exec_config, current_workflow_type);

      if (entry_status.ok()) {
        matrix_entries.append(*entry_status);
      } else {
        LOG(ERROR) << "Failed to create matrix entry for benchmark '"
                   << benchmark_def.name() << "' with hardware '"
                   << HardwareCategoryToString(
                          hw_exec_config.hardware_category())
                   << "': " << entry_status.status();
        // Optionally, continue to generate other valid entries instead of
        // returning error
      }
    }
  }
  return matrix_entries;
}

absl::StatusOr<std::string> FindRegistryFile(
    const std::string& registry_path_or_name) {
  tsl::Env* env = tsl::Env::Default();
  const std::string& path_str = registry_path_or_name;

  if (path_str.empty()) {
    return absl::NotFoundError("Registry file path cannot be empty.");
  }

  // 1. Try as absolute path
  if (tsl::io::IsAbsolutePath(path_str)) {
    VLOG(1) << "Path '" << path_str << "' is absolute. Checking existence...";
    if (env->FileExists(path_str).ok()) {
      VLOG(1) << "Found absolute path: " << path_str;
      return path_str;
    }
    VLOG(1) << "Absolute path not found: " << path_str;
    // Do not return error yet, try other methods if user provided, e.g. a path
    // relative to workspace that happens to start with / on some systems but
    // isn't a true FS root absolute path.
  }

  // 2. Try relative to current working directory
  std::string current_dir_path = std::string(path_str);
  if (!tsl::io::IsAbsolutePath(
          current_dir_path)) {  // If not already absolute from above check
    char* cwd_cstr = getcwd(nullptr, 0);
    if (cwd_cstr) {
      std::string cwd(cwd_cstr);
      free(cwd_cstr);
      current_dir_path = tsl::io::JoinPath(cwd, path_str);
      VLOG(1) << "Path '" << path_str
              << "' is relative. Checking relative to CWD (" << cwd
              << "): " << current_dir_path;
    } else {
      VLOG(1) << "Could not get CWD. Proceeding with path_str as is for "
                 "relative check.";
    }
  }
  // Use ResolvePath for the potentially CWD-ified path
  absl::StatusOr<std::string> resolved_current_path =
      ResolvePath(current_dir_path, env);
  if (resolved_current_path.ok()) {
    return *resolved_current_path;
  }
  VLOG(1) << "Registry file not found relative to CWD (or as absolute): "
          << current_dir_path;

  // 3. Try relative to BUILD_WORKSPACE_DIRECTORY
  VLOG(1) << "Attempting workspace resolution for '" << path_str << "'.";
  const char* build_workspace_dir_cstr =
      std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (build_workspace_dir_cstr != nullptr &&
      build_workspace_dir_cstr[0] != '\0') {
    std::string workspace_dir(build_workspace_dir_cstr);
    std::string workspace_path = tsl::io::JoinPath(workspace_dir, path_str);
    VLOG(1) << "Checking workspace path: " << workspace_path;

    absl::StatusOr<std::string> workspace_resolved_path =
        ResolvePath(workspace_path, env);
    if (workspace_resolved_path.ok()) {
      return *workspace_resolved_path;
    }
    VLOG(1) << "Registry file not found relative to workspace: "
            << workspace_path;
  } else {
    VLOG(1) << "BUILD_WORKSPACE_DIRECTORY not set or empty, skipping workspace "
               "check.";
  }

  return absl::NotFoundError(absl::StrCat(
      "Registry file '", registry_path_or_name,
      "' not found. Tried absolute, relative to CWD (as '", current_dir_path,
      "'), and relative to "
      "BUILD_WORKSPACE_DIRECTORY (if set). Last error for CWD path: ",
      resolved_current_path.status().ToString()));
}

}  // namespace benchmarks
}  // namespace tools
}  // namespace xla

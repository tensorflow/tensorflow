// Copyright 2025 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"
#include "xla/tools/benchmarks/utils/generate_benchmark_matrices.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace {
constexpr char kUsageText[] = R"(
  Usage: bazel run //third_party/tensorflow/compiler/xla/tools/benchmarks/utils:generate_benchmark_matrices_main -- --registry_file=/path/to/registry.yml --workflow_type=PRESUBMIT

  Example output (an array of matrix entries):
  [
    {
      "benchmark_name": "gemma3_1b_flax_call",
      "config_id": "gemma3_1b_flax_call_l4_1h1d_presubmit",
      "container_image": "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest",
      "description": "Benchmarks Gemma3 1b in Flax",
      "github_labels": [ "blocking_presubmit_test" ],
      "hardware_category": "GPU_L4",
      "input_format": "HLO_TEXT",
      "is_gcs_artifact": true,
      "artifact_location": "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo",
      "model_source_info": [ "Gemma3 1B" ],
      "owner": "juliagmt-google@",
      "runner_label": "linux-x86-g2-16-l4-1gpu",
      "runtime_flags": [ "--num_repeats=5" ],
      "target_metrics": [ "GPU_DEVICE_TIME", "GPU_DEVICE_MEMCPY_TIME" ],
      "topology": {
        "devices_per_host": 1, // Corrected key
        "multi_device": false,
        "multi_host": false,
        "num_hosts": 1
      },
      "workflow_type": "PRESUBMIT", // The workflow type this entry is for
      "xla_compilation_flags": [ "--xla_gpu_enable_cudnn_fusion=true", "--xla_gpu_enable_latency_hiding_scheduler=true" ]
    },...
  ]
  )";

// Parses the string argument for workflow_type into the WorkflowType enum.
// Accepts both lowercase (e.g., "presubmit") and uppercase (e.g., "PRESUBMIT")
// for convenience, as GitHub Actions inputs are often lowercase.
absl::StatusOr<xla::WorkflowType> GetWorkflowTypeFromStr(
    std::string workflow_type_arg_str) {
  // Convert to uppercase for matching with enum names
  std::transform(workflow_type_arg_str.begin(), workflow_type_arg_str.end(),
                 workflow_type_arg_str.begin(), ::toupper);

  static const auto* const kWorkflowAliasMap =
      new absl::flat_hash_map<std::string, xla::WorkflowType>{
          {"NIGHTLY", xla::WorkflowType::SCHEDULED},
          {"PRESUBMIT", xla::WorkflowType::PRESUBMIT},
          {"POSTSUBMIT", xla::WorkflowType::POSTSUBMIT},
          {"SCHEDULED", xla::WorkflowType::SCHEDULED},
          {"MANUAL", xla::WorkflowType::MANUAL},
          // Add other aliases if needed
      };
  auto it = kWorkflowAliasMap->find(workflow_type_arg_str);
  if (it != kWorkflowAliasMap->end()) {
    return it->second;
  }

  // If no match was found, return an error.
  return absl::InvalidArgumentError(absl::StrCat(
      "Invalid --workflow_type: ", workflow_type_arg_str,
      ". Valid options are PRESUBMIT, POSTSUBMIT, SCHEDULED, MANUAL, or their "
      "aliases."));
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string registry_file_path_arg;
  // Input string values like "PRESUBMIT", "POSTSUBMIT", "SCHEDULED", "MANUAL"
  // or lowercase versions.
  std::string workflow_type_arg_str;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("registry_file", &registry_file_path_arg,
                "Path to the benchmark registry file."),
      tsl::Flag("workflow_type", &workflow_type_arg_str,
                "Current workflow type (e.g., PRESUBMIT, POSTSUBMIT, "
                "SCHEDULED, MANUAL). Used to filter benchmarks.")};
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << "Failed to parse command-line flags using tsl::Flags::Parse";
  }
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (registry_file_path_arg.empty()) {
    std::string kUsageString =
        absl::StrCat(kUsageText, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
    LOG(QFATAL) << "Required flag --registry_file is missing.\n"
                << kUsageString;
  }
  if (workflow_type_arg_str.empty()) {
    std::string kUsageString =
        absl::StrCat(kUsageText, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
    LOG(QFATAL) << "Required flag --workflow_type is missing.\n"
                << kUsageString;
  }

  // Resolve registry file path.
  absl::StatusOr<std::string> resolved_registry_path =
      xla::tools::benchmarks::FindRegistryFile(registry_file_path_arg);
  TF_QCHECK_OK(resolved_registry_path.status())
      << "Failed to find or access registry file: " << registry_file_path_arg
      << "; Error: " << resolved_registry_path.status();

  absl::StatusOr<xla::WorkflowType> current_workflow_type_status =
      GetWorkflowTypeFromStr(workflow_type_arg_str);
  TF_QCHECK_OK(current_workflow_type_status.status());
  xla::WorkflowType current_workflow_type = *current_workflow_type_status;

  LOG(INFO) << "Loading benchmark suite from: " << *resolved_registry_path;
  LOG(INFO) << "Filtering for workflow type: "
            << xla::WorkflowType_Name(current_workflow_type);

  absl::StatusOr<xla::tools::benchmarks::BenchmarkSuite> suite_status =
      xla::tools::benchmarks::LoadBenchmarkSuiteFromFile(
          *resolved_registry_path);
  TF_QCHECK_OK(suite_status.status())
      << "Failed to load benchmark suite: " << suite_status.status();
  xla::tools::benchmarks::BenchmarkSuite suite = *suite_status;

  absl::StatusOr<Json::Value> matrix_output =
      xla::tools::benchmarks::BuildGitHubActionsMatrix(suite,
                                                       current_workflow_type);
  TF_QCHECK_OK(matrix_output.status())
      << "Failed to build GitHub Actions matrix: " << matrix_output.status();

  Json::StreamWriterBuilder writer_builder;
  writer_builder["indentation"] = "  ";
  std::string output_string = Json::writeString(writer_builder, *matrix_output);

  std::cout << output_string << std::endl;
  return 0;
}

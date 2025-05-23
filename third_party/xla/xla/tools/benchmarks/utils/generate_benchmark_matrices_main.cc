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
Usage: bazel run //third_party/tensorflow/compiler/xla/tools/benchmarks/utils:generate_benchmark_matrices_main -- --registry_file=/path/to/registry.yml --workflow_type=presubmit

Example output:
{
  "benchmarks": [
    {
      "artifact_location" : "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo",
      "benchmark_name" : "gemma3_1b_flax_call",
      "container_image" : "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-cuda12.8-cudnn9.8:latest",
      "github_labels" :
      [
        "blocking_presubmit_test"
      ],
      "hardware_category" : "GPU_B200",
      "input_format" : "HLO_TEXT",
      "is_gcs_artifact" : true,
      "run_frequency" : "PRESUBMIT",
      "runner_label" : "linux-x86-a4-224-b200-1gpu",
      "runtime_flags" :
      [
        "--num_repeat=5"
      ],
      "target_metrics" :
      [
        "WALL_TIME",
        "GPU_DEVICE_TIME",
        "GPU_DEVICE_MEMCPY_TIME",
        "PEAK_GPU_MEMORY"
      ],
      "topology" :
      {
        "multi_device" : false,
        "multi_host" : false,
        "num_devices_per_host" : 1,
        "num_hosts" : 1
      },
      "xla_compilation_flags" : []
    },
)";

absl::StatusOr<xla::RunFrequency> GetRunFrequencyFromStr(
    const std::string& workflow_type_arg_str) {
  static const auto* const workflow_type_to_run_frequency =
      new absl::flat_hash_map<std::string, xla::RunFrequency>{
          {"presubmit", xla::RunFrequency::PRESUBMIT},
          {"postsubmit", xla::RunFrequency::POSTSUBMIT},
          {"nightly", xla::RunFrequency::SCHEDULED},
          {"scheduled", xla::RunFrequency::SCHEDULED},
          {"manual", xla::RunFrequency::MANUAL},
      };
  auto it = workflow_type_to_run_frequency->find(workflow_type_arg_str);
  if (it == workflow_type_to_run_frequency->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid --workflow_type: ", workflow_type_arg_str));
  }
  return it->second;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string registry_file_path_arg;
  // "presubmit", "postsubmit", "nightly", "manual"
  std::string workflow_type_arg_str;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("registry_file", &registry_file_path_arg,
                "Path to the benchmark registry file."),
      tsl::Flag("workflow_type", &workflow_type_arg_str,
                "Current workflow type (e.g., presubmit, postsubmit, nightly, "
                "manual). Used to filter benchmarks.")};
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    // Note: tsl::Flags::Parse usually prints usage on error,
    // but adding a QFATAL provides explicit failure logging.
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
    LOG(QFATAL) << "Required flag --workflow_type is missing.";
  }

  // Convert workflow_type_arg_str to RunFrequency enum.
  absl::StatusOr<xla::RunFrequency> current_run_frequency_status =
      GetRunFrequencyFromStr(workflow_type_arg_str);
  TF_QCHECK_OK(current_run_frequency_status.status());
  xla::RunFrequency current_run_frequency = *current_run_frequency_status;

  absl::StatusOr<xla::tools::benchmarks::BenchmarkSuite> suite_status =
      xla::tools::benchmarks::LoadBenchmarkSuiteFromFile(
          registry_file_path_arg);
  TF_QCHECK_OK(suite_status.status()) << "Failed to load benchmark suite";
  xla::tools::benchmarks::BenchmarkSuite suite = *suite_status;

  absl::StatusOr<Json::Value> matrix_output =
      xla::tools::benchmarks::BuildGitHubActionsMatrix(suite,
                                                       current_run_frequency);
  TF_QCHECK_OK(matrix_output.status())
      << "Failed to build GitHub Actions matrix";

  // Dump JSON matrix to stdout.
  Json::StreamWriterBuilder writer_builder;
  writer_builder["indentation"] = "  ";
  std::string output_string = Json::writeString(writer_builder, *matrix_output);

  std::cout << output_string << std::endl;
  return 0;
}

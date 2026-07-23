/* Copyright 2026 The OpenXLA Authors.

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

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tools/hlo_isolation/hlo_isolation.pb.h"
#include "xla/tools/hlo_isolation/hlo_isolation_api.h"
#include "xla/tools/run_hlo_module.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace hlo_isolation {

constexpr absl::string_view kUsage = R"(
This tool runs an isolation test on an HloModule. It decomposes the module into
submodules (operations) and verifies the correctness of each submodule by
running it on a target platform and comparing results against a reference platform.

Example Usage:
  hlo_isolation_cli --hlo_file=path/to/module.pbtxt \
                    --test_platform=tpu \
                    --reference_platform=interpreter \
                    --filter_by_opcode="multiply" \
                    --skip_by_name="fusion.*"
)";

absl::Status RunMain(
    absl::string_view hlo_path, absl::string_view test_platform_name,
    absl::string_view reference_platform_name, absl::string_view filter_by_name,
    absl::string_view skip_by_name, absl::string_view filter_by_opcode,
    absl::string_view skip_by_opcode, int shard_index, int num_shards,
    double abs_error_bound, double rel_error_bound, bool run_hlo_passes) {
  if (hlo_path.empty()) {
    return absl::InvalidArgumentError("--hlo_file is required");
  }

  // 1. Create Runners
  ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> test_client,
                   GetPjRtClientForPlatform(test_platform_name));
  HloRunner test_runner(std::move(test_client));

  std::unique_ptr<HloRunner> reference_runner_ptr;
  HloRunnerInterface* reference_runner_ptr_raw = nullptr;
  if (!reference_platform_name.empty()) {
    ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> ref_client,
                     GetPjRtClientForPlatform(reference_platform_name));
    reference_runner_ptr = std::make_unique<HloRunner>(std::move(ref_client));
    reference_runner_ptr_raw = reference_runner_ptr.get();
  }

  // 2. Set up Options and Filter
  PipelineIsolationOptions options;
  options.module_options.abs_error_bound = abs_error_bound;
  options.module_options.rel_error_bound = rel_error_bound;
  options.module_options.run_hlo_passes = run_hlo_passes;
  options.shard_index = shard_index;
  options.num_shards = num_shards;
  options.filter_by_name = std::string(filter_by_name);
  options.skip_by_name = std::string(skip_by_name);
  options.filter_by_opcode = std::string(filter_by_opcode);
  options.skip_by_opcode = std::string(skip_by_opcode);

  ASSIGN_OR_RETURN(std::vector<HloIsolationTestResult> results,
                   RunIsolationPipeline(std::string(hlo_path), &test_runner,
                                        reference_runner_ptr_raw, options));

  int run_count = 0;
  int success_count = 0;

  for (const HloIsolationTestResult& result : results) {
    run_count++;
    std::cout << "Submodule: " << result.module_name() << "\n";
    std::cout << "Success: " << (result.state() == SUCCESS ? "YES" : "NO")
              << "\n";
    std::cout << "Reason: " << result.reason() << "\n";

    if (result.state() == SUCCESS) {
      success_count++;
    }
  }

  std::cout << "Total run: " << run_count << ", Success: " << success_count
            << "\n";

  return absl::OkStatus();
}

}  // namespace hlo_isolation
}  // namespace xla

int main(int argc, char** argv) {
  std::string hlo_file;
  int shard_index = -1;
  int num_shards = 1;
  std::string test_platform = "cpu";
  std::string reference_platform = "interpreter";
  std::string filter_by_name = ".*";
  std::string skip_by_name = "";
  std::string filter_by_opcode = ".*";
  std::string skip_by_opcode = "";
  std::string abs_error_bound_str = "0.01";
  std::string rel_error_bound_str = "0.1";
  bool run_hlo_passes = false;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag(
          "hlo_file", &hlo_file,
          "Path to the HLO file to load. Can be text or proto (required)."),
      tsl::Flag("test_platform", &test_platform,
                "Target platform to run the test on (e.g., cpu, gpu, tpu)."),
      tsl::Flag(
          "reference_platform", &reference_platform,
          "Reference platform to run the comparison on (e.g., interpreter). "
          "If empty, reference comparison is disabled."),
      tsl::Flag(
          "filter_by_name", &filter_by_name,
          "Regular expression to match the module name. Only modules matching "
          "this will be run."),
      tsl::Flag(
          "skip_by_name", &skip_by_name,
          "Regular expression to match the module name. Modules matching this "
          "will be skipped."),
      tsl::Flag("filter_by_opcode", &filter_by_opcode,
                "Regular expression to match instruction opcodes. Only modules "
                "containing at least one matching opcode will be run."),
      tsl::Flag("skip_by_opcode", &skip_by_opcode,
                "Regular expression to match instruction opcodes. Modules "
                "containing any matching opcode will be skipped."),
      tsl::Flag("abs_error_bound", &abs_error_bound_str,
                "Absolute error bound for comparison."),
      tsl::Flag("rel_error_bound", &rel_error_bound_str,
                "Relative error bound for comparison."),
      tsl::Flag("run_hlo_passes", &run_hlo_passes,
                "Whether to run HLO passes on the submodules."),
      tsl::Flag("shard_index", &shard_index,
                "The specific shard index to run (zero-based)."),
      tsl::Flag("num_shards", &num_shards, "The total number of shards.")};

  const std::string usage = absl::StrCat(xla::hlo_isolation::kUsage, "\n\n",
                                         tsl::Flags::Usage(argv[0], flag_list));
  const bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << usage << "\n";
    return 1;
  }
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  double abs_error_bound = 0.01;
  double rel_error_bound = 0.1;
  if (!absl::SimpleAtod(abs_error_bound_str, &abs_error_bound)) {
    std::cerr << "Invalid value for --abs_error_bound: " << abs_error_bound_str
              << "\n";
    return 1;
  }
  if (!absl::SimpleAtod(rel_error_bound_str, &rel_error_bound)) {
    std::cerr << "Invalid value for --rel_error_bound: " << rel_error_bound_str
              << "\n";
    return 1;
  }

  absl::Status status = xla::hlo_isolation::RunMain(
      hlo_file, test_platform, reference_platform, filter_by_name, skip_by_name,
      filter_by_opcode, skip_by_opcode, shard_index, num_shards,
      abs_error_bound, rel_error_bound, run_hlo_passes);
  if (!status.ok()) {
    std::cerr << "Error: " << status << "\n";
    return 1;
  }
  return 0;
}

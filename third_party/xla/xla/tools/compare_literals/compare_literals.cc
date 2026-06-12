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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/error_spec.h"
#include "xla/tools/compare_literals/compare_literals_impl.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace {

struct CompareLiteralsConfig {
  std::string file1;
  std::string file2;
  float abs_error_bound = 1e-3;
  float rel_error_bound = 1e-3;
};

// Prints the summary of the diff between the two RunHloModuleLiterals protos
// (i.e. DiffResult).
void PrintDiffSummary(const DiffResult& diff_result) {
  std::cout << "Diff summary:\n";
  for (const IterationDiff& iteration_diff : diff_result.iteration_diffs) {
    if (iteration_diff.has_diff) {
      std::cout << "Iteration " << iteration_diff.iteration_index << "\n";
      if (!iteration_diff.diff_str.empty()) {
        std::cout << "Diff:\n" << iteration_diff.diff_str << "\n";
      }
      if (!iteration_diff.argument_diffs.empty()) {
        std::cout << "Argument diffs:\n"
                  << iteration_diff.argument_diffs << "\n";
      }
    }
  }
}

absl::Status RealMain(const CompareLiteralsConfig& config) {
  if (config.file1.empty() || config.file2.empty()) {
    return absl::InvalidArgumentError(
        "Both --file1 and --file2 must be provided.");
  }

  RunHloModuleLiterals result1, result2;
  RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), config.file1, &result1));
  RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), config.file2, &result2));

  DiffResult diff_result;
  ErrorSpec error_spec(config.abs_error_bound, config.rel_error_bound);
  absl::Status s = CompareResults(result1, result2, error_spec, &diff_result);

  if (!s.ok()) {
    PrintDiffSummary(diff_result);
    return absl::FailedPreconditionError("Literals differ.");
  }

  std::cout << "Literals in the two files are same.\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::CompareLiteralsConfig config;
  std::vector<tsl::Flag> flags = {
      tsl::Flag("file1", &config.file1,
                "Path to the first RunHloModuleLiterals proto file."),
      tsl::Flag("file2", &config.file2,
                "Path to the second RunHloModuleLiterals proto file."),
      tsl::Flag("abs_error_bound", &config.abs_error_bound,
                "Absolute error bound for comparing literals."),
      tsl::Flag("rel_error_bound", &config.rel_error_bound,
                "Relative error bound for comparing literals."),
  };

  std::string usage = tsl::Flags::Usage(argv[0], flags);
  if (!tsl::Flags::Parse(&argc, argv, flags)) {
    std::cerr << usage;
    return EXIT_FAILURE;
  }
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  absl::Status status = xla::RealMain(config);
  if (!status.ok()) {
    std::cout << "Error: " << status << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

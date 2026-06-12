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
#include "absl/flags/flag.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"

ABSL_FLAG(std::string, file1, "",
          "Path to the first RunHloModuleLiterals proto file.");
ABSL_FLAG(std::string, file2, "",
          "Path to the second RunHloModuleLiterals proto file.");
ABSL_FLAG(double, abs_error_bound, 1e-3,
          "Absolute error bound for comparing literals.");
ABSL_FLAG(double, rel_error_bound, 1e-3,
          "Relative error bound for comparing literals.");

namespace xla {
namespace {

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

absl::Status RealMain() {
  std::string file1_path = absl::GetFlag(FLAGS_file1);
  std::string file2_path = absl::GetFlag(FLAGS_file2);

  if (file1_path.empty() || file2_path.empty()) {
    return absl::InvalidArgumentError(
        "Both --file1 and --file2 must be provided.");
  }

  RunHloModuleLiterals result1, result2;
  RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), file1_path, &result1));
  RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), file2_path, &result2));

  DiffResult diff_result;
  ErrorSpec error_spec(absl::GetFlag(FLAGS_abs_error_bound),
                       absl::GetFlag(FLAGS_rel_error_bound));
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
  tsl::port::InitMain("Compare two RunHloModuleLiterals proto files.", &argc,
                      &argv);
  absl::Status status = xla::RealMain();
  if (!status.ok()) {
    std::cout << "Error: " << status << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


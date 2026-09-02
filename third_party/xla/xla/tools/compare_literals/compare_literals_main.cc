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
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "xla/tools/compare_literals/compare_model_literals.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/init_main.h"

ABSL_FLAG(double, abs_error_bound, 1e-3, "Absolute error tolerance bound.");
ABSL_FLAG(double, rel_error_bound, 1e-3, "Relative error tolerance bound.");
ABSL_FLAG(bool, show_histogram, false, "Display 1D relative error histogram.");
ABSL_FLAG(bool, show_heatmap, false, "Display 2D error heatmap.");
ABSL_FLAG(bool, suggest_error_spec, true,
          "Display suggested ErrorSpec in console output.");
ABSL_FLAG(double, heatmap_yellow_pct, 0.5,
          "Failure percentage threshold (0-100) below which heatmap cells are "
          "colored yellow.");
ABSL_FLAG(std::string, output_markdown, "",
          "Path to write Markdown report file.");
ABSL_FLAG(int, max_bar_width, 40, "Maximum character width of histogram bars.");
ABSL_FLAG(bool, color, true, "Use ANSI colors in terminal output.");
ABSL_FLAG(std::string, output_json, "",
          "Optional path to write structured JSON report.");
ABSL_FLAG(std::string, output_tsv, "",
          "Optional path to write flat TSV report.");
ABSL_FLAG(std::string, output_device_tsv, "",
          "Optional path to write detailed per-device TSV report.");
ABSL_FLAG(
    int, threads, 16,
    "Number of parallel worker threads for reading and comparing literals.");

namespace {

using ::xla::compare_literals::CompareLiteralFiles;
using ::xla::compare_literals::CompareModelDirectories;
using ::xla::compare_literals::ComparisonOptions;
using ::xla::compare_literals::ComparisonResult;
using ::xla::compare_literals::ModelComparisonOptions;
using ::xla::compare_literals::ModelComparisonResult;
using ::xla::compare_literals::WriteModelComparisonOutputs;

constexpr int kExitPass = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitError = 2;

int RunFileComparison(tsl::Env* env, absl::string_view clean,
                      absl::string_view dirty) {
  ComparisonOptions options;
  options.abs_error_bound = absl::GetFlag(FLAGS_abs_error_bound);
  options.rel_error_bound = absl::GetFlag(FLAGS_rel_error_bound);
  options.heatmap_yellow_pct = absl::GetFlag(FLAGS_heatmap_yellow_pct);

  absl::StatusOr<ComparisonResult> result_or =
      CompareLiteralFiles(clean, dirty, options);
  if (!result_or.ok()) {
    std::cerr << "Comparison error: " << result_or.status() << "\n";
    return kExitError;
  }

  const ComparisonResult& result = *result_or;

  std::cout << "Comparing:\n";
  std::cout << "  Clean: " << clean << "\n";
  std::cout << "  Dirty: " << dirty << "\n";
  std::cout << "  Bounds: abs = " << options.abs_error_bound
            << ", rel = " << options.rel_error_bound << "\n\n";

  const bool use_color = absl::GetFlag(FLAGS_color);
  std::cout << result.SummaryToString(use_color) << "\n";

  if (result.total_elements > 0) {
    if (absl::GetFlag(FLAGS_show_histogram)) {
      std::cout << result.histogram.ToString(absl::GetFlag(FLAGS_max_bar_width))
                << "\n";
    }
    if (absl::GetFlag(FLAGS_show_heatmap)) {
      std::cout << result.heatmap.ToString(use_color) << "\n";
    }
  }

  const std::string md_path = absl::GetFlag(FLAGS_output_markdown);
  if (!md_path.empty()) {
    std::string md = result.SummaryToMarkdown();
    absl::Status s = tsl::WriteStringToFile(env, md_path, md);
    if (!s.ok()) {
      std::cerr << "Failed to write markdown report to '" << md_path
                << "': " << s << "\n";
      return kExitError;
    }
    std::cout << "Wrote markdown report to: " << md_path << "\n";
  }

  return result.passed ? kExitPass : kExitMismatch;
}

int RunDirectoryComparison(tsl::Env* env, absl::string_view golden_dir,
                           absl::string_view test_dir) {
  ModelComparisonOptions options;
  options.num_threads = absl::GetFlag(FLAGS_threads);
  options.comparison_options.abs_error_bound =
      absl::GetFlag(FLAGS_abs_error_bound);
  options.comparison_options.rel_error_bound =
      absl::GetFlag(FLAGS_rel_error_bound);
  options.comparison_options.heatmap_yellow_pct =
      absl::GetFlag(FLAGS_heatmap_yellow_pct);

  absl::StatusOr<ModelComparisonResult> result_or =
      CompareModelDirectories(golden_dir, test_dir, options);
  if (!result_or.ok()) {
    std::cerr << "CompareModelDirectories failed: " << result_or.status()
              << "\n";
    return kExitError;
  }

  const ModelComparisonResult& result = *result_or;
  std::cout << result.SummaryToString() << "\n";

  const std::string json_path = absl::GetFlag(FLAGS_output_json);
  const std::string tsv_path = absl::GetFlag(FLAGS_output_tsv);
  const std::string device_tsv_path = absl::GetFlag(FLAGS_output_device_tsv);

  absl::Status s =
      WriteModelComparisonOutputs(result, json_path, tsv_path, device_tsv_path);
  if (!s.ok()) {
    std::cerr << "Failed to write outputs: " << s << "\n";
    return kExitError;
  }

  if (!json_path.empty()) {
    std::cout << "Wrote JSON output to: " << json_path << "\n";
  }
  if (!tsv_path.empty()) {
    std::cout << "Wrote TSV output to: " << tsv_path << "\n";
  }
  if (!device_tsv_path.empty()) {
    std::cout << "Wrote Device TSV output to: " << device_tsv_path << "\n";
  }

  bool all_passed =
      (result.missing_in_golden.empty() && result.missing_in_test.empty() &&
       result.summary.differing_literals == 0 &&
       result.summary.failed_device_comparisons == 0);
  return all_passed ? kExitPass : kExitMismatch;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr absl::string_view kUsage =
      "Usage: compare_literals <clean_path> <dirty_path> [flags]";
  tsl::port::InitMain(kUsage.data(), &argc, &argv);
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  if (positional_args.size() != 3) {
    std::cerr << "Error: Exactly two positional paths must be provided.\n";
    std::cerr << "Usage:\n  " << argv[0]
              << " <clean_path> <dirty_path> [flags]\n";
    return kExitError;
  }

  const std::string clean = positional_args[1];
  const std::string dirty = positional_args[2];

  tsl::Env* env = tsl::Env::Default();
  if (!env->FileExists(clean).ok()) {
    std::cerr << "Error: Path does not exist: " << clean << "\n";
    return kExitError;
  }
  if (!env->FileExists(dirty).ok()) {
    std::cerr << "Error: Path does not exist: " << dirty << "\n";
    return kExitError;
  }

  const bool clean_is_dir = env->IsDirectory(clean).ok();
  const bool dirty_is_dir = env->IsDirectory(dirty).ok();

  if (clean_is_dir != dirty_is_dir) {
    std::cerr << "Error: Cannot compare file to directory: clean is "
              << (clean_is_dir ? "directory" : "file") << ", dirty is "
              << (dirty_is_dir ? "directory" : "file") << ".\n";
    return kExitError;
  }

  if (clean_is_dir) {
    return RunDirectoryComparison(env, clean, dirty);
  }

  return RunFileComparison(env, clean, dirty);
}

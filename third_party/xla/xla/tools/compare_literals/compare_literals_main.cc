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

namespace {

using ::xla::compare_literals::CompareLiteralFiles;
using ::xla::compare_literals::ComparisonOptions;
using ::xla::compare_literals::ComparisonResult;

constexpr int kExitPass = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitError = 2;

}  // namespace

int main(int argc, char** argv) {
  constexpr absl::string_view kUsage =
      "Usage: compare_literals <clean_path> <dirty_path> [flags]";
  tsl::port::InitMain(kUsage.data(), &argc, &argv);
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  if (positional_args.size() != 3) {
    std::cerr << "Error: Exactly two positional file paths must be provided.\n";
    std::cerr << "Usage:\n  " << argv[0]
              << " <clean_path> <dirty_path> [flags]\n";
    return kExitError;
  }

  std::string clean = positional_args[1];
  std::string dirty = positional_args[2];

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

  bool use_color = absl::GetFlag(FLAGS_color);

  std::cout << result.SummaryToString(use_color) << "\n";

  if (result.total_elements > 0) {
    if (absl::GetFlag(FLAGS_show_histogram)) {
      std::cout << result.histogram.ToString(absl::GetFlag(FLAGS_max_bar_width))
                << "\n";
    }

    if (absl::GetFlag(FLAGS_show_heatmap)) {
      std::cout << result.heatmap.ToString(absl::GetFlag(FLAGS_color)) << "\n";
    }
  }

  std::string md_path = absl::GetFlag(FLAGS_output_markdown);
  if (!md_path.empty()) {
    std::string md = result.SummaryToMarkdown();
    absl::Status s = tsl::WriteStringToFile(tsl::Env::Default(), md_path, md);
    if (!s.ok()) {
      std::cerr << "Failed to write markdown report to '" << md_path
                << "': " << s << "\n";
      return kExitError;
    }
  }

  return result.passed ? kExitPass : kExitMismatch;
}

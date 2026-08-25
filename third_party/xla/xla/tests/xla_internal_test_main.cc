/* Copyright 2017 The OpenXLA Authors.

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

#include <vector>

#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/util/command_line_flags.h"

// Special test main used to pre-parse XLA flags (such as `--xla_*` debug
// options) before passing remaining arguments to GoogleTest.
//
// Background:
// When defining test rules (e.g., `xla_test`), XLA debug flags can be specified
// in the `args` attribute. However, standard `gunit_main` delegates flag
// parsing to Abseil Flags (`absl::ParseCommandLine`), which does not recognize
// internal XLA debug flags and results in an "Unknown command line flag" error.
// This test main uses `tsl::Flags::Parse` to extract and consume XLA debug
// options from `argv` before initializing GoogleTest.
//
// Note: XLA debug options can also be passed using the `XLA_FLAGS` environment
// variable (e.g., via the `env` attribute), but historically the team
// used `args`. Prefer using standard `gunit_main` (with
// flags passed via `env`) over this file whenever possible.
GTEST_API_ int main(int argc, char** argv) {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::vector<tsl::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  auto usage = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  // If the --benchmark_filter flag is passed in then only run the benchmarks,
  // not the tests.
  for (int i = 1; i < argc; i++) {
    absl::string_view arg(argv[i]);
    if (arg == "--benchmark_filter" ||
        absl::StartsWith(arg, "--benchmark_filter=")) {
      if (arg == "--benchmark_filter") {
        // Handle flag of the form '--benchmark_filter foo' (no '=').
        if (i + 1 >= argc || absl::StartsWith(argv[i + 1], "--")) {
          LOG(ERROR) << "--benchmark_filter flag requires an argument.";
          return 2;
        }
      }
      tsl::testing::InitializeBenchmarks(&argc, argv);
      testing::InitGoogleTest(&argc, argv);
      tsl::testing::RunBenchmarks();
      return 0;
    }
  }

  testing::InitGoogleTest(&argc, argv);

  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}

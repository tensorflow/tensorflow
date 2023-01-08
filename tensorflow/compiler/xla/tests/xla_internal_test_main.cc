/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

GTEST_API_ int main(int argc, char** argv) {
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
      const char* pattern = nullptr;
      if (absl::StartsWith(arg, "--benchmark_filter=")) {
        pattern = argv[i] + strlen("--benchmark_filter=");
      } else {
        // Handle flag of the form '--benchmark_filter foo' (no '=').
        if (i + 1 >= argc || absl::StartsWith(argv[i + 1], "--")) {
          LOG(ERROR) << "--benchmark_filter flag requires an argument.";
          return 2;
        }
        pattern = argv[i + 1];
      }
      ::benchmark::Initialize(&argc, argv);
      testing::InitGoogleTest(&argc, argv);
      benchmark::RunSpecifiedBenchmarks();
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

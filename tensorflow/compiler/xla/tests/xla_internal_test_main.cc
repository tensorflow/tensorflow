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

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

GTEST_API_ int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  auto usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  // If the --benchmarks flag is passed in then only run the benchmarks, not the
  // tests.
  for (int i = 1; i < argc; i++) {
    tensorflow::StringPiece arg(argv[i]);
    if (arg == "--benchmarks" || arg.starts_with("--benchmarks=")) {
      const char* pattern = nullptr;
      if (arg.starts_with("--benchmarks=")) {
        pattern = argv[i] + strlen("--benchmarks=");
      } else {
        // Handle flag of the form '--benchmarks foo' (no '=').
        if (i + 1 >= argc ||
            tensorflow::StringPiece(argv[i + 1]).starts_with("--")) {
          LOG(ERROR) << "--benchmarks flag requires an argument.";
          return 2;
        }
        pattern = argv[i + 1];
      }
      // Unfortunately Google's internal benchmark infrastructure has a
      // different API than Tensorflow's.
#if defined(PLATFORM_GOOGLE)
      base::SetFlag(&FLAGS_benchmarks, pattern);
      RunSpecifiedBenchmarks();
#else
      tensorflow::testing::Benchmark::Run(pattern);
#endif
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

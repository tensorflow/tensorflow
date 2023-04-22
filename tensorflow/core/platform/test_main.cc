/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// A program with a main that is suitable for unittests, including those
// that also define microbenchmarks.  Based on whether the user specified
// the --benchmark_filter flag which specifies which benchmarks to run,
// we will either run benchmarks or run the gtest tests in the program.

#include "tensorflow/core/platform/platform.h"

#if defined(PLATFORM_GOOGLE) || defined(__ANDROID__)
// main() is supplied by gunit_main
#else

#include <iostream>

#include "absl/strings/match.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

GTEST_API_ int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";

  tensorflow::testing::InstallStacktraceHandler();
  testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; i++) {
    if (absl::StartsWith(argv[i], "--benchmarks=")) {
      const char* pattern = argv[i] + strlen("--benchmarks=");
      tensorflow::testing::Benchmark::Run(pattern);
      return 0;
    }
  }
  return RUN_ALL_TESTS();
}
#endif

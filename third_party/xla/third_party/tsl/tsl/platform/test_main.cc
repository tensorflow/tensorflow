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

#include <string>

#include "absl/strings/match.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/stacktrace_handler.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

GTEST_API_ int main(int argc, char** argv) {
  tsl::testing::InstallStacktraceHandler();

  for (int i = 1; i < argc; i++) {
    if (absl::StartsWith(argv[i], "--benchmark_filter=")) {
      ::benchmark::Initialize(&argc, argv);

      // XXX: Must be called after benchmark's init because
      // InitGoogleTest eventually calls absl::ParseCommandLine() which would
      // complain that benchmark_filter flag is not known because that flag is
      // defined by the benchmark library via its own command-line flag
      // facility, which is not known to absl flags.
      // FIXME(vyng): Fix this mess once we make benchmark use absl flags
      testing::InitGoogleTest(&argc, argv);
      ::benchmark::RunSpecifiedBenchmarks();
      return 0;
    }
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

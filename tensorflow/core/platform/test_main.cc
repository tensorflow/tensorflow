// A program with a main that is suitable for unittests, including those
// that also define microbenchmarks.  Based on whether the user specified
// the --benchmark_filter flag which specifies which benchmarks to run,
// we will either run benchmarks or run the gtest tests in the program.

#include <iostream>

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
// main() is supplied by gunit_main
#else
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/test_benchmark.h"

GTEST_API_ int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";

  testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; i++) {
    if (tensorflow::StringPiece(argv[i]).starts_with("--benchmarks=")) {
      const char* pattern = argv[i] + strlen("--benchmarks=");
      tensorflow::testing::Benchmark::Run(pattern);
      return 0;
    }
  }
  return RUN_ALL_TESTS();
}
#endif

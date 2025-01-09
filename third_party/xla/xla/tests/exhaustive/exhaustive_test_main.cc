/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gtest/gtest.h>
#include "xla/tests/exhaustive/exhaustive_op_test_base.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"

GTEST_API_ int main(int argc, char** argv) {
  std::vector<tsl::Flag> flag_list;
  xla::exhaustive_op_test::AddExhaustiveFlags(flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

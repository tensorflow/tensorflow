/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/metrics.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(MetricsTest, RecordsGpuCompilerStacktrace) {
  // clang-format off
  // NOLINTBEGIN(whitespace/line_length)
  std::vector<std::string> expected_stack_trace = {
    "xla::RecordGpuCompilerStacktrace();",
    "xla::gpu::(anonymous namespace)::MetricsTest_RecordsGpuCompilerStacktrace_Test::TestBody();",
    "testing::Test::Run();",
    "testing::TestInfo::Run();",
    "testing::TestSuite::Run();",
    "testing::internal::UnitTestImpl::RunAllTests();",
    "testing::UnitTest::Run();",
    "main;",
    "__libc_start_main;",
    "_start",
  };
  // NOLINTEND(whitespace/line_length)
  // clang-format on

  RecordGpuCompilerStacktrace();

  std::string expected_stacktrace = absl::StrJoin(expected_stack_trace, "\n");
  EXPECT_EQ(GetGpuCompilerStacktraceCount(expected_stacktrace), 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

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

#include "xla/error_util.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(ErrorUtilTest, WrapWithPythonStacktraceTruncation) {
  absl::Status status = absl::InvalidArgumentError("original error");
  std::string stack_trace;
  for (int i = 0; i < 60; ++i) {
    stack_trace += absl::StrFormat("frame %d\n", i);
  }

  absl::Status wrapped_status = WrapWithPythonStacktrace(status, stack_trace);

  absl::string_view message = wrapped_status.message();
  EXPECT_TRUE(absl::StrContains(message, "[... truncated to 50 lines ...]"));
  EXPECT_TRUE(absl::StrContains(message, "frame 0\n"));
  EXPECT_TRUE(absl::StrContains(message, "frame 49\n"));
  EXPECT_FALSE(absl::StrContains(message, "frame 50\n"));
}

TEST(ErrorUtilTest, WrapWithPythonStacktraceNoTruncation) {
  absl::Status status = absl::InvalidArgumentError("original error");
  std::string stack_trace;
  for (int i = 0; i < 40; ++i) {
    stack_trace += absl::StrFormat("frame %d\n", i);
  }

  absl::Status wrapped_status = WrapWithPythonStacktrace(status, stack_trace);

  absl::string_view message = wrapped_status.message();
  EXPECT_FALSE(absl::StrContains(message, "[... truncated to 50 lines ...]"));
  EXPECT_TRUE(absl::StrContains(message, "frame 39\n"));
}

TEST(ErrorUtilTest, WrapWithPythonStacktraceExactly50Lines) {
  absl::Status status = absl::InvalidArgumentError("original error");
  std::string stack_trace;
  for (int i = 0; i < 50; ++i) {
    stack_trace += absl::StrFormat("frame %d\n", i);
  }

  absl::Status wrapped_status = WrapWithPythonStacktrace(status, stack_trace);

  absl::string_view message = wrapped_status.message();
  EXPECT_FALSE(absl::StrContains(message, "[... truncated to 50 lines ...]"));
  EXPECT_TRUE(absl::StrContains(message, "frame 49\n"));
}

TEST(ErrorUtilTest, WrapWithPythonStacktrace51Lines) {
  absl::Status status = absl::InvalidArgumentError("original error");
  std::string stack_trace;
  for (int i = 0; i < 51; ++i) {
    stack_trace += absl::StrFormat("frame %d\n", i);
  }

  absl::Status wrapped_status = WrapWithPythonStacktrace(status, stack_trace);

  absl::string_view message = wrapped_status.message();
  EXPECT_TRUE(absl::StrContains(message, "[... truncated to 50 lines ...]"));
  EXPECT_TRUE(absl::StrContains(message, "frame 49\n"));
  EXPECT_FALSE(absl::StrContains(message, "frame 50\n"));
}

}  // namespace
}  // namespace xla

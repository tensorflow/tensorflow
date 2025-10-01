/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/error/fatal_error_sink.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "xla/error/debug_me_context_util.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla::error {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

// A matcher that counts the number of times a substring appears in a reference
// string.
MATCHER_P2(CountSubstr, substr, expected_count, "") {
  const std::string substr_as_string(substr);
  int count = 0;
  std::string::size_type pos = 0;

  // Use the guaranteed std::string in the loop.
  while ((pos = arg.find(substr_as_string, pos)) != std::string::npos) {
    ++count;
    pos += substr_as_string.length();
  }

  return count == expected_count;
}

// Test fixture for FatalErrorSink tests.
class FatalErrorSinkTest : public ::testing::Test {
 protected:
  void SetUp() override { absl::AddLogSink(&sink_); }
  void TearDown() override { absl::RemoveLogSink(&sink_); }

 private:
  FatalErrorSink sink_;
};

TEST_F(FatalErrorSinkTest, ContextPresent_NonFatalIsIgnored) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");
  testing::internal::CaptureStderr();

  LOG(ERROR) << "This is a test error.";

  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr("DebugMeContext")));
  EXPECT_THAT(stderr_output, Not(HasSubstr("MyTestPass")));
  EXPECT_THAT(stderr_output, HasSubstr("This is a test error."));
}

using FatalErrorSinkDeathTest = FatalErrorSinkTest;

TEST_F(FatalErrorSinkDeathTest, NoContextPresent_NoLog) {
  EXPECT_DEATH(LOG(FATAL) << "test", Not(HasSubstr("DebugMeContext")));
}

TEST_F(FatalErrorSinkDeathTest, ContextPresent_LogExactlyOnce) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");

  EXPECT_DEATH(LOG(FATAL) << "test", CountSubstr("MyTestPass", 1));
}

TEST_F(FatalErrorSinkDeathTest, ContextPresent_CHECK_LogExactlyOnce) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");

  EXPECT_DEATH(CHECK(false) << "test", CountSubstr("MyTestPass", 1));
}

}  // namespace
}  // namespace xla::error

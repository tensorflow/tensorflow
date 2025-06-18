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

#include "xla/status_macros.h"

#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/log_sink.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::Status RetCheckFail() {
  TF_RET_CHECK(2 > 3);
  return absl::OkStatus();
}

absl::Status RetCheckFailWithExtraMessage() {
  TF_RET_CHECK(2 > 3) << "extra message";
  return absl::OkStatus();
}

absl::Status RetCheckFailWithLogSeverity(absl::LogSeverity severity) {
  TF_RET_CHECK(1 == 2).with_log_severity(severity) << "extra message";
  return absl::OkStatus();
}

absl::Status RetCheckSuccess() {
  TF_RET_CHECK(3 > 2);
  return absl::OkStatus();
}

absl::Status XlaRetCheckFailLogWarning() {
  XLA_RET_CHECK_FAIL().with_log_severity(absl::LogSeverity::kWarning)
      << "xla ret check fail message";
}

// A type that has `AbslStringify` but not `operator<<`.
struct HasOnlyAbslStringify {
  int i;

  bool operator==(const HasOnlyAbslStringify& h) const { return i == h.i; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HasOnlyAbslStringify& h) {
    absl::Format(&sink, "Stringify-%v", h.i);
  }
};

absl::Status RetCheckPrintAbslStringify() {
  HasOnlyAbslStringify h = {123};
  TF_RET_CHECK(false) << h;
  return absl::OkStatus();
}

TEST(StatusMacros, RetCheckFailing) {
  absl::Status status = RetCheckFail();
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("RET_CHECK failure.*2 > 3"));
}

TEST(StatusMacros, RetCheckFailingWithExtraMessage) {
  absl::Status status = RetCheckFailWithExtraMessage();
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("RET_CHECK.*2 > 3 extra message"));
}

TEST(StatusMacros, RetCheckLogWarning) {
  // absl::ScopedMockLog only works if we're actually using ABSL logging, and
  // TSL supports a homegrown logging implementation, so we should only check
  // the log is emitted when ABSL logging is used.
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  const std::string kExpectedRegex = "RET_CHECK.*1 == 2 extra message";
  if constexpr (std::is_same_v<absl::LogSink, tsl::TFLogSink>) {
    EXPECT_CALL(mock_log, Log(absl::LogSeverity::kWarning, ::testing::_,
                              ::testing::ContainsRegex(kExpectedRegex)));
  }
  mock_log.StartCapturingLogs();
  absl::Status status =
      RetCheckFailWithLogSeverity(absl::LogSeverity::kWarning);
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(), ::testing::ContainsRegex(kExpectedRegex));
}

TEST(StatusMacros, RetCheckLogInfo) {
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  const std::string kExpectedRegex = "RET_CHECK.*1 == 2 extra message";
  if constexpr (std::is_same_v<absl::LogSink, tsl::TFLogSink>) {
    EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                              ::testing::ContainsRegex(kExpectedRegex)));
  }
  mock_log.StartCapturingLogs();
  absl::Status status = RetCheckFailWithLogSeverity(absl::LogSeverity::kInfo);
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(), ::testing::ContainsRegex(kExpectedRegex));
}

TEST(StatusMacros, RetCheckSucceeding) {
  absl::Status status = RetCheckSuccess();
  EXPECT_IS_OK(status);
}

absl::StatusOr<int> CreateIntSuccessfully() { return 42; }

absl::StatusOr<int> CreateIntUnsuccessfully() {
  return tsl::errors::Internal("foobar");
}

TEST(StatusMacros, AssignOrAssertOnOK) {
  TF_ASSERT_OK_AND_ASSIGN(int result, CreateIntSuccessfully());
  EXPECT_EQ(42, result);
}

absl::Status ReturnStatusOK() { return absl::OkStatus(); }

absl::Status ReturnStatusError() { return (tsl::errors::Internal("foobar")); }

using StatusReturningFunction = std::function<absl::Status()>;

absl::StatusOr<int> CallStatusReturningFunction(
    const StatusReturningFunction& func) {
  TF_RETURN_IF_ERROR(func());
  return 42;
}

TEST(StatusMacros, ReturnIfErrorOnOK) {
  absl::StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusOK);
  EXPECT_IS_OK(rc);
  EXPECT_EQ(42, std::move(rc).value());
}

TEST(StatusMacros, ReturnIfErrorOnError) {
  absl::StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusError);
  EXPECT_FALSE(rc.ok());
  EXPECT_EQ(rc.status().code(), tsl::error::INTERNAL);
}

TEST(StatusMacros, AssignOrReturnSuccessfully) {
  absl::Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntSuccessfully());
    EXPECT_EQ(value, 42);
    return absl::OkStatus();
  }();
  EXPECT_IS_OK(status);
}

TEST(StatusMacros, AssignOrReturnUnsuccessfully) {
  absl::Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntUnsuccessfully());
    (void)value;
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
}

TEST(StatusMacros, XlaRetCheckFailLogWarning) {
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  const std::string kExpectedLog = "xla ret check fail message";
  if constexpr (std::is_same_v<absl::LogSink, tsl::TFLogSink>) {
    EXPECT_CALL(mock_log, Log(absl::LogSeverity::kWarning, ::testing::_,
                              ::testing::HasSubstr(kExpectedLog)));
  }
  mock_log.StartCapturingLogs();
  absl::Status status = XlaRetCheckFailLogWarning();
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(), ::testing::HasSubstr(kExpectedLog));
}

TEST(StatusMacros, RetCheckPrintAbslStringify) {
  absl::Status status = RetCheckPrintAbslStringify();
  EXPECT_EQ(status.code(), tsl::error::INTERNAL);
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("RET_CHECK.*Stringify-123"));
}

}  // namespace xla

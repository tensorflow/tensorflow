// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/test/matchers.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

using litert::Error;
using litert::Expected;
using litert::Unexpected;
using testing::Not;
using testing::StrEq;
using testing::litert::IsError;
using testing::litert::IsOk;

namespace {

struct CopyOnly {
  CopyOnly() = default;
  CopyOnly(const CopyOnly&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
};

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
};

TEST(IsOkMatcherTest, Works) {
  const Expected<int> error = Error(kLiteRtStatusErrorNotFound, "not found");
  EXPECT_THAT(kLiteRtStatusOk, IsOk());
  EXPECT_THAT(Expected<int>(3), IsOk());

  EXPECT_THAT(error, Not(IsOk()));
  EXPECT_THAT(Unexpected(kLiteRtStatusErrorFileIO), Not(IsOk()));
  EXPECT_THAT(Error(kLiteRtStatusErrorInvalidArgument), Not(IsOk()));

  EXPECT_THAT(kLiteRtStatusErrorUnsupported, Not(IsOk()));

  EXPECT_THAT(testing::DescribeMatcher<Expected<int>>(IsOk()), StrEq("is ok."));
  EXPECT_THAT(testing::DescribeMatcher<Expected<int>>(Not(IsOk())),
              StrEq("is not ok."));

  testing::StringMatchResultListener listener;
  EXPECT_FALSE(testing::ExplainMatchResult(
      IsOk(), kLiteRtStatusErrorUnsupported, &listener));
  EXPECT_THAT(listener.str(), StrEq("status is kLiteRtStatusErrorUnsupported"));

  listener.Clear();
  EXPECT_FALSE(testing::ExplainMatchResult(IsOk(), error, &listener));
  EXPECT_THAT(listener.str(), StrEq(""));

  listener.Clear();
  EXPECT_FALSE(testing::ExplainMatchResult(IsOk(), error.Error(), &listener));
  EXPECT_THAT(listener.str(), StrEq(""));
}

// No, I'm not creating a templated test fixture just for that. This only
// contains non-fatal failures that are propagated to the test.
//
// The type of the error wrapper that fails is the test failure stack trace when
// debug options are specified.
template <class ErrorWrapper>
void TestErrorWrapper() {
  const ErrorWrapper error = Error(kLiteRtStatusErrorNotFound, "not found");
  EXPECT_THAT(error, IsError());
  EXPECT_THAT(error, IsError(kLiteRtStatusErrorNotFound));
  EXPECT_THAT(error, IsError(kLiteRtStatusErrorNotFound, "not found"));
  // This checks against the wrong status.
  EXPECT_THAT(error, Not(IsError(kLiteRtStatusErrorInvalidArgument)));
  // This checks against the wrong message.
  EXPECT_THAT(error, Not(IsError(kLiteRtStatusErrorNotFound, "oob")));

  testing::StringMatchResultListener listener;
  EXPECT_FALSE(testing::ExplainMatchResult(
      IsError(kLiteRtStatusErrorInvalidArgument), error, &listener));
  EXPECT_THAT(listener.str(), StrEq("status doesn't match"));

  listener.Clear();
  EXPECT_FALSE(testing::ExplainMatchResult(
      IsError(kLiteRtStatusErrorNotFound, "oob"), error, &listener));
  EXPECT_THAT(listener.str(), StrEq("message doesn't match"));
}

TEST(IsErrorMatcherTest, Works) {
  TestErrorWrapper<Expected<int>>();
  TestErrorWrapper<Unexpected>();
  TestErrorWrapper<Error>();

  EXPECT_THAT(kLiteRtStatusErrorUnsupported, IsError());
  EXPECT_THAT(kLiteRtStatusOk, Not(IsError()));
  EXPECT_THAT(Expected<int>(3), Not(IsError()));

  EXPECT_THAT(testing::DescribeMatcher<Expected<int>>(IsError()),
              StrEq("is an error."));
  EXPECT_THAT(testing::DescribeMatcher<Expected<int>>(Not(IsError())),
              StrEq("is not an error."));
  EXPECT_THAT(
      testing::DescribeMatcher<Expected<int>>(
          IsError(kLiteRtStatusErrorUnsupported)),
      testing::StrEq("is an error with status kLiteRtStatusErrorUnsupported."));
  EXPECT_THAT(testing::DescribeMatcher<Expected<int>>(
                  IsError(kLiteRtStatusErrorUnsupported, "unsupported")),
              testing::StrEq("is an error with status "
                             "kLiteRtStatusErrorUnsupported and message "
                             "matching: 'unsupported'."));

  testing::StringMatchResultListener listener;
  EXPECT_FALSE(
      testing::ExplainMatchResult(IsError(), kLiteRtStatusOk, &listener));
  EXPECT_THAT(listener.str(), StrEq("status doesn't match"));

  listener.Clear();
  EXPECT_FALSE(
      testing::ExplainMatchResult(IsError(), Expected<int>(3), &listener));
  EXPECT_THAT(listener.str(),
              StrEq("expected holds a value (but should hold an error)"));
}

TEST(LitertAssertOk, Works) {
  LITERT_ASSERT_OK(Expected<int>(3));
  LITERT_ASSERT_OK(kLiteRtStatusOk);
  EXPECT_FATAL_FAILURE(
      LITERT_ASSERT_OK(Error(kLiteRtStatusErrorInvalidArgument)), "is ok");
}
TEST(LitertExpectOk, Works) {
  LITERT_EXPECT_OK(Expected<int>(3));
  LITERT_EXPECT_OK(kLiteRtStatusOk);
  EXPECT_NONFATAL_FAILURE(
      LITERT_EXPECT_OK(Error(kLiteRtStatusErrorInvalidArgument)), "is ok");
}

TEST(AssertOkAndAssign, DefineAVariableWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto expected, Expected<int>(3));
  static_assert(std::is_same_v<decltype(expected), int>,
                "Type should be deduced to int.");
  EXPECT_EQ(expected, 3);

  LITERT_ASSERT_OK_AND_ASSIGN([[maybe_unused]] auto copy_only,
                              Expected<CopyOnly>(CopyOnly()));
  LITERT_ASSERT_OK_AND_ASSIGN([[maybe_unused]] auto move_only,
                              Expected<MoveOnly>(MoveOnly()));
}

TEST(AssertOkAndAssign, AssignAVariableWorks) {
  int expected = 0;
  LITERT_ASSERT_OK_AND_ASSIGN(expected, Expected<int>(3));
  EXPECT_EQ(expected, 3);

  [[maybe_unused]] CopyOnly copy_only;
  [[maybe_unused]] MoveOnly move_only;
  LITERT_ASSERT_OK_AND_ASSIGN(copy_only, Expected<CopyOnly>(CopyOnly()));
  LITERT_ASSERT_OK_AND_ASSIGN(move_only, Expected<MoveOnly>(MoveOnly()));
}

void TestAssertOkAndAssignFailure() {
  LITERT_ASSERT_OK_AND_ASSIGN(
      [[maybe_unused]] int expected,
      Expected<int>(Unexpected(kLiteRtStatusErrorInvalidArgument)));
}

TEST(AssertOkAndAssign, FailuresStopsExecution) {
  EXPECT_FATAL_FAILURE(TestAssertOkAndAssignFailure(), "is ok");
}

}  // namespace

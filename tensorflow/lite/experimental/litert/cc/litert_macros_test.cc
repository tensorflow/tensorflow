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

#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace {

using testing::AllOf;
using testing::Property;

TEST(LiteRtReturnIfErrorTest, ConvertsResultToLiteRtStatus) {
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(
            Expected<int>(Unexpected(kLiteRtStatusErrorNotFound)));
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(Unexpected(kLiteRtStatusErrorNotFound));
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
  EXPECT_EQ(
      []() -> LiteRtStatus {
        LITERT_RETURN_IF_ERROR(kLiteRtStatusErrorNotFound);
        return kLiteRtStatusOk;
      }(),
      kLiteRtStatusErrorNotFound);
}

TEST(LiteRtReturnIfErrorTest, ConvertsResultToExpectedHoldingAnError) {
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(
            Expected<void>(Unexpected(kLiteRtStatusErrorNotFound)));
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(Unexpected(kLiteRtStatusErrorNotFound));
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
  EXPECT_THAT(
      []() -> Expected<void> {
        LITERT_RETURN_IF_ERROR(kLiteRtStatusErrorNotFound);
        return {};
      }(),
      AllOf(Property(&Expected<void>::HasValue, false),
            Property(&Expected<void>::Error,
                     Property(&Error::Status, kLiteRtStatusErrorNotFound))));
}

TEST(LiteRtReturnIfErrorTest, DoesntReturnOnSuccess) {
  int canary_value = 0;
  auto ReturnExpectedIfError = [&canary_value]() -> Expected<void> {
    LITERT_RETURN_IF_ERROR(Expected<void>());
    canary_value = 1;
    return {};
  };
  EXPECT_THAT(ReturnExpectedIfError(),
              Property(&Expected<void>::HasValue, true));
  EXPECT_EQ(canary_value, 1);

  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(kLiteRtStatusOk);
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 2);
}

TEST(LiteRtReturnIfErrorTest, ExtraLoggingWorks) {
  int canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false) << "Successful default level logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogVerbose() << "Successful verbose logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogInfo() << "Successful info logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogWarning() << "Successful warning logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).LogError() << "Successful error logging.";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);

  canary_value = 0;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(false).NoLog() << "This should never be printed";
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 0);
}

TEST(LiteRtAssignOrReturnTest, VariableAssignmentWorks) {
  int canary_value = 0;
  auto ChangeCanaryValue = [&canary_value]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, Expected<int>(1));
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ChangeCanaryValue(), kLiteRtStatusOk);
  EXPECT_EQ(canary_value, 1);
}

TEST(LiteRtAssignOrReturnTest, MoveOnlyVariableAssignmentWorks) {
  struct MoveOnly {
    explicit MoveOnly(int val) : val(val) {};
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;
    int val = 1;
  };

  MoveOnly canary_value{0};
  auto ChangeCanaryValue = [&canary_value]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, Expected<MoveOnly>(1));
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ChangeCanaryValue(), kLiteRtStatusOk);
  EXPECT_EQ(canary_value.val, 1);
}

TEST(LiteRtAssignOrReturnTest, ReturnsOnFailure) {
  const Expected<int> InvalidArgumentError =
      Expected<int>(Unexpected(kLiteRtStatusErrorInvalidArgument));

  int canary_value = 0;
  auto ErrorWithStatus = [&]() -> LiteRtStatus {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError);
    return kLiteRtStatusOk;
  };
  EXPECT_EQ(ErrorWithStatus(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithCustomStatus = [&]() -> int {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError, 42);
    return 1;
  };
  EXPECT_EQ(ErrorWithCustomStatus(), 42);
  EXPECT_EQ(canary_value, 0);

  auto ErrorWithExpected = [&]() -> Expected<void> {
    LITERT_ASSIGN_OR_RETURN(canary_value, InvalidArgumentError);
    return {};
  };
  auto expected_return = ErrorWithExpected();
  ASSERT_FALSE(expected_return.HasValue());
  EXPECT_EQ(expected_return.Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(canary_value, 0);
}

}  // namespace
}  // namespace litert

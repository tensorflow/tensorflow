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
  EXPECT_THAT(
      []() -> Expected<int> {
        LITERT_RETURN_IF_ERROR(Expected<int>(3));
        return 4;
      }(),
      AllOf(Property(&Expected<int>::HasValue, true),
            Property(&Expected<int>::Value, 4)));
  int canary_value = 1;
  [&canary_value]() -> LiteRtStatus {
    LITERT_RETURN_IF_ERROR(kLiteRtStatusOk);
    canary_value = 2;
    return kLiteRtStatusOk;
  }();
  EXPECT_EQ(canary_value, 2);
}

}  // namespace
}  // namespace litert

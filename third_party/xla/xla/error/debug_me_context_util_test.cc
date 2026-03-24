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

#include "xla/error/debug_me_context_util.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/debug_me_context.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace {

using ::testing::EndsWith;

TEST(DebugMeContextUtil, StringCheck) {
  constexpr absl::string_view kCompilerName{"MyCompiler"};

  tsl::DebugMeContext<error::DebugMeContextKey> ctx(
      error::DebugMeContextKey::kCompiler, std::string(kCompilerName));

  const std::string error_message =
      error::DebugMeContextToErrorMessageString();

  EXPECT_TRUE(absl::StrContains(error_message, kCompilerName));
}

TEST(FlattenDebugPayloadIntoMessage, StatusWithPayloadIsFlattened) {
  absl::Status status = absl::InternalError("Original message.");
  status.SetPayload(error::kDebugContextPayloadUrl, absl::Cord("Debug info."));

  status = error::FlattenDebugPayloadIntoMessage(status);

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Original message.\nDebug info."));
  EXPECT_FALSE(status.GetPayload(error::kDebugContextPayloadUrl).has_value());
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(status.GetSourceLocations().front().file_name(),
              EndsWith("debug_me_context_util_test.cc"));
#endif  // defined(PLATFORM_GOOGLE)
}

TEST(FlattenDebugPayloadIntoMessage, StatusWithOtherPayloadsIsPreserved) {
  constexpr absl::string_view kOtherPayloadUrl = "other_payload";
  constexpr absl::string_view kOtherPayloadContent = "preserved";
  absl::Status status = absl::InternalError("Original message.");
  status.SetPayload(error::kDebugContextPayloadUrl, absl::Cord("Debug info."));
  status.SetPayload(kOtherPayloadUrl, absl::Cord(kOtherPayloadContent));

  status = error::FlattenDebugPayloadIntoMessage(status);

  // Assert the debug payload was flattened.
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Original message.\nDebug info."));
  EXPECT_FALSE(status.GetPayload(error::kDebugContextPayloadUrl).has_value());
  std::optional<absl::Cord> other_payload = status.GetPayload(kOtherPayloadUrl);
  ASSERT_TRUE(other_payload.has_value());
  EXPECT_EQ(other_payload.value(), kOtherPayloadContent);
}

TEST(FlattenDebugPayloadIntoMessage, StatusWithoutPayloadIsUnchanged) {
  absl::Status status = absl::InternalError("Original message.");
  absl::Status original_status = status;

  status = error::FlattenDebugPayloadIntoMessage(status);

  EXPECT_EQ(status, original_status);
}

TEST(FlattenDebugPayloadIntoMessage, OkStatusIsUnchanged) {
  absl::Status status = absl::OkStatus();

  status = error::FlattenDebugPayloadIntoMessage(status);

  EXPECT_TRUE(status.ok());
}

TEST(FlattenDebugPayloadIntoMessage, StatusOrWithPayloadIsFlattened) {
  absl::Status status = absl::InternalError("Original message.");
  status.SetPayload(error::kDebugContextPayloadUrl, absl::Cord("Debug info."));
  absl::StatusOr<int> status_or = status;

  status_or = error::FlattenDebugPayloadIntoMessage(status_or);

  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status_or.status().message(),
                                "Original message.\nDebug info."));
  EXPECT_FALSE(status_or.status()
                   .GetPayload(error::kDebugContextPayloadUrl)
                   .has_value());
}

}  // namespace
}  // namespace xla

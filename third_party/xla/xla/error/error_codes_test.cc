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

#include "xla/error/error_codes.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tsl/platform/platform.h"

namespace xla::error {
namespace {

// We will use kInvalidArgument for all single-value tests.
constexpr ErrorCode kTestCode = ErrorCode::kInvalidArgument;

TEST(ErrorCodesTest, ErrorCodeToStringIdentifier) {
  // E0002 is the string_id for InvalidArgument in the macro list.
  EXPECT_EQ(ErrorCodeToStringIdentifier(kTestCode), "E0002");
}

TEST(ErrorCodesTest, ErrorCodeToName) {
  // InvalidArgument is the enum_name for kInvalidArgument.
  EXPECT_EQ(ErrorCodeToName(kTestCode), "InvalidArgument");
}

TEST(ErrorCodesTest, GetErrorCodeAndName) {
  // Should combine the string_id and enum_name.
  EXPECT_EQ(GetErrorCodeAndName(kTestCode), "E0002: InvalidArgument");
}

TEST(ErrorCodesTest, GetErrorUrl) {
  // Should produce the documentation URL using the string_id.
  EXPECT_EQ(GetErrorUrl(kTestCode),
            "https://openxla.org/xla/errors/error_0002");
}

TEST(ErrorCodesTest, FactoryFunctionWithNoArgs) {
  // Test one of the generated factory functions.
  absl::Status status = InvalidArgument("My Test error");

  // Check the absl::StatusCode defined in the macro.
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // Check the full formatted error message.
  EXPECT_EQ(status.message(),
            "E0002: InvalidArgument:\nMy Test error\nSee "
            "https://openxla.org/xla/errors/error_0002 for more details.");
#if defined(PLATFORM_GOOGLE)
  auto location = status.GetSourceLocations().front();
  EXPECT_THAT(location.file_name(), testing::EndsWith("error_codes_test.cc"));
#endif  // PLATFORM_GOOGLE
}

TEST(ErrorCodesTest, FactoryFunctionWithArgs) {
  // Test one of the generated factory functions.
  std::string detail = "Something was wrong";
  absl::Status status = InvalidArgument("Test error: %s", detail);

  // Check the absl::StatusCode defined in the macro.
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // Check the full formatted error message.
  EXPECT_EQ(status.message(),
            "E0002: InvalidArgument:\nTest error: Something was wrong\nSee "
            "https://openxla.org/xla/errors/error_0002 for more details.");

#if defined(PLATFORM_GOOGLE)
  auto location = status.GetSourceLocations().front();
  EXPECT_THAT(location.file_name(), testing::EndsWith("error_codes_test.cc"));
#endif  // PLATFORM_GOOGLE
}

}  // namespace
}  // namespace xla::error

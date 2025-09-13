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

#include "xla/errors/error_codes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

#define TEST_ERROR_CODE(string_id, enum_name, status_code, description)      \
  TEST(ErrorCodeTest, Test##enum_name) {                                     \
    EXPECT_EQ(ErrorCodeToStringIdentifier(ErrorCode::enum_name), string_id); \
    EXPECT_EQ(ErrorCodeToName(ErrorCode::enum_name), description);           \
    EXPECT_EQ(GetErrorCodeAndName(ErrorCode::enum_name),                     \
              absl::StrCat(string_id, ": ", description));                   \
    EXPECT_EQ(GetErrorUrl(ErrorCode::enum_name),                             \
              absl::StrCat("https://openxla.org/xla/errors/", string_id));   \
    absl::Status status = enum_name("Test message: %s", "detail");           \
    EXPECT_THAT(status, absl_testing::StatusIs(status_code));                \
    EXPECT_THAT(status.message(),                                            \
                HasSubstr(absl::StrCat(string_id, ": ", description)));      \
    EXPECT_THAT(status.message(), HasSubstr("Test message: detail"));        \
  }

XLA_ERROR_CODE_LIST(TEST_ERROR_CODE)
#undef TEST_ERROR_CODE

}  // namespace
}  // namespace xla

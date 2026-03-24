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

#include "xla/pjrt/errors.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(ErrorsTest, SetCompilationErrorWithPayload) {
  absl::Status status = absl::InvalidArgumentError("test error");
  absl::Status status_with_payload = SetCompilationErrorWithPayload(status);
  EXPECT_THAT(status_with_payload, StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("test error")));
  EXPECT_TRUE(HasCompilationErrorPayload(status_with_payload));
}

}  // namespace xla

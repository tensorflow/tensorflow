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

#include "xla/stream_executor/sycl/sycl_status.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(SyclStatusTest, ToStatusReturnsExpectedStatusCodes) {
  // We only promise SyclError::kSyclSuccess to map to Ok, everything
  // else to Internal.
  EXPECT_THAT(ToStatus(SyclError::kSyclSuccess), IsOk());
  EXPECT_THAT(ToStatus(SyclError::kSyclErrorInvalidDevice),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(SyclStatusTest, ToStatusIncludesDetailMessage) {
  constexpr absl::string_view kMyMessage = "Some arbitrary message";
  EXPECT_THAT(ToStatus(SyclError::kSyclErrorInvalidDevice, kMyMessage),
              StatusIs(absl::StatusCode::kInternal, HasSubstr(kMyMessage)));
}

}  // namespace
}  // namespace stream_executor::gpu

/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_status_utils.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace pjrt {
namespace {

TEST(PjRtCApiStatusUtilsTest, PjrtErrorToStatusPayloadTest) {
  absl::Status status = absl::InternalError("test error");
  status.SetPayload("test_payload", absl::Cord("test_value"));
  PJRT_Error* error = StatusToPjRtError(status);

  const PJRT_Api* api = GetPjrtApi();
  absl::Status result = PjrtErrorToStatus(error, api);
  DestroyPjRtError(error);
  EXPECT_EQ(result.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(result.message(), "test error");
  auto payload = result.GetPayload("test_payload");
  ASSERT_TRUE(payload.has_value());
  EXPECT_EQ(*payload, "test_value");
}

TEST(PjRtCApiStatusUtilsTest, PjrtErrorToStatusNullError) {
  const PJRT_Api* api = GetPjrtApi();
  absl::Status result = PjrtErrorToStatus(nullptr, api);
  EXPECT_TRUE(result.ok());
}

}  // namespace
}  // namespace pjrt

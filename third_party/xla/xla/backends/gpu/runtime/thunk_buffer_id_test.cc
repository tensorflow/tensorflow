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

#include "xla/backends/gpu/runtime/thunk_buffer_id.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/tsl/platform/statusor.h"

namespace {

TEST(ThunkBufferIdTest, CreateFailsForLargeBufferIndex) {
  EXPECT_THAT(xla::gpu::ThunkBufferId::Create(xla::gpu::ThunkId(123),
                                              /*buffer_idx=*/256),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ThunkBufferIdTest, CreateSucceedsForSmallBufferIndex) {
  EXPECT_THAT(xla::gpu::ThunkBufferId::Create(xla::gpu::ThunkId(123),
                                              /*buffer_idx=*/255),
              absl_testing::IsOk());
}

TEST(ThunkBufferIdTest, CorrectlyStoresAndExtractsThunkIdAndBufferIndex) {
  TF_ASSERT_OK_AND_ASSIGN(
      xla::gpu::ThunkBufferId buffer_id,
      xla::gpu::ThunkBufferId::Create(xla::gpu::ThunkId(123),
                                      /*buffer_idx=*/45));

  EXPECT_THAT(buffer_id.thunk_id(), xla::gpu::ThunkId(123));
  EXPECT_THAT(buffer_id.buffer_idx(), 45);
}

}  // namespace

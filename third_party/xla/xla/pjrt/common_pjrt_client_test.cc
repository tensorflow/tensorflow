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

#include "xla/pjrt/common_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/runtime/device_id.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla

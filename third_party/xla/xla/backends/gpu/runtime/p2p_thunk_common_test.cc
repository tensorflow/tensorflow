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

#include "xla/backends/gpu/runtime/p2p_thunk_common.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(P2PThunkCommonTest, SerializeDeserializePopulatedP2PConfig) {
  P2PConfigProto proto = ParseTextProtoOrDie<P2PConfigProto>(R"pb(
    config {
      operand_element_type: F32
      replica_groups { replica_ids: 0 replica_ids: 1 }
      group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
    }
    id_to_source_target {
      key: 0
      value { target: 1 }
    }
    id_to_source_target {
      key: 1
      value { source: 0 }
    }
  )pb");

  TF_ASSERT_OK_AND_ASSIGN(P2PConfig deserialized, P2PConfigFromProto(proto));
  P2PConfigProto round_trip_proto = P2PConfigToProto(deserialized);
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(P2PThunkCommonTest, GetSortedSourceTargetPairs) {
  P2PConfig::IdToSourceTargetMap id_to_source_target = {
      {0, {3, std::nullopt}},  // replica 0 receives from 3 (3->0)
      {1, {std::nullopt, 2}},  // replica 1 sends to 2 (1->2)
      {2, {1, 3}},  // replica 2 receives from 1 (1->2), 2 sends to 3 (2->3)
      {3, {2, 0}},  // replica3 receives from 2 (2->3), 3 sends to 0 (3->0)
  };

  std::vector<SourceTarget> sorted_pairs =
      GetSortedSourceTargetPairs(id_to_source_target);

  ASSERT_EQ(sorted_pairs.size(), 3);
  EXPECT_EQ(sorted_pairs[0].source(), 1);
  EXPECT_EQ(sorted_pairs[0].target(), 2);
  EXPECT_EQ(sorted_pairs[1].source(), 2);
  EXPECT_EQ(sorted_pairs[1].target(), 3);
  EXPECT_EQ(sorted_pairs[2].source(), 3);
  EXPECT_EQ(sorted_pairs[2].target(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

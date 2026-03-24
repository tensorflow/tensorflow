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

#include "xla/backends/gpu/runtime/collective_thunk.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(CollectiveConfigTest, ToProto) {
  CollectiveConfig config{
      /*operand_element_type=*/{PrimitiveType::F32, PrimitiveType::BF16},
      /*replica_groups=*/
      {ParseTextProtoOrDie<ReplicaGroup>(
           R"pb(replica_ids: 0 replica_ids: 1)pb"),
       ParseTextProtoOrDie<ReplicaGroup>(
           R"pb(replica_ids: 2 replica_ids: 3)pb")},
      /*group_mode=*/
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
      /*use_symmetric_buffer=*/true,
  };

  EXPECT_THAT(config.ToProto(), EqualsProto(R"pb(
                operand_element_type: F32
                operand_element_type: BF16
                replica_groups { replica_ids: 0 replica_ids: 1 }
                replica_groups { replica_ids: 2 replica_ids: 3 }
                group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION
                use_symmetric_buffer: true
              )pb"));
}

TEST(CollectiveConfigTest, FromProto) {
  CollectiveConfigProto proto = ParseTextProtoOrDie<CollectiveConfigProto>(
      R"pb(
        operand_element_type: F32
        operand_element_type: BF16
        replica_groups { replica_ids: 0 replica_ids: 1 }
        replica_groups { replica_ids: 2 replica_ids: 3 }
        group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION
        use_symmetric_buffer: true
      )pb");

  CollectiveConfig config = CollectiveConfig::FromProto(proto);

  EXPECT_THAT(config.operand_element_type,
              ElementsAre(PrimitiveType::F32, PrimitiveType::BF16));
  EXPECT_THAT(config.replica_groups,
              ElementsAre(EqualsProto(R"pb(replica_ids: 0 replica_ids: 1)pb"),
                          EqualsProto(R"pb(replica_ids: 2 replica_ids: 3)pb")));
  EXPECT_EQ(config.group_mode,
            CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION);
  EXPECT_TRUE(config.use_symmetric_buffer);
}

}  // namespace
}  // namespace xla::gpu

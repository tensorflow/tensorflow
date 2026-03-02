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

}  // namespace
}  // namespace gpu
}  // namespace xla

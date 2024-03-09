/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/convert_async_collectives_to_sync.h"

#include <memory>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {

namespace {

namespace m = xla::testing::opcode_matchers;

// Note: The pass only processes modules that are already scheduled. If the test
// does not work as epxected, make sure to check if "is_scheduled=true" is added
// to the HLO module string.
class ConvertAsyncCollectivesToSyncTest : public HloTestBase {
 public:
  Status RunPass(HloModule *module, bool expect_change,
                 HloPredicate is_nop = {}) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        ConvertAsyncCollectivesToSync{is_nop}.Run(module));
    EXPECT_EQ(changed, expect_change);
    return OkStatus();
  }

  absl::string_view GetAsyncName(const HloInstruction *inst) {
    const auto &map = inst->frontend_attributes().map();
    return map.at(
        ConvertAsyncCollectivesToSync::kAsyncCollectiveNameAttributeName);
  }

  HloPredicate is_nop_simple_ =
      HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement,
                       HloOpcode::kParameter>;
};

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleAllReduce) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::ReplicaId()));
  const auto *ar = Cast<HloAllReduceInstruction>(root);
  EXPECT_TRUE(ar->channel_id().has_value());
  EXPECT_EQ(ar->channel_id().value(), 3);
  EXPECT_EQ(GetAsyncName(ar), "start");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleAllReduceWithNop) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3, replica_groups={{0,1}, {2,3}}
        id2 = f32[] bitcast(id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, is_nop_simple_));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::ReplicaId()));
  const auto *ar = Cast<HloAllReduceInstruction>(root);
  EXPECT_TRUE(ar->channel_id().has_value());
  EXPECT_EQ(ar->channel_id().value(), 3);
  EXPECT_THAT(ar, m::ReplicaGroups({{0, 1}, {2, 3}}));
  EXPECT_EQ(GetAsyncName(ar), "start");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleAllReduceWithNonNop) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        id2 = u32[] add(id, id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false));
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleAllGather) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test_computation {
    a1 = u32[1, 2] parameter(0)
    ags = (u32[1, 2], u32[2, 2]) all-gather-start(a1), dimensions={0}, channel_id=3
    ROOT allgather = u32[2,2] all-gather-done(ags)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllGather(m::Parameter(0)));
  const auto *ag = Cast<HloAllGatherInstruction>(root);
  EXPECT_TRUE(ag->channel_id().has_value());
  EXPECT_EQ(ag->channel_id().value(), 3);
  EXPECT_EQ(ag->all_gather_dimension(), 0);
  EXPECT_EQ(GetAsyncName(ag), "ags");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleCollectivePermute) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  ENTRY test_computation {
    p = u32[2] parameter(0)
    start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p), source_target_pairs={{0,1}, {1,0}}
    ROOT done = u32[2] collective-permute-done(start)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::CollectivePermute(m::Parameter(0)));
  const auto *cp = Cast<HloCollectivePermuteInstruction>(root);
  EXPECT_THAT(cp, m::SourceTargetPairs({{0, 1}, {1, 0}}));
  EXPECT_EQ(GetAsyncName(cp), "start");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleReduceScatter) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  add {
    lhs = u32[] parameter(0)
    rhs = u32[] parameter(1)
    ROOT add = u32[] add(lhs, rhs)
  }

  reduce_scatter {
    p0 = u32[8] parameter(0)
    ROOT result = u32[4] reduce-scatter(p0), replica_groups={{0,3}, {1,2}},
                      dimensions={0}, to_apply=add
  }

  ENTRY main {
    data = u32[8] parameter(0)
    rs-start = ((u32[8]{0}), u32[4]{0}) async-start(u32[8]{0} %data), calls=reduce_scatter
    ROOT %ars = u32[4]{0} async-done(((u32[8]{0}), u32[4]{0}) %rs-start), calls=reduce_scatter
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::ReduceScatter(m::Parameter(0)));
  const auto *rs = Cast<HloReduceScatterInstruction>(root);
  EXPECT_THAT(rs, m::ReplicaGroups({{0, 3}, {1, 2}}));
  EXPECT_EQ(rs->scatter_dimension(), 0);
  EXPECT_EQ(GetAsyncName(rs), "rs-start");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, SimpleAllToAll) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  all_to_all {
    p0 = u32[2] parameter(0)
    ROOT result = u32[2] all-to-all(p0), dimensions={0}, replica_groups={{0,1},{2,3}}
  }

  ENTRY test_computation {
    a1 = u32[2] parameter(0)
    a2a-start = ((u32[2]), u32[2]) async-start(u32[2] a1), calls=all_to_all
    ROOT a2s = u32[2] async-done(a2a-start), calls=all_to_all
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllToAll(m::Parameter(0)));
  const auto *a2a = Cast<HloAllToAllInstruction>(root);
  EXPECT_THAT(a2a, m::ReplicaGroups({{0, 1}, {2, 3}}));
  EXPECT_TRUE(a2a->split_dimension().has_value());
  EXPECT_EQ(a2a->split_dimension().value(), 0);
  EXPECT_EQ(GetAsyncName(a2a), "a2a-start");
}

TEST_F(ConvertAsyncCollectivesToSyncTest, ControlDeps) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        done1 = u32[] all-reduce-done(start1)
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4, control-predecessors={done1}
        done2 = u32[] all-reduce-done(start2)
        ROOT x = u32[] add(done1, done2)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Add(m::AllReduce(), m::AllReduce()));
}

// Test multiple in-flight collectives that are ordered in a streaming fashion:
// i.e., ends are in start order (FIFO).
TEST_F(ConvertAsyncCollectivesToSyncTest, MultipleInFlightStreaming) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done1 = u32[] all-reduce-done(start1)
        done2 = u32[] all-reduce-done(start2)
        ROOT x = u32[] add(done1, done2)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Add(m::AllReduce(), m::AllReduce()));
}

// Test multiple in-flight collectives that are nested: {s0,{s1,e1},e0}
TEST_F(ConvertAsyncCollectivesToSyncTest, MultipleInFlightNested) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done2 = u32[] all-reduce-done(start2)
        done1 = u32[] all-reduce-done(start1)
        ROOT x = u32[] add(done1, done2)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Add(m::AllReduce(), m::AllReduce()));
}

// Test multiple in-flight collectives that are nested: {s0,{s1,e1},e0} where
// inner pair can be converted but not outer.
TEST_F(ConvertAsyncCollectivesToSyncTest, MultipleInFlightNestedPartial) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done2 = u32[] all-reduce-done(start2)
        id2 = u32[] add(done2, done2)
        done1 = u32[] all-reduce-done(start1)
        ROOT x = u32[] add(done1, done2)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  const HloInstruction *root = module->entry_computation()->root_instruction();
  // We expect start2/done2 to be converted to async, start1/done1 will stay
  // unchanged.
  EXPECT_THAT(root, m::Add(m::AllReduceDone(), m::AllReduce()));
}

}  // namespace

}  // namespace xla

/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/async_collective_creator.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

using ::testing::NotNull;
using ::testing::SizeIs;

using AsyncCollectiveCreatorTest = HloHardwareIndependentTestBase;

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleAllReduce) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[1024] parameter(0)
    ROOT ar = f32[1024] all-reduce(p0), to_apply=add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = HloPredicateTrue;
  config.all_reduce_min_threshold_in_bytes = 4096;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllReduceDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllReduceStart);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleAllGather) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[1] parameter(0)
    ROOT ag = f32[8] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_gather = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllGatherDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllGatherStart);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleCollectivePermute) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    %p0 = bf16[8]{0} parameter(0)
    ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} p0), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleInPlaceCollectivePermute) {
  std::string hlo_string = std::string(R"(
HloModule module

ENTRY %module_spmd () -> f32[4,4,128] {
  %constant.8 = u32[] constant(0)
  %constant.5 = u32[] constant(2)
  %tuple.1 = (u32[], u32[], u32[]) tuple(u32[] %constant.8, u32[] %constant.8, u32[] %constant.8)
  %tuple = (u32[], u32[], u32[]) tuple(u32[] %constant.5, u32[] %constant.8, u32[] %constant.8)
  %custom-call = f32[4,4,128]{2,1,0:T(4,128)} custom-call(), custom_call_target="SomeCustomCall"
  ROOT %collective-permute = f32[4,4,128]{2,1,0:T(4,128)} collective-permute(f32[4,4,128]{2,1,0:T(4,128)} %custom-call, f32[4,4,128]{2,1,0:T(4,128)} %custom-call, (u32[], u32[], u32[]) %tuple, (u32[], u32[], u32[]) %tuple.1), channel_id=958, source_target_pairs={{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, slice_sizes={{2,4,128}}
}
)");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 7);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleCollectivePermuteScheduled) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY entry {
    %p0 = bf16[8]{0} parameter(0)
    ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} p0), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const int64_t original_instr_sequence_size =
      hlo_module->schedule().sequence(hlo_module->entry_computation()).size();

  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
  EXPECT_EQ(
      hlo_module->schedule().sequence(hlo_module->entry_computation()).size(),
      original_instr_sequence_size + 1);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleCollectiveBroadcast) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[8,16] parameter(0)
    ROOT cb = f32[8,16] collective-broadcast(p0), replica_groups={{7,0,1,2,3,4,5,6}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_broadcast = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);
  ASSERT_THAT(start->async_wrapped_instruction(), NotNull());
  EXPECT_THAT(start->async_wrapped_opcode(), HloOpcode::kCollectiveBroadcast);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleAllToAll) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[8,16] parameter(0)
    ROOT ata = f32[8,16] all-to-all(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_to_all = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());
  XLA_VLOG_LINES(0, hlo_module->ToString());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);
  ASSERT_THAT(start->async_wrapped_instruction(), NotNull());
  EXPECT_THAT(start->async_wrapped_opcode(), HloOpcode::kAllToAll);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleReduceScatter) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[8,16] parameter(0)
    ROOT ata = f32[1,16] reduce-scatter(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_reduce_scatter = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());
  XLA_VLOG_LINES(0, hlo_module->ToString());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);
  ASSERT_THAT(start->async_wrapped_instruction(), NotNull());
  EXPECT_THAT(start->async_wrapped_opcode(), HloOpcode::kReduceScatter);
}

TEST_F(AsyncCollectiveCreatorTest, SplitsSingleRaggedAllToAll) {
  constexpr absl::string_view hlo_string = R"(
HloModule RaggedAllToAll

ENTRY RA2A {
  input = f32[64,8,128]{2,1,0:T(8,128)} parameter(0)
  c0 = f32[] constant(0)
  output = f32[64,8,128]{2,1,0:T(8,128)S(1)} broadcast(c0), dimensions={}
  input_offsets = s32[8]{0} parameter(1)
  send_sizes = s32[8]{0} parameter(2)
  output_offsets = s32[8]{0} parameter(3)
  recv_sizes = s32[8]{0} parameter(4)
  ROOT ra2a = f32[64,8,128]{2,1,0:T(8,128)S(1)} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3,4,5,6,7}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_ragged_all_to_all = HloPredicateTrue;
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());
  XLA_VLOG_LINES(0, hlo_module->ToString());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 9);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);
  ASSERT_THAT(start->async_wrapped_instruction(), NotNull());
  EXPECT_THAT(start->async_wrapped_opcode(), HloOpcode::kRaggedAllToAll);
}

TEST_F(AsyncCollectiveCreatorTest, ControlPredecessor) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ag = f32[1024] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, control-predecessors={p0}
    p1 = f32[128] parameter(1), control-predecessors={ag}
    ROOT sum = add(ag, ag)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_gather = HloPredicateTrue;
  config.all_gather_min_threshold_in_bytes = 4096;
  TF_ASSERT_OK(
      RunHloPass(AsyncCollectiveCreator(config), hlo_module.get()).status());
  SCOPED_TRACE(hlo_module->ToString());

  HloInstruction* start;
  HloInstruction* done;
  ASSERT_THAT(
      hlo_module->entry_computation()->root_instruction(),
      GmockMatch(m::Add(m::Op(),
                        m::Op(&done)
                            .WithOpcode(HloOpcode::kAllGatherDone)
                            .WithOperand(0, m::Op(&start).WithOpcode(
                                                HloOpcode::kAllGatherStart)))));
  EXPECT_EQ(start->control_successors().size(), 0);
  ASSERT_EQ(start->control_predecessors().size(), 1);
  EXPECT_THAT(start->control_predecessors()[0], GmockMatch(m::Parameter(0)));

  EXPECT_EQ(done->control_predecessors().size(), 0);
  ASSERT_EQ(done->control_successors().size(), 1);
  EXPECT_THAT(done->control_successors()[0], GmockMatch(m::Parameter(1)));
}

TEST_F(AsyncCollectiveCreatorTest, PreserveFrontendAttributesAllGather) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT ag = f32[1024] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, frontend_attributes={_scheduling_group_id="0"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_gather = HloPredicateTrue;
  TF_ASSERT_OK(
      RunHloPass(AsyncCollectiveCreator(config), hlo_module.get()).status());

  HloInstruction* done = hlo_module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_TRUE(
      done->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(done->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
  EXPECT_TRUE(
      start->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(start->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
}

TEST_F(AsyncCollectiveCreatorTest, PreserveFrontendAttributesAllReduce) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[1024] parameter(0)
    ROOT ar = f32[1024] all-reduce(p0), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add, frontend_attributes={_scheduling_group_id="0"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = HloPredicateTrue;
  TF_ASSERT_OK(
      RunHloPass(AsyncCollectiveCreator(config), hlo_module.get()).status());

  HloInstruction* done = hlo_module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_TRUE(
      done->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(done->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
  EXPECT_TRUE(
      start->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(start->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
}

TEST_F(AsyncCollectiveCreatorTest,
       PreserveFrontendAttributesCollectivePermute) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[1024] parameter(0)
    ROOT cp = f32[1024] collective-permute(p0), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}, frontend_attributes={_scheduling_group_id="0"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(
      RunHloPass(AsyncCollectiveCreator(config), hlo_module.get()).status());

  HloInstruction* done = hlo_module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_TRUE(
      done->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(done->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
  EXPECT_TRUE(
      start->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(start->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
}

TEST_F(AsyncCollectiveCreatorTest, PreserveFrontendAttributesAllToAll) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[1024] parameter(0)
    ROOT a2a = f32[1024] all-to-all(p0), replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, frontend_attributes={_scheduling_group_id="0"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_to_all = HloPredicateTrue;
  TF_ASSERT_OK(
      RunHloPass(AsyncCollectiveCreator(config), hlo_module.get()).status());

  HloInstruction* done = hlo_module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_TRUE(
      done->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(done->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
  EXPECT_TRUE(
      start->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(start->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "0");
}
}  // namespace
}  // namespace xla

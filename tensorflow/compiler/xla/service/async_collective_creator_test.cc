/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_collective_creator.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

using AsyncAllReduceCreatorTest = HloTestBase;

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllReduce) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[8] parameter(0)
    ROOT ar = f32[8] all-reduce(p0), to_apply=add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = [](const HloInstruction*) { return true; };
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

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllGather) {
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
  config.convert_all_gather = [](const HloInstruction*) { return true; };
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

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleCollectivePermute) {
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
  config.convert_collective_permute = [](const HloInstruction*) {
    return true;
  };
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

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleInPlaceCollectivePermute) {
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
  config.convert_collective_permute = [](const HloInstruction*) {
    return true;
  };
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

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleCollectivePermuteScheduled) {
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
  config.convert_collective_permute = [](const HloInstruction*) {
    return true;
  };
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

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllToAll) {
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
  config.convert_all_to_all = [](const HloInstruction*) { return true; };
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

}  // namespace
}  // namespace xla

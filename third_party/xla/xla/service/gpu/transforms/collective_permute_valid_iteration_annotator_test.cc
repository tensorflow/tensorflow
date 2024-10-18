/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using CollectivePermuteValidIterationAnnotatorTest = HloTestBase;

TEST_F(CollectivePermuteValidIterationAnnotatorTest, NoChange) {
  // We expect no changes here because the while loop is not labelled as
  // `is_pipelined_while_loop`.
  absl::string_view hlo_string = R"(
    HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %permute)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple), condition=%Cond, body=%Body
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, 1, 4));

  HloPassPipeline pipeline("my-pass-pipeline");

  pipeline.AddPass<WhileLoopTripCountAnnotator>();
  pipeline.AddPass<CollectivePermuteValidIterationAnnotator>();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_FALSE(changed);

  HloCollectivePermuteInstruction* cp =
      DynCastOrNull<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), HloOpcode::kCollectivePermute));

  ASSERT_NE(cp, nullptr);

  auto sendRecvValidationIt =
      cp->frontend_attributes().map().find(kSendRecvValidationAttr);
  ASSERT_EQ(sendRecvValidationIt, cp->frontend_attributes().map().end());
}

TEST_F(CollectivePermuteValidIterationAnnotatorTest, ForwardCycle) {
  absl::string_view hlo_string = R"(
    HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %i_plus_one)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple), condition=%Cond, body=%Body, frontend_attributes={is_pipelined_while_loop="true"}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, 1, 4));

  HloPassPipeline pipeline("my-pass-pipeline");

  pipeline.AddPass<WhileLoopTripCountAnnotator>();
  pipeline.AddPass<CollectivePermuteValidIterationAnnotator>();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  HloCollectivePermuteInstruction* cp =
      DynCastOrNull<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), HloOpcode::kCollectivePermute));

  ASSERT_NE(cp, nullptr);

  auto sendRecvValidationIt =
      cp->frontend_attributes().map().find(kSendRecvValidationAttr);
  ASSERT_NE(sendRecvValidationIt, cp->frontend_attributes().map().end());
  std::string sendRecvValidationAttr = sendRecvValidationIt->second;
  EXPECT_EQ(sendRecvValidationAttr, "{{0,6},{1,7},{2,8},{3,9}}");
}

TEST_F(CollectivePermuteValidIterationAnnotatorTest, BackwardCycle) {
  absl::string_view hlo_string = R"(
    HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %i_plus_one)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple), condition=%Cond, body=%Body, frontend_attributes={is_pipelined_while_loop="true"}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, 1, 4));

  HloPassPipeline pipeline("my-pass-pipeline");

  pipeline.AddPass<WhileLoopTripCountAnnotator>();
  pipeline.AddPass<CollectivePermuteValidIterationAnnotator>();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  HloCollectivePermuteInstruction* cp =
      DynCastOrNull<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), HloOpcode::kCollectivePermute));

  ASSERT_NE(cp, nullptr);

  auto sendRecvValidationIt =
      cp->frontend_attributes().map().find(kSendRecvValidationAttr);
  ASSERT_NE(sendRecvValidationIt, cp->frontend_attributes().map().end());
  std::string sendRecvValidationAttr = sendRecvValidationIt->second;
  EXPECT_EQ(sendRecvValidationAttr, "{{3,9},{2,8},{1,7},{0,6}}");
}
}  // namespace
}  // namespace xla

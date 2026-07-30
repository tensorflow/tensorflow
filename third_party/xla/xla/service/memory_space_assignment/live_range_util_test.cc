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

#include "xla/service/memory_space_assignment/live_range_util.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_buffer.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class LiveRangeCalculatorTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<absl::flat_hash_map<const HloInstruction*, int64_t>>
  GetInstToIndex(const HloModule* module,
                 const HloAliasAnalysis& alias_analysis) {
    ASSIGN_OR_RETURN(auto live_range,
                     HloLiveRange::Run(module->schedule(), alias_analysis,
                                       module->entry_computation()));
    return live_range->instruction_schedule();
  }

  const HloBuffer* GetBufferForInstruction(
      const HloAliasAnalysis& alias_analysis, const HloInstruction* inst,
      const ShapeIndex& index = {}) {
    return &alias_analysis.GetUniqueBufferAt(inst, index);
  }
};

// Tests live range calculation for a simple linear HLO sequence without
// control flow.
TEST_F(LiveRangeCalculatorTest, SimpleLinear) {
  const char* hlo_text = R"(
    HloModule Simple, is_scheduled=true
    ENTRY %entry (p0: f32[2,3], p1: f32[2,3]) -> f32[2,3] {
      %p0 = f32[2,3]{1,0} parameter(0)
      %p1 = f32[2,3]{1,0} parameter(1)
      %add = f32[2,3]{1,0} add(%p0, %p1)
      ROOT %neg = f32[2,3]{1,0} negate(%add)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* add_inst = FindInstruction(module.get(), "add");
  ASSERT_NE(add_inst, nullptr);
  const HloBuffer* add_buffer =
      GetBufferForInstruction(*alias_analysis, add_inst);
  ASSERT_NE(add_buffer, nullptr);

  LiveRangeCalculator calculator(*add_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: add
  // 3: neg

  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start_time, 2);
  EXPECT_EQ(ranges[0].end_time, 3);
}

// Tests Case 1: Conditional input that has no subsequent use after the
// conditional. The live range should propagate into the branches and result in
// disjoint live ranges for each branch.
TEST_F(LiveRangeCalculatorTest, ConditionalInputNoSubsequentUse) {
  const char* hlo_text = R"(
    HloModule Cond, is_scheduled=true
    true_comp (p3: (f32[2,3])) -> (f32[2,3]) {
      p3 = (f32[2,3]) parameter(0)
      gte1 = f32[2,3] get-tuple-element(p3), index=0
      neg0 = f32[2,3] negate(gte1)
      ROOT tuple2 = (f32[2,3]) tuple(neg0)
    }
    false_comp (p4: (f32[2,3])) -> (f32[2,3]) {
      p4 = (f32[2,3]) parameter(0)
      gte2 = f32[2,3] get-tuple-element(p4), index=0
      neg1 = f32[2,3] negate(gte2)
      ROOT tuple3 = (f32[2,3]) tuple(neg1)
    }
    ENTRY entry (p0: pred[], p1: f32[2,3]) -> f32[2,3] {
      p0 = pred[] parameter(0)
      p1 = f32[2,3] parameter(1)
      tanh0 = f32[2,3] tanh(p1)
      tuple0 = (f32[2,3]) tuple(tanh0)
      conditional = (f32[2,3]) conditional(p0, tuple0, tuple0), true_computation=true_comp, false_computation=false_comp
      gte3 = f32[2,3] get-tuple-element(conditional), index=0
      ROOT neg2 = f32[2,3] negate(gte3)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: tanh0
  // 3: tuple0
  // -- true_comp --
  // 4: p3
  // 5: gte1
  // 6: neg0
  // 7: tuple2
  // -- false_comp --
  // 8: p4
  // 9: gte2
  // 10: neg1
  // 11: tuple3
  // -- entry --
  // 12: conditional
  // 13: gte3
  // 14: neg2

  // tanh0 buffer should be live:
  // - in entry: from tanh0 (2) to tuple0 (3) -> [2, 3]
  // - in true_comp: from p3 (4) to neg0 (6) -> [4, 6]
  // - in false_comp: from p4 (8) to neg1 (10) -> [8, 10]
  // Merged ranges should be: [2, 6] and [8, 10]
  // Wait, tanh0 value itself is live [2, 4] and [8, 8] (due to conditional at
  // 12, earliest called comp is 4, so [2, 4]. And false_comp starts at 8, so
  // [8, 8]). p3 index 0 is live [4, 6]. p4 index 0 is live [8, 10]. Union: [2,
  // 4] U [8, 8] U [4, 6] U [8, 10] = [2, 6] and [8, 10].

  ASSERT_EQ(ranges.size(), 2);
  EXPECT_EQ(ranges[0].start_time, 2);
  EXPECT_EQ(ranges[0].end_time, 6);
  EXPECT_EQ(ranges[1].start_time, 8);
  EXPECT_EQ(ranges[1].end_time, 10);
}

// Tests Case 2: Conditional input that has a subsequent use after the
// conditional. The live range should cover the entire region from definition to
// the subsequent use.
TEST_F(LiveRangeCalculatorTest, ConditionalInputWithSubsequentUse) {
  const char* hlo_text = R"(
    HloModule Cond, is_scheduled=true
    true_comp (p3: (f32[2,3])) -> (f32[2,3]) {
      p3 = (f32[2,3]) parameter(0)
      gte1 = f32[2,3] get-tuple-element(p3), index=0
      neg0 = f32[2,3] negate(gte1)
      ROOT tuple2 = (f32[2,3]) tuple(neg0)
    }
    false_comp (p4: (f32[2,3])) -> (f32[2,3]) {
      p4 = (f32[2,3]) parameter(0)
      gte2 = f32[2,3] get-tuple-element(p4), index=0
      neg1 = f32[2,3] negate(gte2)
      ROOT tuple3 = (f32[2,3]) tuple(neg1)
    }
    ENTRY entry (p0: pred[], p1: f32[2,3]) -> f32[2,3] {
      p0 = pred[] parameter(0)
      p1 = f32[2,3] parameter(1)
      tanh0 = f32[2,3] tanh(p1)
      tuple0 = (f32[2,3]) tuple(tanh0)
      conditional = (f32[2,3]) conditional(p0, tuple0, tuple0), true_computation=true_comp, false_computation=false_comp
      gte3 = f32[2,3] get-tuple-element(conditional), index=0
      neg2 = f32[2,3] negate(gte3)
      add0 = f32[2,3] add(tanh0, neg2)
      ROOT neg3 = f32[2,3] negate(add0)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: tanh0
  // 3: tuple0
  // -- true_comp --
  // 4: p3
  // 5: gte1
  // 6: neg0
  // 7: tuple2
  // -- false_comp --
  // 8: p4
  // 9: gte2
  // 10: neg1
  // 11: tuple3
  // -- entry --
  // 12: conditional
  // 13: gte3
  // 14: neg2
  // 15: add0
  // 16: neg3

  // tanh0 buffer should be live:
  // - from tanh0 (2) to add0 (15) because of subsequent use at add0.
  // - inside true_comp: [4, 6] (p3 to neg0)
  // - inside false_comp: [8, 10] (p4 to neg1)
  // Union of [2, 15], [4, 6], [8, 10] is [2, 15].

  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start_time, 2);
  EXPECT_EQ(ranges[0].end_time, 15);
}

// Tests Case 3: Conditional output where branch outputs and conditional output
// are mapped to the same buffer, resulting in separate, non-overlapping live
// ranges.
TEST_F(LiveRangeCalculatorTest, ConditionalOutput) {
  const char* hlo_text = R"(
    HloModule Cond, is_scheduled=true
    true_comp (p3: (f32[2,3])) -> (f32[2,3]) {
      p3 = (f32[2,3]) parameter(0)
      gte1 = f32[2,3] get-tuple-element(p3), index=0
      neg0 = f32[2,3] negate(gte1)
      ROOT tuple2 = (f32[2,3]) tuple(neg0)
    }
    false_comp (p4: (f32[2,3])) -> (f32[2,3]) {
      p4 = (f32[2,3]) parameter(0)
      gte2 = f32[2,3] get-tuple-element(p4), index=0
      neg1 = f32[2,3] negate(gte2)
      ROOT tuple3 = (f32[2,3]) tuple(neg1)
    }
    ENTRY entry (p0: pred[], p1: f32[2,3]) -> f32[2,3] {
      p0 = pred[] parameter(0)
      p1 = f32[2,3] parameter(1)
      tanh0 = f32[2,3] tanh(p1)
      tuple0 = (f32[2,3]) tuple(tanh0)
      conditional = (f32[2,3]) conditional(p0, tuple0, tuple0), true_computation=true_comp, false_computation=false_comp
      gte3 = f32[2,3] get-tuple-element(conditional), index=0
      ROOT neg2 = f32[2,3] negate(gte3)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* cond_inst =
      FindInstruction(module.get(), "conditional");
  ASSERT_NE(cond_inst, nullptr);
  const HloBuffer* cond_output_buffer =
      GetBufferForInstruction(*alias_analysis, cond_inst, {0});
  ASSERT_NE(cond_output_buffer, nullptr);

  LiveRangeCalculator calculator(*cond_output_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: tanh0
  // 3: tuple0
  // -- true_comp --
  // 4: p3
  // 5: gte1
  // 6: neg0
  // 7: tuple2
  // -- false_comp --
  // 8: p4
  // 9: gte2
  // 10: neg1
  // 11: tuple3
  // -- entry --
  // 12: conditional
  // 13: gte3
  // 14: neg2

  // The conditional output buffer (element 0) contains:
  // - neg0 (def 6) in true_comp, used as ROOT. Live range: [6, 7] (tuple2 is at
  // 7)
  // - neg1 (def 10) in false_comp, used as ROOT. Live range: [10, 11] (tuple3
  // is at 11)
  // - conditional output (def 12) in entry, used at neg2 (14). Live range: [12,
  // 14] Since they don't merge (7 < 8, 11 < 12), we expect 3 ranges: [6, 7],
  // [10, 11], [12, 14].

  ASSERT_EQ(ranges.size(), 3);
  EXPECT_EQ(ranges[0].start_time, 6);
  EXPECT_EQ(ranges[0].end_time, 7);
  EXPECT_EQ(ranges[1].start_time, 10);
  EXPECT_EQ(ranges[1].end_time, 11);
  EXPECT_EQ(ranges[2].start_time, 12);
  EXPECT_EQ(ranges[2].end_time, 14);
}

// Tests Case 4: Conditional input that aliases the output, merging them into a
// single continuous live range.
TEST_F(LiveRangeCalculatorTest, ConditionalInputAliasOutput) {
  const char* hlo_text = R"(
    HloModule Cond, is_scheduled=true
    true_comp (p3: (f32[2,3])) -> (f32[2,3]) {
      p3 = (f32[2,3]) parameter(0)
      gte1 = f32[2,3] get-tuple-element(p3), index=0
      update = f32[1,1] constant({{1}})
      index = s32[] constant(0)
      dus = f32[2,3] dynamic-update-slice(gte1, update, index, index)
      ROOT tuple2 = (f32[2,3]) tuple(dus)
    }
    false_comp (p4: (f32[2,3])) -> (f32[2,3]) {
      p4 = (f32[2,3]) parameter(0)
      gte2 = f32[2,3] get-tuple-element(p4), index=0
      update = f32[1,1] constant({{1}})
      index = s32[] constant(0)
      dus = f32[2,3] dynamic-update-slice(gte2, update, index, index)
      ROOT tuple3 = (f32[2,3]) tuple(dus)
    }
    ENTRY entry (p0: pred[], p1: f32[2,3]) -> f32[2,3] {
      p0 = pred[] parameter(0)
      p1 = f32[2,3] parameter(1)
      tanh0 = f32[2,3] tanh(p1)
      tuple0 = (f32[2,3]) tuple(tanh0)
      conditional = (f32[2,3]) conditional(p0, tuple0, tuple0), true_computation=true_comp, false_computation=false_comp
      gte3 = f32[2,3] get-tuple-element(conditional), index=0
      ROOT neg2 = f32[2,3] negate(gte3)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  const HloInstruction* cond_inst =
      FindInstruction(module.get(), "conditional");
  ASSERT_NE(cond_inst, nullptr);
  const HloBuffer* cond_output_buffer =
      GetBufferForInstruction(*alias_analysis, cond_inst, {0});
  ASSERT_EQ(tanh0_buffer, cond_output_buffer);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: tanh0
  // 3: tuple0
  // -- true_comp --
  // 4: p3
  // 5: gte1
  // 6: update (true)
  // 7: index (true)
  // 8: dus (true)
  // 9: tuple2
  // -- false_comp --
  // 10: p4
  // 11: gte2
  // 12: update (false)
  // 13: index (false)
  // 14: dus (false)
  // 15: tuple3
  // -- entry --
  // 16: conditional
  // 17: gte3
  // 18: neg2

  // - [2, 9] covers tanh0 definition to true_comp end.
  // - [10, 15] covers false_comp start to false_comp end.
  // - [16, 18] covers conditional output to neg2.
  ASSERT_EQ(ranges.size(), 3);
  EXPECT_EQ(ranges[0].start_time, 2);
  EXPECT_EQ(ranges[0].end_time, 9);
  EXPECT_EQ(ranges[1].start_time, 10);
  EXPECT_EQ(ranges[1].end_time, 15);
  EXPECT_EQ(ranges[2].start_time, 16);
  EXPECT_EQ(ranges[2].end_time, 18);
}

// Tests live range calculation for loop-carried parameters in a while loop,
// resulting in disjoint live ranges for loop iterations.
TEST_F(LiveRangeCalculatorTest, WhileLoop) {
  const char* hlo_text = R"(
    HloModule While, is_scheduled=true
    cond_comp (p0: (f32[2,3], s32[])) -> pred[] {
      p0 = (f32[2,3], s32[]) parameter(0)
      limit = s32[] constant(10)
      iteration = s32[] get-tuple-element(p0), index=1
      ROOT cond = pred[] compare(iteration, limit), direction=LT
    }
    body_comp (p1: (f32[2,3], s32[])) -> (f32[2,3], s32[]) {
      p1 = (f32[2,3], s32[]) parameter(0)
      val = f32[2,3] get-tuple-element(p1), index=0
      iteration = s32[] get-tuple-element(p1), index=1
      one = s32[] constant(1)
      new_iteration = s32[] add(iteration, one)
      new_val = f32[2,3] negate(val)
      ROOT tuple2 = (f32[2,3], s32[]) tuple(new_val, new_iteration)
    }
    ENTRY entry (p2: f32[2,3], p3: s32[]) -> (f32[2,3], s32[]) {
      p2 = f32[2,3] parameter(0)
      p3 = s32[] parameter(1)
      tanh0 = f32[2,3] tanh(p2)
      init_tuple = (f32[2,3], s32[]) tuple(tanh0, p3)
      ROOT while_loop = (f32[2,3], s32[]) while(init_tuple), condition=cond_comp, body=body_comp
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p2
  // 1: p3
  // 2: tanh0
  // 3: init_tuple
  // -- cond_comp --
  // 4: p0
  // 5: limit
  // 6: iteration
  // 7: cond
  // -- body_comp --
  // 8: p1
  // 9: val
  // 10: iteration
  // 11: one
  // 12: new_iteration
  // 13: new_val
  // 14: tuple2
  // -- entry --
  // 15: while_loop

  // Expected ranges:
  // - tanh0 (def 2) to start of loop comps (4) -> [2, 4]
  // - p0 index 0 (def 4) -> [4, 4] (no uses)
  // - p1 index 0 (def 8) to new_val (13) -> [8, 13]
  // - new_val (def 13) to tuple2 (14) -> [13, 14]
  // - while_loop output element 0 (def 15) -> [15, 15]
  // Merged: [2, 4], [8, 14], [15, 15]

  ASSERT_EQ(ranges.size(), 3);
  EXPECT_EQ(ranges[0].start_time, 2);
  EXPECT_EQ(ranges[0].end_time, 7);
  EXPECT_EQ(ranges[1].start_time, 8);
  EXPECT_EQ(ranges[1].end_time, 14);
  EXPECT_EQ(ranges[2].start_time, 15);
  EXPECT_EQ(ranges[2].end_time, 15);
}

// Tests live range calculation for async instructions, ensuring the input
// buffer remains live from async-start until async-done.
TEST_F(LiveRangeCalculatorTest, AsyncInstruction) {
  const char* hlo_text = R"(
    HloModule Async, is_scheduled=true
    async_comp (p0: f32[2,3]) -> f32[2,3] {
      p0 = f32[2,3] parameter(0)
      ROOT neg0 = f32[2,3] negate(p0)
    }
    ENTRY entry (p1: f32[2,3]) -> f32[2,3] {
      p1 = f32[2,3] parameter(0)
      tanh0 = f32[2,3] tanh(p1)
      async_start = ((f32[2,3]), f32[2,3], s32[]) async-start(tanh0), calls=async_comp
      async_done = f32[2,3] async-done(async_start)
      ROOT neg1 = f32[2,3] negate(async_done)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p1
  // 1: tanh0
  // 2: p0
  // 3: neg0
  // 4: async_start
  // 5: async_done
  // 6: neg1

  // The input buffer must remain live from tanh0 (1) until async_done (5)
  // completes.
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start_time, 1);
  EXPECT_EQ(ranges[0].end_time, 5);
}

TEST_F(LiveRangeCalculatorTest, NestedConditional) {
  const char* hlo_text = R"(
    HloModule NestedCond, is_scheduled=true
    inner_true (p5: (f32[2,3])) -> (f32[2,3]) {
      p5 = (f32[2,3]) parameter(0)
      gte_in = f32[2,3] get-tuple-element(p5), index=0
      neg_in = f32[2,3] negate(gte_in)
      ROOT tuple_in = (f32[2,3]) tuple(neg_in)
    }

    inner_false (p6: (f32[2,3])) -> (f32[2,3]) {
      p6 = (f32[2,3]) parameter(0)
      gte_in2 = f32[2,3] get-tuple-element(p6), index=0
      abs_in = f32[2,3] abs(gte_in2)
      ROOT tuple_in2 = (f32[2,3]) tuple(abs_in)
    }

    outer_true (p3: (f32[2,3], pred[])) -> (f32[2,3]) {
      p3 = (f32[2,3], pred[]) parameter(0)
      gte_out = f32[2,3] get-tuple-element(p3), index=0
      pred_out = pred[] get-tuple-element(p3), index=1
      tuple_out = (f32[2,3]) tuple(gte_out)
      cond_in = (f32[2,3]) conditional(pred_out, tuple_out, tuple_out), true_computation=inner_true, false_computation=inner_false
      gte_out2 = f32[2,3] get-tuple-element(cond_in), index=0
      ROOT tuple_out2 = (f32[2,3]) tuple(gte_out2)
    }

    outer_false (p4: (f32[2,3])) -> (f32[2,3]) {
      p4 = (f32[2,3]) parameter(0)
      gte_out3 = f32[2,3] get-tuple-element(p4), index=0
      neg_out = f32[2,3] negate(gte_out3)
      ROOT tuple_out3 = (f32[2,3]) tuple(neg_out)
    }

    ENTRY entry (p0: pred[], p1: f32[2,3], p2: pred[]) -> f32[2,3] {
      p0 = pred[] parameter(0)
      p1 = f32[2,3] parameter(1)
      p2 = pred[] parameter(2)
      tanh0 = f32[2,3] tanh(p1)
      tuple0 = (f32[2,3], pred[]) tuple(tanh0, p2)
      tuple1 = (f32[2,3]) tuple(tanh0)
      conditional = (f32[2,3]) conditional(p0, tuple0, tuple1), true_computation=outer_true, false_computation=outer_false
      gte_entry = f32[2,3] get-tuple-element(conditional), index=0
      ROOT neg_entry = f32[2,3] negate(gte_entry)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_TRUE(module->has_schedule());

  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  ASSERT_OK_AND_ASSIGN(auto inst_to_index,
                       GetInstToIndex(module.get(), *alias_analysis));

  const HloInstruction* tanh0_inst = FindInstruction(module.get(), "tanh0");
  ASSERT_NE(tanh0_inst, nullptr);
  const HloBuffer* tanh0_buffer =
      GetBufferForInstruction(*alias_analysis, tanh0_inst);
  ASSERT_NE(tanh0_buffer, nullptr);

  LiveRangeCalculator calculator(*tanh0_buffer, inst_to_index);
  auto ranges = calculator.CalculateBufferLiveRange().ranges;

  // Expected flattened schedule:
  // 0: p0
  // 1: p1
  // 2: p2
  // 3: tanh0
  // 4: tuple0
  // 5: tuple1
  // -- outer_true --
  // 6: p3
  // 7: gte_out
  // 8: pred_out
  // 9: tuple_out
  // -- inner_true --
  // 10: p5
  // 11: gte_in
  // 12: neg_in
  // 13: tuple_in
  // -- inner_false --
  // 14: p6
  // 15: gte_in2
  // 16: abs_in
  // 17: tuple_in2
  // -- outer_true (cont) --
  // 18: cond_in
  // 19: gte_out2
  // 20: tuple_out2
  // -- outer_false --
  // 21: p4
  // 22: gte_out3
  // 23: neg_out
  // 24: tuple_out3
  // -- entry (cont) --
  // 25: conditional
  // 26: gte_entry
  // 27: neg_entry

  // tanh0 buffer should be live:
  // - in entry: [3, 6] (tanh0 (3) -> conditional (25), earliest called comp is
  // outer_true (6))
  // - in outer_true: [6, 10] (p3 (6) -> cond_in (18), earliest called comp is
  // inner_true (10))
  // - in inner_true: [10, 12] (p5 (10) -> neg_in (12))
  // - in inner_false: [14, 16] (p6 (14) -> abs_in (16))
  // - in outer_false: [21, 23] (p4 (21) -> neg_out (23))
  // Merged ranges: [3, 12], [14, 16], [21, 23]
  ASSERT_EQ(ranges.size(), 3);
  EXPECT_EQ(ranges[0].start_time, 3);
  EXPECT_EQ(ranges[0].end_time, 12);
  EXPECT_EQ(ranges[1].start_time, 14);
  EXPECT_EQ(ranges[1].end_time, 16);
  EXPECT_EQ(ranges[2].start_time, 21);
  EXPECT_EQ(ranges[2].end_time, 23);
}

}  // namespace
}  // namespace xla

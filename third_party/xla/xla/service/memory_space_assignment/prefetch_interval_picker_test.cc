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

#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/testing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {
namespace {

using CostAnalysisPrefetchIntervalPickerTest = HloTestBase;

TEST_F(CostAnalysisPrefetchIntervalPickerTest, PrefetchIntervalOrder) {
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] negate(param0)
    b = f32[2,4] negate(a)
    c = f32[2,4] negate(b)
    d = f32[2,4] negate(c)
    e = f32[2,4] negate(d)
    f = f32[2,4] negate(e)
    g = f32[2,4] negate(f)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[2,4] negate(n)
    p = f32[2,4] negate(o)
    q = f32[2,4] negate(p)
    r = f32[2,4] negate(q)
    s = f32[2,4] negate(r)
    t = f32[2,4] negate(s)
    u = f32[2,4] negate(t)
    ROOT v = f32[2,4] add(u, param0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis;
  CostAnalysisOptions options;
  HloCostAnalysisCosts hlo_cost_analysis_costs(hlo_cost_analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cost_analysis,
      FakeCostAnalysis::Create(hlo_cost_analysis_costs, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/4.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/22, std::nullopt);

  // Expect that the first interval is (15, 22), which has elapsed time of 6.0,
  // twice of the async copy elased (3.0). Then we expect that intervals will be
  // visited in alternating increasing and decreasing orders until hitting the
  // min and max async copy overlap ratios, which are the intervals (18, 22)
  // and (9, 22) respectively.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 15);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 16);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 14);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 17);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 13);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 18);  // Min async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 12);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 11);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 10);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 9);  // Max async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());

  // Expect that if the time between start_time and end_time is too short, there
  // won't be any available intervals.
  interval_picker.Begin(use, /*start_time=*/19, /*end_time=*/22, std::nullopt);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, PrefetchIntervalOrderWhile) {
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4]) parameter(0)    // 19
    ROOT cond = pred[] constant(true)   // 20
  }

  while_body {
    param2 = (f32[2,4]) parameter(0)    // 21
    gte2 = f32[2,4] get-tuple-element(param2), index=0  // 22
    add = f32[2,4] add(gte2, gte2)      // 23
    ROOT tuple2 = (f32[2,4]) tuple(add) // 24
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)  // 0
    a = f32[2,4] negate(param0)     // 1
    b = f32[2,4] negate(a)          // 2
    c = f32[2,4] negate(b)          // 3
    d = f32[2,4] negate(c)          // 4
    e = f32[2,4] negate(d)          // 5
    f = f32[2,4] negate(e)          // 6
    g = f32[2,4] negate(f)          // 7
    h = f32[2,4] negate(g)          // 8
    i = f32[2,4] negate(h)          // 9
    j = f32[2,4] negate(i)          // 10
    k = f32[2,4] negate(j)          // 11
    l = f32[2,4] negate(k)          // 12
    m = f32[2,4] negate(l)          // 13
    n = f32[2,4] negate(m)          // 14
    o = f32[2,4] negate(n)          // 15
    p = f32[2,4] negate(o)          // 16
    q = f32[2,4] negate(p)          // 17
    tuple = (f32[2,4]) tuple(q)     // 18
    while = (f32[2,4]) while(tuple), condition=while_condition, body=while_body  // 25
    gte1 = f32[2,4] get-tuple-element(while), index=0  // 26
    r = f32[2,4] negate(gte1)       // 27
    s = f32[2,4] negate(r)          // 28
    t = f32[2,4] negate(s)          // 29
    u = f32[2,4] negate(t)          // 30
    ROOT v = f32[2,4] add(u, param0)  // 31
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis;
  CostAnalysisOptions options;
  HloCostAnalysisCosts hlo_cost_analysis_costs(hlo_cost_analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cost_analysis,
      FakeCostAnalysis::Create(hlo_cost_analysis_costs, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  EXPECT_EQ(cost_analysis->GetWhileNestMultiplier(1), 5.0);
  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/31, std::nullopt);

  // Because there are while loop computations between [19, 24], we ensure that
  // the interval picker avoids this interval.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 25);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 26);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 18);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 27);  // Min async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 17);  // Max async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, NestedWhile) {
  // This test is to check against a bug where we didn't assign
  // while_nest_level_ for while instructions, and defaulting to 0. This could
  // cause the prefetch interval logic to think a nested while instruction is
  // the same level as the outermost computation.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition.2 {
    param1 = (f32[2,4]) parameter(0)    // 11
    ROOT cond = pred[] constant(true)   // 12
  }

  while_body.2 {
    param2 = (f32[2,4]) parameter(0)    // 13
    gte2 = f32[2,4] get-tuple-element(param2), index=0  // 14
    add = f32[2,4] add(gte2, gte2)      // 15
    ROOT tuple2 = (f32[2,4]) tuple(add) // 16
  }

  while_condition.1 {
    param3 = (f32[2,4]) parameter(0)    // 5
    ROOT cond = pred[] constant(true)   // 6
  }

  while_body.1 {
    param4 = (f32[2,4]) parameter(0)    // 7
    gte1 = f32[2,4] get-tuple-element(param4), index=0  // 8
    add1 = f32[2,4] add(gte1, gte1)     // 9
    tuple1 = (f32[2,4]) tuple(add1)     // 10
    while = (f32[2,4]) while(tuple1), condition=while_condition.2, body=while_body.2  // 17
    gte2 = f32[2,4] get-tuple-element(while), index=0  // 18
    add2 = f32[2,4] add(gte2, gte2)     // 19
    ROOT tuple2 = (f32[2,4]) tuple(add2)  // 20
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)  // 0
    a = f32[2,4] negate(param0)     // 1
    b = f32[2,4] negate(a)          // 2
    c = f32[2,4] negate(b)          // 3
    tuple = (f32[2,4]) tuple(c)     // 4
    while = (f32[2,4]) while(tuple), condition=while_condition.1, body=while_body.1  // 21
    gte1 = f32[2,4] get-tuple-element(while), index=0  // 22
    ROOT root = f32[2,4] add(gte1, param0)  // 23
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis;
  CostAnalysisOptions options;
  HloCostAnalysisCosts hlo_cost_analysis_costs(hlo_cost_analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cost_analysis,
      FakeCostAnalysis::Create(hlo_cost_analysis_costs, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  const Shape& shape = root->operand(1)->shape();

  // We expect the root's latest prefetch start time to be before the while loop
  // (logical time 4).
  EXPECT_EQ(interval_picker.LatestPrefetchStartTime(shape, /*start_time=*/0,
                                                    /*end_time=*/23, &use),
            4);
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, ConsecutiveConditionals) {
  // This is a test for b/170668492, where prefetching for consecutive
  // conditionals can cause the prefetch to start in the conditional's
  // computation.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  true_computation.0 {
    p0 = (f32[3]{0}) parameter(0)                   // 5
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 6
    ROOT neg1 = f32[3]{0} negate(gte)               // 7
  }

  false_computation.0 {
    p0 = (f32[3]{0}) parameter(0)                   // 8
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 9
    ROOT neg2 = f32[3]{0} negate(gte)               // 10
  }

  true_computation.1 {
    p0 = (f32[3]{0}) parameter(0)                   // 12
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 13
    ROOT neg1 = f32[3]{0} negate(gte)               // 14
  }

  false_computation.1 {
    p0 = (f32[3]{0}) parameter(0)                   // 15
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 16
    ROOT neg2 = f32[3]{0} negate(gte)               // 17
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)       // 0
    p1 = f32[3]{0} parameter(1)       // 1
    p2 = pred[] parameter(2)          // 2
    tuple0 = (f32[3]{0}) tuple(p0)    // 3
    tuple1 = (f32[3]{0}) tuple(p1)    // 4
    conditional0 = f32[3]{0} conditional(p2, tuple0, tuple0), true_computation=true_computation.0, false_computation=false_computation.0  // 11
    conditional1 = f32[3]{0} conditional(p2, tuple1, tuple1), true_computation=true_computation.1, false_computation=false_computation.1  // 18
    ROOT tuple2 = (f32[3]{0}, f32[3]{0}) tuple(conditional0, conditional1)  // 19
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis;
  CostAnalysisOptions options;
  HloCostAnalysisCosts hlo_cost_analysis_costs(hlo_cost_analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cost_analysis,
      FakeCostAnalysis::Create(hlo_cost_analysis_costs, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  LOG(INFO) << module->ToString();

  HloInstruction* conditional1 =
      module->entry_computation()->GetInstructionWithName("conditional1");
  const HloUse use{conditional1, /*operand_number=*/1, /*operand_index=*/{0}};
  const Shape& shape =
      module->entry_computation()->parameter_instruction(0)->shape();

  // Expect that the prefetch to start before conditional0's called
  // computations.
  EXPECT_LT(interval_picker.LatestPrefetchStartTime(shape, /*start_time=*/0,
                                                    /*end_time=*/11, &use),
            5);
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, EarliestLatestWindowTooSmall) {
  // This tests the scenario where there is an op that takes a long time (tanh
  // in this example) and as a result the earliest and latest times both fall
  // inside this long-running op. In this case, we should still return a valid
  // prefetch interval just before the long-running op.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    negate = f32[2,4] negate(param0)
    tanh = f32[2,4] tanh(param0)
    ROOT add = f32[2,4] add(tanh, negate)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis;
  CostAnalysisOptions options;
  HloCostAnalysisCosts hlo_cost_analysis_costs(hlo_cost_analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto cost_analysis,
      FakeCostAnalysis::Create(hlo_cost_analysis_costs, *module, options));
  cost_analysis->SetOverrideForGetInstructionElapsed(
      [](const HloInstruction& hlo) {
        if (hlo.opcode() == HloOpcode::kTanh) {
          return 20.0;
        }
        return 1.0;
      });
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/1, /*end_time=*/3, std::nullopt);

  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_FALSE(interval_picker.Done());
  EXPECT_EQ(interval_picker.Next(), 1);
  EXPECT_TRUE(interval_picker.Done());
}

}  // namespace
}  // namespace memory_space_assignment
}  // namespace xla

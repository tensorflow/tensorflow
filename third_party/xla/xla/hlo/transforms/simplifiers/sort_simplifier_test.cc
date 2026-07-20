/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"

#include <cstdint>

#include "absl/status/status_matchers.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
namespace m = match;

using SortSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(SortSimplifierTest, RemoveUnusedSortOperandArrayResult) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} parameter(1)
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  uint64_t num_executions = 0;
  do {
    num_executions++;
  } while (simplifier.Run(module.get()).value());
  EXPECT_EQ(num_executions, 2);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Sort(m::Parameter(0))));
}

TEST_F(SortSimplifierTest, RemoveUnusedSortOperandTuple) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     p.2.lhs = u32[] parameter(4)
     p.2.rhs = u32[] parameter(5)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,87] parameter(0)
     values.0 = s32[64,87] parameter(1)
     values.1 = u32[64,87] parameter(2)
     sort = (f32[64,87], s32[64,87], u32[64,87]) sort(
         keys, values.0, values.1),
       dimensions={1}, to_apply=compare
     gte.0 = f32[64,87] get-tuple-element(sort), index=0
     gte.1 = u32[64,87] get-tuple-element(sort), index=2
     ROOT tuple = (f32[64,87], u32[64,87]) tuple(gte.0, gte.1)
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Sort(m::Parameter(0), m::Parameter(2)), 0),
          m::GetTupleElement(m::Sort(m::Parameter(0), m::Parameter(2)), 1))));
}

TEST_F(SortSimplifierTest, DontRemoveUnusedSortKey) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} parameter(1)
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=compare
     ROOT gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(SortSimplifierTest, RemoveUnusedFirstOperand) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.1.lhs, p.1.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} parameter(1)
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare
     ROOT gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  uint64_t num_executions = 0;
  do {
    num_executions++;
  } while (simplifier.Run(module.get()).value());
  EXPECT_EQ(num_executions, 2);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Sort(m::Parameter(1))));
}

TEST_F(SortSimplifierTest, DontRemoveUnusedSortOperandWhenSortIsRoot) {
  const char* hlo_string = R"(
   HloModule sort_root

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64] parameter(0)
     values = s32[64] parameter(1)
     ROOT sort = (f32[64], s32[64]) sort(keys, values),
       dimensions={0}, to_apply=compare
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(SortSimplifierTest, DoesNotRemoveOldComparatorWhenShared) {
  const char* hlo_string = R"(
   HloModule shared_comparator

   // CHECK-DAG: %compare ({{.*}}p.0.lhs: f32[], {{.*}}p.0.rhs: f32[], {{.*}}p.1.lhs: s32[], {{.*}}p.1.rhs: s32[]) -> pred[]
   // CHECK-DAG: %compare.clone ({{.*}}) -> pred[]
   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   // CHECK: ENTRY %sort_computation
   ENTRY sort_computation {
     keys1 = f32[64] parameter(0)
     values1 = s32[64] parameter(1)
     // CHECK: %[[NEW_SORT:.*]] = f32[64]{{.*}}sort(%keys1), dimensions={0}, to_apply=%compare.clone
     sort1 = (f32[64], s32[64]) sort(keys1, values1),
       dimensions={0}, to_apply=compare
     gte1 = f32[64] get-tuple-element(sort1), index=0

     keys2 = f32[64] parameter(2)
     values2 = s32[64] parameter(3)
     // CHECK: %sort2 = (f32[64]{{.*}}, s32[64]{{.*}}) sort(%keys2, %values2), dimensions={0}, to_apply=%compare
     sort2 = (f32[64], s32[64]) sort(keys2, values2),
       dimensions={0}, to_apply=compare
     gte2.0 = f32[64] get-tuple-element(sort2), index=0
     gte2.1 = s32[64] get-tuple-element(sort2), index=1

     // CHECK: ROOT %tuple = (f32[64]{{.*}}, f32[64]{{.*}}, s32[64]{{.*}}) tuple(%[[NEW_SORT]], %sort2#0, %sort2#1)
     ROOT tuple = (f32[64], f32[64], s32[64]) tuple(gte1, gte2.0, gte2.1)
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}
}  // namespace
}  // namespace xla

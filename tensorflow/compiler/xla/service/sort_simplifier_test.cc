/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sort_simplifier.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

using SortSimplifierTest = HloTestBase;

TEST_F(SortSimplifierTest, RemoveUnusedSortOperandArrayResult) {
  const char* hlo_string = R"(
   HloModule permutation_sort

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} parameter(1)
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
        dimensions={1}
      ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  uint64 num_executions = 0;
  do {
    num_executions++;
  } while (simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(num_executions, 2);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Sort(m::Parameter(0))));
}

TEST_F(SortSimplifierTest, RemoveUnusedSortOperandTuple) {
  const char* hlo_string = R"(
   HloModule permutation_sort

    ENTRY sort_computation {
      keys = f32[64,87] parameter(0)
      values.0 = s32[64,87] parameter(1)
      values.1 = u32[64,87] parameter(2)
      sort = (f32[64,87], s32[64,87], u32[64,87]) sort(
          keys, values.0, values.1),
        dimensions={1}
      gte.0 = f32[64,87] get-tuple-element(sort), index=0
      gte.1 = u32[64,87] get-tuple-element(sort), index=2
      ROOT tuple = (f32[64,87], u32[64,87]) tuple(gte.0, gte.1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
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

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} parameter(1)
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}
      ROOT gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SortSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}
}  // namespace
}  // namespace xla

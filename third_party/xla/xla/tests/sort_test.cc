/* Copyright 2018 The OpenXLA Authors.

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

#include <cstddef>

#include <gtest/gtest.h>
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

using SortOpTest = HloTestBase;

TEST_F(SortOpTest, SimpleSort) {
  const char* hlo_string = R"(
   HloModule simple_sort

   compare {
     p0 = f32[] parameter(0)
     p1 = f32[] parameter(1)
     ROOT lt = pred[] compare(p0, p1), direction=LT
   }

   ENTRY sort_computation {
     param = f32[5]{0} constant({3, 1, 4, 1, 5})
     ROOT sorted = f32[5]{0} sort(param), dimensions={0}, to_apply=compare
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto root = module->entry_computation()->root_instruction();

  EXPECT_EQ(root->opcode(), HloOpcode::kSort);
  EXPECT_THAT(root, GmockMatch(m::Sort(m::Constant())));

  // Evaluate the module to get the sorted result.
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(auto result, evaluator.Evaluate(*module, {}));

  // Get the sorted values from result
  auto sorted = result.data<float>();

  // Verify that the result is sorted
  for (size_t i = 0; i < sorted.size() - 1; ++i) {
    EXPECT_LE(sorted[i], sorted[i + 1]);
  }
}

}  // namespace
}  // namespace xla

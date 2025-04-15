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

#include "xla/hlo/transforms/collectives/all_reduce_contiguous.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

using AllReduceContiguousTest = HloHardwareIndependentTestBase;

TEST_F(AllReduceContiguousTest, Simple) {
  const absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[128] parameter(0)
  p1 = f32[4,4] parameter(1)
  ROOT crs = (f32[128], f32[4,4]) all-reduce(p0, p1), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceContiguous pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  auto crs =
      AllOf(op::Shape("f32[144]"),
            op::AllReduce(op::Concatenate(op::Bitcast(op::Parameter(0)),
                                          op::Bitcast(op::Parameter(1)))));
  ASSERT_THAT(
      root,
      op::Tuple(AllOf(op::Shape("f32[128]"), op::Bitcast(op::Slice(crs))),
                AllOf(op::Shape("f32[4,4]"), op::Bitcast(op::Slice(crs)))));

  EXPECT_EQ(root->operand(0)->operand(0)->slice_starts(0), 0);
  EXPECT_EQ(root->operand(0)->operand(0)->slice_limits(0), 128);
  EXPECT_EQ(root->operand(1)->operand(0)->slice_starts(0), 128);
  EXPECT_EQ(root->operand(1)->operand(0)->slice_limits(0), 128 + 4 * 4);
}

}  // namespace
}  // namespace xla

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

#include "xla/hlo/transforms/collectives/all_to_all_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using AllToAllDecomposerTest = HloHardwareIndependentTestBase;

constexpr absl::string_view kArrayAllToAll = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,8] parameter(0)
  ROOT a2a = f32[4,8] all-to-all(p0), replica_groups={{0,1}}, dimensions={0},
    metadata={op_name="a2a_op"}, sharding={replicated}
})";

TEST_F(AllToAllDecomposerTest, DecomposeToTuple) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kArrayAllToAll));
  AllToAllDecomposer pass(/*decompose_to_tuple=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Concatenate(
                  op::GetTupleElement(op::AllToAll(op::Slice(op::Parameter(0)),
                                                   op::Slice(op::Parameter(0))),
                                      0),
                  op::GetTupleElement(op::AllToAll(), 1)));
}

TEST_F(AllToAllDecomposerTest, DecomposeToTuplePropagatesDerivedState) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kArrayAllToAll));
  AllToAllDecomposer pass(/*decompose_to_tuple=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* concat =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(concat->opcode(), HloOpcode::kConcatenate);
  EXPECT_EQ(concat->metadata().op_name(), "a2a_op");
  ASSERT_TRUE(concat->has_sharding());
  EXPECT_TRUE(concat->sharding().IsReplicated());

  ASSERT_EQ(concat->operand_count(), 2);
  for (const HloInstruction* gte : concat->operands()) {
    ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    // Each GTE must be set up as a derived instruction of the original
    // all-to-all, i.e. inherit its metadata and (array) sharding.
    EXPECT_EQ(gte->metadata().op_name(), "a2a_op");
    ASSERT_TRUE(gte->has_sharding());
    EXPECT_TRUE(gte->sharding().IsReplicated());
  }

  const HloInstruction* tuple_all_to_all = concat->operand(0)->operand(0);
  ASSERT_EQ(tuple_all_to_all->opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(tuple_all_to_all->metadata().op_name(), "a2a_op");
  for (const HloInstruction* slice : tuple_all_to_all->operands()) {
    EXPECT_EQ(slice->metadata().op_name(), "a2a_op");
  }
}

TEST_F(AllToAllDecomposerTest, DecomposeToMinArrayRank) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kArrayAllToAll));
  AllToAllDecomposer pass(/*decompose_to_tuple=*/false, /*min_array_rank=*/4);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::AllToAll(op::Reshape(op::Parameter(0)))));
  EXPECT_EQ(root->operand(0)->shape().dimensions().size(), 4);
  EXPECT_EQ(root->metadata().op_name(), "a2a_op");
}

}  // namespace
}  // namespace xla

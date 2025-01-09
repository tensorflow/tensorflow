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

#include "xla/service/gpu/reduction_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

using ReductionUtilsTest = HloTestBase;

const char kModulePrefix[] = R"(
    HloModule test_module
    scalar_add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    })";

TEST_F(ReductionUtilsTest, ReductionsAreMultioutputFusionCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_sibling1 {
      p_0 = f32[32,64]{1,0} parameter(0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(p_0, constant), dimensions={1}, to_apply=scalar_add
    }

    fused_sibling2 {
      p_0 = f32[32,64]{1,0} parameter(0)
      neg = f32[32,64]{1,0} negate(p_0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(neg, constant), dimensions={1}, to_apply=scalar_add
    }

    ENTRY entry {
      p_0 = f32[32,64]{1,0} parameter(0)
      fusion1 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling1
      fusion2 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling2
      ROOT root = (f32[32]{0}, f32[32]{0}) tuple(fusion1, fusion2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion1 = root->operand(0);
  const HloInstruction* fusion2 = root->operand(1);
  EXPECT_TRUE(AreReductionsMultiOutputFusionCompatible(
      fusion1->fused_expression_root(), fusion2->fused_expression_root()));
}

TEST_F(ReductionUtilsTest,
       ReductionsWithSameCanonicalizedDimsAreMultioutputFusionCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_sibling1 {
      p_0 = f32[32,64]{1,0} parameter(0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(p_0, constant), dimensions={1}, to_apply=scalar_add
    }

    fused_sibling2 {
      p_0 = f32[32,64]{1,0} parameter(0)
      bitcast = f32[32,8,8]{2,1,0} bitcast(p_0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(bitcast, constant), dimensions={1,2}, to_apply=scalar_add
    }

    ENTRY entry {
      p_0 = f32[32,64]{1,0} parameter(0)
      fusion1 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling1
      fusion2 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling2
      ROOT root = (f32[32]{0}, f32[32]{0}) tuple(fusion1, fusion2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion1 = root->operand(0);
  const HloInstruction* fusion2 = root->operand(1);
  EXPECT_TRUE(AreReductionsMultiOutputFusionCompatible(
      fusion1->fused_expression_root(), fusion2->fused_expression_root()));
}

TEST_F(ReductionUtilsTest,
       ReductionsAreNotMultioutputFusionCompatible_DifferentOperandShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_sibling1 {
      p_0 = f32[32,64]{1,0} parameter(0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(p_0, constant), dimensions={1}, to_apply=scalar_add
    }

    fused_sibling2 {
      p_0 = f32[64,32]{1,0} parameter(0)
      neg = f32[64,32]{1,0} negate(p_0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(neg, constant), dimensions={0}, to_apply=scalar_add
    }

    ENTRY entry {
      p_0 = f32[32,64]{1,0} parameter(0)
      p_1 = f32[64,32]{1,0} parameter(1)
      fusion1 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling1
      fusion2 = f32[32]{0} fusion(p_1), kind=kInput, calls=fused_sibling2
      ROOT root = (f32[32]{0}, f32[32]{0}) tuple(fusion1, fusion2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion1 = root->operand(0);
  const HloInstruction* fusion2 = root->operand(1);
  EXPECT_FALSE(AreReductionsMultiOutputFusionCompatible(
      fusion1->fused_expression_root(), fusion2->fused_expression_root()));
}

TEST_F(ReductionUtilsTest,
       ReductionsAreNotMultioutputFusionCompatible_DifferentOutputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_sibling1 {
      p_0 = f32[32,64]{1,0} parameter(0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(p_0, constant), dimensions={1}, to_apply=scalar_add
    }

    fused_sibling2 {
      p_0 = f32[64,32]{1,0} parameter(0)
      neg = f32[64,32]{1,0} negate(p_0)
      constant = f32[] constant(0)
      ROOT reduce = f32[64]{0} reduce(neg, constant), dimensions={1}, to_apply=scalar_add
    }

    ENTRY entry {
      p_0 = f32[32,64]{1,0} parameter(0)
      p_1 = f32[64,32]{1,0} parameter(1)
      fusion1 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling1
      fusion2 = f32[64]{0} fusion(p_1), kind=kInput, calls=fused_sibling2
      ROOT root = (f32[32]{0}, f32[64]{0}) tuple(fusion1, fusion2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion1 = root->operand(0);
  const HloInstruction* fusion2 = root->operand(1);
  EXPECT_FALSE(AreReductionsMultiOutputFusionCompatible(
      fusion1->fused_expression_root(), fusion2->fused_expression_root()));
}

TEST_F(ReductionUtilsTest,
       ReductionsAreNotMultioutputFusionCompatible_DifferentReduceDimensions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_sibling1 {
      p_0 = f32[32,32]{1,0} parameter(0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(p_0, constant), dimensions={0}, to_apply=scalar_add
    }

    fused_sibling2 {
      p_0 = f32[32,32]{1,0} parameter(0)
      neg = f32[32,32]{1,0} negate(p_0)
      constant = f32[] constant(0)
      ROOT reduce = f32[32]{0} reduce(neg, constant), dimensions={1}, to_apply=scalar_add
    }

    ENTRY entry {
      p_0 = f32[32,32]{1,0} parameter(0)
      fusion1 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling1
      fusion2 = f32[32]{0} fusion(p_0), kind=kInput, calls=fused_sibling2
      ROOT root = (f32[32]{0}, f32[32]{0}) tuple(fusion1, fusion2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion1 = root->operand(0);
  const HloInstruction* fusion2 = root->operand(1);
  EXPECT_FALSE(AreReductionsMultiOutputFusionCompatible(
      fusion1->fused_expression_root(), fusion2->fused_expression_root()));
}

TEST(ReductionDimensionsTest, GetOutputShape) {
  ReductionDimensions row_reduction{/*is_row_reduction=*/true, {1, 2, 3}};
  ReductionDimensions col_reduction{/*is_row_reduction=*/false, {1, 2, 3}};

  EXPECT_THAT(row_reduction.GetOutputShape(), ElementsAre(2));
  EXPECT_THAT(col_reduction.GetOutputShape(), ElementsAre(1, 3));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/onehot_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

class OneHotGatherRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[32,8,16]{2,1,0} parameter(0)
  %weights = bf16[1024,2,3072] parameter(1)

  // One-hot logic approximation for test
  %iota = s32[1024] iota(), iota_dimension=0
  
  // Full shapes: [32,8,16,1024]
  %indices_broadcast = s32[32,8,16,1024]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[32,8,16,1024]{3,2,1,0} broadcast(%iota), dimensions={3}
  
  %compare = pred[32,8,16,1024] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[32,8,16,1024] convert(%compare)
  
  ROOT %dot = bf16[32,8,16,2,3072] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(
      Match(root, m::Gather(m::Parameter(1), m::Reshape(m::Parameter(0)))));

  auto* gather = Cast<HloGatherInstruction>(root);
  const auto& dnums = gather->gather_dimension_numbers();
  EXPECT_THAT(dnums.offset_dims(), testing::ElementsAre(3, 4));
  EXPECT_THAT(dnums.collapsed_slice_dims(), testing::ElementsAre(0));
  EXPECT_THAT(dnums.start_index_map(), testing::ElementsAre(0));
  EXPECT_EQ(dnums.index_vector_dim(), 3);
}

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGatherWithCall) {
  absl::string_view hlo_string = R"(
HloModule module

%_one_hot_1296.528 (Arg_0.320: s32[32,8,16]) -> pred[32,8,16,1024] {
  %Arg_0.320 = s32[32,8,16]{2,1,0} parameter(0)
  %broadcast_in_dim.2564 = s32[32,8,16,1]{3,2,1,0} reshape(%Arg_0.320)
  %eq.327 = s32[32,8,16,1]{3,2,1,0} broadcast(%broadcast_in_dim.2564), dimensions={0,1,2,3}
  %eq.328 = s32[32,8,16]{2,1,0} reshape(%eq.327)
  %eq.329 = s32[32,8,16,1024]{3,2,1,0} broadcast(%eq.328), dimensions={0,1,2}
  %iota.205 = s32[1024]{0} iota(), iota_dimension=0
  %iota.206 = s32[1,1,1,1024]{3,2,1,0} reshape(%iota.205)
  %eq.330 = s32[1,1,1,1024]{3,2,1,0} broadcast(%iota.206), dimensions={0,1,2,3}
  %eq.331 = s32[1024]{0} reshape(%eq.330)
  %eq.332 = s32[32,8,16,1024]{3,2,1,0} broadcast(%eq.331), dimensions={3}
  ROOT %eq.333 = pred[32,8,16,1024]{3,2,1,0} compare(%eq.329, %eq.332), direction=EQ
}

ENTRY %main {
  %jit__where_.369 = s32[32,8,16]{2,1,0} parameter(0)
  %jit__one_hot_.3 = pred[32,8,16,1024]{3,2,1,0} call(%jit__where_.369), to_apply=%_one_hot_1296.528
  %dot_general.206 = bf16[32,8,16,1024]{3,2,1,0} convert(%jit__one_hot_.3)
  %Arg_9.23 = bf16[1024,2,8,384]{3,2,1,0} parameter(2)
  %reshape.520 = bf16[1024,2,3072]{2,1,0} reshape(%Arg_9.23)
  %dot_general.207 = bf16[32,8,16,2,3072]{4,3,2,1,0} dot(%dot_general.206, %reshape.520), lhs_contracting_dims={3}, rhs_contracting_dims={0}
  %dot_general.201 = bf16[32,8,16,1024]{3,2,1,0} convert(%jit__one_hot_.3)
  %Arg_8.23 = bf16[1024,2,8,384]{3,2,1,0} parameter(1)
  %reshape.519 = bf16[1024,2,3072]{2,1,0} reshape(%Arg_8.23)
  %dot_general.202 = bf16[32,8,16,2,3072]{4,3,2,1,0} dot(%dot_general.201, %reshape.519), lhs_contracting_dims={3}, rhs_contracting_dims={0}
  ROOT %tuple.208 = tuple(%dot_general.207, %dot_general.202)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  ASSERT_TRUE(Match(root->operand(0), m::Gather()));
  ASSERT_TRUE(Match(root->operand(1), m::Gather()));
  // Check that reshape instructions for indices are reused.
  EXPECT_EQ(root->operand(0)->operand(1), root->operand(1)->operand(1));
}

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGather_RHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[32,8,16]{2,1,0} parameter(0)
  %weights = bf16[1024,2,3072] parameter(1)

  // One-hot logic (transposed effectively compared to LHS test)
  %iota = s32[1024] iota(), iota_dimension=0
  
  // Weights: [2, 3072, 1024]
  // OneHot: [1024, 32, 8, 16] (Assuming Iota is dim 0)
  
  %weights_param = bf16[2,3072,1024] parameter(2)
  
  %indices_b = s32[1024,32,8,16] broadcast(%indices), dimensions={1,2,3}
  %iota_b = s32[1024,32,8,16] broadcast(%iota), dimensions={0}
  
  %comp = pred[1024,32,8,16] compare(%indices_b, %iota_b), direction=EQ
  %oh = bf16[1024,32,8,16] convert(%comp)
  
  ROOT %dot = bf16[2,3072,32,8,16] dot(%weights_param, %oh), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(
      Match(root, m::Gather(m::Parameter(2), m::Reshape(m::Parameter(0)))));
}

TEST_F(OneHotGatherRewriterTest, MismatchContractingDim) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[32,8,16]{2,1,0} parameter(0)
  %weights = bf16[1024,2,3072] parameter(1)

  %iota = s32[1024] iota(), iota_dimension=0
  
  %indices_broadcast = s32[32,8,16,1024]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[32,8,16,1024]{3,2,1,0} broadcast(%iota), dimensions={3}
  
  %compare = pred[32,8,16,1024] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[32,8,16,1024] convert(%compare)
  
  // Dot contracting dim 2 (size 16) vs OneHot Iota dim 3 (size 1024).
  // This is invalid dot if sizes don't match, so let's make sizes match.
  // Change weights to have contracting dim size 16.
  
  %weights_mismatch = bf16[16, 2, 3072] parameter(2)
  
  // Contract on dim 2 (size 16) of one_hot.
  ROOT %dot = bf16[32,8,1024,2,3072] dot(%one_hot, %weights_mismatch), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";
  // This should NOT rewrite because the contracting dim (2) corresponds to
  // %indices dimension, not %iota dimension (3).

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(OneHotGatherRewriterTest, DoesNotRewriteRankChangingBitcast) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2]{0} parameter(0)
  %weights = bf16[2,8]{1,0} parameter(1)
  %iota = s32[4]{0} iota(), iota_dimension=0
  %bc = s32[2,2]{1,0} bitcast(%iota)
  %indices_b = s32[2,2]{1,0} broadcast(%indices), dimensions={0}
  %compare = pred[2,2]{1,0} compare(%indices_b, %bc), direction=EQ
  %one_hot = bf16[2,2]{1,0} convert(%compare)
  ROOT %dot = bf16[2,8]{1,0} dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(OneHotGatherRewriterTest, DepthLimitExceeded) {
  // Construct a chain deeper than kMaxTraceDepth (20).
  // Iota -> Bitcast -> ... -> Bitcast -> Compare

  auto builder = HloComputation::Builder("entry");

  Shape iota_shape = ShapeUtil::MakeShape(S32, {128});
  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(iota_shape, 0));

  HloInstruction* current = iota;
  // 21 levels of depth.
  for (int i = 0; i < 21; ++i) {
    current = builder.AddInstruction(
        HloInstruction::CreateBitcast(iota_shape, current));
  }

  Shape indices_shape = ShapeUtil::MakeShape(S32, {128});
  HloInstruction* indices = builder.AddInstruction(
      HloInstruction::CreateParameter(0, indices_shape, "indices"));

  HloInstruction* compare = builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {128}), indices,
                                    current, ComparisonDirection::kEq));

  HloInstruction* one_hot = builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {128}), compare));

  HloInstruction* weights =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {128, 32}), "weights"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(0);
  dnums.add_rhs_contracting_dimensions(0);

  builder.AddInstruction(
      HloInstruction::CreateDot(ShapeUtil::MakeShape(F32, {32}), one_hot,
                                weights, dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  OneHotGatherRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  // Should NOT rewrite because depth > 20.
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

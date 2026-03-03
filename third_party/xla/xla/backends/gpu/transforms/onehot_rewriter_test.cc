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

#include "xla/backends/gpu/transforms/onehot_rewriter.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_utils.h"

namespace xla {
namespace gpu {
namespace {

class OneHotGatherRewriterTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  // Runs the OneHotGatherRewriter pass on the given HLO string and verifies:
  // 1. The pass makes changes to the module.
  // 2. The transformed HLO matches the provided FileCheck pattern.
  // 3. The execution results are consistent with the original HLO (semantics
  //    preservation), using either provided arguments or generated ones.
  void RunPassAndVerify(
      absl::string_view hlo_string, absl::string_view check_pattern,
      std::optional<std::vector<Literal>> custom_arguments = std::nullopt) {
    auto test_preprocessor = [&](HloModule* module) {
      module->mutable_config().mutable_debug_options().set_xla_enable_fast_math(
          true);
      OneHotGatherRewriter rewriter;
      ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(rewriter, module));
      EXPECT_TRUE(changed);
      absl::StatusOr<bool> filecheck_result =
          RunFileCheck(module->ToString(), check_pattern);
      ASSERT_OK(filecheck_result.status());
      EXPECT_TRUE(filecheck_result.value()) << "FileCheck failed. Module:\n"
                                            << module->ToString();
    };

    if (custom_arguments.has_value()) {
      ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_string));
      std::vector<const Literal*> args_ptrs;
      args_ptrs.reserve(custom_arguments->size());
      for (const auto& arg : *custom_arguments) {
        args_ptrs.push_back(&arg);
      }
      EXPECT_TRUE(
          RunAndCompare(std::move(module), args_ptrs, ErrorSpec{1e-3, 1e-3},
                        /*reference_preprocessor=*/nullptr, test_preprocessor));
    } else {
      EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-3, 1e-3},
                                /*reference_preprocessor=*/nullptr,
                                test_preprocessor));
    }
  }

  void RunPassAndVerifyNoChange(absl::string_view hlo_string) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
    module->mutable_config().mutable_debug_options().set_xla_enable_fast_math(
        true);
    OneHotGatherRewriter rewriter;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(rewriter, module.get()));
    EXPECT_FALSE(changed);
  }
};

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[8,2,4] parameter(1)

  %iota = s32[8] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,8] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,8] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,4] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: %gather = {{.*}} gather
    CHECK: offset_dims={3,4}
    CHECK: collapsed_slice_dims={0}
    CHECK: start_index_map={0}
    CHECK: index_vector_dim=3
  )");
}

TEST_F(OneHotGatherRewriterTest, Correctness_InBound) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[4,2,8] parameter(1)

  %iota = s32[4] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,4]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,4]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,4] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,4] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,8] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));

  // Replace indices (arg 0)
  args[0] =
      LiteralUtil::CreateR3<int32_t>({{{0, 1}, {2, 3}}, {{1, 2}, {3, 0}}});

  RunPassAndVerify(hlo_string,
                   R"(
    CHECK: %gather = {{.*}} gather
    CHECK: offset_dims={3,4}
    CHECK: collapsed_slice_dims={0}
    CHECK: start_index_map={0}
    CHECK: index_vector_dim=3
  )",
                   std::move(args));
}

TEST_F(OneHotGatherRewriterTest, Correctness_OutOfBound) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[4,2,8] parameter(1)

  %iota = s32[4] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,4]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,4]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,4] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,4] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,8] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));

  // Replace indices (arg 0) with some out of bound values (depth is 4, so [0,3]
  // is valid)
  args[0] = LiteralUtil::CreateR3<int32_t>(
      {{{-1, 4}, {5, 0}}, {{-100, 3}, {2, 100}}});

  RunPassAndVerify(hlo_string,
                   R"(
    CHECK: %gather = {{.*}} gather
    CHECK: offset_dims={3,4}
    CHECK: collapsed_slice_dims={0}
    CHECK: start_index_map={0}
    CHECK: index_vector_dim=3
  )",
                   std::move(args));
}

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGather_Inlined) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %indices_r1 = s32[2,2,2,1]{3,2,1,0} reshape(%indices)
  %indices_b1 = s32[2,2,2,1]{3,2,1,0} broadcast(%indices_r1), dimensions={0,1,2,3}
  %indices_r2 = s32[2,2,2]{2,1,0} reshape(%indices_b1)
  %indices_b_final = s32[2,2,2,8]{3,2,1,0} broadcast(%indices_r2), dimensions={0,1,2}
  %iota = s32[8]{0} iota(), iota_dimension=0
  %iota_r1 = s32[1,1,1,8]{3,2,1,0} reshape(%iota)
  %iota_b1 = s32[1,1,1,8]{3,2,1,0} broadcast(%iota_r1), dimensions={0,1,2,3}
  %iota_r2 = s32[8]{0} reshape(%iota_b1)
  %iota_b_final = s32[2,2,2,8]{3,2,1,0} broadcast(%iota_r2), dimensions={3}
  %one_hot = pred[2,2,2,8]{3,2,1,0} compare(%indices_b_final, %iota_b_final), direction=EQ

  %one_hot_c1 = bf16[2,2,2,8]{3,2,1,0} convert(%one_hot)
  %weights2 = bf16[8,2,2,2]{3,2,1,0} parameter(2)
  %weights2_r = bf16[8,2,4]{2,1,0} reshape(%weights2)
  %dot1 = bf16[2,2,2,2,4]{4,3,2,1,0} dot(%one_hot_c1, %weights2_r), lhs_contracting_dims={3}, rhs_contracting_dims={0}
  %one_hot_c2 = bf16[2,2,2,8]{3,2,1,0} convert(%one_hot)
  %weights1 = bf16[8,2,2,2]{3,2,1,0} parameter(1)
  %weights1_r = bf16[8,2,4]{2,1,0} reshape(%weights1)
  %dot2 = bf16[2,2,2,2,4]{4,3,2,1,0} dot(%one_hot_c2, %weights1_r), lhs_contracting_dims={3}, rhs_contracting_dims={0}
  ROOT %tuple = tuple(%dot1, %dot2)
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: gather
    CHECK: select
    CHECK: gather
    CHECK: select
    CHECK: tuple
  )");
}

TEST_F(OneHotGatherRewriterTest, RewriteOneHotDotToGather_RHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[8,2,4] parameter(1)

  // One-hot logic (transposed effectively compared to LHS test)
  %iota = s32[8] iota(), iota_dimension=0

  %weights_param = bf16[2,4,8] parameter(2)

  %indices_b = s32[8,2,2,2] broadcast(%indices), dimensions={1,2,3}
  %iota_b = s32[8,2,2,2] broadcast(%iota), dimensions={0}

  %comp = pred[8,2,2,2] compare(%indices_b, %iota_b), direction=EQ
  %oh = bf16[8,2,2,2] convert(%comp)

  ROOT %dot = bf16[2,4,2,2,2] dot(%weights_param, %oh), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: %[[GATHER:.*]] = {{.*}} gather
    CHECK: offset_dims={0,1}
    CHECK: collapsed_slice_dims={2}
    CHECK: select({{.*}}, %[[GATHER]], {{.*}})
  )");
}

TEST_F(OneHotGatherRewriterTest, MetadataPreserved) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[8,2,4] parameter(1)

  %iota = s32[8] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,8] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,8] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,4] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}, metadata={op_type="MyOp" op_name="my_op"}
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: %[[GATHER:.*]] = {{.*}} gather{{.*}}metadata={op_type="MyOp" op_name="my_op"}
    CHECK: select{{.*}}metadata={op_type="MyOp" op_name="my_op"}
  )");
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

  %weights_mismatch = bf16[16, 2, 3072] parameter(2)

  ROOT %dot = bf16[32,8,1024,2,3072] dot(%one_hot, %weights_mismatch), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";
  // This should NOT rewrite because the contracting dim (2) corresponds to
  // %indices dimension, not %iota dimension (3).

  RunPassAndVerifyNoChange(hlo_string);
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

  RunPassAndVerifyNoChange(hlo_string);
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
  module->mutable_config().mutable_debug_options().set_xla_enable_fast_math(
      true);

  OneHotGatherRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(rewriter, module.get()));
  // Should NOT rewrite because depth > 20.
  EXPECT_FALSE(changed);
}

TEST_F(OneHotGatherRewriterTest, Negative_MergedIota) {
  absl::string_view hlo_string = R"(
HloModule Negative_MergedIota

ENTRY Main {
  indices = s32[100] parameter(0)
  iota = s32[100] iota(), iota_dimension=0
  compare = pred[100] compare(indices, iota), direction=EQ
  one_hot = f32[100] convert(compare)
  reshaped_one_hot = f32[10, 10] reshape(one_hot)
  weights = f32[10, 128] parameter(1)
  ROOT dot = f32[10, 128] dot(reshaped_one_hot, weights),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  // Contracting over a merged dimension: The rewriter should reject this.
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, Negative_ReshapedIndices) {
  absl::string_view hlo_string = R"(
HloModule Negative_ReshapedIndices

ENTRY Main {
  indices = s32[2, 5] parameter(0)
  flattened_indices = s32[10] reshape(indices)
  iota = s32[10] iota(), iota_dimension=0
  compare = pred[10] compare(flattened_indices, iota), direction=EQ
  one_hot = f32[10] convert(compare)
  weights = f32[10, 128] parameter(1)
  ROOT dot = f32[128] dot(one_hot, weights),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  // This should not change because the indices shape [2, 5] does not match
  // the dot rank structure implied by flattening.
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, InterdependentOneHotDots) {
  // dot1 matches OneHot pattern.
  // dot2 uses dot1 as weights.
  absl::string_view hlo_string = R"(
HloModule InterdependentOneHotDots

ENTRY Main {
  indices = s32[] parameter(0)
  iota1 = s32[10] iota(), iota_dimension=0
  indices_b = s32[10] broadcast(indices), dimensions={}
  compare1 = pred[10] compare(indices_b, iota1), direction=EQ
  one_hot1 = f32[10] convert(compare1)
  weights1 = f32[10, 20] parameter(1)
  dot1 = f32[20] dot(one_hot1, weights1),
         lhs_contracting_dims={0}, rhs_contracting_dims={0}

  indices2 = s32[5] parameter(2)
  iota2 = s32[20] iota(), iota_dimension=0
  indices2_b = s32[5, 20] broadcast(indices2), dimensions={0}
  iota2_b = s32[5, 20] broadcast(iota2), dimensions={1}
  compare2 = pred[5, 20] compare(indices2_b, iota2_b), direction=EQ
  one_hot2 = f32[5, 20] convert(compare2)

  ROOT dot2 = f32[5] dot(one_hot2, dot1),
              lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: %[[GATHER1:.*]] = {{.*}} gather
    CHECK: %[[SELECT1:.*]] = {{.*}} select({{.*}}, %[[GATHER1]], {{.*}})
    CHECK: %[[GATHER2:.*]] = {{.*}} gather(%[[SELECT1]], {{.*}})
    CHECK: select({{.*}}, %[[GATHER2]], {{.*}})
  )");
}

TEST_F(OneHotGatherRewriterTest, HeuristicSkip) {
  // depth=128 (< 256), indices elements=2048 (> 1024).
  // This triggers the heuristic to skip rewrite.
  absl::string_view hlo_string = R"(
HloModule HeuristicSkip

ENTRY Main {
  %indices = s32[2048] parameter(0)
  %weights = f32[128, 32] parameter(1)

  %iota = s32[128] iota(), iota_dimension=0
  %indices_b = s32[2048,128] broadcast(%indices), dimensions={0}
  %iota_b = s32[2048,128] broadcast(%iota), dimensions={1}

  %compare = pred[2048,128] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2048,128] convert(%compare)

  ROOT %dot = f32[2048,32] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, RankMismatch_ReshapeIndices) {
  absl::string_view hlo_string = R"(
HloModule RankMismatch
ENTRY main {
  %indices = s32[2,2] parameter(0)
  %weights = f32[4,2] parameter(1)
  %indices_r = s32[4] reshape(%indices)
  %iota = s32[4,4] iota(), iota_dimension=1
  %indices_b = s32[4,4] broadcast(%indices_r), dimensions={0}
  %compare = pred[4,4] compare(%indices_b, %iota), direction=EQ
  %one_hot = f32[4,4] convert(%compare)
  ROOT %dot = f32[4,2] dot(%one_hot, %weights),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  RunPassAndVerify(hlo_string, R"(
    CHECK: %[[WEIGHTS:.*]] = f32[4,2]{{.*}}parameter(1)
    CHECK: %[[RESHAPE:.*]] = s32[4,1]{{.*}}reshape
    CHECK: %[[GATHER:.*]] = f32[4,2]{{.*}}gather(%[[WEIGHTS]], %[[RESHAPE]])
    CHECK: ROOT %[[SELECT:.*]] = f32[4,2]{{.*}}select({{.*}}, %[[GATHER]], {{.*}})
  )");
}

TEST_F(OneHotGatherRewriterTest, FloatIndices_NoRewrite) {
  absl::string_view hlo_string = R"(
HloModule FloatIndices

ENTRY main {
  %indices = f32[2] parameter(0)
  %weights = f32[4, 2] parameter(1)

  %iota = f32[4] iota(), iota_dimension=0
  %indices_b = f32[2,4] broadcast(%indices), dimensions={0}
  %iota_b = f32[2,4] broadcast(%iota), dimensions={1}

  %compare = pred[2,4] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2,4] convert(%compare)

  ROOT %dot = f32[2,2] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, S64Indices_CorrectConstants) {
  absl::string_view hlo_string = R"(
HloModule S64Indices

ENTRY main {
  %indices = s64[2] parameter(0)
  %weights = f32[4, 2] parameter(1)

  %iota = s64[4] iota(), iota_dimension=0
  %indices_b = s64[2,4] broadcast(%indices), dimensions={0}
  %iota_b = s64[2,4] broadcast(%iota), dimensions={1}

  %compare = pred[2,4] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2,4] convert(%compare)

  ROOT %dot = f32[2,2] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, Bug_TypeMismatch_S64Indices_Converted) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices_s64 = s64[2,2,2]{2,1,0} parameter(0)
  %indices = s32[2,2,2]{2,1,0} convert(%indices_s64)
  %weights = bf16[8,2,4] parameter(1)

  %iota = s32[8] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,8] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,8] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,4] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";
  RunPassAndVerify(hlo_string, R"(
    CHECK: gather
  )");
}

TEST_F(OneHotGatherRewriterTest, TypeMismatch_F32Indices_Converted) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices_f32 = f32[2] parameter(0)
  %indices = s32[2] convert(%indices_f32)
  %weights = f32[4, 2] parameter(1)

  %iota = s32[4] iota(), iota_dimension=0
  %indices_b = s32[2,4] broadcast(%indices), dimensions={0}
  %iota_b = s32[2,4] broadcast(%iota), dimensions={1}

  %compare = pred[2,4] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2,4] convert(%compare)

  ROOT %dot = f32[2,2] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerify(hlo_string, R"(
    CHECK: gather
  )");
}

TEST_F(OneHotGatherRewriterTest, ShapeMismatch_BroadcastBatchDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2] parameter(0)
  %weights = f32[2, 2] parameter(1)

  %iota = s32[2] iota(), iota_dimension=0
  %indices_b = s32[2,2] broadcast(%indices), dimensions={1}
  %iota_b = s32[2,2] broadcast(%iota), dimensions={1}

  %compare = pred[2,2] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2,2] convert(%compare)

  ROOT %dot = f32[2,2] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().mutable_debug_options().set_xla_enable_fast_math(
      true);
  OneHotGatherRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(rewriter, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(OneHotGatherRewriterTest, BroadcastPermutation_ShapeMismatch) {
  absl::string_view hlo_string = R"(
HloModule PermutationBug

ENTRY main {
  %indices = s32[2, 3] parameter(0)
  %weights = f32[4, 5] parameter(1)

  %iota = s32[4] iota(), iota_dimension=0
  %iota_b = s32[3, 4, 2] broadcast(%iota), dimensions={1}

  // Permute [2, 3] to [3, 2] and add dim 1.
  %indices_b = s32[3, 4, 2] broadcast(%indices), dimensions={2, 0}

  %compare = pred[3, 4, 2] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[3, 4, 2] convert(%compare)

  ROOT %dot = f32[3, 2, 5] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, TypeMismatch_MixedPrecision) {
  absl::string_view hlo_string = R"(
HloModule TypeMismatch

ENTRY entry {
  %indices = s32[2] parameter(0)
  %weights = f16[10, 5] parameter(1)

  %iota = s32[2, 10] iota(), iota_dimension=1
  %indices_b = s32[2, 10] broadcast(%indices), dimensions={0}
  %compare = pred[2, 10] compare(%iota, %indices_b), direction=EQ
  %one_hot = f16[2, 10] convert(%compare)

  // Dot accumulates F16 into F32. Output is F32.
  // Rewrite creates Gather(Weights=F16) -> Output=F32.
  // HLO Verifier fails: Gather output element type must match operand element type.
  ROOT %dot = f32[2, 5] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, ZeroDepth_NoRewrite) {
  absl::string_view hlo_string = R"(
HloModule ZeroDepth

ENTRY main {
  %indices = s32[2] parameter(0)
  %weights = f32[0, 2] parameter(1)

  %iota = s32[0] iota(), iota_dimension=0
  %indices_b = s32[2,0] broadcast(%indices), dimensions={0}
  %iota_b = s32[2,0] broadcast(%iota), dimensions={1}

  %compare = pred[2,0] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[2,0] convert(%compare)

  // Dot contracts dim 0 (size 0). Result should be 0.
  ROOT %dot = f32[2,2] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, LargeDepth_NoRewrite) {
  absl::string_view hlo_string = R"(
HloModule LargeDepth

ENTRY main {
  %indices = s32[1] parameter(0)
  // Depth 3,000,000,000 > INT32_MAX
  %weights = f32[3000000000, 1] parameter(1)

  %iota = s32[3000000000] iota(), iota_dimension=0
  %indices_b = s32[1, 3000000000] broadcast(%indices), dimensions={0}
  %iota_b = s32[1, 3000000000] broadcast(%iota), dimensions={1}

  %compare = pred[1, 3000000000] compare(%indices_b, %iota_b), direction=EQ
  %one_hot = f32[1, 3000000000] convert(%compare)

  ROOT %dot = f32[1, 1] dot(%one_hot, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  RunPassAndVerifyNoChange(hlo_string);
}

TEST_F(OneHotGatherRewriterTest, RequiresFastMath) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  %indices = s32[2,2,2]{2,1,0} parameter(0)
  %weights = bf16[8,2,4] parameter(1)

  %iota = s32[8] iota(), iota_dimension=0

  %indices_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%indices), dimensions={0,1,2}
  %iota_broadcast = s32[2,2,2,8]{3,2,1,0} broadcast(%iota), dimensions={3}

  %compare = pred[2,2,2,8] compare(%indices_broadcast, %iota_broadcast), direction=EQ
  %one_hot = bf16[2,2,2,8] convert(%compare)

  ROOT %dot = bf16[2,2,2,2,4] dot(%one_hot, %weights), lhs_contracting_dims={3}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  // xla_enable_fast_math is false by default.
  EXPECT_FALSE(module->config().debug_options().xla_enable_fast_math());
  OneHotGatherRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(rewriter, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

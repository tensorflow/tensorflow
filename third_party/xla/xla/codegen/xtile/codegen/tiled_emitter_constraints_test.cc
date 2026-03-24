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

#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class TiledEmitterConstraintsTest : public HloHardwareIndependentTestBase {
 public:
  TiledEmitterConstraintsTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(
      HloModule* module, bool with_tiled_emitter_specific_constraints = true) {
    EmitterSpecificConstraintsBuilder constraints_builder = nullptr;

    if (with_tiled_emitter_specific_constraints) {
      constraints_builder = TiledEmitterConstraints::GetBuilder();
    }

    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_, constraints_builder);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
    }
    VLOG(1) << "Cannot analyze module: "
            << std::get<FusionDecision>(analysis_or_error).Explain();
    return std::nullopt;
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(TiledEmitterConstraintsTest, CustomReshapeConstraintsAreEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_computation {
  p = s8[36] parameter(0)
  ROOT bitcast = s8[6,6] bitcast(p)
}

ENTRY entry_computation {
  p = s8[36] parameter(0)
  ROOT fusion = s8[6,6] fusion(p), kind=kCustom, calls=triton_computation
})"));

  std::optional<SymbolicTileAnalysis> analysis_without_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_tiling_constraints.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  Tiling tiling({{fusion_root, FlatTiling({2, 6})}});

  // (2, 6) is a theoretically valid tiling for this reshape, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_tiling_constraints->ParametersSatisfyConstraints(tiling),
      absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_tiling_constraints.has_value());

  // (2, 6) is a theoretically valid tiling for this reshape, but it won't
  // work because of Triton's power of two restriction. Thus, we should reject
  // it here.
  EXPECT_THAT(
      analysis_with_tiling_constraints->ParametersSatisfyConstraints(tiling),
      absl_testing::IsOkAndHolds(false));

  // However, (1, 6) is valid and should still work.
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 6})}})),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(TiledEmitterConstraintsTest,
       CustomConcatenateSizeConstraintsAreEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[4] parameter(2)
  ROOT concatenate = bf16[20] concatenate(p0, p1, p2), dimensions={0}
}

ENTRY main {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[4] parameter(2)
  ROOT fusion = bf16[20] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_tiling_constraints.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // (16,) is a theoretically valid tiling for this concatenate, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(analysis_without_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({16})}})),
              absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_tiling_constraints.has_value());

  // (16,) is a theoretically valid tiling for this concatenate, but it won't
  // work in our lowering for now, because we want to be loading from a single
  // operand at a time, and it doesn't divide each operand's concatenation
  // dimension. We want to reject it here.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({16})}})),
              absl_testing::IsOkAndHolds(false));

  // However, (8,) is valid and should still work.
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8})}})),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(TiledEmitterConstraintsTest,
       ConcatenateConstrainsOffsetToBeZeroAlongConcatenationDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  concatenate = bf16[48] concatenate(p0, p1, p2), dimensions={0}
  ROOT slice = bf16[24] slice(concatenate), slice={[24:48]}
}

ENTRY main {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  ROOT fusion = bf16[24] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_tiling_constraints.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // (8,) is a theoretically valid tiling for this concatenate, and one that
  // works for all operands, so SymbolicTileAnalysis should allow it.
  EXPECT_THAT(analysis_without_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8})}})),
              absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_tiling_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, but the
  // constraints enforce that the offset along the concatenation dimension be 0.
  // Here, it is 24, so we expect the tiling to be rejected.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8})}})),
              absl_testing::IsOkAndHolds(false));

  // Even the smallest tiling, (1,) should be rejected here. (This is
  // unnecessary in theory, but a sanity check for the implementation).
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1})}})),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(TiledEmitterConstraintsTest,
       ConcatenateConstrainsStrideToBeOneAlongConcatenationDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  concatenate = bf16[48] concatenate(p0, p1, p2), dimensions={0}
  ROOT slice = bf16[24] slice(concatenate), slice={[0:48:2]}
}

ENTRY main {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  ROOT fusion = bf16[24] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_tiling_constraints.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // (8,) is a theoretically valid tiling for this concatenate, and one that
  // works for all operands, so SymbolicTileAnalysis should allow it.
  EXPECT_THAT(analysis_without_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8})}})),
              absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_tiling_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_tiled_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_tiling_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, but the
  // constraints enforce that the stride along the concatenation dimension be 1.
  // Here, it is 2, so we expect the tiling to be rejected.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8})}})),
              absl_testing::IsOkAndHolds(false));

  // Even the smallest tiling, (1,) should be rejected here. (This is
  // unnecessary in theory, but a sanity check for the implementation).
  EXPECT_THAT(analysis_with_tiling_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1})}})),
              absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla

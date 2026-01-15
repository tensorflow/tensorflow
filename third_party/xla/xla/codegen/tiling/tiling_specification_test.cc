/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/tiling_specification.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

MATCHER_P2(InstructionMapping, instruction, num_tiling_parameters,
           "A matcher for "
           "`TilingSpecification::InstructionAndNumTilingParameters`s.") {
  return ExplainMatchResult(instruction, arg.instruction, result_listener) &&
         ExplainMatchResult(num_tiling_parameters, arg.num_tiling_parameters,
                            result_listener);
}

class TilingSpecificationTest : public HloHardwareIndependentTestBase {
 public:
  TilingSpecificationTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  SymbolicTileAnalysis AnalyzeModule(HloModule* module) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_,
            /*emitter_specific_constraints_builder=*/nullptr);

    CHECK(std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error));
    return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(TilingSpecificationTest, TilingSpecificationDerivesOutputParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  ROOT p0 = f32[137,115] parameter(0)
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  ROOT fusion = f32[137,115] fusion(p0), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      tiling_spec.parameter_mapping(),
      ElementsAre(InstructionMapping(root->fused_expression_root(), 2)));
}

TEST_F(TilingSpecificationTest, TilingSpecificationDerivesHiddenDotParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT abs = f32[137,137] abs(dot)
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* abs = root->fused_expression_root();
  const HloInstruction* dot = abs->operand(0);

  EXPECT_THAT(
      tiling_spec.parameter_mapping(),
      ElementsAre(InstructionMapping(abs, 2), InstructionMapping(dot, 1)));
}

TEST_F(TilingSpecificationTest,
       TilingSpecificationDerivesOutputAndHiddenParametersOnTheSameOperation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot = root->fused_expression_root();

  EXPECT_THAT(tiling_spec.parameter_mapping(),
              ElementsAre(InstructionMapping(dot, 3)));
}

TEST_F(TilingSpecificationTest,
       TilingSpecificationDerivesHiddenParametersInNestedFusions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
nested_computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  dot_output = f32[137,137] fusion(p0, p1), kind=kLoop, calls=nested_computation
  ROOT abs = f32[137,137] abs(dot_output)
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* abs = root->fused_expression_root();
  const HloInstruction* dot = abs->operand(0)->fused_expression_root();

  EXPECT_THAT(
      tiling_spec.parameter_mapping(),
      ElementsAre(InstructionMapping(abs, 2), InstructionMapping(dot, 1)));
}

TEST_F(TilingSpecificationTest,
       TilingWithIncorrectSetOfNestedTileSizesDoesNotConformToSpecification) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot = root->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);

  // An underspecified tiling does not conform to a tiling specification.
  Tiling underspecified_nested_tiling(Tiling::TileMapping{{dot, {1}}});
  EXPECT_FALSE(underspecified_nested_tiling.ConformsTo(tiling_spec));

  // An overspecified tiling does not conform to a tiling specification either.
  Tiling overspecified_nested_tiling(Tiling::TileMapping{{dot, {1, 1, 1, 1}}});
  EXPECT_FALSE(overspecified_nested_tiling.ConformsTo(tiling_spec));
}

TEST_F(TilingSpecificationTest,
       TilingWithIncorrectSetOfOutputTileSizesDoesNotConformToSpecification) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT abs = f32[137,137] abs(dot)
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* abs = root->fused_expression_root();
  const HloInstruction* dot = abs->operand(0);

  // An underspecified tiling does not conform to a tiling specification.
  Tiling underspecified_output_tiling(
      Tiling::TileMapping{{dot, {1}}, {abs, {1}}});
  EXPECT_FALSE(underspecified_output_tiling.ConformsTo(tiling_spec));

  // An overspecified tiling does not conform to a tiling specification either.
  Tiling overspecified_output_tiling(
      Tiling::TileMapping{{dot, {1}}, {abs, {1, 1, 1}}});
  EXPECT_FALSE(overspecified_output_tiling.ConformsTo(tiling_spec));
}

TEST_F(TilingSpecificationTest,
       TilingWithIncorrectSetOfTiledInstructionsDoesNotConformToSpecification) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot = root->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);

  // An underspecified tiling does not conform to a tiling specification.
  Tiling underspecified_tiling(Tiling::TileMapping{{}});
  EXPECT_FALSE(underspecified_tiling.ConformsTo(tiling_spec));

  // A tiling along of irrelevant operations does not conform to a tiling
  // specification.
  Tiling off_topic_tiling(Tiling::TileMapping{{dot->operand(0), {1}}});
  EXPECT_FALSE(off_topic_tiling.ConformsTo(tiling_spec));

  // An overspecified tiling does not conform to a tiling specification either.
  Tiling overspecified_tiling(
      Tiling::TileMapping{{dot, {1, 1, 1}}, {dot->operand(0), {1}}});

  EXPECT_FALSE(overspecified_tiling.ConformsTo(tiling_spec));
}

TEST_F(TilingSpecificationTest,
       TilingWithExactlyConformantSetOfParametersConformsToSpecification) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
computation {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT dot = f32[137,137] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[115,137] parameter(1)
  ROOT fusion = f32[137,137] fusion(p0, p1), kind=kLoop, calls=computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = AnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot = root->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);

  Tiling exact_tiling(Tiling::TileMapping{{dot, {1, 1, 1}}});
  EXPECT_TRUE(exact_tiling.ConformsTo(tiling_spec));
}

}  // namespace
}  // namespace xla

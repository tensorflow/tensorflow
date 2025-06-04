/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/symbolic_tile_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

using detail::GetGoodTilings;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::ExplainMatchResult;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Not;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;
using TilingVector = std::vector<SymbolicTileAnalysis::Tiling>;

MATCHER_P3(MatchTiledHloInstructionImpl, tile_sizes, tile_strides,
           tile_offsets_indexing, "") {
  return ExplainMatchResult(ElementsAreArray(tile_sizes), arg.tile_sizes(),
                            result_listener) &&
         ExplainMatchResult(ElementsAreArray(tile_strides), arg.tile_strides(),
                            result_listener) &&
         ExplainMatchResult(
             IsOkAndHolds(MatchIndexingMap(tile_offsets_indexing)),
             arg.tile_offsets_indexing(), result_listener);
}

Matcher<const TiledHloInstruction> MatchTiledHloInstruction(
    std::vector<int64_t> tile_sizes, std::vector<int64_t> tile_strides,
    std::string tile_offsets_indexing) {
  return MatchTiledHloInstructionImpl(tile_sizes, tile_strides,
                                      tile_offsets_indexing);
}

MATCHER_P(MatchConstraintExpressionString, constraint_expression_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(constraint_expression_string, arg.ToString()),
      result_listener);
}

MATCHER_P2(InstructionMapping, instruction, num_tiling_parameters,
           "A matcher for "
           "`TilingSpecification::InstructionAndNumTilingParameters`s.") {
  return ExplainMatchResult(instruction, arg.instruction, result_listener) &&
         ExplainMatchResult(num_tiling_parameters, arg.num_tiling_parameters,
                            result_listener);
}

// Fake emitter-specific constraints for testing. Requires that the tile size
// along the first dimension is exactly half the size of the axis.
class FakeEmitterSpecificConstraints : public EmitterSpecificConstraints {
 public:
  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const override {
    return tile_parameters[0] == dim0_tile_size_;
  }

  static EmitterSpecificConstraintsBuilder GetBuilder() {
    return [](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                  instructions,
              const HloFusionAdaptor&) {
      const SymbolicTiledHloInstruction* root = instructions[0].get();
      int64_t dim0_size = root->hlo()->shape().dimensions(0);
      return std::make_unique<FakeEmitterSpecificConstraints>(
          /*dim0_tile_size=*/dim0_size / 2);
    };
  }

  explicit FakeEmitterSpecificConstraints(int64_t dim0_tile_size)
      : dim0_tile_size_(dim0_tile_size) {}

 private:
  int64_t dim0_tile_size_;
};

class SymbolicTileAnalysisTest : public HloHardwareIndependentTestBase {
 public:
  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(
      HloModule* module,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_, emitter_specific_constraints_builder);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
    }
    VLOG(1) << "Cannot analyze module: "
            << std::get<FusionDecision>(analysis_or_error).Explain();
    return std::nullopt;
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(SymbolicTileAnalysisTest, SimpleNormalizationDiamondIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
max {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0, p1)
}

fusion {
  p0 = f32[2,97]{1,0} parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[2] reduce(p0, constant), dimensions={1}, to_apply=max
  broadcast = f32[2,97]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[2,97]{1,0} subtract(p0, broadcast)
}

ENTRY main {
  p0 = f32[2,97]{1,0} parameter(0)
  ROOT fusion = f32[2,97]{1,0} fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{1, 10},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];

  EXPECT_THAT(*root, MatchTiledHloInstruction(/*tile_sizes=*/{1, 10},
                                              /*tile_strides=*/{1, 1},
                                              /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 10, (pid_0 mod 10) * 10),
    domain:
    pid_0 in [0, 19]
  )"));

  auto p0_from_subtract0 = root->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0)->operand(0);

  EXPECT_THAT(*p0_from_subtract0, MatchTiledHloInstruction(
                                      /*tile_sizes=*/{1, 10},
                                      /*tile_strides=*/{1, 1},
                                      /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 10, (pid_0 mod 10) * 10),
    domain:
    pid_0 in [0, 19]
  )"));

  EXPECT_THAT(*p0_from_subtract1, MatchTiledHloInstruction(
                                      /*tile_sizes=*/{1, 97},
                                      /*tile_strides=*/{1, 1},
                                      /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 10, 0),
    domain:
    pid_0 in [0, 19]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, ElementwiseDiamondCSEIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[2,97] parameter(0)
  exp = f32[2,97] exponential(p0)
  log = f32[2,97] log(p0)
  ROOT subtract = f32[2,97] subtract(exp, log)
}

ENTRY main {
  p0 = f32[2,97] parameter(0)
  ROOT fusion = f32[2,97] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis->ComputeTiledHloInstructions(/*output_tile_sizes=*/{1, 10}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];

  auto p0_from_subtract0 = root->operand(0)->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0);

  EXPECT_EQ(p0_from_subtract0, p0_from_subtract1);
}

TEST_F(SymbolicTileAnalysisTest,
       ExpandingReshapeIsSupportedWithTileParamsOutsideBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  param_0 = f32[20] parameter(0)
  abs = f32[20] abs(param_0)
  ROOT reshape = f32[4,5] reshape(abs)
}

ENTRY entry_computation {
  param_0 = f32[20] parameter(0)
  ROOT fusion = f32[4, 5] fusion(param_0), kind=kCustom, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{1, 8},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];
  auto parameter = root->operand(0)->operand(0);
  EXPECT_THAT(*parameter, MatchTiledHloInstruction(
                              /*tile_sizes=*/{8},
                              /*tile_strides=*/{1},
                              /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 * 5),
    domain:
    pid_0 in [0, 3]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, ProducerConsumerFusionIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT m = f32[] maximum(p0, p1)
}

fusion.1 {
  p0 = f32[2,97] parameter(0)
  constant = f32[] constant(-inf)
  exp = f32[2,97] exponential(p0)
  ROOT reduce = f32[2] reduce(exp, constant), dimensions={1}, to_apply=max
}

fusion.2 {
  p0 = f32[2] parameter(0)
  p1 = f32[2,97] parameter(1)
  broadcast = f32[2,97]{1,0} broadcast(p0), dimensions={0}
  ROOT subtract = f32[2,97] subtract(p1, broadcast)
}

ENTRY main {
  p0 = f32[2,97] parameter(0)
  producer = f32[2] fusion(p0), kind=kLoop, calls=fusion.1
  ROOT consumer = f32[2,97] fusion(producer, p0), kind=kLoop, calls=fusion.2
})"));

  const auto* consumer = module->entry_computation()->root_instruction();
  const auto* producer = consumer->operand(0);

  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(*fusion, &mlir_context_);
  ASSERT_TRUE(std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error));
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis.ComputeTiledHloInstructions(/*output_tile_sizes=*/{1, 97}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];

  const TiledHloInstruction* p0_from_producer =
      root->operand(1)->operand(0)->operand(0)->operand(0);
  const TiledHloInstruction* p0_from_consumer = root->operand(0);

  EXPECT_EQ(p0_from_producer, p0_from_consumer);

  EXPECT_THAT(*p0_from_producer,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{1, 97}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0, 0),
    domain:
    pid_0 in [0, 1]
  )"));
}

TEST_F(SymbolicTileAnalysisTest,
       ProducerConsumerFusionWithExtraOutputIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation {
  param_0.1 = f32[2048,8192] parameter(0)
  ROOT abs.1 = f32[2048,8192] abs(param_0.1)
}

region {
  param_0.2 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0.2, param_1)
}

fused_computation.1 {
  param_0.3 = f32[2048,8192] parameter(0)
  constant = f32[] constant(-inf)
  ROOT reduce = f32[8192] reduce(param_0.3, constant), dimensions={0}, to_apply=region
}

ENTRY entry_computation {
  param_0.4 = f32[2048,8192] parameter(0)
  fusion = f32[2048,8192] fusion(param_0.4), kind=kCustom, calls=fused_computation
  fusion.1 = f32[8192] fusion(fusion), kind=kCustom, calls=fused_computation.1
  ROOT tuple = (f32[8192], f32[2048,8192]) tuple(fusion.1, fusion)
})"));

  const auto* consumer =
      module->entry_computation()->root_instruction()->operand(0);
  const auto* producer = consumer->operand(0);

  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      producer, consumer, /*with_extra_outputs=*/true);

  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(*fusion, &mlir_context_);
  ASSERT_TRUE(std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error));
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
  EXPECT_THAT(analysis.GetRoots(),
              ::testing::ElementsAre(consumer->fused_expression_root(),
                                     producer->fused_expression_root()));
}

TEST_F(SymbolicTileAnalysisTest, ExtraOutputNeedsToSelectTheRightTile) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[64,64] parameter(0)
  abs = f32[64,64] abs(param_0.1)
  slice = f32[1,1] slice(abs), slice={[0:1], [0:1]}
  reshape = f32[] reshape(slice)
  constant = f32[] constant(1)
  add.1 = f32[] add(reshape, constant)
  broadcast = f32[64,64] broadcast(add.1), dimensions={}
  add.2 = f32[64,64] add(abs, broadcast)
  ROOT tuple = (f32[64,64], f32[64,64]) tuple(add.2, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[64,64] parameter(0)
  ROOT fusion = (f32[64,64], f32[64,64]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{2, 4},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/false));
  const auto& roots = tiled_hlo_computation.GetRoots();
  EXPECT_EQ(roots.size(), 2);
  EXPECT_THAT(*roots[0], MatchTiledHloInstruction(
                             /*tile_sizes=*/{2, 4}, /*tile_strides=*/{1, 1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 16) * 2, (pid_0 mod 16) * 4),
    domain:
    pid_0 in [0, 511]
  )"));
  EXPECT_THAT(*roots[1], MatchTiledHloInstruction(
                             /*tile_sizes=*/{2, 4}, /*tile_strides=*/{1, 1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 16) * 2, (pid_0 mod 16) * 4),
    domain:
    pid_0 in [0, 511]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, ExtraOutputCannotReuseTileDueToTileOverlaps) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.1)
  broadcast = f32[64,64] broadcast(abs), dimensions={1}
  ROOT tuple = (f32[64,64], f32[64]) tuple(broadcast, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[64] parameter(0)
  ROOT fusion = (f32[64,64], f32[64]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  auto maybe_tiled_hlo_computation = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{1, 4},
      /*constraints_are_known_satisfied=*/false,
      /*compute_all_tile_offset_indexing_maps=*/false);
  EXPECT_THAT(
      maybe_tiled_hlo_computation.status(),
      tsl::testing::StatusIs(
          tsl::error::UNIMPLEMENTED,
          ::testing::HasSubstr("Unsupported case of multi-output fusion")));
}

TEST_F(SymbolicTileAnalysisTest, ExtraOutputCanReuseTileForExpandingReshape) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.1)
  reshape = f32[8,8] reshape(abs)
  ROOT tuple = (f32[8,8], f32[64]) tuple(reshape, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[64] parameter(0)
  ROOT fusion = (f32[8,8], f32[64]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{1, 4},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/false));
  const auto& roots = tiled_hlo_computation.GetRoots();
  EXPECT_EQ(roots.size(), 2);
  EXPECT_THAT(*roots[0], MatchTiledHloInstruction(
                             /*tile_sizes=*/{1, 4}, /*tile_strides=*/{1, 1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 2, (pid_0 mod 2) * 4),
    domain:
    pid_0 in [0, 15]
  )"));
  EXPECT_THAT(*roots[1], MatchTiledHloInstruction(
                             /*tile_sizes=*/{4}, /*tile_strides=*/{1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 * 4),
    domain:
    pid_0 in [0, 15]
  )"));
}

TEST_F(
    SymbolicTileAnalysisTest,
    ExtraOutputCannotReuseTileForExpandingReshapeDueToDifferentNumberOfBlocks) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[6400] parameter(0)
  abs = f32[6400] abs(param_0.1)
  reshape = f32[64,100] reshape(abs)
  ROOT tuple = (f32[64,100], f32[6400]) tuple(reshape, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[6400] parameter(0)
  ROOT fusion = (f32[64,100], f32[6400]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  auto maybe_tiled_hlo_computation = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{1, 16},
      /*constraints_are_known_satisfied=*/false,
      /*compute_all_tile_offset_indexing_maps=*/false);
  EXPECT_THAT(
      maybe_tiled_hlo_computation.status(),
      tsl::testing::StatusIs(
          tsl::error::UNIMPLEMENTED,
          ::testing::HasSubstr("Unsupported case of multi-output fusion")));
}

TEST_F(SymbolicTileAnalysisTest,
       ExtraOutputCannotReuseTileForExpandingReshapeDueToDifferentTileOffsets) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  reshape = f32[3,12] reshape(abs)
  ROOT tuple = (f32[3,12], f32[36]) tuple(reshape, abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = (f32[3,12], f32[36]) fusion(param_0), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  auto maybe_tiled_hlo_computation = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{1, 16},
      /*constraints_are_known_satisfied=*/false,
      /*compute_all_tile_offset_indexing_maps=*/false);
  EXPECT_THAT(
      maybe_tiled_hlo_computation.status(),
      tsl::testing::StatusIs(
          tsl::error::UNIMPLEMENTED,
          ::testing::HasSubstr("Unsupported case of multi-output fusion")));
}

TEST_F(SymbolicTileAnalysisTest, ExtraOutputCanReuseTileForCollapsingReshape) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[8,8] parameter(0)
  abs = f32[8,8] abs(param_0.1)
  reshape = f32[64] reshape(abs)
  ROOT tuple = (f32[64], f32[8,8]) tuple(reshape, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[8,8] parameter(0)
  ROOT fusion = (f32[64], f32[8,8]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{4},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/false));
  const auto& roots = tiled_hlo_computation.GetRoots();
  EXPECT_EQ(roots.size(), 2);
  EXPECT_THAT(*roots[0], MatchTiledHloInstruction(
                             /*tile_sizes=*/{4}, /*tile_strides=*/{1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 * 4),
    domain:
    pid_0 in [0, 15]
  )"));
  EXPECT_THAT(*roots[1], MatchTiledHloInstruction(
                             /*tile_sizes=*/{1, 4}, /*tile_strides=*/{1, 1},
                             /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 2, (pid_0 mod 2) * 4),
    domain:
    pid_0 in [0, 15]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, TransposeOffsetIndexingIsCorrect) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[8,16,4] parameter(0)
  ROOT transpose = f32[4,8,16] transpose(p0), dimensions={2,0,1}
}

ENTRY main {
  p0 = f32[8,16,4] parameter(0)
  ROOT fusion = f32[4,8,16] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{2, 4, 2},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];

  EXPECT_THAT(*root, MatchTiledHloInstruction(
                         /*tile_sizes=*/{2, 4, 2}, /*tile_strides=*/{1, 1, 1},
                         /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 16) * 2, ((pid_0 floordiv 8) mod 2) * 4, (pid_0 mod 8) * 2),
    domain:
    pid_0 in [0, 31]
  )"));

  EXPECT_THAT(*root->operand(0),
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{4, 2, 2}, /*tile_strides=*/{1, 1, 1},
                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> (((pid_0 floordiv 8) mod 2) * 4, (pid_0 mod 8) * 2, (pid_0 floordiv 16) * 2),
    domain:
    pid_0 in [0, 31]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, SliceOffsetIndexingIsCorrect) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[8,16] parameter(0)
  slice.0 = f32[4,8] slice(p0), slice={[0:4], [2:10]}
  slice.1 = f32[4,8] slice(p0), slice={[3:7], [4:12]}
  ROOT add = f32[4,8] add(slice.0, slice.1)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = f32[4,8] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{2, 2},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];
  const TiledHloInstruction* p0_from_slice0 = root->operand(0)->operand(0);
  const TiledHloInstruction* p0_from_slice1 = root->operand(1)->operand(0);

  EXPECT_THAT(*root, MatchTiledHloInstruction(
                         /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                         /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 4) * 2, (pid_0 mod 4) * 2),
    domain:
    pid_0 in [0, 7]
  )"));

  EXPECT_THAT(*p0_from_slice0,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 4) * 2, (pid_0 mod 4) * 2 + 2),
    domain:
    pid_0 in [0, 7]
  )"));

  EXPECT_THAT(*p0_from_slice1,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 4) * 2 + 3, (pid_0 mod 4) * 2 + 4),
    domain:
    pid_0 in [0, 7]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, DotOffsetIndexingIsCorrect) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[4,8] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT dot = f32[4,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[4,8] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[4,16] fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const HloInstruction* dot_hlo =
      module->entry_computation()->root_instruction()->fused_expression_root();
  Tiling tiling(Tiling::TileMapping{{dot_hlo, {8, 2, 2}}});
  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*tiling=*/tiling,
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* dot = tiled_hlo_computation.GetRoots()[0];
  EXPECT_THAT(*dot, MatchTiledHloInstruction(
                        /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                        /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 8) * 2, (pid_0 mod 8) * 2),
    domain:
    pid_0 in [0, 15]
  )"));

  const TiledHloInstruction* lhs = dot->operand(0);
  EXPECT_THAT(*lhs, MatchTiledHloInstruction(
                        /*tile_sizes=*/{2, 8}, /*tile_strides=*/{1, 1},
                        /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 8) * 2, 0),
    domain:
    pid_0 in [0, 15]
  )"));

  const TiledHloInstruction* rhs = dot->operand(1);
  EXPECT_THAT(*rhs, MatchTiledHloInstruction(
                        /*tile_sizes=*/{8, 2}, /*tile_strides=*/{1, 1},
                        /*tile_offsets_indexing=*/R"(
    (pid_0) -> (0, (pid_0 mod 8) * 2),
    domain:
    pid_0 in [0, 15]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, DoesNotBailOutOnConstrainedReshape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[4,2]{1,0} parameter(0)
  ROOT reshape = f32[8] reshape(p0)
}

ENTRY main {
  p0 = f32[4,2]{1,0} parameter(0)
  ROOT fusion = f32[8] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const ConstraintExpression& constraints =
      analysis->GetTilingSpecification().constraints();
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "2 mod d0 in [0, 0] || d0 mod 2 in [0, 0]"));
}

TEST_F(SymbolicTileAnalysisTest, DoesNotBailOutOnConstrainedBitcast) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[4,2]{1,0} parameter(0)
  ROOT bitcast = f32[8] bitcast(p0)
}

ENTRY main {
  p0 = f32[4,2]{1,0} parameter(0)
  ROOT fusion = f32[8] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const ConstraintExpression& constraints =
      analysis->GetTilingSpecification().constraints();
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "2 mod d0 in [0, 0] || d0 mod 2 in [0, 0]"));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedConcatenate) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,3]{1,0} parameter(0)
  p1 = f32[1,3]{1,0} parameter(1)
  ROOT concatenate = f32[2,3] concatenate(p0, p1), dimensions={0}
}

ENTRY main {
  p0 = f32[1,3]{1,0} parameter(0)
  p1 = f32[1,3]{1,0} parameter(1)
  ROOT fusion = f32[2,3] fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  EXPECT_FALSE(TryAnalyzeModule(module.get()).has_value());
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedNegativeStrides) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[16] parameter(0)
  ROOT reverse = f32[16] reverse(p0), dimensions={0}
}

ENTRY main {
  p0 = f32[16] parameter(0)
  ROOT fusion = f32[16] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  auto result = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{2},
      /*constraints_are_known_satisfied=*/false,
      /*compute_all_tile_offset_indexing_maps=*/true);
  ASSERT_THAT(result.status(),
              tsl::testing::StatusIs(tsl::error::UNIMPLEMENTED,
                                     ::testing::HasSubstr("negative stride")));
}

TEST_F(SymbolicTileAnalysisTest, MultiOutputFusionIsNotSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[32] parameter(0)
  p1 = f32[32] parameter(1)
  add = f32[32] add(p0, p1)
  subtract = f32[32] subtract(p0, p1)
  ROOT tuple = (f32[32], f32[32]) tuple(add, subtract)
}

ENTRY main {
  p0 = f32[32] parameter(0)
  p1 = f32[32] parameter(1)
  ROOT fusion = (f32[32], f32[32]) fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  EXPECT_FALSE(TryAnalyzeModule(module.get()).has_value());
}

TEST_F(SymbolicTileAnalysisTest, ConstraintSatisfactionIsEvaluatedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,8,6,4,8]{4,3,2,1,0} parameter(0)
  ROOT bitcast = f32[48,32]{1,0} bitcast(p0)
}

ENTRY main {
  p0 = f32[1,8,6,4,8]{4,3,2,1,0} parameter(0)
  ROOT fusion = f32[48,32]{1,0} fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const ConstraintExpression& constraints =
      analysis->GetTilingSpecification().constraints();
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "6 mod d0 in [0, 0] && 8 mod d1 in [0, 0] || "
                               "6 mod d0 in [0, 0] && d1 mod 8 in [0, 0] || "
                               "8 mod d1 in [0, 0] && d0 mod 6 in [0, 0] || "
                               "d0 mod 6 in [0, 0] && d1 mod 8 in [0, 0]"));

  // We expect the constraints here to be
  //    6 mod d0 in [0, 0] && 8 mod s1 in [0, 0] ||
  //    6 mod d0 in [0, 0] && d1 mod 8 in [0, 0] ||
  //    8 mod d1 in [0, 0] && d0 mod 6 in [0, 0] ||
  //    d0 mod 6 in [0, 0] && d1 mod 8 in [0, 0],
  // Tile sizes {6, 8} satisfy these constraints.
  std::vector<int64_t> possible_tile_parameters({6, 8});
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(possible_tile_parameters),
              IsOkAndHolds(true));

  // However, tile sizes {6, 7} do not satisfy these constraints.
  std::vector<int64_t> impossible_tile_parameters({6, 7});
  EXPECT_THAT(
      analysis->ParametersSatisfyConstraints(impossible_tile_parameters),
      IsOkAndHolds(false));

  // Passing too few tile parameters results in an error since constraints can
  // not be properly evaluated.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(/*tile_parameters==*/{6}),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Passing tile parameters that satisfy the constraints should let us compute
  // a TiledHloComputation.
  TF_EXPECT_OK(
      analysis->ParametersSatisfyConstraints(possible_tile_parameters));

  // Passing tile parameters that do not satisfy the constraints should result
  // in an error...
  EXPECT_THAT(analysis->ComputeTiledHloInstructions(impossible_tile_parameters),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // ... unless we pinky-promise (lie) that they satisfy the constraints ;)
  TF_EXPECT_OK(analysis->ComputeTiledHloInstructions(
      impossible_tile_parameters, /*constraints_are_known_satisfied=*/true));
}

TEST_F(SymbolicTileAnalysisTest, EmitterSpecificConstraintsAreUsedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
  fusion {
    p0 = f32[16,32] parameter(0)
    ROOT add = f32[16,32] add(p0, p0)
  }

  ENTRY main {
    p0 = f32[16,32] parameter(0)
    ROOT fusion = f32[16,32] fusion(p0), kind=kLoop, calls=fusion
  })"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(
      module.get(), FakeEmitterSpecificConstraints::GetBuilder());

  ASSERT_TRUE(analysis.has_value());

  // FakeEmitterSpecificConstraints require that the tile size along the first
  // dimension is exactly half the size of the axis. Tile sizes {5, 32} do not
  // satisfy emitter-specific constraints.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({5, 32}),
              IsOkAndHolds(false));

  // However, tile sizes {8, 32} do satisfy emitter-specific constraints.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({8, 32}),
              IsOkAndHolds(true));
}

TEST_F(SymbolicTileAnalysisTest, ConstraintsAreAggregatedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,48,4,8]{3,2,1,0} parameter(0)
  p1 = f32[1,8,6,32]{3,2,1,0} parameter(1)
  bitcast_p0 = f32[48,32]{1,0} bitcast(p0)
  bitcast_p1 = f32[48,32]{1,0} bitcast(p1)
  ROOT add = f32[48,32]{1,0} add(bitcast_p0, bitcast_p1)
}

ENTRY main {
  p0 = f32[1,48,4,8]{3,2,1,0} parameter(0)
  p1 = f32[1,8,6,32]{3,2,1,0} parameter(1)
  ROOT fusion = f32[48,32]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  // Each bitcast in the above module introduces one disjoint constraint. Once
  // they are aggregated, we have four disjoint constraints!
  const ConstraintExpression& constraints =
      analysis->GetTilingSpecification().constraints();
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "6 mod d0 in [0, 0] && 8 mod d1 in [0, 0] || "
                               "6 mod d0 in [0, 0] && d1 mod 8 in [0, 0] || "
                               "8 mod d1 in [0, 0] && d0 mod 6 in [0, 0] || "
                               "d0 mod 6 in [0, 0] && d1 mod 8 in [0, 0]"));
}

bool AlwaysValid(absl::Span<const int64_t>) { return true; }

TEST(GetGoodTilingsTest, ReturnsOneTilingWhenRankIsZero) {
  EXPECT_EQ(GetGoodTilings({}, AlwaysValid),
            TilingVector{SymbolicTileAnalysis::Tiling{}});
}

TEST(GetGoodTilingsTest, ReturnsPowersOfTwoAndTheDimSizeForRankOne) {
  EXPECT_EQ(GetGoodTilings({1}, AlwaysValid), TilingVector{{1}});
  EXPECT_EQ(GetGoodTilings({2}, AlwaysValid), TilingVector({{1}, {2}}));
  EXPECT_EQ(GetGoodTilings({3}, AlwaysValid), TilingVector({{1}, {2}, {3}}));
  EXPECT_EQ(GetGoodTilings({4}, AlwaysValid), TilingVector({{1}, {2}, {4}}));
  EXPECT_EQ(GetGoodTilings({5}, AlwaysValid),
            TilingVector({{1}, {2}, {4}, {5}}));
  EXPECT_EQ(GetGoodTilings({11}, AlwaysValid),
            TilingVector({{1}, {2}, {4}, {8}, {11}}));
}

TEST(GetGoodTilingsTest, CreatesCartesianProductForRankTwo) {
  EXPECT_EQ(GetGoodTilings({3, 4}, AlwaysValid), TilingVector({{1, 1},
                                                               {1, 2},
                                                               {1, 4},
                                                               {2, 1},
                                                               {2, 2},
                                                               {2, 4},
                                                               {3, 1},
                                                               {3, 2},
                                                               {3, 4}}));
}

TEST(GetGoodTilingsTest, CreatesCartesianProductForRankThree) {
  EXPECT_EQ(GetGoodTilings({3, 4, 2}, AlwaysValid), TilingVector({{1, 1, 1},
                                                                  {1, 1, 2},
                                                                  {1, 2, 1},
                                                                  {1, 2, 2},
                                                                  {1, 4, 1},
                                                                  {1, 4, 2},
                                                                  {2, 1, 1},
                                                                  {2, 1, 2},
                                                                  {2, 2, 1},
                                                                  {2, 2, 2},
                                                                  {2, 4, 1},
                                                                  {2, 4, 2},
                                                                  {3, 1, 1},
                                                                  {3, 1, 2},
                                                                  {3, 2, 1},
                                                                  {3, 2, 2},
                                                                  {3, 4, 1},
                                                                  {3, 4, 2}}));
}

TEST(GetGoodTilingsTest, FiltersTheTilingsUsingThePredicate) {
  auto all_even = [](absl::Span<const int64_t> tile_sizes) {
    return absl::c_all_of(tile_sizes,
                          [](int64_t tile_size) { return tile_size % 2 == 0; });
  };

  EXPECT_EQ(GetGoodTilings({3, 4}, all_even), TilingVector({{2, 2}, {2, 4}}));

  auto all_equal = [](absl::Span<const int64_t> tile_sizes) {
    return absl::c_all_of(tile_sizes, [&](int64_t tile_size) {
      return tile_size == tile_sizes.at(0);
    });
  };

  EXPECT_EQ(GetGoodTilings({3, 3, 3}, all_equal),
            TilingVector({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}));
}

TEST_F(SymbolicTileAnalysisTest,
       GetGoodTilingsWorksTakingConstraintsIntoAccount) {
  // The module was chosen (from SymbolicTileTest) because it has a constraint
  // on the tile sizes.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,8,6,1]{3,2,1,0} parameter(0)
  ROOT bitcast = f32[48,1]{1,0} bitcast(p0)
}

ENTRY main {
  p0 = f32[1,8,6,1]{3,2,1,0} parameter(0)
  ROOT fusion = f32[48,1]{1,0} fusion(p0), kind=kLoop, calls=fusion
})"));

  std::optional<SymbolicTileAnalysis> opt_analysis =
      TryAnalyzeModule(module.get());
  ASSERT_TRUE(opt_analysis.has_value());

  const SymbolicTileAnalysis& analysis = opt_analysis.value();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicTileAnalysis::Tiling> good_tilings,
      analysis.GetGoodTilings());
  // The constraint on the 1st dimension is
  //   6 mod d0 in [0, 0] || d0 mod 6 in [0, 0],
  // and only 48, 1, and 2 fulfill it from the set of possible tile sizes
  // (1, 2, 4, 8, 16, 32, 48).
  // There is no constraint on the 2nd dimension.
  EXPECT_EQ(good_tilings, std::vector<SymbolicTileAnalysis::Tiling>(
                              {{1, 1}, {2, 1}, {48, 1}}));
}

// Logs the tilings if VLOG level 1 is enabled.
//
// Use these arguments to see the log:
// --test_output=all
// --test_arg=--logtostderr
// --test_arg=--vmodule=symbolic_tile_analysis_test=1
void LogTilingsIfVlog1(absl::Span<const SymbolicTileAnalysis::Tiling> tilings) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Tilings: {";
    for (const SymbolicTileAnalysis::Tiling& tiling : tilings) {
      LOG(INFO) << "{" << absl::StrJoin(tiling, ",") << "},";
    }
    LOG(INFO) << "}";
  }
}

TEST_F(SymbolicTileAnalysisTest, GetGoodTilingsWorksForSoftmaxExample) {
  // The example is from
  // https://github.com/google/paxml/blob/91893818862645f5e9f23b84f530e611551745f6/paxml/contrib/gpu/scripts_gpu/configs.py#L107-L120.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

add_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  bitcast = f32[4,2048,50304] bitcast(param_0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  bitcast.1 = f32[4,2048] bitcast(reduce)
  broadcast = f32[4,2048,50304] broadcast(bitcast.1), dimensions={0,1}
  subtract = f32[4,2048,50304] subtract(bitcast, broadcast)
  exponential = f32[4,2048,50304] exponential(subtract)
  constant.1 = f32[] constant(0)
  reduce.1 = f32[4,2048] reduce(exponential, constant.1), dimensions={2}, to_apply=add_computation
  log = f32[4,2048] log(reduce.1)
  broadcast.1 = f32[4,2048,50304] broadcast(log), dimensions={0,1}
  ROOT subtract.1 = f32[4,2048,50304] subtract(subtract, broadcast.1)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[4,2048,50304] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> opt_analysis =
      TryAnalyzeModule(module.get());
  ASSERT_TRUE(opt_analysis.has_value());
  const SymbolicTileAnalysis& analysis = opt_analysis.value();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicTileAnalysis::Tiling> good_tilings,
      analysis.GetGoodTilings());
  EXPECT_THAT(good_tilings, Not(IsEmpty()));
  LogTilingsIfVlog1(good_tilings);
}

TEST_F(SymbolicTileAnalysisTest,
       GetGoodTilingsWorksForSoftmaxAndReduceExample) {
  // The example is from
  // https://github.com/google/paxml/blob/91893818862645f5e9f23b84f530e611551745f6/paxml/contrib/gpu/scripts_gpu/configs.py#L107-L120.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

add_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  param_1 = s32[4,2048] parameter(1)
  broadcast = s32[4,2048,50304] broadcast(param_1), dimensions={0,1}
  iota = s32[4,2048,50304] iota(), iota_dimension=2
  compare = pred[4,2048,50304] compare(broadcast, iota), direction=EQ
  bitcast = f32[4,2048,50304] bitcast(param_0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  bitcast.1 = f32[4,2048] bitcast(reduce)
  broadcast.1 = f32[4,2048,50304] broadcast(bitcast.1), dimensions={0,1}
  subtract = f32[4,2048,50304] subtract(bitcast, broadcast.1)
  exponential = f32[4,2048,50304] exponential(subtract)
  constant.1 = f32[] constant(0)
  reduce.1 = f32[4,2048] reduce(exponential, constant.1), dimensions={2}, to_apply=add_computation
  log = f32[4,2048] log(reduce.1)
  broadcast.2 = f32[4,2048,50304] broadcast(log), dimensions={0,1}
  subtract.1 = f32[4,2048,50304] subtract(subtract, broadcast.2)
  constant.2 = f32[] constant(0)
  broadcast.3 = f32[4,2048,50304] broadcast(constant.2), dimensions={}
  select = f32[4,2048,50304] select(compare, subtract.1, broadcast.3)
  bitcast.2 = f32[4,2048,393,128] bitcast(select)
  ROOT reduce.2 = f32[4,2048,393] reduce(bitcast.2, constant.2), dimensions={3}, to_apply=add_computation
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  param_1 = s32[4,2048] parameter(1)
  ROOT fusion = f32[4,2048,393] fusion(param_0, param_1), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton_softmax"}}
}
)"));

  std::optional<SymbolicTileAnalysis> opt_analysis =
      TryAnalyzeModule(module.get());
  ASSERT_TRUE(opt_analysis.has_value());
  const SymbolicTileAnalysis& analysis = opt_analysis.value();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicTileAnalysis::Tiling> good_tilings,
      analysis.GetGoodTilings());
  EXPECT_THAT(good_tilings, Not(IsEmpty()));
  LogTilingsIfVlog1(good_tilings);
}

// This test means to catch integer overflow errors when run with ASan build.
TEST_F(SymbolicTileAnalysisTest,
       FusionWithNumberOfTilesLargerThanInt32MaxIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule softmax

fused_computation {
  param_0 = f16[65538,32768]{1,0} parameter(0)
  ROOT log = f16[65538,32768]{1,0} log(param_0)
}

ENTRY main {
  param_0 = f16[65538,32768]{1,0} parameter(0)
  ROOT fusion = f16[65538,32768]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{1, 1},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  EXPECT_THAT(*tiled_hlo_computation.GetRoots()[0],
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{1, 1},
                  /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> (pid_0 floordiv 32768, pid_0 mod 32768),
    domain:
    pid_0 in [0, 2147549183]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, CanComputeTiledHloInstructionsWithRTVars) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  ROOT maximum = s32[] maximum(param_0, param_1)
}

fused_computation {
  src = s32[2,2,258] parameter(0)
  of1 = s32[] parameter(1)
  of2 = s32[] parameter(2)
  of3 = s32[] parameter(3)
  ds = s32[1,2,32] dynamic-slice(s32[2,2,258] src, s32[] of1, s32[] of2, s32[] of3),
    dynamic_slice_sizes={1, 2, 32}
  c0 = s32[] constant(0)
  ROOT reduce = s32[1,2] reduce(ds, c0), dimensions={2}, to_apply=max_computation
}

ENTRY main {
  param_0 = s32[2,2,258] parameter(0)
  param_1 = s32[] parameter(1)
  param_2 = s32[] parameter(2)
  param_3 = s32[] parameter(3)
  ROOT fusion = s32[1,2] fusion(param_0, param_1, param_2, param_3), kind=kLoop,
      calls=fused_computation
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{1, 1},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* dynamic_slice =
      tiled_hlo_computation.GetRoots()[0]->operand(0);
  const TiledHloInstruction* param_0_tile = dynamic_slice->operand(0);

  EXPECT_THAT(*dynamic_slice, MatchTiledHloInstruction(
                                  /*tile_sizes=*/{1, 1, 32},
                                  /*tile_strides=*/{0, 1, 1},
                                  /*tile_offsets_indexing=*/R"(
    (pid_0) -> (0, pid_0, 0),
    domain:
    pid_0 in [0, 1]
  )"));

  EXPECT_THAT(*param_0_tile, MatchTiledHloInstruction(
                                 /*tile_sizes=*/{1, 1, 32},
                                 /*tile_strides=*/{0, 1, 1},
                                 /*tile_offsets_indexing=*/R"(
    (pid_0){rt0, rt1} -> (rt0, pid_0, rt1),
    domain:
    pid_0 in [0, 1],
    rt0 in [0, 1],
    rt1 in [0, 226]
  )"));
}

TEST_F(SymbolicTileAnalysisTest,
       BailsOutOnReshapeWhenStandaloneSymbolicTileDerivationFails) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

add_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation {
  p0 = f32[2,128,128] parameter(0)
  // We use two successive bitcasts here as a hack to produce the right
  // failure---otherwise, the derivation failure may occur on the parameter
  // instruction.
  bitcast_fix = f32[16384,1,2] bitcast(p0)
  bitcast = f32[2,128,128] bitcast(bitcast_fix)
  c0 = f32[] constant(0)
  reduce = f32[2,128] reduce(bitcast, c0), dimensions={2},
    to_apply=add_computation
}

ENTRY main {
  p0 = f32[2,128,128] parameter(0)
  ROOT fusion = f32[2,128] fusion(p0), kind=kLoop, calls=fused_computation
})"));

  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeComputation(
          *module->entry_computation()
               ->root_instruction()
               ->fused_instructions_computation(),
          &mlir_context_, /*emitter_specific_constraints_builder=*/nullptr);

  ASSERT_TRUE(std::holds_alternative<FusionDecision>(analysis_or_error));
  EXPECT_THAT(std::get<FusionDecision>(analysis_or_error).Explain(),
              ::testing::HasSubstr("Bailing out on reshape"));
}

TEST_F(SymbolicTileAnalysisTest,
       DoesNotBailOutOnFilteredOutHloIfThatHloIsOnlyAnOperand) {
  // This is a regression test for a bug where we would refuse to tile a
  // computation if its operand could not be tiled according to
  // `SymbolicTileAnalysis`.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation {
  p0 = f32[10,10] parameter(0)
  ROOT reshape = f32[100] reshape(p0)
}

ENTRY main {
  p0 = f32[10,2] parameter(0)
  p1 = f32[2,10] parameter(1)
  // Note: this will need upgrading once `SymbolicTileAnalysis` stops filtering
  // out dots.
  untileable_dot = f32[10,10] dot(p0, p1),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT fusion = f32[100] fusion(untileable_dot),
    kind=kLoop, calls=fused_computation
})"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  EXPECT_TRUE(analysis.has_value());
}

TEST_F(SymbolicTileAnalysisTest, IotaAlwaysHasTileOffsetsIndexingSet) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  ROOT iota = s32[100] iota(), iota_dimension=0
}

ENTRY main {
  ROOT fusion = s32[100] fusion(), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{4},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/false));

  const TiledHloInstruction* iota = tiled_hlo_computation.GetRoots()[0];
  EXPECT_THAT(iota->tile_offsets_indexing().status(), ::tsl::testing::IsOk());
}

TEST_F(SymbolicTileAnalysisTest, TileNestedDotFusions) {
  // Tile a dot of [8192,256] x [256,512] = [8192,512].
  // [M, K] * [K, N] = [M, N].
  // M is tiled to 128, K: 8, N: 32.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
lhs {
  lhs.p0 = bf16[8192,256]{1,0} parameter(0)
  ROOT lhs.root = bf16[8192,256]{1,0} negate(lhs.p0)
}

rhs {
  ROOT rhs.p0 = bf16[256,512]{1,0} parameter(0)
}

dot {
  dot.p0 = bf16[8192,256]{1,0} parameter(0)
  dot.p1 = bf16[256,512]{1,0} parameter(1)

  dot.lhs = bf16[8192,256]{1,0} fusion(dot.p0),
    kind=kCustom, calls=lhs, backend_config={
      "fusion_backend_config":{
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","8"]}]}}}
  dot.rhs = bf16[256,512]{1,0} fusion(dot.p1),
    kind=kCustom, calls=rhs, backend_config={
      "fusion_backend_config":{
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["8","32"]}]}}}

  ROOT dot = bf16[8192,512]{1,0} dot(dot.lhs, dot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  main.p0 = bf16[8192,256]{1,0} parameter(0)
  main.p1 = bf16[256,512]{1,0} parameter(1)
  ROOT fusion = bf16[8192,512]{1,0} fusion(main.p0, main.p1),
    kind=kCustom, calls=dot
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  const HloInstruction* dot_hlo =
      module->entry_computation()->root_instruction()->fused_expression_root();
  ASSERT_TRUE(analysis.has_value());
  Tiling dot_tiling = Tiling(Tiling::TileMapping({{dot_hlo, {8, 128, 32}}}));
  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*tiling=*/dot_tiling,
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  auto match_dot = MatchTiledHloInstruction(
      /*tile_sizes=*/{128, 32},
      /*tile_strides=*/{1, 1},
      /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 16) * 128, (pid_0 mod 16) * 32),
    domain:
    pid_0 in [0, 1023]
  )");
  const TiledHloInstruction* dot = tiled_hlo_computation.GetRoots().front();
  EXPECT_THAT(*dot, match_dot);

  // LHS nested fusion.
  const TiledHloInstruction* lhs = dot->operand(0);
  EXPECT_THAT(lhs->operands(), IsEmpty()) << "operand shouldn't be analyzed.";
  const TiledHloComputation* lhs_nested_computation =
      static_cast<const TiledHloFusionInstruction*>(lhs)->called_computation();
  auto match_lhs = MatchTiledHloInstruction(
      /*tile_sizes=*/{128, 8},
      /*tile_strides=*/{1, 1},
      /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 floordiv 512) * 128, (pid_0 mod 32) * 8),
    domain:
    pid_0 in [0, 32767]
  )");
  const TiledHloInstruction* negate =
      lhs_nested_computation->GetRoots().front();
  EXPECT_THAT(*negate, match_lhs);
  const TiledHloInstruction* lhs_p0 = negate->operand(0);
  EXPECT_THAT(*lhs_p0, match_lhs);
  EXPECT_EQ(lhs_p0->hlo(), lhs->hlo()->operand(0))
      << "tiled parameter is not the operand of the nested fusion instruction.";

  // RHS nested fusion.
  const TiledHloInstruction* rhs = dot->operand(1);
  EXPECT_THAT(rhs->operands(), IsEmpty()) << "operand shouldn't be analyzed.";
  const TiledHloComputation* rhs_nested_computation =
      static_cast<const TiledHloFusionInstruction*>(rhs)->called_computation();
  auto match_rhs = MatchTiledHloInstruction(
      /*tile_sizes=*/{8, 32},
      /*tile_strides=*/{1, 1},
      /*tile_offsets_indexing=*/R"(
    (pid_0) -> ((pid_0 mod 32) * 8, ((pid_0 floordiv 32) mod 16) * 32),
    domain:
    pid_0 in [0, 32767]
  )");
  const TiledHloInstruction* rhs_p0 =
      rhs_nested_computation->GetRoots().front();
  EXPECT_THAT(*rhs_p0, match_rhs);
  EXPECT_EQ(rhs_p0->hlo(), rhs->hlo()->operand(0))
      << "tiled parameter is not the operand of the nested fusion instruction.";
}

TEST_F(SymbolicTileAnalysisTest, EmptyFusionsAreSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  ROOT fusion.p0 = f32[8] parameter(0)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  ROOT fusion = f32[8] fusion(p0), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{2},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));
  const TiledHloInstruction* root = tiled_hlo_computation.GetRoots()[0];
  EXPECT_THAT(*root, MatchTiledHloInstruction(
                         /*tile_sizes=*/{2}, /*tile_strides=*/{1},
                         /*tile_offsets_indexing=*/
                         "(pid_0) -> (pid_0 * 2), domain: pid_0 in [0, 3]"));
}

TEST_F(SymbolicTileAnalysisTest, BailsOutOnRootFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
nested {
  ROOT nested.root = f32[] parameter(0)
}

fusion {
  fusion.p0 = f32[] parameter(0)
  ROOT fusion.root = f32[] fusion(fusion.p0), kind=kLoop, calls=nested
}

ENTRY main {
  main.p0 = f32[] parameter(0)
  ROOT main.root = f32[] fusion(main.p0), kind=kLoop, calls=fusion
})"));
  EXPECT_FALSE(TryAnalyzeModule(module.get()).has_value());
}

TEST_F(SymbolicTileAnalysisTest,
       ConcatenatesInNestedGemmFusionsAllowSymbolicTileDerivation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[6] parameter(0)
  p1 = bf16[6] parameter(1)
  p2 = bf16[6] parameter(2)
  ROOT concatenate = bf16[18] concatenate(p0, p1, p2), dimensions={0}
}

ENTRY main {
  p0 = bf16[6] parameter(0)
  p1 = bf16[6] parameter(1)
  p2 = bf16[6] parameter(2)
  ROOT fusion = bf16[18] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  EXPECT_TRUE(analysis.has_value());
}

TEST_F(SymbolicTileAnalysisTest,
       ConcatenatesOutsideOfNestedGemmFusionsForbidSymbolicTileDerivation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[6] parameter(0)
  p1 = bf16[6] parameter(1)
  p2 = bf16[6] parameter(2)
  ROOT concatenate = bf16[18] concatenate(p0, p1, p2), dimensions={0}
}

ENTRY main {
  p0 = bf16[6] parameter(0)
  p1 = bf16[6] parameter(1)
  p2 = bf16[6] parameter(2)
  ROOT fusion = bf16[18] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  EXPECT_FALSE(analysis.has_value());
}

TEST_F(SymbolicTileAnalysisTest,
       ConcatenateOperandsInNestedGemmFusionsAreProvidedCorrectTilingBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
nest0 {
  ROOT p0 = s32[128] parameter(0)
}

nest1 {
  ROOT p0 = s32[128] parameter(0)
}

nest2 {
  ROOT p0 = s32[128] parameter(0)
}

concatenate {
  p0 = s32[128] parameter(0)
  p1 = s32[128] parameter(1)
  p2 = s32[128] parameter(2)

  fusion0 = s32[128] fusion(p0), kind=kCustom, calls=nest0
  fusion1 = s32[128] fusion(p1), kind=kCustom, calls=nest1
  fusion2 = s32[128] fusion(p2), kind=kCustom, calls=nest2

  ROOT concatenate = s32[384] concatenate(fusion0, fusion1, fusion2), dimensions={0}
}

ENTRY main {
  p0 = s32[128] parameter(0)
  p1 = s32[128] parameter(1)
  p2 = s32[128] parameter(2)
  ROOT fusion = s32[384] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis->ComputeTiledHloInstructions(
          /*output_tile_sizes=*/{32}, /*constraints_are_known_satisfied=*/false,
          /*compute_all_tile_offset_indexing_maps=*/true));

  // Gather the three nested fusions present in the module, in order.
  std::vector<const TiledHloFusionInstruction*> nested_fusions(3, nullptr);
  for (const TiledHloInstruction* tiled_instr :
       tiled_hlo_computation.instructions()) {
    if (auto tiled_fusion =
            dynamic_cast<const TiledHloFusionInstruction*>(tiled_instr)) {
      nested_fusions[tiled_fusion->hlo()->operand(0)->parameter_number()] =
          tiled_fusion;
    }
  }

  // Ensure that each parameter has domain bounds with the proper offsets.
  // Concatenate creates partial functions.
  const TiledHloInstruction* nested_p0 =
      nested_fusions[0]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p0,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{32}, /*tile_strides=*/{1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> (pid_0 * 32), domain: pid_0 in [0, 3]"));
  const TiledHloInstruction* nested_p1 =
      nested_fusions[1]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p1,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{32}, /*tile_strides=*/{1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> (pid_0 * 32 - 128), domain: pid_0 in [4, 7]"));

  const TiledHloInstruction* nested_p2 =
      nested_fusions[2]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p2,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{32}, /*tile_strides=*/{1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> (pid_0 * 32 - 256), domain: pid_0 in [8, 11]"));

  // Ensure that providing tile sizes that do not divide the resulting offsets
  // results in the tiling being rejected, even if we pretend that `33`
  // satisfies the constraints.
  auto tiled_hlo_computation_or = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{33}, /*constraints_are_known_satisfied=*/true,
      /*compute_all_tile_offset_indexing_maps=*/false);

  EXPECT_THAT(tiled_hlo_computation_or,
              tsl::testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  ::testing::HasSubstr("not divisible by tile size")));
}

// Same as ConcatenateOperandsInNestedGemmFusionsAreProvidedCorrectTilingBounds,
// but with rank 2.
TEST_F(SymbolicTileAnalysisTest,
       2DConcatenateOperandsInNestedGemmFusionsAreProvidedCorrectTilingBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
nest0 {
  ROOT p0 = s32[64,128] parameter(0)
}

nest1 {
  ROOT p0 = s32[64,128] parameter(0)
}

nest2 {
  ROOT p0 = s32[64,128] parameter(0)
}

concatenate {
  p0 = s32[64,128] parameter(0)
  p1 = s32[64,128] parameter(1)
  p2 = s32[64,128] parameter(2)

  fusion0 = s32[64,128] fusion(p0), kind=kCustom, calls=nest0
  fusion1 = s32[64,128] fusion(p1), kind=kCustom, calls=nest1
  fusion2 = s32[64,128] fusion(p2), kind=kCustom, calls=nest2

  ROOT concatenate = s32[64,384] concatenate(fusion0, fusion1, fusion2), dimensions={1}
}

ENTRY main {
  p0 = s32[64,128] parameter(0)
  p1 = s32[64,128] parameter(1)
  p2 = s32[64,128] parameter(2)
  ROOT fusion = s32[64,384] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*output_tile_sizes=*/{16, 32},
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  // Gather the three nested fusions present in the module, in order.
  std::vector<const TiledHloFusionInstruction*> nested_fusions(3, nullptr);
  for (const TiledHloInstruction* tiled_instr :
       tiled_hlo_computation.instructions()) {
    if (auto tiled_fusion =
            dynamic_cast<const TiledHloFusionInstruction*>(tiled_instr)) {
      nested_fusions[tiled_fusion->hlo()->operand(0)->parameter_number()] =
          tiled_fusion;
    }
  }

  // Ensure that each parameter has domain bounds with the proper offsets.
  // Concatenate creates partial functions.
  const TiledHloInstruction* nested_p0 =
      nested_fusions[0]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p0,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{16, 32}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> ((pid_0 floordiv 12) * 16, (pid_0 mod 12) * 32), "
                  "domain: pid_0 in [0, 47], pid_0 mod 12 in [0, 3]"));
  const TiledHloInstruction* nested_p1 =
      nested_fusions[1]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p1,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{16, 32}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> ((pid_0 floordiv 12) * 16, (pid_0 mod 12) * 32 - "
                  "128), domain: pid_0 in [0, 47], pid_0 mod 12 in [4, 7]"));

  const TiledHloInstruction* nested_p2 =
      nested_fusions[2]->called_computation()->GetRoots()[0];
  EXPECT_THAT(*nested_p2,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{16, 32}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> ((pid_0 floordiv 12) * 16, (pid_0 mod 12) * 32 - "
                  "256), domain: pid_0 in [0, 47], pid_0 mod 12 in [8, 11]"));

  // Ensure that providing tile sizes that do not divide the resulting offsets
  // results in the tiling being rejected, even if we pretend that `33`
  // satisfies the constraints.
  auto tiled_hlo_computation_or = analysis->ComputeTiledHloInstructions(
      /*output_tile_sizes=*/{16, 33}, /*constraints_are_known_satisfied=*/true,
      /*compute_all_tile_offset_indexing_maps=*/false);

  EXPECT_THAT(tiled_hlo_computation_or,
              tsl::testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  ::testing::HasSubstr("not divisible by tile size")));
}

TEST_F(SymbolicTileAnalysisTest, TrivialDimensionParametersArePreserved) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
lhs {
  ROOT p0 = f32[137,115] parameter(0)
}

rhs {
  p0 = f32[1,115] parameter(0)
  ROOT root = f32[1,115] convert(p0)
}

dot {
  p0 = f32[137,115] parameter(0)
  p1 = f32[1,115] parameter(1)

  lhs = f32[137,115] fusion(p0),
    kind=kCustom, calls=lhs, backend_config={
      "fusion_backend_config":{
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","32"]}]}}}
  rhs = f32[1,115] fusion(p1),
    kind=kCustom, calls=rhs, backend_config={
      "fusion_backend_config":{
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","32"]}]}}}

  ROOT dot = f32[137,1] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  p0 = f32[137,115] parameter(0)
  p1 = f32[1,115] parameter(1)
  ROOT fusion = f32[137,1] fusion(p0, p1),
    kind=kCustom, calls=dot
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const HloInstruction* dot_hlo =
      module->entry_computation()->root_instruction()->fused_expression_root();
  Tiling tiling(Tiling::TileMapping{{dot_hlo, {32, 16, 16}}});
  TF_ASSERT_OK_AND_ASSIGN(TiledHloComputation tiled_hlo_computation,
                          analysis->ComputeTiledHloInstructions(
                              /*tiling=*/tiling,
                              /*constraints_are_known_satisfied=*/false,
                              /*compute_all_tile_offset_indexing_maps=*/true));

  const TiledHloInstruction* dot = tiled_hlo_computation.GetRoots().front();
  ASSERT_EQ(dot->hlo()->opcode(), HloOpcode::kDot);

  const TiledHloFusionInstruction* lhs_fusion =
      static_cast<const TiledHloFusionInstruction*>(dot->operand(0));
  const TiledHloFusionInstruction* rhs_fusion =
      static_cast<const TiledHloFusionInstruction*>(dot->operand(1));

  EXPECT_THAT(
      *lhs_fusion->called_computation()->GetRoots().front(),
      MatchTiledHloInstruction(
          /*tile_sizes=*/{16, 32}, /*tile_strides=*/{1, 1},
          /*tile_offsets_indexing=*/
          "(pid_0) -> ((pid_0 floordiv 4) * 16, (pid_0 mod 4) * 32), domain: "
          "pid_0 in [0, 35]"));

  // RHS has a trivial dimension. We make sure here that the requested padding
  // is propagated as requested, and not simplified away (which would result in
  // an invalid tile size of size "1").
  // The trivial argument is still expected to be eliminated in the
  // `tile_offsets_indexing` map, since this allows for more effective CSE.
  EXPECT_THAT(*rhs_fusion->called_computation()->GetRoots().front(),
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{16, 32}, /*tile_strides=*/{1, 1},
                  /*tile_offsets_indexing=*/
                  "(pid_0) -> (0, (pid_0 mod 4) * 32), domain: "
                  "pid_0 in [0, 35]"));
}

using TilingSpecificationTest = SymbolicTileAnalysisTest;

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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
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
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const TilingSpecification& tiling_spec = analysis->GetTilingSpecification();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot = root->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);

  Tiling exact_tiling(Tiling::TileMapping{{dot, {1, 1, 1}}});
  EXPECT_TRUE(exact_tiling.ConformsTo(tiling_spec));
}

TEST_F(SymbolicTileAnalysisTest,
       SymbolicTilesAlwaysDependOnAllTheHiddenParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[4,8] parameter(0)
  p1 = f32[8,16] parameter(1)
  dot1 = f32[4,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot2 = f32[4,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT add = f32[4,16] add(dot1, dot2)
}

ENTRY main {
  p0 = f32[4,8] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[4,16] fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  // Here we have 4 parameters: 1 nested parameter for each dot, and 2 output
  // parameters for the add.
  constexpr int64_t kNumTilingParameters = 4;

  EXPECT_EQ(analysis->GetTilingSpecification().num_parameters(),
            kNumTilingParameters);

  for (const auto& instruction : analysis->GetSymbolicTiledHloComputation()) {
    EXPECT_EQ(instruction->symbolic_tile().size_map().getNumDims(),
              kNumTilingParameters);
    // Symbols should also have been completely eliminated from all maps.
    EXPECT_EQ(instruction->symbolic_tile().size_map().getNumSymbols(), 0);
  }
}

TEST_F(SymbolicTileAnalysisTest,
       NestedConstraintsArePropagatedToTheOutermostAnalysis) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
lhs {
  lhs.p0 = bf16[8,1024,256] parameter(0)
  ROOT lhs.root = bf16[8192,256] reshape(lhs.p0)
}

rhs {
  ROOT rhs.p0 = bf16[256,512] parameter(0)
}

dot {
  dot.p0 = bf16[8,1024,256] parameter(0)
  dot.p1 = bf16[256,512] parameter(1)

  dot.lhs = bf16[8192,256] fusion(dot.p0), kind=kCustom, calls=lhs
  dot.rhs = bf16[256,512] fusion(dot.p1), kind=kCustom, calls=rhs

  ROOT dot = bf16[8192,512]{1,0} dot(dot.lhs, dot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  main.p0 = bf16[8,1024,256] parameter(0)
  main.p1 = bf16[256,512] parameter(1)
  ROOT fusion = bf16[8192,512] fusion(main.p0, main.p1), kind=kCustom, calls=dot
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  EXPECT_FALSE(
      analysis->GetTilingSpecification().constraints().IsAlwaysSatisfied());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

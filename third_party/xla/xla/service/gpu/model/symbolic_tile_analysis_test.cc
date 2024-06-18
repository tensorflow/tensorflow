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
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using detail::GetGoodTilings;
using ::testing::ElementsAreArray;
using ::testing::ExplainMatchResult;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using TilingVector = std::vector<SymbolicTileAnalysis::Tiling>;

MATCHER_P3(MatchTiledHloInstructionImpl, tile_sizes, tile_strides,
           block_id_to_tile_offsets_indexing, "") {
  return ExplainMatchResult(ElementsAreArray(tile_sizes), arg.tile_sizes(),
                            result_listener) &&
         ExplainMatchResult(ElementsAreArray(tile_strides), arg.tile_strides(),
                            result_listener) &&
         ExplainMatchResult(MatchIndexingMap(block_id_to_tile_offsets_indexing),
                            arg.block_id_to_tile_offsets_indexing(),
                            result_listener);
}

Matcher<const TiledHloInstruction> MatchTiledHloInstruction(
    absl::Span<const int64_t> tile_sizes,
    absl::Span<const int64_t> tile_strides,
    absl::string_view block_id_to_tile_offsets_indexing) {
  return MatchTiledHloInstructionImpl(tile_sizes, tile_strides,
                                      block_id_to_tile_offsets_indexing);
}

class SymbolicTileAnalysisTest : public HloTestBase {
 public:
  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(HloModule* module) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_);

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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 10}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();

  EXPECT_THAT(root->block_id_to_tile_offsets_indexing(), MatchIndexingMap(R"(
    (d0) -> (d0 floordiv 10, (d0 mod 10) * 10)
    domain:
    d0 in [0, 19]
  )"));

  auto p0_from_subtract0 = root->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0)->operand(0);

  EXPECT_THAT(*p0_from_subtract0, MatchTiledHloInstruction(
                                      /*tile_sizes=*/{1, 10},
                                      /*tile_strides=*/{1, 1},
                                      /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> (d0 floordiv 10, (d0 mod 10) * 10)
    domain:
    d0 in [0, 19]
  )"));

  EXPECT_THAT(*p0_from_subtract1, MatchTiledHloInstruction(
                                      /*tile_sizes=*/{1, 97},
                                      /*tile_strides=*/{1, 1},
                                      /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> (d0 floordiv 10, 0)
    domain:
    d0 in [0, 19]
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
      analysis->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 10}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();

  auto p0_from_subtract0 = root->operand(0)->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0);

  EXPECT_EQ(p0_from_subtract0, p0_from_subtract1);
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
      analysis.ComputeTiledHloInstructions(/*tile_parameters=*/{1, 97}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();

  const TiledHloInstruction* p0_from_producer =
      root->operand(1)->operand(0)->operand(0)->operand(0);
  const TiledHloInstruction* p0_from_consumer = root->operand(0);

  EXPECT_EQ(p0_from_producer, p0_from_consumer);

  EXPECT_THAT(*p0_from_producer,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{1, 97}, /*tile_strides=*/{1, 1},
                  /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> (d0, 0)
    domain: d0 in [0, 1]
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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis->ComputeTiledHloInstructions(/*tile_parameters=*/{2, 4, 2}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();

  EXPECT_THAT(*root, MatchTiledHloInstruction(
                         /*tile_sizes=*/{2, 4, 2}, /*tile_strides=*/{1, 1, 1},
                         /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> ((d0 floordiv 16) * 2, ((d0 floordiv 8) mod 2) * 4, (d0 mod 8) * 2)
    domain:
    d0 in [0, 31]
  )"));

  EXPECT_THAT(*root->operand(0),
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{4, 2, 2}, /*tile_strides=*/{1, 1, 1},
                  /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> (((d0 floordiv 8) mod 2) * 4, (d0 mod 8) * 2, (d0 floordiv 16) * 2)
    domain:
    d0 in [0, 31]
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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis->ComputeTiledHloInstructions(/*tile_parameters=*/{2, 2}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();
  const TiledHloInstruction* p0_from_slice0 = root->operand(0)->operand(0);
  const TiledHloInstruction* p0_from_slice1 = root->operand(1)->operand(0);

  EXPECT_THAT(*root, MatchTiledHloInstruction(
                         /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                         /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> ((d0 floordiv 4) * 2, (d0 mod 4) * 2)
    domain:
    d0 in [0, 7]
  )"));

  EXPECT_THAT(*p0_from_slice0,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                  /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> ((d0 floordiv 4) * 2, (d0 mod 4) * 2 + 2)
    domain:
    d0 in [0, 7]
  )"));

  EXPECT_THAT(*p0_from_slice1,
              MatchTiledHloInstruction(
                  /*tile_sizes=*/{2, 2}, /*tile_strides=*/{1, 1},
                  /*block_id_to_tile_offsets_indexing=*/R"(
    (d0) -> ((d0 floordiv 4) * 2 + 3, (d0 mod 4) * 2 + 4)
    domain:
    d0 in [0, 7]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,2]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  ROOT dot = f32[1,3]{1,0} dot(p0, p1),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[1,2]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  ROOT fusion = f32[1,3]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  EXPECT_FALSE(TryAnalyzeModule(module.get()).has_value());
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
  EXPECT_THAT(analysis->GetConstraints(), SizeIs(1));
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
  EXPECT_THAT(analysis->GetConstraints(), SizeIs(1));
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
  EXPECT_THAT(analysis->GetConstraints(), SizeIs(2));

  // We expect the constraints here to be
  //    s0 mod 6 in [0, 0]
  //    s1 mod 8 in [0, 0]
  // We expect tile sizes {6, 8} to satisfy these constraints.
  std::vector<int64_t> possible_tile_parameters({6, 8});
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(possible_tile_parameters),
              IsOkAndHolds(true));

  // However, we do not expect tile sizes {6, 7} to satisfy these constraints.
  std::vector<int64_t> impossible_tile_parameters({6, 7});
  EXPECT_THAT(
      analysis->ParametersSatisfyConstraints(impossible_tile_parameters),
      IsOkAndHolds(false));

  // Passing too few tile parameters results in an error since constraints can
  // not be properly evaluated.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(/*tile_parameters==*/{6}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Passing tile parameters that satisfy the constraints should let us compute
  // a TiledHloComputation.
  EXPECT_OK(analysis->ParametersSatisfyConstraints(possible_tile_parameters));

  // Passing tile parameters that do not satisfy the constraints should result
  // in an error...
  EXPECT_THAT(analysis->ComputeTiledHloInstructions(impossible_tile_parameters),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // ... unless we pinky-promise (lie) that they satisfy the constraints ;)
  EXPECT_OK(analysis->ComputeTiledHloInstructions(
      impossible_tile_parameters, /*constraints_are_known_satisfied=*/true));
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
  // Each bitcast in the above module introduces one constraint. Once they are
  // aggregated, we have two!
  EXPECT_THAT(analysis->GetConstraints(), SizeIs(2));
}

TEST_F(SymbolicTileAnalysisTest, BailsOutWhenConstraintsCanNotBeMerged) {
  // TODO(bchetioui): allow merging a constraint with itself.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion {
  p0 = f32[1,48,4,8]{3,2,1,0} parameter(0)
  p1 = f32[1,48,4,8]{3,2,1,0} parameter(1)
  bitcast_p0 = f32[48,32]{1,0} bitcast(p0)
  bitcast_p1 = f32[48,32]{1,0} bitcast(p1)
  ROOT add = f32[48,32]{1,0} add(bitcast_p0, bitcast_p1)
}

ENTRY main {
  p0 = f32[1,48,4,8]{3,2,1,0} parameter(0)
  p1 = f32[1,48,4,8]{3,2,1,0} parameter(1)
  ROOT fusion = f32[48,32]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
})"));
  EXPECT_FALSE(TryAnalyzeModule(module.get()).has_value());
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
  p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
  ROOT bitcast = f32[48,4]{1,0} bitcast(p0)
}

ENTRY main {
  p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
  ROOT fusion = f32[48,4]{1,0} fusion(p0), kind=kLoop, calls=fusion
})"));

  std::optional<SymbolicTileAnalysis> opt_analysis =
      TryAnalyzeModule(module.get());
  ASSERT_TRUE(opt_analysis.has_value());

  const SymbolicTileAnalysis& analysis = opt_analysis.value();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicTileAnalysis::Tiling> good_tilings,
      analysis.GetGoodTilings());
  // The constraint on the 1st dimension is "s0 mod 6 in [0, 0]", and only 48
  // fulfills that from the set of possible tile sizes (1, 2, 4, 8, 16, 32, 48).
  // There is no constraint on the 2nd dimension.
  EXPECT_EQ(good_tilings, std::vector<SymbolicTileAnalysis::Tiling>(
                              {{48, 1}, {48, 2}, {48, 4}}));
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

region {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

region.1 {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  bitcast = f32[4,2048,50304] bitcast(param_0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=region
  bitcast.1 = f32[4,2048] bitcast(reduce)
  broadcast = f32[4,2048,50304] broadcast(bitcast.1), dimensions={0,1}
  subtract = f32[4,2048,50304] subtract(bitcast, broadcast)
  exponential = f32[4,2048,50304] exponential(subtract)
  constant.1 = f32[] constant(0)
  reduce.1 = f32[4,2048] reduce(exponential, constant.1), dimensions={2}, to_apply=region.1
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

}  // namespace
}  // namespace gpu
}  // namespace xla

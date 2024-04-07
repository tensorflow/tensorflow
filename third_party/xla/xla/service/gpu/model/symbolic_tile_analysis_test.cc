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

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

class SymbolicTileAnalysisTest : public HloTestBase {
 public:
  bool SetAnalysis(HloModule* module) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(*module->entry_computation(),
                                                 &mlir_context_);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      analysis_ = std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
      return true;
    }
    return false;
  }

  mlir::MLIRContext mlir_context_;
  std::optional<SymbolicTileAnalysis> analysis_;
};

TEST_F(SymbolicTileAnalysisTest, SimpleNormalizationDiamondIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
max {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0, p1)
}

ENTRY main {
  p0 = f32[2,97]{1,0} parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[2] reduce(p0, constant), dimensions={1}, to_apply=max
  broadcast = f32[2,97]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[2,97]{1,0} subtract(p0, broadcast)
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  const SymbolicTiledHloInstruction* root = analysis_->GetRoot();

  analysis_->SetTileSizes(/*sizes=*/{1, 10});

  EXPECT_THAT(*analysis_->ComputeBlockIdToTileOffsetIndexing(*root),
              MatchIndexingMap(R"(
    (d0) -> (d0 floordiv 10, (d0 mod 10) * 10)
    domain:
    d0 in [0, 19]
  )"));

  auto p0_from_subtract0 = root->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0)->operand(0);

  EXPECT_THAT(analysis_->TileOffsets(*p0_from_subtract0), ElementsAre(0, 0));
  EXPECT_THAT(analysis_->TileSizes(*p0_from_subtract0), ElementsAre(1, 10));
  EXPECT_THAT(analysis_->TileStrides(*p0_from_subtract0), ElementsAre(1, 1));

  EXPECT_THAT(
      *analysis_->ComputeBlockIdToTileOffsetIndexing(*p0_from_subtract0),
      MatchIndexingMap(R"(
    (d0) -> (d0 floordiv 10, (d0 mod 10) * 10)
    domain:
    d0 in [0, 19]
  )"));

  EXPECT_THAT(analysis_->TileOffsets(*p0_from_subtract1), ElementsAre(0, 0));
  EXPECT_THAT(analysis_->TileSizes(*p0_from_subtract1), ElementsAre(1, 97));
  EXPECT_THAT(analysis_->TileStrides(*p0_from_subtract1), ElementsAre(1, 1));

  EXPECT_THAT(
      *analysis_->ComputeBlockIdToTileOffsetIndexing(*p0_from_subtract1),
      MatchIndexingMap(R"(
    (d0) -> (d0 floordiv 10, 0)
    domain:
    d0 in [0, 19]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, ElementwiseDiamondCSEIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[2,97] parameter(0)
  exp = f32[2,97] exponential(p0)
  log = f32[2,97] log(p0)
  ROOT subtract = f32[2,97] subtract(exp, log)
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  const SymbolicTiledHloInstruction* root = analysis_->GetRoot();

  auto p0_from_subtract0 = root->operand(0)->operand(0);
  auto p0_from_subtract1 = root->operand(1)->operand(0);

  EXPECT_EQ(p0_from_subtract0, p0_from_subtract1);
}

TEST_F(SymbolicTileAnalysisTest, TransposeOffsetIndexingIsCorrect) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[8,16,4] parameter(0)
  ROOT transpose = f32[4,8,16] transpose(p0), dimensions={2,0,1}
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  analysis_->SetTileSizes(/*sizes=*/{2, 4, 2});

  const SymbolicTiledHloInstruction* root = analysis_->GetRoot();

  EXPECT_THAT(*analysis_->ComputeBlockIdToTileOffsetIndexing(*root),
              MatchIndexingMap(R"(
    (d0) -> ((d0 floordiv 16) * 2, ((d0 floordiv 8) mod 2) * 4, (d0 mod 8) * 2)
    domain:
    d0 in [0, 31]
  )"));

  EXPECT_THAT(*analysis_->ComputeBlockIdToTileOffsetIndexing(*root->operand(0)),
              MatchIndexingMap(R"(
    (d0) -> (((d0 floordiv 8) mod 2) * 4, (d0 mod 8) * 2, (d0 floordiv 16) * 2)
    domain:
    d0 in [0, 31]
  )"));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[1,2]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  ROOT dot = f32[1,3]{1,0} dot(p0, p1),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

  EXPECT_FALSE(SetAnalysis(module.get()));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedReshape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[1,2]{1,0} parameter(0)
  ROOT reshape = f32[2] reshape(p0)
})"));

  EXPECT_FALSE(SetAnalysis(module.get()));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedBitcast) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[1,2]{1,0} parameter(0)
  ROOT bitcast = f32[2] bitcast(p0)
})"));

  mlir::MLIRContext mlir_ctx;
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeComputation(*module->entry_computation(),
                                               &mlir_ctx);
  EXPECT_FALSE(SetAnalysis(module.get()));
}

TEST_F(SymbolicTileAnalysisTest, BailOutOnUnsupportedConcatenate) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[1,3]{1,0} parameter(0)
  p1 = f32[1,3]{1,0} parameter(1)
  ROOT concatenate = f32[2,3] concatenate(p0, p1), dimensions={0}
})"));

  EXPECT_FALSE(SetAnalysis(module.get()));
}

TEST_F(SymbolicTileAnalysisTest, ComputingIndexingMapFailsWithoutTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY main {
  p0 = f32[4,8]{1,0} parameter(0)
  ROOT exponential = f32[4,8]{1,0} exponential(p0)
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  const SymbolicTiledHloInstruction* root = analysis_->GetRoot();

  EXPECT_THAT(
      analysis_->ComputeBlockIdToTileOffsetIndexing(*root).status().message(),
      ::testing::HasSubstr("SetTileSizes() must be called before"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

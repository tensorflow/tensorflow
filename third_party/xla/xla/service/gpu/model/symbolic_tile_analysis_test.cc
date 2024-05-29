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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::ExplainMatchResult;
using ::testing::Matcher;
using ::testing::Pointee;

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

std::vector<const TiledHloInstruction*> GetInstructionsWithName(
    const TiledHloComputation& tiled_hlo_computation,
    absl::string_view instruction_name) {
  std::vector<const TiledHloInstruction*> result;
  for (const TiledHloInstruction* instruction :
       tiled_hlo_computation.instructions()) {
    if (instruction->hlo()->name() == instruction_name) {
      result.push_back(instruction);
    }
  }
  return result;
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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 10}));

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

TEST_F(SymbolicTileAnalysisTest,
       NormalizationDiamondWithBroadcastAndReshapeIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
max {
  p1.1 = f32[] parameter(1)
  p0.1 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0.1, p1.1)
}

ENTRY main {
  p0 = f32[48,97] parameter(0)
  bitcast = f32[4,12,97] bitcast(p0)

  p1 = pred[4,97] parameter(1)
  broadcast.1 = pred[4,12,97]{2,1,0} broadcast(p1), dimensions={0,2}

  constant.3 = f32[] constant(-2.38197633e+38)
  broadcast.5 = f32[4,12,97] broadcast(constant.3), dimensions={}

  select = f32[4,12,97] select(broadcast.1, bitcast, broadcast.5)
  constant = f32[] constant(-inf)
  reduce = f32[4,12] reduce(select, constant), dimensions={2}, to_apply=max
  broadcast = f32[4,12,97] broadcast(reduce), dimensions={0,1}
  ROOT subtract = f32[4,12,97] subtract(select, broadcast)
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  {
    TF_ASSERT_OK_AND_ASSIGN(
        TiledHloComputation tiled_hlo_computation,
        analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 1, 97}));

    EXPECT_THAT(GetInstructionsWithName(tiled_hlo_computation, "p0"),
                ElementsAre(Pointee(MatchTiledHloInstruction(
                    /*tile_sizes=*/{1, 97}, /*tile_strides=*/{0, 1},
                    /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (d0, 0)
      domain:
      d0 in [0, 47]
    )"))));

    EXPECT_THAT(GetInstructionsWithName(tiled_hlo_computation, "p1"),
                ElementsAre(Pointee(MatchTiledHloInstruction(
                    /*tile_sizes=*/{1, 97}, /*tile_strides=*/{1, 1},
                    /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (d0 floordiv 12, 0)
      domain:
      d0 in [0, 47]
    )"))));
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(
        TiledHloComputation tiled_hlo_computation,
        analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 2, 10}));

    EXPECT_THAT(GetInstructionsWithName(tiled_hlo_computation, "p0"),
                ElementsAre(Pointee(MatchTiledHloInstruction(
                                /*tile_sizes=*/{2, 10}, /*tile_strides=*/{1, 1},
                                /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (((d0 floordiv 10) mod 6) * 2 + (d0 floordiv 60) * 12, (d0 mod 10) * 10)
      domain:
      d0 in [0, 239]
    )")),
                            Pointee(MatchTiledHloInstruction(
                                /*tile_sizes=*/{2, 97}, /*tile_strides=*/{1, 1},
                                /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (((d0 floordiv 10) mod 6) * 2 + (d0 floordiv 60) * 12, 0)
      domain:
      d0 in [0, 239]
    )"))));

    EXPECT_THAT(GetInstructionsWithName(tiled_hlo_computation, "p1"),
                ElementsAre(Pointee(MatchTiledHloInstruction(
                                /*tile_sizes=*/{1, 10}, /*tile_strides=*/{1, 1},
                                /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (d0 floordiv 60, (d0 mod 10) * 10)
      domain: d0 in [0, 239]
    )")),
                            Pointee(MatchTiledHloInstruction(
                                /*tile_sizes=*/{1, 97}, /*tile_strides=*/{1, 1},
                                /*block_id_to_tile_offsets_indexing=*/R"(
      (d0) -> (d0 floordiv 60, 0)
      domain: d0 in [0, 239]
    )"))));
  }
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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{1, 10}));

  const TiledHloInstruction* root = tiled_hlo_computation.GetRoot();

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

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{2, 4, 2}));

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
ENTRY main {
  p0 = f32[8,16] parameter(0)
  slice.0 = f32[4,8] slice(p0), slice={[0:4], [2:10]}
  slice.1 = f32[4,8] slice(p0), slice={[3:7], [4:12]}
  ROOT add = f32[4,8] add(slice.0, slice.1)
})"));

  EXPECT_TRUE(SetAnalysis(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      analysis_->ComputeTiledHloInstructions(/*tile_parameters=*/{2, 2}));

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
ENTRY main {
  p0 = f32[1,2]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  ROOT dot = f32[1,3]{1,0} dot(p0, p1),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

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

}  // namespace
}  // namespace gpu
}  // namespace xla

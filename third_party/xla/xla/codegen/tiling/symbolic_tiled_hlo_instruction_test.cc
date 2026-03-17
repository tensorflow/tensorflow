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

#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using SymbolicTiledHloInstructionTest = HloHardwareIndependentTestBase;

TEST_F(SymbolicTiledHloInstructionTest, TransposeTileSizesAreSupported) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
fused_computation {
  p0 = f32[16,32] parameter(0)
  p1 = f32[32,16] parameter(1)
  transpose = f32[32,16] transpose(p0), dimensions={1,0}
  ROOT subtract = f32[32,16] subtract(transpose, p1)
}

ENTRY main {
  p0 = f32[16,32] parameter(0)
  p1 = f32[32,16] parameter(1)
  ROOT root = f32[32,16] fusion(p0, p1), kind=kLoop, calls=fused_computation
}
)"));

  mlir::MLIRContext mlir_ctx;
  RegisterSymbolicExprStorage(&mlir_ctx);
  auto fusion = module->entry_computation()->root_instruction();
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);

  auto output_to_input_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_ctx);

  HloInstruction* subtract = fusion->fused_expression_root();
  HloInstruction* p0 = subtract->mutable_operand(0)->mutable_operand(0);
  HloInstruction* p1 = subtract->mutable_operand(1);

  // We use `fusion->operand(0)` to get indexing from the map instead of `p0`,
  // because `HloFusionAdaptor` and `ComputeGroupedOutputToInputIndexing` ignore
  // kParameter instructions inside the fusion and produces indexing for fusion
  // operands.
  IndexingMap p0_indexing =
      output_to_input_indexing[fusion->operand(0)].begin()->map();
  std::optional<SymbolicTile> p0_symbolic_tile =
      SymbolicTile::FromIndexingMap(p0_indexing);
  ASSERT_TRUE(p0_symbolic_tile.has_value());
  SymbolicTiledHloInstruction tiled_p0(p0, p0_indexing);
  tiled_p0.set_symbolic_tile(*p0_symbolic_tile);
  ASSERT_TRUE(p0_symbolic_tile.has_value());

  IndexingMap p1_indexing =
      output_to_input_indexing[fusion->operand(1)].begin()->map();
  std::optional<SymbolicTile> p1_symbolic_tile =
      SymbolicTile::FromIndexingMap(p1_indexing);
  ASSERT_TRUE(p1_symbolic_tile.has_value());
  SymbolicTiledHloInstruction tiled_p1(p1, p1_indexing);
  tiled_p1.set_symbolic_tile(*p1_symbolic_tile);

  std::vector<int64_t> output_tile_sizes = {8, 4};

  EXPECT_THAT(EvaluateTileSizes(tiled_p0.symbolic_tile(), output_tile_sizes),
              ElementsAre(4, 8));
  EXPECT_THAT(EvaluateTileSizes(tiled_p1.symbolic_tile(), output_tile_sizes),
              ElementsAre(8, 4));
}

TEST_F(SymbolicTiledHloInstructionTest, ToString) {
  mlir::MLIRContext mlir_ctx;
  RegisterSymbolicExprStorage(&mlir_ctx);

  std::unique_ptr<HloInstruction> p0 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(xla::F32, {16, 32}), "p0");

  std::unique_ptr<HloInstruction> p1 = HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(xla::F32, {4}), "p1");
  IndexingMap p1_indexing_map(
      mlir::AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                           {mlir::getAffineDimExpr(0, &mlir_ctx)}, &mlir_ctx),
      {IndexingMap::Variable{0, 3}}, {}, {});
  auto tiled_p1 =
      std::make_unique<SymbolicTiledHloInstruction>(p1.get(), p1_indexing_map);

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlir_ctx);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &mlir_ctx);
  mlir::AffineMap affine_map = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/1, {d1, d0}, &mlir_ctx);
  IndexingMap indexing_map(
      affine_map,
      /*dimensions=*/
      {IndexingMap::Variable{0, 31}, IndexingMap::Variable{0, 15}},
      /*range_vars=*/{},
      /*rt_vars=*/{IndexingMap::Variable{0, 3}});

  std::optional<SymbolicTile> symbolic_tile =
      SymbolicTile::FromIndexingMap(indexing_map);
  ASSERT_TRUE(symbolic_tile.has_value());

  SymbolicTiledHloInstruction tiled_p0(p0.get(), indexing_map,
                                       {tiled_p1.get()});
  tiled_p0.set_symbolic_tile(*symbolic_tile);

  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>> region;
  region.push_back(std::move(tiled_p1));
  tiled_p0.AddRegion(std::move(region));

  EXPECT_EQ(tiled_p0.ToString(/*field_separator=*/"\n  "),
            R"""(hlo: %p0 = f32[16,32]{1,0} parameter(0)
  Symbolic tile with
	offset_map: (d0, d1) -> (0, 0)
	size_map: (d0, d1) -> (d1, d0)
	stride_map: (d0, d1) -> (1, 1)
  indexing map: (d0, d1){rt0} -> (d1, d0), domain: d0 in [0, 31], d1 in [0, 15], rt0 in [0, 3]
  runtime variables: (
  hlo: %p1 = f32[4]{0} parameter(1)
  (no symbolic tile)
  indexing map: (d0) -> (d0), domain: d0 in [0, 3])
  regions: (
  #0 size: 1))""");
}

}  // namespace

}  // namespace xla

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

#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using SymbolicTiledHloInstructionTest = HloTestBase;

TEST_F(SymbolicTiledHloInstructionTest, TransposeTileSizesAreSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
      *output_to_input_indexing[fusion->operand(0)].begin();
  std::optional<SymbolicTile> p0_symbolic_tile =
      SymbolicTile::FromIndexingMap(p0_indexing);
  ASSERT_TRUE(p0_symbolic_tile.has_value());
  SymbolicTiledHloInstruction tiled_p0(p0, p0_indexing);
  tiled_p0.set_symbolic_tile(*p0_symbolic_tile);
  ASSERT_TRUE(p0_symbolic_tile.has_value());

  IndexingMap p1_indexing =
      *output_to_input_indexing[fusion->operand(1)].begin();
  std::optional<SymbolicTile> p1_symbolic_tile =
      SymbolicTile::FromIndexingMap(p1_indexing);
  ASSERT_TRUE(p1_symbolic_tile.has_value());
  SymbolicTiledHloInstruction tiled_p1(p1, p1_indexing);
  tiled_p1.set_symbolic_tile(*p1_symbolic_tile);

  std::vector<int64_t> output_tile_sizes = {8, 4};

  auto p0_tile_sizes = tiled_p0.TileSizes(output_tile_sizes);
  EXPECT_THAT(tiled_p0.TileSizes(output_tile_sizes), ElementsAre(4, 8));
  EXPECT_THAT(tiled_p1.TileSizes(output_tile_sizes), ElementsAre(8, 4));
}

}  // namespace

}  // namespace gpu
}  // namespace xla

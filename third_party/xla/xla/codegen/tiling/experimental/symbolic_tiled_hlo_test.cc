/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/experimental/symbolic_tiled_hlo.h"

#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/symbolic_tile_propagation.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::gpu::experimental {
namespace {

using ::mlir::MLIRContext;

class SymbolicTiledHloTest : public HloHardwareIndependentTestBase {
 public:
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(SymbolicTiledHloTest, TestPrinting) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,30] parameter(0)
      ROOT broadcast = f32[10,20,30] broadcast(p0), dimensions={0,2}
    }
  )");
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  std::optional<SymbolicTiles> tiled_operands = PropagateTileToInput(
      *tiling_space, *root,
      GetTestSymbolicTile(*tiling_space, root->shape().dimensions()), 0);

  ASSERT_TRUE(tiled_operands.has_value());
  SymbolicTiledHloInstruction tiled_hlo_instruction(root, (*tiled_operands)[0]);
  EXPECT_THAT(tiled_hlo_instruction, MatchString(R"(
    hlo: %broadcast = f32[10,20,30]{2,1,0} broadcast(%p0), dimensions={0,2}
    tile: (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 3]
         upper bounds [10, 30]
  )"));
}

TEST_F(SymbolicTiledHloTest, TestReduceWithRegionPrinting) {
  HloInstruction* reduce = ParseAndGetRoot(R"(
    HloModule m

    max {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT maximum = f32[] maximum(x, y)
    }

    ENTRY e {
      p0 = f32[2,97]{1,0} parameter(0)
      constant = f32[] constant(-inf)
      ROOT reduce = f32[2]{0} reduce(p0, constant), dimensions={1}, to_apply=max
    }
  )");

  auto tiling_space_reduce = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(reduce), &mlir_context_);
  auto tiled_reduce = std::make_unique<SymbolicTiledHloInstruction>(
      reduce,
      GetTestSymbolicTile(*tiling_space_reduce, reduce->shape().dimensions()));
  std::optional<SymbolicTiles> operands_tiles = PropagateTileToInput(
      *tiling_space_reduce, *reduce,
      GetTestSymbolicTile(*tiling_space_reduce, reduce->shape().dimensions()),
      0);

  SymbolicTiledHloInstruction::Region region;
  for (const auto& [operand, tile] :
       llvm::zip(reduce->operands(), (*operands_tiles))) {
    region.push_back(
        std::make_unique<SymbolicTiledHloInstruction>(operand, tile));
  }
  tiled_reduce->AddRegion(std::move(region));

  EXPECT_THAT(*tiled_reduce, MatchString(R"(
    hlo: %reduce = f32[2]{0} reduce(%p0, %constant), dimensions={1}, to_apply=%max
    tile: (tid_0, tid_1)[ts_0, ts_1]
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [2]
    region #0 {
      hlo: %p0 = f32[2,97]{1,0} parameter(0)
      tile: (tid_0, tid_1)[ts_0, ts_1]
        -> offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [2, 97]
      hlo: %constant = f32[] constant(-inf)
      tile: (tid_0, tid_1)[ts_0, ts_1]
        -> offsets [] sizes [] strides [] upper bounds []
    }
  )"));
}

}  // namespace
}  // namespace xla::gpu::experimental

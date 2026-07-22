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

#include "xla/codegen/tiling/experimental/tile_propagation.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

using ::absl_testing::StatusIs;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

class TilePropagationTest : public HloHardwareIndependentTestBase {
 public:
  TilePropagationTest() = default;

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(TilePropagationTest, CanPropagateToInputsOfElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT add0 = f32[10,20] add(p0, p1)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
    1) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputsOfElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT add0 = f32[10,20] add(p0, p1)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  constexpr absl::string_view kExpected = R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
  )";

  ASSERT_OK_AND_ASSIGN(
      auto from_operand_0,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(from_operand_0, MatchToString(kExpected));
  ASSERT_OK_AND_ASSIGN(
      auto from_operand_1,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 1));
  EXPECT_THAT(from_operand_1, MatchToString(kExpected));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfAllReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    %add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %a = f32[] add(p0, p1)
    }
    ENTRY %module {
      %p0 = f32[2,8,256] parameter(0)
      ROOT %ar = f32[2,8,256] all-reduce(p0), replica_groups={{0,1}},
        to_apply=%add
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto ar_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(ar_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [2, 8, 256]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfAllGatherOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[64,256] parameter(0)
      ROOT all_gather = f32[128,256] all-gather(p0), replica_groups={{0,1}}, dimensions={0}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [(tid_0 * ts_0) mod 64, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [64, 256]
         replica ids {
           offsets [(tid_0 * ts_0) / 64]
           sizes [1]
           strides [1]
           upper bounds [2]
         }
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputOfBroadcastOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,30] parameter(0)
      ROOT broadcast = f32[10,20,30] broadcast(p0), dimensions={0,2}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 3]
         upper bounds [10, 30]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfBroadcastOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,30] parameter(0)
      ROOT broadcast = f32[10,20,30] broadcast(p0), dimensions={0,2}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
      0) (tid_0, tid_1, tid_2)
         -> offsets [tid_0 * ts_0, 0, tid_1 * ts_1]
            sizes [ts_0, 32, ts_1]
            strides [1, 1, 2]
            upper bounds [10, 20, 30]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateThroughBitcastTransposeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = f32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto input_tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(input_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_3 * ts_3, tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_0, ts_3, ts_1, ts_2]
         strides [1, 4, 2, 3]
         upper bounds [3, 12288, 6, 128]
  )"));
  ASSERT_OK_AND_ASSIGN(
      auto output_tiled_operands,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(output_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2, tid_3 * ts_3, tid_1 * ts_1]
         sizes [ts_0, ts_2, ts_3, ts_1]
         strides [1, 3, 4, 2]
         upper bounds [3, 6, 128, 12288]
  )"));
}

TEST_F(TilePropagationTest, CanNotPropagateThroughBitcastReshapeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 32] parameter(0)
      ROOT bitcast = f32[4, 8, 4] bitcast(p0)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK(tiling_space->AssignTileSizes({2, 2, 2}));
  EXPECT_THAT(PropagateTileToInput(
                  *tiling_space, *root,
                  GetTestTile(*tiling_space, root->shape().dimensions()), 0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(TilePropagationTest, CanNotPropagateThroughBitcastTrtInput) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = bf16[16, 4096, 8, 256]{3, 2, 1, 0} parameter(0)
      ROOT bitcast = bf16[16, 2048, 4096]{1, 2, 0} bitcast(p0)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK(tiling_space->AssignTileSizes({2, 2, 2}));
  EXPECT_THAT(PropagateTileToInput(
                  *tiling_space, *root,
                  GetTestTile(*tiling_space, root->shape().dimensions()), 0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(TilePropagationTest, AllowSymbolicTilingOfRehsapeButRejectsConcrete) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 4, 4, 4] parameter(0)
      ROOT bitcast = f32[256] bitcast(p0)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  // Before AssignTileSizes: ACCEPTED as the tiling space is symbolic.
  EXPECT_TRUE(PropagateTileToInput(*tiling_space, *root,
                                   tiling_space->tiled_roots()[0], 0)
                  .ok());

  // After AssignTileSizes: REJECTED as "Multiple dimensions are partially tiled
  // tile_size [1, 1, 3, 2], dims [4, 4, 4, 4]".
  ASSERT_OK(tiling_space->AssignTileSizes({10}));
  Tile input_tile = tiling_space->tiled_roots()[0];
  input_tile.Simplify();
  EXPECT_THAT(PropagateTileToInput(*tiling_space, *root, input_tile, 0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfConcatenateOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      ROOT concatenate = f32[60] concatenate(p0, p1, p2), dimensions={0}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0)
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [10]
    1) (tid_0)
      -> offsets [tid_0 * ts_0 - 10]
         sizes [ts_0]
         strides [1]
         upper bounds [20]
    2) (tid_0)
      -> offsets [tid_0 * ts_0 - 30]
         sizes [ts_0]
         strides [1]
         upper bounds [30]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputsOfConcatenateOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 5] parameter(0)
      p1 = f32[10, 8] parameter(1)
      p2 = f32[10, 2] parameter(2)
      ROOT concatenate = f32[10, 15] concatenate(p0, p1, p2), dimensions={1}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  // Operand 0
  ASSERT_OK_AND_ASSIGN(
      auto from_operand_0,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(from_operand_0, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 5]
  )"));

  // Operand 1
  ASSERT_OK_AND_ASSIGN(
      auto from_operand_1,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(1)->shape().dimensions()),
          1));
  EXPECT_THAT(from_operand_1, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1 + 5]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 13]
  )"));

  // Operand 2
  ASSERT_OK_AND_ASSIGN(
      auto from_operand_2,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(2)->shape().dimensions()),
          2));
  EXPECT_THAT(from_operand_2, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1 + 13]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 15]
  )"));
}

TEST_F(TilePropagationTest,
       CanPropagateToInputsOfConcatenateOpWithNonDefaultUpperBound) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      ROOT concatenate = f32[60] concatenate(p0, p1, p2), dimensions={0}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  llvm::SmallVector<SymbolicExpr, 1> upper_bounds{
      CreateSymbolicConstant(25, &mlir_context_)};
  tile = Tile{*tiling_space, tile.offsets(), tile.sizes(), tile.strides(),
              upper_bounds};
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0)
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [10]
    1) (tid_0)
      -> offsets [tid_0 * ts_0 - 10]
         sizes [ts_0]
         strides [1]
         upper bounds [15]
    2) (tid_0)
      -> offsets [tid_0 * ts_0 - 30]
         sizes [ts_0]
         strides [1]
         upper bounds [0]
  )"));
}

TEST_F(TilePropagationTest,
       CanPropagateToInputsOfConcatenateOpWithNonConstantUpperBound) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      ROOT concatenate = f32[60] concatenate(p0, p1, p2), dimensions={0}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  llvm::SmallVector<SymbolicExpr, 1> upper_bounds{
      CreateDimExpr(0, &mlir_context_) * 30};
  tile = Tile{*tiling_space, tile.offsets(), tile.sizes(), tile.strides(),
              upper_bounds};
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0)
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [max(min(tid_0 * 30, 10), 0)]
    1) (tid_0)
      -> offsets [tid_0 * ts_0 - 10]
         sizes [ts_0]
         strides [1]
         upper bounds [max(min(tid_0 * 30 - 10, 20), 0)]
    2) (tid_0)
      -> offsets [tid_0 * ts_0 - 30]
         sizes [ts_0]
         strides [1]
         upper bounds [max(min(tid_0 * 30 - 30, 30), 0)]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfPadOpWithEdgePadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[12,13] pad(p0, p1), padding=1_7x0_9
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()),
          /*output_index=*/0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0 - 1, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [4, 4]
    1) (tid_0, tid_1)
      -> offsets [] sizes [] strides [] upper bounds []
  )"));
}

TEST_F(TilePropagationTest, CanNotPropagateToInputsOfPadOpWithInteriorPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[30,13] pad(p0, p1), padding=1_4_7x0_9
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  EXPECT_THAT(PropagateTileToInput(
                  *tiling_space, *root,
                  GetTestTile(*tiling_space, root->shape().dimensions()),
                  /*output_index=*/0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfTransposeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,5,1,3] parameter(0)
      ROOT transpose = f32[1,2,3,5] transpose(p0), dimensions={2,0,3,1}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_1 * ts_1, tid_3 * ts_3, tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_1, ts_3, ts_0, ts_2]
         strides [2, 4, 1, 3]
         upper bounds [2, 5, 1, 3]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfTransposeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,5,1,3] parameter(0)
      ROOT transpose = f32[1,2,3,5] transpose(p0), dimensions={2,0,3,1}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_2 * ts_2, tid_0 * ts_0, tid_3 * ts_3, tid_1 * ts_1]
         sizes [ts_2, ts_0, ts_3, ts_1]
         strides [3, 1, 4, 2]
         upper bounds [1, 2, 3, 5]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfSliceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[5,7,13] parameter(0)
      ROOT slice = f32[2,7,4] slice(p0), slice={[1:5:2], [0:7], [5:13:2]}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0 * 2 + 1, tid_1 * ts_1, tid_2 * ts_2 * 2 + 5]
         sizes [ts_0, ts_1, ts_2]
         strides [2, 2, 6]
         upper bounds [5, 7, 13]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfDynSliceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[20,2,258] parameter(0)
      c4 = s32[] constant(4)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      ROOT ds = s32[1,2,32] dynamic-slice(p0, c4, p1, p2),
        dynamic_slice_sizes={1, 2, 32}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2){rt_0, rt_1, rt_2}
      -> offsets [tid_0 * ts_0 + 4, rt_1 + tid_1 * ts_1, rt_2 + tid_2 * ts_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [5, rt_1 + 2, rt_2 + 32]
    1) (tid_0, tid_1, tid_2){rt_0, rt_1, rt_2}
      -> offsets [] sizes [] strides [] upper bounds []
    2) (tid_0, tid_1, tid_2){rt_0, rt_1, rt_2}
      -> offsets [] sizes [] strides [] upper bounds []
    3) (tid_0, tid_1, tid_2){rt_0, rt_1, rt_2}
      -> offsets [] sizes [] strides [] upper bounds []
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfDotOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,38,17,11,18,10] parameter(0)
      p1 = f32[17,10,16,18,22,38] parameter(1)
      ROOT dot = f32[10,38,4,11,16,22] dot(p0, p1),
        lhs_batch_dims={5,1}, rhs_batch_dims={1,5},
        lhs_contracting_dims={4,2}, rhs_contracting_dims={3,0}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  tile = Tile{*tiling_space, tile.offsets(), tile.sizes(), tile.strides(),
              tile.upper_bounds()};
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
         -> offsets [tid_2 * ts_2, tid_1 * ts_1, tid_7 * ts_7,
                     tid_3 * ts_3, tid_6 * ts_6, tid_0 * ts_0]
            sizes [ts_2, ts_1, ts_7, ts_3, ts_6, ts_0]
            strides [3, 2, 1, 4, 1, 1]
            upper bounds [4, 38, 17, 11, 18, 10]
    1) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
         -> offsets [tid_7 * ts_7, tid_0 * ts_0, tid_4 * ts_4,
                     tid_6 * ts_6, tid_5 * ts_5, tid_1 * ts_1]
            sizes [ts_7, ts_0, ts_4, ts_6, ts_5, ts_1]
            strides [1, 1, 5, 1, 6, 2]
            upper bounds [17, 10, 16, 18, 22, 38]
  )"));

  EXPECT_OK(tiling_space->AssignTileSizes({16, 16, 16, 16, 16, 16, 16, 16}));
  ASSERT_OK_AND_ASSIGN(auto concrete_tiled_operands,
                       PropagateTileToInput(*tiling_space, *root,
                                            tiling_space->tiled_roots()[0], 0));

  EXPECT_THAT(concrete_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
         -> offsets [0, tid_1 * 16, tid_7 * 16, 0, tid_6 * 16, 0]
            sizes [16, 16, 16, 16, 16, 16]
            strides [1, 1, 1, 1, 1, 1]
            upper bounds [4, 38, 17, 11, 18, 10]
    1) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
         -> offsets [tid_7 * 16, 0, 0, tid_6 * 16, tid_5 * 16, tid_1 * 16]
            sizes [16, 16, 16, 16, 16, 16]
            strides [1, 1, 1, 1, 1, 1]
            upper bounds [17, 10, 16, 18, 22, 38]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsForScaledDotOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule module

      ENTRY main {
        lhs = f32[1024,512] parameter(0)
        rhs = f32[64,512] parameter(1)
        lhs_scale = f32[32,2] parameter(2)
        rhs_scale = f32[64,512] parameter(3)
        ROOT dot = f32[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1}
      }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 1]
         upper bounds [1024, 512]
    1) (tid_0, tid_1, tid_2)
      -> offsets [tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_1, ts_2]
         strides [2, 1]
         upper bounds [64, 512]
    2) (tid_0, tid_1, tid_2)
      -> offsets [(tid_0 * ts_0) / 32, (tid_2 * ts_2) / 256]
         sizes [(tid_0 * ts_0 + ts_0 - 1) / 32 - (tid_0 * ts_0) / 32 + 1, (tid_2 * ts_2 + ts_2 - 1) / 256 - (tid_2 * ts_2) / 256 + 1]
         strides [1, 1]
         upper bounds [32, 2]
    3) (tid_0, tid_1, tid_2)
      -> offsets [tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_1, ts_2]
         strides [2, 1]
         upper bounds [64, 512]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateReplicaIdThroughBroadcast) {
  // Pick an arbitrary op that is a bit more complicated than elementwise
  // and test that replica_id is propagated correctly for fused ops.
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 32] parameter(0)
      broadcast = f32[10, 32, 5] broadcast(p0), dimensions={0, 1}
      ROOT all_gather = f32[10, 64, 5] all-gather(broadcast), replica_groups={{0,1}}, dimensions={1}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  ASSERT_OK_AND_ASSIGN(
      auto tiled_ag_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));

  EXPECT_THAT(tiled_ag_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, (tid_1 * ts_1) mod 32, tid_2 * ts_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [10, 32, 5]
         replica ids {
           offsets [(tid_1 * ts_1) / 32]
           sizes [1]
           strides [1]
           upper bounds [2]
         }
  )"));
  // operand(0) is the broadcast, tile_ag_operands[0] is its output tile.
  // This should preserve the replica_id and drop dimension 2.
  ASSERT_OK_AND_ASSIGN(auto tiled_broadcast_operands,
                       PropagateTileToInput(*tiling_space, *root->operand(0),
                                            tiled_ag_operands[0], 0));
  EXPECT_THAT(tiled_broadcast_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, (tid_1 * ts_1) mod 32]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 32]
         replica ids {
           offsets [(tid_1 * ts_1) / 32]
           sizes [1]
           strides [1]
           upper bounds [2]
         }
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      p0 = f32[150, 20, 10, 50] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT reduce = f32[150, 10] reduce(p0, p0_init),
        dimensions={3, 1}, to_apply=max
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root,
                                            tiling_space->tiled_roots()[0], 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_3 * ts_3, tid_1 * ts_1, tid_2 * ts_2]
        sizes [ts_0, ts_3, ts_1, ts_2]
        strides [1, 1, 1, 1]
        upper bounds [150, 20, 10, 50]
    1) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [] sizes [] strides [] upper bounds []
  )"));

  EXPECT_OK(tiling_space->AssignTileSizes({8, 16, 32, 64}));
  ASSERT_OK_AND_ASSIGN(auto concrete_tiled_operands,
                       PropagateTileToInput(*tiling_space, *root,
                                            tiling_space->tiled_roots()[0], 0));
  EXPECT_THAT(concrete_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * 8, 0, 0, tid_2 * 32]
        sizes [8, 64, 16, 32]
        strides [1, 1, 1, 1]
        upper bounds [150, 20, 10, 50]
    1) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [] sizes [] strides [] upper bounds []
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      p0 = f32[150, 20, 10, 50] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT reduce = f32[150, 10] reduce(p0, p0_init),
        dimensions={3, 1}, to_apply=max
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
        sizes [ts_0, ts_2]
        strides [1, 3]
        upper bounds [150, 10]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToInputsOfVariadicReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
   HloModule m
    min {
      tmp_0 = f32[] parameter(0)
      tmp_1 = f32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      cmp = pred[] compare(tmp_0, tmp_1), direction=GE
      select1 = f32[] select(cmp, tmp_0, tmp_1)
      select2 = s32[] select(cmp, tmp_2, tmp_3)
      ROOT tmp_4 = (f32[], s32[]) tuple(select1, select2)
    }
    ENTRY e {
      p0 = f32[256,10] parameter(0)
      p0_init = f32[] constant(-inf)
      p1 = s32[256,10] parameter(1)
      p1_init = s32[] constant(0)
      ROOT reduce = (f32[10], s32[10]) reduce(p0, p1, p0_init, p1_init),
        dimensions={0}, to_apply=min
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  MLIRContext mlir_context;
  Tile tile = GetTestTile(*tiling_space, GetFirstShape(root).dimensions());
  tile = Tile{*tiling_space, tile.offsets(), tile.sizes(), tile.strides(),
              tile.upper_bounds()};
  ASSERT_OK_AND_ASSIGN(auto tiled_operands,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1) -> offsets [tid_1 * ts_1, tid_0 * ts_0]
      sizes [ts_1, ts_0] strides [1, 1] upper bounds [256, 10]
    1) (tid_0, tid_1) -> offsets [tid_1 * ts_1, tid_0 * ts_0]
      sizes [ts_1, ts_0] strides [1, 1] upper bounds [256, 10]
    2) (tid_0, tid_1)
      -> offsets [] sizes [] strides [] upper bounds []
    3) (tid_0, tid_1)
      -> offsets [] sizes [] strides [] upper bounds []
  )"));
}

TEST_F(TilePropagationTest, ConcatenateOpSupportsShiftedConstantBaseOffset) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      concat = f32[60] concatenate(p0, p1, p2), dimensions={0}
      ROOT slice = f32[30] slice(concat), slice={[13:43]}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());

  // Symbolic tiling.
  ASSERT_OK_AND_ASSIGN(auto operands_of_slice,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  const Tile& concat_tile = operands_of_slice[0];
  const HloInstruction* concat = root->operand(0);
  ASSERT_OK_AND_ASSIGN(
      auto operands_of_concat,
      PropagateTileToInput(*tiling_space, *concat, concat_tile, 0));

  EXPECT_OK(tiling_space->AssignTileSizes({17}));
  const Tile& root_tile = tiling_space->tiled_roots()[0];
  ASSERT_OK_AND_ASSIGN(
      auto operands_of_slice2,
      PropagateTileToInput(*tiling_space, *root, root_tile, 0));
  const Tile& concat_tile_17 = operands_of_slice2[0];
  ASSERT_OK(
      PropagateTileToInput(*tiling_space, *concat, concat_tile_17, 0).status());
}

TEST_F(TilePropagationTest,
       ConcatenateOpRejectsShiftedOffsetWhenRemainingSizeNotDivisible) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      concat = f32[60] concatenate(p0, p1, p2), dimensions={0}
      ROOT slice = f32[30] slice(concat), slice={[13:43]}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  EXPECT_OK(tiling_space->AssignTileSizes({5}));

  const Tile& root_tile = tiling_space->tiled_roots()[0];

  ASSERT_OK_AND_ASSIGN(
      auto operands_of_slice,
      PropagateTileToInput(*tiling_space, *root, root_tile, 0));
  const Tile& concat_tile = operands_of_slice[0];
  const HloInstruction* concat = root->operand(0);

  EXPECT_THAT(
      PropagateTileToInput(*tiling_space, *concat, concat_tile, 0).status(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ::testing::HasSubstr(
                   "The remaining dimension size 17 in the concatenate operand "
                   "1 must be a clean multiple of its tile size 5")));
}

TEST_F(TilePropagationTest, CanPropagateToInputOfScanOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m

    scan_add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      add1 = f32[] add(p0, p1)
      ROOT tuple = (f32[], f32[]) tuple(add1, add1)
    }

    ENTRY e {
      p0 = f32[4] parameter(0)
      p1 = f32[] parameter(1)
      scan = (f32[4], f32[]) scan(p0, p1), dimensions={0}, num_carries=1, to_apply=scan_add
      ROOT gte = f32[4] get-tuple-element(scan), index=0
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  const HloInstruction* scan = root->operand(0);

  // Create a tile for an array output (output_index = 0)
  Tile tile_arr = GetTestTile(*tiling_space, {4});
  ASSERT_OK_AND_ASSIGN(auto tiled_operands_arr,
                       PropagateTileToInput(*tiling_space, *scan, tile_arr, 0));

  EXPECT_THAT(tiled_operands_arr, MatchToString(R"(
    0) (tid_0)
         -> offsets [tid_0 * ts_0]
            sizes [ts_0]
            strides [1]
            upper bounds [4]
    1) (tid_0)
         -> offsets []
            sizes []
            strides []
            upper bounds []
  )"));

  // Create a tile for a scalar carry output (output_index = 1)
  Tile tile_carry = GetTestTile(*tiling_space, {});
  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands_carry,
      PropagateTileToInput(*tiling_space, *scan, tile_carry, 1));

  EXPECT_THAT(tiled_operands_carry, MatchToString(R"(
    0) (tid_0)
         -> offsets []
            sizes []
            strides []
            upper bounds []
    1) (tid_0)
         -> offsets []
            sizes []
            strides []
            upper bounds []
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfScanOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m

    scan_add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      add1 = f32[] add(p0, p1)
      ROOT tuple = (f32[], f32[]) tuple(add1, add1)
    }

    ENTRY e {
      p0 = f32[4] parameter(0)
      p1 = f32[] parameter(1)
      scan = (f32[4], f32[]) scan(p0, p1), dimensions={0}, num_carries=1, to_apply=scan_add
      ROOT gte = f32[4] get-tuple-element(scan), index=0
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  const HloInstruction* scan = root->operand(0);

  // Create a tile for an array input (input_index = 0)
  Tile tile_arr = GetTestTile(*tiling_space, {4});
  ASSERT_OK_AND_ASSIGN(
      auto output_tiles,
      PropagateTileToOutput(*tiling_space, *scan, tile_arr, 0));

  EXPECT_THAT(output_tiles, MatchToString(R"(
    0) (tid_0)
         -> offsets [tid_0 * ts_0]
            sizes [ts_0]
            strides [1]
            upper bounds [4]
    1) (tid_0)
         -> offsets []
            sizes []
            strides []
            upper bounds []
  )"));

  // Create a tile for a scalar carry input (input_index = 1)
  Tile tile_carry = GetTestTile(*tiling_space, {});
  EXPECT_THAT(PropagateTileToOutput(*tiling_space, *scan, tile_carry, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TilePropagationTest, CanPropagateToGetTupleElementOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = (f32[4], f32[]) parameter(0)
      ROOT gte = f32[4] get-tuple-element(p0), index=0
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  Tile tile = GetTestTile(*tiling_space, {4});
  ASSERT_OK_AND_ASSIGN(auto input_tiles,
                       PropagateTileToInput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(input_tiles, MatchToString(R"(
    0) (tid_0)
         -> offsets [tid_0 * ts_0]
            sizes [ts_0]
            strides [1]
            upper bounds [4]
  )"));

  ASSERT_OK_AND_ASSIGN(auto output_tiles,
                       PropagateTileToOutput(*tiling_space, *root, tile, 0));
  EXPECT_THAT(output_tiles, MatchToString(R"(
    0) (tid_0)
         -> offsets [tid_0 * ts_0]
            sizes [ts_0]
            strides [1]
            upper bounds [4]
  )"));
}

TEST_F(TilePropagationTest, FailsToPropagateToConcatenateThroughBitcast) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 512] parameter(0)
      p1 = f32[10, 2, 512] parameter(1)
      reshape = f32[10, 1024] bitcast(p1)
      ROOT concatenate = f32[10, 1536] concatenate(p0, reshape), dimensions={1}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));

  std::vector<int64_t> test_sizes(tiling_space->num_dimensions(), 1);
  test_sizes[1] = 2;
  ASSERT_OK(tiling_space->AssignTileSizes(test_sizes));

  auto root_tile = tiling_space->tiled_roots()[0];
  root_tile.Simplify();

  ASSERT_OK_AND_ASSIGN(
      auto tiled_operands,
      PropagateTileToInput(*tiling_space, *root, root_tile, 0));

  ASSERT_EQ(tiled_operands.size(), 2);

  ASSERT_OK_AND_ASSIGN(auto output_tiles,
                       PropagateTileToInput(*tiling_space, *(root->operand(1)),
                                            tiled_operands[1], 0));
  EXPECT_THAT(output_tiles, MatchToString(R"(
    0) (tid_0, tid_1)
         -> offsets [tid_0, tid_1 / 256 - 1, (tid_1 mod 256) * 2]
            sizes [1, 1, 2]
            strides [1, 1, 1]
            upper bounds [10, tid_1 / 256, (tid_1 mod 256) * 2 + 2]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfGatherOpDataOperand) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[32,16] parameter(1)
      ROOT gather = f32[32,16,64] gather(p0, p1),
        offset_dims={2}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto from_data_operand,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_THAT(from_data_operand, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [0, 0, tid_1 * ts_1]
         sizes [32, 16, ts_1]
         strides [1, 1, 2]
         upper bounds [32, 16, 64]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateToOutputOfGatherOpIndicesOperand) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[32,16] parameter(1)
      ROOT gather = f32[32,16,64] gather(p0, p1),
        offset_dims={2}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto from_indices_operand,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(1)->shape().dimensions()),
          1));
  EXPECT_THAT(from_indices_operand, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1, 0]
         sizes [ts_0, ts_1, 64]
         strides [1, 2, 1]
         upper bounds [32, 16, 64]
  )"));
}

TEST_F(TilePropagationTest, PropagateToOutputOfGatherOpInvalidIndex) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[32,16] parameter(1)
      ROOT gather = f32[32,16,64] gather(p0, p1),
        offset_dims={2}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  EXPECT_THAT(
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          2),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TilePropagationTest, GatherOpIndexVectorDimAtPositionZero) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[1,32,16] parameter(1)
      ROOT gather = f32[32,16,64] gather(p0, p1),
        offset_dims={2}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=0,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto input_tiles,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_EQ(input_tiles.size(), 2);
}

TEST_F(TilePropagationTest, GatherOpScalarIndexVectorDim) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[32] parameter(1)
      ROOT gather = f32[32,64] gather(p0, p1),
        offset_dims={1}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=1,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto input_tiles,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_EQ(input_tiles.size(), 2);
}

TEST_F(TilePropagationTest, GatherOpMultipleCollapsedAndOffsetDims) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,20,30,40] parameter(0)
      p1 = s32[5,2] parameter(1)
      ROOT gather = f32[5,20,40] gather(p0, p1),
        offset_dims={1,2}, collapsed_slice_dims={0,2},
        start_index_map={0,2}, index_vector_dim=1,
        slice_sizes={1,20,1,40}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  ASSERT_OK_AND_ASSIGN(
      auto input_tiles,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_EQ(input_tiles.size(), 2);

  ASSERT_OK_AND_ASSIGN(
      auto from_data,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          0));
  EXPECT_EQ(from_data.size(), 1);

  ASSERT_OK_AND_ASSIGN(
      auto from_indices,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(1)->shape().dimensions()),
          1));
  EXPECT_EQ(from_indices.size(), 1);
}

TEST_F(TilePropagationTest, GatherOpOutOfBoundsInputIndex) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,64] parameter(0)
      p1 = s32[32,16] parameter(1)
      ROOT gather = f32[32,16,64] gather(p0, p1),
        offset_dims={2}, collapsed_slice_dims={0},
        start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,64}
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root),
                          &mlir_context_));
  EXPECT_THAT(
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          2),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->operand(0)->shape().dimensions()),
          -1),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu::experimental

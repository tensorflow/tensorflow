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

#include "xla/service/gpu/model/experimental/symbolic_tile_propagation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/service/gpu/model/experimental/tiling_space.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::MLIRContext;
using ::testing::Optional;

MATCHER_P(MatchString, symbolic_tile_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(symbolic_tile_string, arg.ToString()),
      result_listener);
}

class SymbolicTilePropagationTest : public HloHardwareIndependentTestBase {
 public:
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

ExperimentalSymbolicTile GetTestSymbolicTile(MLIRContext* mlir_context,
                                             absl::Span<const int64_t> shape,
                                             int64_t num_rt_vars = 0) {
  int64_t rank = shape.size();
  SmallVector<DimTile> one_dim_tiles;
  one_dim_tiles.reserve(rank);
  CHECK(mlir_context);
  for (auto [index, dim] : llvm::enumerate(shape)) {
    auto tid = getAffineDimExpr(index, mlir_context);
    auto ts = getAffineSymbolExpr(index, mlir_context);
    one_dim_tiles.push_back(DimTile{
        tid * ts, ts, mlir::getAffineConstantExpr(index + 1, mlir_context),
        mlir::getAffineConstantExpr(dim, mlir_context)});
  }
  return ExperimentalSymbolicTile{mlir_context, /*num_tile_ids=*/rank,
                                  num_rt_vars, std::move(one_dim_tiles)};
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT add0 = f32[10,20] add(p0, p1)
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1)[ts_0, ts_1]
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
    1) (tid_0, tid_1)[ts_0, ts_1]
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
  )")));
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfBroadcastOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,30] parameter(0)
      ROOT broadcast = f32[10,20,30] broadcast(p0), dimensions={0,2}
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 3]
         upper bounds [10, 30]
  )")));
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfConcatenateOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10] parameter(0)
      p1 = f32[20] parameter(1)
      p2 = f32[30] parameter(2)
      ROOT concatenate = f32[60] concatenate(p0, p1, p2), dimensions={0}
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [10]
    1) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0 - 10]
         sizes [ts_0]
         strides [1]
         upper bounds [20]
    2) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0 - 30]
         sizes [ts_0]
         strides [1]
         upper bounds [30]
  )")));
}

TEST_F(SymbolicTilePropagationTest,
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
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  ExperimentalSymbolicTile symbolic_tile =
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions());
  llvm::SmallVector<AffineExpr, 1> upper_bounds{
      mlir::getAffineConstantExpr(25, &mlir_context)};
  symbolic_tile =
      ExperimentalSymbolicTile{&mlir_context,         /*num_tile_ids=*/1,
                               /*num_rt_vars=*/0,     symbolic_tile.offsets(),
                               symbolic_tile.sizes(), symbolic_tile.strides(),
                               upper_bounds};
  std::optional<TiledOperands> tiled_operands =
      PropagateTileToInput(tiling_space, *root, symbolic_tile, 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0]
         sizes [ts_0]
         strides [1]
         upper bounds [10]
    1) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0 - 10]
         sizes [ts_0]
         strides [1]
         upper bounds [15]
    2) (tid_0)[ts_0]
      -> offsets [tid_0 * ts_0 - 30]
         sizes [ts_0]
         strides [1]
         upper bounds [0]
  )")));
}

TEST_F(SymbolicTilePropagationTest,
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
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  ExperimentalSymbolicTile symbolic_tile =
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions());
  llvm::SmallVector<AffineExpr, 1> upper_bounds{
      mlir::getAffineDimExpr(0, &mlir_context) * 30};
  symbolic_tile =
      ExperimentalSymbolicTile{&mlir_context,         /*num_tile_ids=*/1,
                               /*num_rt_vars=*/0,     symbolic_tile.offsets(),
                               symbolic_tile.sizes(), symbolic_tile.strides(),
                               upper_bounds};
  std::optional<TiledOperands> tiled_operands =
      PropagateTileToInput(tiling_space, *root, symbolic_tile, 0);
  EXPECT_EQ(tiled_operands, std::nullopt);
}

TEST_F(SymbolicTilePropagationTest,
       CanPropagateToInputsOfPadOpWithEdgePadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[12,13] pad(p0, p1), padding=1_7x0_9
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()),
      /*result_index=*/0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1)[ts_0, ts_1]
      -> offsets [tid_0 * ts_0 - 1, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [4, 4]
    1) (tid_0, tid_1)[ts_0, ts_1]
      -> offsets [] sizes [] strides [] upper bounds []
  )")));
}

TEST_F(SymbolicTilePropagationTest,
       CanNotPropagateToInputsOfPadOpWithInteriorPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[30,13] pad(p0, p1), padding=1_4_7x0_9
    }
  )");
  MLIRContext mlir_context;
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()),
      /*result_index=*/0);
  EXPECT_EQ(tiled_operands, std::nullopt);
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfTransposeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,5,1,3] parameter(0)
      ROOT transpose = f32[1,2,3,5] transpose(p0), dimensions={2,0,3,1}
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)[ts_0, ts_1, ts_2, ts_3]
      -> offsets [tid_1 * ts_1, tid_3 * ts_3, tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_1, ts_3, ts_0, ts_2]
         strides [2, 4, 1, 3]
         upper bounds [2, 5, 1, 3]
  )")));
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfSliceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[5,7,13] parameter(0)
      ROOT slice = f32[2,7,4] slice(p0), slice={[1:5:2], [0:7], [5:13:2]}
    }
  )");
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      tiling_space, *root,
      GetTestSymbolicTile(&mlir_context, root->shape().dimensions()), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [(tid_0 * ts_0) * 2 + 1, tid_1 * ts_1, (tid_2 * ts_2) * 2 + 5]
         sizes [ts_0, ts_1, ts_2]
         strides [2, 2, 6]
         upper bounds [5, 7, 13]
  )")));
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfDynSliceOp) {
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
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  auto symbolic_tile = GetTestSymbolicTile(
      &mlir_context, root->shape().dimensions(), /*num_rt_vars=*/3);
  std::optional<TiledOperands> tiled_operands =
      PropagateTileToInput(tiling_space, *root, symbolic_tile, 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]{rt_0, rt_1, rt_2}
      -> offsets [tid_0 * ts_0 + 4, tid_1 * ts_1 + rt_1, tid_2 * ts_2 + rt_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [5, rt_1 + 2, rt_2 + 32]
    1) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [] sizes [] strides [] upper bounds []
    2) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [] sizes [] strides [] upper bounds []
    3) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [] sizes [] strides [] upper bounds []
  )")));
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfDotOp) {
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
  auto tiling_space =
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(root));
  MLIRContext mlir_context;
  auto symbolic_tile = GetTestSymbolicTile(
      &mlir_context, root->shape().dimensions(), /*num_rt_vars=*/0);
  symbolic_tile = ExperimentalSymbolicTile{&mlir_context,
                                           /*num_tile_ids=*/8,
                                           /*num_rt_vars=*/0,
                                           symbolic_tile.offsets(),
                                           symbolic_tile.sizes(),
                                           symbolic_tile.strides(),
                                           symbolic_tile.upper_bounds()};
  std::optional<TiledOperands> tiled_operands =
      PropagateTileToInput(tiling_space, *root, symbolic_tile, 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
       [ts_0, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7]
         -> offsets [tid_2 * ts_2, tid_1 * ts_1, tid_7 * ts_7,
                     tid_3 * ts_3, tid_6 * ts_6, tid_0 * ts_0]
            sizes [ts_2, ts_1, ts_7, ts_3, ts_6, ts_0]
            strides [3, 2, 1, 4, 1, 1]
            upper bounds [4, 38, 17, 11, 18, 10]
    1) (tid_0, tid_1, tid_2, tid_3, tid_4, tid_5, tid_6, tid_7)
       [ts_0, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7]
         -> offsets [tid_7 * ts_7, tid_0 * ts_0, tid_4 * ts_4,
                     tid_6 * ts_6, tid_5 * ts_5, tid_1 * ts_1]
            sizes [ts_7, ts_0, ts_4, ts_6, ts_5, ts_1]
            strides [1, 1, 5, 1, 6, 2]
            upper bounds [17, 10, 16, 18, 22, 38]
  )")));
}

}  // namespace
}  // namespace xla::gpu

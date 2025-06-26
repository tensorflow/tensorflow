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
#include "xla/service/gpu/model/experimental/symbolic_tile.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
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

ExperimentalSymbolicTile GetTestSymbolicTile(absl::Span<const int64_t> shape,
                                             MLIRContext* mlir_context) {
  int64_t rank = shape.size();
  SmallVector<AffineExpr> offsets, strides, sizes, upper_bounds;
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);
  upper_bounds.reserve(rank);
  CHECK(mlir_context);
  for (auto [index, dim] : llvm::enumerate(shape)) {
    auto tid = getAffineDimExpr(index, mlir_context);
    auto ts = getAffineSymbolExpr(index, mlir_context);
    offsets.push_back(tid * ts);
    sizes.push_back(ts);
    strides.push_back(mlir::getAffineConstantExpr(index + 1, mlir_context));
    upper_bounds.push_back(mlir::getAffineConstantExpr(dim, mlir_context));
  }
  return ExperimentalSymbolicTile{
      mlir_context, /*num_tile_ids=*/rank, offsets,       sizes,
      strides,      upper_bounds,          /*rt_vars=*/{}};
}

TEST_F(SymbolicTilePropagationTest, CanPropagateToInputsOfElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )");
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context), 0);
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
      p0 = f32[10, 30] parameter(0)
      ROOT broadcast = f32[10, 20, 30] broadcast(p0), dimensions={0,2}
    }
  )");
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 3]
         upper bounds [10, 30]
  )")));
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
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context),
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
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context),
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
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context), 0);
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
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands = PropagateTileToInput(
      *root, GetTestSymbolicTile(root->shape().dimensions(), &mlir_context), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1, tid_2)[ts_0, ts_1, ts_2]
      -> offsets [(tid_0 * ts_0) * 2 + 1, tid_1 * ts_1, (tid_2 * ts_2) * 2 + 5]
         sizes [ts_0, ts_1, ts_2]
         strides [2, 2, 6]
         upper bounds [5, 7, 13]
  )")));
}

}  // namespace
}  // namespace xla::gpu

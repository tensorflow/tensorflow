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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu::experimental {
namespace {

using ::absl_testing::StatusIs;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

MATCHER_P(MatchToString, test_string, "") {
  return ExplainMatchResult(true, ApproximateMatch(test_string, ToString(arg)),
                            result_listener);
}

class TilePropagationTest : public HloHardwareIndependentTestBase {
 public:
  TilePropagationTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

struct ReshapeTestCase {
  std::string name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> input_tile_sizes;  // Empty means remain symbolic.
  std::vector<int64_t> input_tile_strides;
  std::vector<int64_t> output_shape;
  std::string expected_output;

  // TODO(b/477615292) - Add checks for upper bounds.
};

class ReshapeTilePropagationTest
    : public TilePropagationTest,
      public ::testing::WithParamInterface<ReshapeTestCase> {};

TEST_P(ReshapeTilePropagationTest, PropagateReshape) {
  const auto& param = GetParam();
  Shape input_shape = ShapeUtil::MakeShape(F32, param.input_shape);
  Shape output_shape = ShapeUtil::MakeShape(F32, param.output_shape);

  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(output_shape, p0));

  auto tiling_space = TilingSpace::Create(*HloFusionAdaptor::ForInstruction(p0),
                                          &mlir_context_);
  if (!param.input_tile_sizes.empty()) {
    CHECK_EQ(param.input_tile_sizes.size(), tiling_space->num_dimensions());
    tiling_space->AssignTileSizes(param.input_tile_sizes);
  }
  SmallVector<DimTile> input_dim_tiles =
      llvm::to_vector(tiling_space->tiled_roots()[0].dim_tiles());
  CHECK_EQ(param.input_tile_strides.size(), tiling_space->num_dimensions());
  for (int i = 0; i < input_dim_tiles.size(); ++i) {
    input_dim_tiles[i].stride =
        CreateSymbolicConstant(param.input_tile_strides[i], &mlir_context_);
  }
  Tile input_tile = Tile(*tiling_space, std::move(input_dim_tiles));
  auto output_tiles =
      PropagateTileToOutput(*tiling_space, *reshape, input_tile, 0);

  if (param.expected_output.empty()) {
    ASSERT_FALSE(output_tiles.ok());
  } else {
    ASSERT_TRUE(output_tiles.ok());
    EXPECT_THAT(output_tiles.value(), MatchToString(param.expected_output));
  }
}

// TODO(b/491727659): Convert this to lit infra.
INSTANTIATE_TEST_SUITE_P(
    ReshapeTilePropagationTests, ReshapeTilePropagationTest,
    ::testing::ValuesIn<ReshapeTestCase>({
        {"Identity",
         /*input_shape=*/{10, 20},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2},
         /*output_shape=*/{10, 20},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
  )"},
        {"IncreaseRank",
         /*input_shape=*/{10},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1},
         /*output_shape=*/{1, 10, 1},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [0, tid_0 * ts_0, 0]
         sizes [1, ts_0, 1]
         strides [1, 1, 1]
         upper bounds [1, 10, 1]
  )"},
        {"DecreaseRank",
         /*input_shape=*/{1, 10, 1},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2, 3},
         /*output_shape=*/{10},
         /*expected_output=*/R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_1 * ts_1]
         sizes [ts_1]
         strides [2]
         upper bounds [10]
  )"},
        {"Generic",
         /*input_shape=*/{2, 5, 7},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2, 3},
         /*output_shape=*/{7, 5, 2},
         /*expected_output=*/""},
        {"SupportedMultiSegment",
         /*input_shape=*/{12, 1, 8},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2, 3},
         /*output_shape=*/{1, 12, 8},
         /*expected_output=*/R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [0, tid_0 * ts_0, tid_2 * ts_2]
         sizes [1, ts_0, ts_2]
         strides [1, 1, 3]
         upper bounds [1, 12, 8]
  )"},
        {"UnsupportedMultiSegment",
         /*input_shape=*/{12, 4},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2},
         /*output_shape=*/{1, 12, 2, 2},
         /*expected_output=*/""},
        {"ExpandShape",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{1},
         /*input_tile_strides=*/{1},
         /*output_shape=*/{3, 4},
         /*expected_output=*/""},
        // Example (tid_0, tid_1) -> (offset, upper bound):
        // (0, 0) -> (0,  3), (0, 1) -> ( 3,  4)
        // (1, 0) -> (4,  7), (1, 1) -> ( 7,  8)
        // (2, 0) -> (8, 11), (2, 1) -> (11, 12)
        {"CollapseShapeCase1_Stride1_LastDimPartialTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3]
         sizes [3]
         strides [1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1]
  )"},
        {"CollapseShapeCase2_Stride1_LastDimFullTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 4},
         /*input_tile_strides=*/{1, 1},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 8 + tid_1 * 4]
         sizes [8]
         strides [1]
         upper bounds [min(tid_0 * 2 + 1, 2) * 4 + min(tid_1 * 4 + 3, 3) + 1]
  )"},
        // Example (tid_0, tid_1) -> (offset, upper bound):
        // (0, 0) -> (0,  4), (0, 1) -> ( 3,  4)
        // (1, 0) -> (4,  8), (1, 1) -> ( 7,  8)
        // (2, 0) -> (8, 12), (2, 1) -> (11, 12)
        {"CollapseShapeCase3_StrideNot1_LastDimPartialTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 2},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3]
         sizes [3]
         strides [2]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 4, 3) + 1]
  )"},
        {"CollapseShape_WithLeadingOneInOutput",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*output_shape=*/{1, 12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [0, tid_0 * 4 + tid_1 * 3]
         sizes [1, 3]
         strides [1, 1]
         upper bounds [1, min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1]
  )"},
        {"CollapseShape_WithTrailingOneInOutput",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*output_shape=*/{12, 1},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3, 0]
         sizes [3, 1]
         strides [1, 1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1, 1]
  )"},
        {"CollapseShape_WithMiddleOneInInput",
         /*input_shape=*/{3, 1, 4},
         /*input_tile_sizes=*/{1, 1, 3},
         /*input_tile_strides=*/{1, 1, 1},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * 4 + tid_2 * 3]
         sizes [3]
         strides [1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_2 * 3 + 2, 3) + 1]
  )"},
    }),
    [](const ::testing::TestParamInfo<ReshapeTilePropagationTest::ParamType>&
           info) { return info.param.name; });

TEST_F(TilePropagationTest, CanPropagateToInputsOfElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT add0 = f32[10,20] add(p0, p1)
    }
  )");
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
      %ar-start = f32[2,8,256] all-reduce-start(p0), replica_groups={{0,1}},
        to_apply=%add
      ROOT %ar-done = f32[2,8,256] all-reduce-done(%ar-start)
    }
  )");
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  ASSERT_OK_AND_ASSIGN(
      auto ar_done_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(ar_done_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [2, 8, 256]
  )"));
  ASSERT_OK_AND_ASSIGN(
      auto ar_start_operands,
      PropagateTileToInput(
          *tiling_space, *root->operand(0),
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(ar_start_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_0, ts_1, ts_2]
         strides [1, 2, 3]
         upper bounds [2, 8, 256]
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  ASSERT_OK_AND_ASSIGN(
      auto input_tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(input_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2, tid_3 * ts_3, tid_1 * ts_1]
         sizes [ts_0, ts_2, ts_3, ts_1]
         strides [1, 3, 4, 2]
         upper bounds [3, 128, 12288, 6]
  )"));
  ASSERT_OK_AND_ASSIGN(
      auto output_tiled_operands,
      PropagateTileToOutput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));
  EXPECT_THAT(output_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * ts_0, tid_3 * ts_3, tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_0, ts_3, ts_1, ts_2]
         strides [1, 4, 2, 3]
         upper bounds [3, 12288, 6, 128]
  )"));
}

TEST_F(TilePropagationTest, CanPropagateThroughBitcastReshapeOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 32] parameter(0)
      ROOT bitcast = f32[4, 8, 4] bitcast(p0)
    }
  )");
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  EXPECT_THAT(PropagateTileToInput(
                  *tiling_space, *root,
                  GetTestTile(*tiling_space, root->shape().dimensions()), 0),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(PropagateTileToInput(
                  *tiling_space, *root,
                  GetTestTile(*tiling_space, root->shape().dimensions()), 0),
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);

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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  Tile tile = GetTestTile(*tiling_space, root->shape().dimensions());
  llvm::SmallVector<SymbolicExpr, 1> upper_bounds{
      CreateDimExpr(0, &mlir_context_) * 30};
  tile = Tile{*tiling_space, tile.offsets(), tile.sizes(), tile.strides(),
              upper_bounds};
  EXPECT_THAT(PropagateTileToInput(*tiling_space, *root, tile, 0),
              StatusIs(absl::StatusCode::kUnimplemented));
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  auto tile = GetTestTile(*tiling_space, root->shape().dimensions());
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  auto tile = GetTestTile(*tiling_space, root->shape().dimensions());
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  auto tile = GetTestTile(*tiling_space, root->shape().dimensions());
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
      -> offsets [(tid_0 * ts_0) floordiv 32, (tid_2 * ts_2) floordiv 256]
         sizes [(tid_0 * ts_0 + ts_0 - 1) floordiv 32 - (tid_0 * ts_0) floordiv 32 + 1, (tid_2 * ts_2 + ts_2 - 1) floordiv 256 - (tid_2 * ts_2) floordiv 256 + 1]
         strides [1, 1]
         upper bounds [32, 2]
    3) (tid_0, tid_1, tid_2)
      -> offsets [tid_1 * ts_1, tid_2 * ts_2]
         sizes [ts_1, ts_2]
         strides [2, 1]
         upper bounds [64, 512]
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);

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

  tiling_space->AssignTileSizes({8, 16, 32, 64});
  ASSERT_OK_AND_ASSIGN(auto concrete_tiled_operands,
                       PropagateTileToInput(*tiling_space, *root,
                                            tiling_space->tiled_roots()[0], 0));
  EXPECT_THAT(concrete_tiled_operands, MatchToString(R"(
    0) (tid_0, tid_1, tid_2, tid_3)
      -> offsets [tid_0 * 8, tid_3 * 64, tid_1 * 16, tid_2 * 32]
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
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
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  MLIRContext mlir_context;
  auto tile = GetTestTile(*tiling_space, GetFirstShape(root).dimensions());
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

}  // namespace
}  // namespace xla::gpu::experimental

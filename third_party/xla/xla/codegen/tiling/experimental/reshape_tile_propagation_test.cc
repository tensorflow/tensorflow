/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::SmallVector;
using ::mlir::MLIRContext;

struct ReshapeTestCase {
  std::string name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> input_tile_sizes;  // Empty means remain symbolic.
  std::vector<int64_t> input_tile_strides;
  std::vector<int64_t> input_tile_offsets;
  std::vector<int64_t> output_shape;
  std::string expected_output;
  // TODO(b/477615292) - Add checks for upper bounds.
};

class ReshapeTilePropagationTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<ReshapeTestCase> {
 public:
  ReshapeTilePropagationTest() = default;

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_P(ReshapeTilePropagationTest, PropagateReshape) {
  const auto& param = GetParam();
  Shape input_shape = ShapeUtil::MakeShape(F32, param.input_shape);
  Shape output_shape = ShapeUtil::MakeShape(F32, param.output_shape);

  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(output_shape, p0));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(p0),
                          &mlir_context_));
  if (!param.input_tile_sizes.empty()) {
    CHECK_EQ(param.input_tile_sizes.size(), tiling_space->num_dimensions());
    ASSERT_OK(tiling_space->AssignTileSizes(param.input_tile_sizes));
  }
  SmallVector<DimTile> input_dim_tiles =
      llvm::to_vector(tiling_space->tiled_roots()[0].dim_tiles());
  CHECK_EQ(param.input_tile_strides.size(), tiling_space->num_dimensions());
  bool has_offsets = input_dim_tiles.size() == param.input_tile_offsets.size();
  for (int i = 0; i < input_dim_tiles.size(); ++i) {
    if (has_offsets) {
      input_dim_tiles[i].offset =
          CreateSymbolicConstant(param.input_tile_offsets[i], &mlir_context_);
    }
    input_dim_tiles[i].stride =
        CreateSymbolicConstant(param.input_tile_strides[i], &mlir_context_);
  }
  Tile input_tile = Tile(*tiling_space, std::move(input_dim_tiles));
  auto output_tiles =
      PropagateTileToOutput(*tiling_space, *reshape, input_tile, 0);

  input_tile.Simplify();
  if (output_tiles.ok()) {
    ASSERT_EQ(output_tiles->size(), 1);
    auto output_tile = output_tiles.value()[0];
    output_tile.Simplify();
    ASSERT_OK(VerifyTileEquivalence(input_tile, input_shape, output_tile,
                                    output_shape, tiling_space.get()));
  }
  if (param.expected_output.empty()) {
    ASSERT_FALSE(output_tiles.ok());
  } else {
    ASSERT_TRUE(output_tiles.ok())
        << "Failed for " << param.name << ": " << output_tiles.status();
    EXPECT_THAT(output_tiles.value(), MatchToString(param.expected_output));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeTilePropagationTests, ReshapeTilePropagationTest,
    ::testing::ValuesIn<ReshapeTestCase>({
        // =====================================================================
        // General / Other Reshapes
        // =====================================================================
        {"Identity",
         /*input_shape=*/{10, 20},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{10, 20},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * ts_0, tid_1 * ts_1]
         sizes [ts_0, ts_1]
         strides [1, 2]
         upper bounds [10, 20]
  )"},
        {"IdentityConcrete",
         /*input_shape=*/{10, 20},
         /*input_tile_sizes=*/{2, 2},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{10, 20},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 2, tid_1 * 2]
         sizes [2, 2]
         strides [1, 1]
         upper bounds [10, 20]
  )"},
        {"IncreaseRank",
         /*input_shape=*/{10},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{},
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
         /*input_tile_offsets=*/{},
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
         /*input_tile_offsets=*/{},
         /*output_shape=*/{7, 5, 2},
         /*expected_output=*/""},
        {"SupportedMultiSegment",
         /*input_shape=*/{12, 1, 8},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2, 3},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 12, 8},
         /*expected_output=*/R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [0, tid_0 * ts_0, tid_2 * ts_2]
         sizes [1, ts_0, ts_2]
         strides [1, 1, 3]
         upper bounds [1, 12, 8]
  )"},
        {"UnsupportedMultiSegment",
         /*input_shape=*/{12, 2, 5, 7},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{1, 2, 3, 4},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 12, 7, 5, 2},
         /*expected_output=*/""},

        // =====================================================================
        // CollapseShapeContiguous
        // =====================================================================
        // Example (tid_0, tid_1) -> (offset, upper bound):
        // (0, 0) -> (0,  3), (0, 1) -> ( 3,  4)
        // (1, 0) -> (4,  7), (1, 1) -> ( 7,  8)
        // (2, 0) -> (8, 11), (2, 1) -> (11, 12)
        {"CollapseShapeContiguous_Stride1_LastDimPartialTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3]
         sizes [3]
         strides [1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1]
  )"},
        {"CollapseShapeContiguous_Stride1_LastDimFullTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 4},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 8]
         sizes [8]
         strides [1]
         upper bounds [min(tid_0 * 2 + 1, 2) * 4 + 4]
  )"},
        {"CollapseShapeContiguous_10x4_1x4",
         /*input_shape=*/{10, 4},
         /*input_tile_sizes=*/{1, 4},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{40},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4]
         sizes [4]
         strides [1]
         upper bounds [min(tid_0, 9) * 4 + 4]
  )"},
        // Example (tid_0, tid_1) -> (offset, upper bound):
        // (0, 0) -> (0,  4), (0, 1) -> ( 3,  4)
        // (1, 0) -> (4,  8), (1, 1) -> ( 7,  8)
        // (2, 0) -> (8, 12), (2, 1) -> (11, 12)
        {"CollapseShapeContiguous_StrideNot1_LastDimPartialTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 2},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3]
         sizes [3]
         strides [2]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 4, 3) + 1]
  )"},
        {"CollapseShapeContiguous_WithLeadingOneInOutput",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [0, tid_0 * 4 + tid_1 * 3]
         sizes [1, 3]
         strides [1, 1]
         upper bounds [1, min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1]
  )"},
        {"CollapseShapeContiguous_WithTrailingOneInOutput",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12, 1},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [tid_0 * 4 + tid_1 * 3, 0]
         sizes [3, 1]
         strides [1, 1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_1 * 3 + 2, 3) + 1, 1]
  )"},
        {"CollapseShapeContiguous_WithMiddleOneInInput",
         /*input_shape=*/{3, 1, 4},
         /*input_tile_sizes=*/{1, 1, 3},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * 4 + tid_2 * 3]
         sizes [3]
         strides [1]
         upper bounds [min(tid_0, 2) * 4 + min(tid_2 * 3 + 2, 3) + 1]
  )"},
        {"CollapseShapeContiguous_3DCollapseWithTrivialInnerDim",
         /*input_shape=*/{2, 32, 128},
         /*input_tile_sizes=*/{1, 16, 1},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{8192},
         /*expected_output=*/R"(
      0) (tid_0, tid_1, tid_2)
        -> offsets [tid_0 * 4096 + tid_1 * 2048 + tid_2]
           sizes [16]
           strides [128]
           upper bounds [min(tid_1 * 16 + 15, 31) * 128 + min(tid_0, 1) * 4096 + min(tid_2, 127) + 1]
    )"},
        {"CollapseShapeContiguous_3DCollapseWithTrivialInnerDim_Strided",
         /*input_shape=*/{2, 32, 128},
         /*input_tile_sizes=*/{1, 16, 1},
         /*input_tile_strides=*/{1, 1, 2},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{8192},
         /*expected_output=*/R"(
      0) (tid_0, tid_1, tid_2)
        -> offsets [tid_0 * 4096 + tid_1 * 2048 + tid_2]
           sizes [16]
           strides [128]
           upper bounds [min(tid_1 * 16 + 15, 31) * 128 + min(tid_0, 1) * 4096 + min(tid_2, 127) + 1]
    )"},
        {"CollaseShapeNonContinousTile1",
         /*input_shape=*/{17, 2, 4},
         /*input_tile_sizes=*/{4, 1, 4},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{0, 0, 0},
         /*output_shape=*/{136},
         /*expected_output=*/""},
        {"CollapseShapeContiguous_FullySpannedInnermost",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{3, 2},
         /*input_tile_strides=*/{1, 2},
         /*input_tile_offsets=*/{0, 0},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [0]
         sizes [6]
         strides [2]
         upper bounds [11]
  )"},
        {"CollapseShapeContiguous_PreserveInnermostStride",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 2},
         /*input_tile_strides=*/{1, 2},
         /*input_tile_offsets=*/{1, 0},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [4]
         sizes [2]
         strides [2]
         upper bounds [7]
  )"},

        // =====================================================================
        // CollapseShapeNonContiguous
        // =====================================================================
        {"CollapseShapeNonContiguous_SteppedOuterDimension",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 1},
         /*input_tile_strides=*/{2, 1},
         /*input_tile_offsets=*/{0, 0},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [0]
         sizes [2]
         strides [8]
         upper bounds [9]
  )"},
        {"CollapseShapeNonContiguous_MultipleSteppedOuterDimensions",
         /*input_shape=*/{3, 4, 5},
         /*input_tile_sizes=*/{2, 2, 1},
         /*input_tile_strides=*/{2, 2, 1},
         /*input_tile_offsets=*/{0, 0, 0},
         /*output_shape=*/{60},
         /*expected_output=*/""},
        {"CollapseShapeNonContiguous_SteppedOuterDimensionAndAnotherTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 2},
         /*input_tile_strides=*/{2, 1},
         /*input_tile_offsets=*/{0, 0},
         /*output_shape=*/{12},
         /*expected_output=*/""},
        {"CollapseShapeNonContiguous_SteppedOuterAndInnermostStrideNot1_"
         "InnermostSize1",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 1},
         /*input_tile_strides=*/{2, 2},
         /*input_tile_offsets=*/{0, 0},
         /*output_shape=*/{12},
         /*expected_output=*/R"(
    0) (tid_0, tid_1)
      -> offsets [0]
         sizes [2]
         strides [8]
         upper bounds [9]
  )"},
        {"CollapseShapeNonContiguous_SteppedOuterAndInnermostStrideNot1_"
         "BothTiled",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{2, 2},
         /*input_tile_strides=*/{2, 2},
         /*input_tile_offsets=*/{0, 0},
         /*output_shape=*/{12},
         /*expected_output=*/""},
        {"CollapseShapeNonContiguous_ZeroStride",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{0, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/""},
        {"CollapseShapeNonContiguous_NegativeStride",
         /*input_shape=*/{3, 4},
         /*input_tile_sizes=*/{1, 3},
         /*input_tile_strides=*/{-1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{12},
         /*expected_output=*/""},
        {"CollapseShapeTrivialTiledDim",
         /*input_shape=*/{1, 4},
         /*input_tile_sizes=*/{2, 2},
         /*input_tile_strides=*/{1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{4},
         /*expected_output=*/R"(
         0) (tid_0, tid_1) ->
          offsets [tid_1 * 2] sizes [2] strides [1] upper bounds [4] )"},
        {"CollapseShapeWithTrivialTiledDimInGroup",
         /*input_shape=*/{2, 1, 2},
         /*input_tile_sizes=*/{2, 2, 2},
         /*input_tile_strides=*/{1, 2, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{4},
         /*expected_output=*/R"(
         0) (tid_0, tid_1, tid_2) ->
          offsets [0] sizes [4] strides [1] upper bounds [4]
        )"},
        {"CollapseToSingleElement",
         /*input_shape=*/{1, 1, 1},
         /*input_tile_sizes=*/{1, 1, 1},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1},
         /*expected_output=*/R"(
         0) (tid_0, tid_1, tid_2) ->
          offsets [0] sizes [1] strides [1] upper bounds [1]
        )"},
        {"CollapseToSingleElementTiled",
         /*input_shape=*/{1, 1, 1},
         /*input_tile_sizes=*/{2, 1, 1},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1},
         /*expected_output=*/R"(
         0) (tid_0, tid_1, tid_2) ->
          offsets [0] sizes [1] strides [1] upper bounds [1]
        )"},
        {"CollapseToScalar",
         /*input_shape=*/{1, 1, 1},
         /*input_tile_sizes=*/{2, 2, 2},
         /*input_tile_strides=*/{1, 1, 1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{},
         /*expected_output=*/R"(
         0) (tid_0, tid_1, tid_2) ->
          offsets [] sizes [] strides [] upper bounds []
        )"},
        // =====================================================================
        // ExpandShapeContiguous
        // =====================================================================
        {"ExpandShapeContiguous_FullTargetInnerDim",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{4},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{3, 4},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [tid_0, 0]
         sizes [1, 4]
         strides [1, 1]
         upper bounds [tid_0 + 1, 4]
  )"},
        {"ExpandShapeContiguous_PartialTargetInnerDim",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{2},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{1},
         /*output_shape=*/{3, 4},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [0, 1]
         sizes [1, 2]
         strides [1, 1]
         upper bounds [1, 3]
  )"},
        {"ExpandShapeContiguous_MultipleTargetInnerDims",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{8},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{4},
         /*output_shape=*/{3, 4},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [1, 0]
         sizes [2, 4]
         strides [1, 1]
         upper bounds [3, 4]
  )"},
        {"ExpandShapeContiguous_Unsupported_NonBox",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{5},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{0},
         /*output_shape=*/{3, 4},
         /*expected_output=*/""},
        {"ExpandShapeContiguous_WithUnitDim",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{4},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{3, 1, 4},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [tid_0, 0, 0]
         sizes [1, 1, 4]
         strides [1, 1, 1]
         upper bounds [tid_0 + 1, 1, 4]
  )"},
        {"ExpandShapeContiguous_To1DIdentity",
         /*input_shape=*/{12},
         /*input_tile_sizes=*/{4},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{4},
         /*output_shape=*/{1, 12},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [0, 4]
         sizes [1, 4]
         strides [1, 1]
         upper bounds [1, 12]
  )"},
        {"ExpandSingleElement",
         /*input_shape=*/{1},
         /*input_tile_sizes=*/{1},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 1, 1},
         /*expected_output=*/R"(
         0) (tid_0) ->
          offsets [0, 0, 0]
          sizes [1, 1, 1]
          strides [1, 1, 1]
          upper bounds [1, 1, 1]
        )"},
        {"ExpandSingleTiledElement",
         /*input_shape=*/{1},
         /*input_tile_sizes=*/{1},
         /*input_tile_strides=*/{1},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 1, 1},
         /*expected_output=*/R"(
         0) (tid_0) ->
          offsets [0, 0, 0]
          sizes [1, 1, 1]
          strides [1, 1, 1]
          upper bounds [1, 1, 1]
        )"},
        {"ExpandScalar",
         /*input_shape=*/{},
         /*input_tile_sizes=*/{},
         /*input_tile_strides=*/{},
         /*input_tile_offsets=*/{},
         /*output_shape=*/{1, 1, 1},
         /*expected_output=*/R"(
         0) () ->
          offsets [0, 0, 0]
          sizes [1, 1, 1]
          strides [1, 1, 1]
          upper bounds [1, 1, 1]
        )"},

        // =====================================================================
        // ExpandShapeNonContiguous
        // =====================================================================
        {"ExpandShapeNonContiguous_SteppedSource",
         /*input_shape=*/{128},
         /*input_tile_sizes=*/{2},
         /*input_tile_strides=*/{64},
         /*input_tile_offsets=*/{0},
         /*output_shape=*/{1, 2, 64},
         /*expected_output=*/R"(
    0) (tid_0)
      -> offsets [0, 0, 0]
         sizes [1, 2, 1]
         strides [1, 1, 1]
         upper bounds [1, 2, 1]
  )"},
    }),
    [](const ::testing::TestParamInfo<ReshapeTilePropagationTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
}  // namespace xla::gpu::experimental

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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/analysis/symbolic_map_serialization.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::SmallVector;
using ::mlir::MLIRContext;

// Test case parameters for reshape tile propagation tests.
struct ReshapeTestCase {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> to_shape;
  // tile_sizes can be left empty, to keep the tiling symbolic.
  std::vector<int64_t> tile_sizes;
  // `offsets` and `upper_bounds` are symbolic expressions containing
  // dimension indexes `tid_{x}`, tile size variables `ts_{y}`.
  std::vector<std::string> offsets;
  std::vector<int64_t> strides;
  std::vector<std::string> upper_bounds;
  // If not empty, expected error message.
  std::string expected_error;
};

absl::StatusOr<Tile> CreateTile(absl::Span<const int64_t> sizes,
                                absl::Span<const std::string> offsets,
                                absl::Span<const int64_t> strides,
                                absl::Span<const std::string> upper_bounds,
                                const TilingSpace& tiling_space) {
  int64_t num_dims = offsets.size();
  TF_RET_CHECK(offsets.size() == num_dims);
  TF_RET_CHECK(upper_bounds.size() == num_dims);
  TF_RET_CHECK(strides.size() == num_dims);
  std::vector<std::string> ts_strings(num_dims);
  std::vector<std::string> tid_strings(num_dims);
  std::vector<SymbolicExpr> ts_exprs(num_dims);
  llvm::DenseMap<llvm::StringRef, SymbolicExpr> variable_map;
  mlir::MLIRContext* ctx = tiling_space.mlir_context();
  for (int64_t i = 0; i < num_dims; ++i) {
    ts_strings[i] = absl::StrCat("ts_", i);
    tid_strings[i] = absl::StrCat("tid_", i);
    if (sizes.size() > i) {
      ts_exprs[i] = CreateSymbolicConstant(sizes[i], ctx);
    } else {
      ts_exprs[i] = CreateSymbolExpr(i, num_dims, ctx);
    }
    variable_map[ts_strings[i]] = ts_exprs[i];
    variable_map[tid_strings[i]] = CreateDimExpr(i, ctx);
  }

  llvm::SmallVector<DimTile> input_dim_tiles;
  input_dim_tiles.reserve(num_dims);
  for (int64_t i = 0; i < num_dims; ++i) {
    SymbolicExpr offset = ParseSymbolicExpr(offsets[i], ctx, variable_map);
    if (!offset) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse offset: ", offsets[i]));
    }
    SymbolicExpr stride = CreateSymbolicConstant(strides[i], ctx);
    auto upper_bound = ParseSymbolicExpr(upper_bounds[i], ctx, variable_map);
    if (!upper_bound) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse upper bound: ", upper_bounds[i]));
    }
    input_dim_tiles.push_back(
        DimTile{offset, ts_exprs[i], stride, upper_bound});
  }

  return Tile(tiling_space, std::move(input_dim_tiles));
}

absl::StatusOr<Tile> RunPropagation(const HloInstruction& reshape,
                                    const Tile& input_tile,
                                    TilingSpace& tiling_space) {
  ABSL_ASSIGN_OR_RETURN(Tiles output_tiles,
                   PropagateTileToOutput(tiling_space, reshape, input_tile, 0));
  TF_RET_CHECK(output_tiles.size() == 1)
      << "Expected exactly one output tile, got " << output_tiles.size();

  return std::move(output_tiles[0]);
}

class ReshapeExamplesTilePropagationTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<ReshapeTestCase> {
 public:
  ReshapeExamplesTilePropagationTest() = default;

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_P(ReshapeExamplesTilePropagationTest, PropagateReshape) {
  const ReshapeTestCase& param = GetParam();
  Shape shape = ShapeUtil::MakeShape(F32, param.shape);
  Shape to_shape = ShapeUtil::MakeShape(F32, param.to_shape);

  HloComputation::Builder builder("entry");
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(to_shape, p0));

  module_ = CreateNewVerifiedModule();
  module_->AddEntryComputation(builder.Build());

  auto tiling_space_statusor = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(p0), &mlir_context_);
  ASSERT_OK(tiling_space_statusor.status());
  std::unique_ptr<TilingSpace> tiling_space =
      std::move(tiling_space_statusor).value();

  if (!param.tile_sizes.empty()) {
    ASSERT_EQ(param.tile_sizes.size(), tiling_space->num_dimensions());
    ASSERT_OK(tiling_space->AssignTileSizes(param.tile_sizes));
  }

  ASSERT_OK_AND_ASSIGN(
      Tile input_tile,
      CreateTile(param.tile_sizes, param.offsets, param.strides,
                 param.upper_bounds, *tiling_space));
  absl::StatusOr<Tile> output_tile =
      RunPropagation(*reshape, input_tile, *tiling_space);

  if (!param.expected_error.empty()) {
    ASSERT_FALSE(output_tile.ok());
    EXPECT_THAT(output_tile.status(),
                absl_testing::StatusIs(
                    testing::_, testing::HasSubstr(param.expected_error)));
  } else {
    ASSERT_OK(output_tile);
    output_tile->Simplify();
    input_tile.Simplify();
    ASSERT_OK(VerifyTileEquivalence(input_tile, shape, *output_tile, to_shape,
                                    tiling_space.get()));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeTilePropagationTests, ReshapeExamplesTilePropagationTest,
    ::testing::ValuesIn<ReshapeTestCase>({
        {"IdentityConcrete",
         /*shape=*/{10, 20},
         /*to_shape=*/{10, 20},
         /*tile_sizes=*/{2, 2},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"10", "20"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_Stride1_LastDimPartialTiled",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_Stride1_LastDimFullTiled",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{2, 4},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_10x4_1x4",
         /*shape=*/{10, 4},
         /*to_shape=*/{40},
         /*tile_sizes=*/{1, 4},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"10", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_StrideNot1_LastDimPartialTiled",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 2},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_WithLeadingOneInOutput",
         /*shape=*/{3, 4},
         /*to_shape=*/{1, 12},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_WithTrailingOneInOutput",
         /*shape=*/{3, 4},
         /*to_shape=*/{12, 1},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_WithMiddleOneInInput",
         /*shape=*/{3, 1, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 1, 3},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"3", "1", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_3DCollapseWithTrivialInnerDim",
         /*shape=*/{2, 32, 128},
         /*to_shape=*/{8192},
         /*tile_sizes=*/{1, 16, 1},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"2", "32", "128"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_3DCollapseWithTrivialInnerDim_Strided",
         /*shape=*/{2, 32, 128},
         /*to_shape=*/{8192},
         /*tile_sizes=*/{1, 16, 1},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 2},
         /*upper_bounds=*/{"2", "32", "128"},
         /*expected_error=*/""},
        {"CollapseShapeNonContinousTile1",
         /*shape=*/{17, 2, 4},
         /*to_shape=*/{136},
         /*tile_sizes=*/{4, 1, 4},
         /*offsets=*/{"0", "0", "0"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"17", "2", "4"},
         /*expected_error=*/"Multiple dimensions are partially tiled"},
        {"CollapseShapeContiguous_FullySpannedInnermost",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{3, 2},
         /*offsets=*/{"0", "0"},
         /*strides=*/{1, 2},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeContiguous_PreserveInnermostStride",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 2},
         /*offsets=*/{"1", "0"},
         /*strides=*/{1, 2},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeNonContiguous_SteppedOuterDimension",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{2, 1},
         /*offsets=*/{"0", "0"},
         /*strides=*/{2, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/""},
        {"CollapseShapeNonContiguous_MultipleSteppedOuterDimensions",
         /*shape=*/{3, 4, 5},
         /*to_shape=*/{60},
         /*tile_sizes=*/{2, 2, 1},
         /*offsets=*/{"0", "0", "0"},
         /*strides=*/{2, 2, 1},
         /*upper_bounds=*/{"3", "4", "5"},
         /*expected_error=*/"At most one dimension can have stride >1"},
        {"CollapseShapeNonContiguous_SteppedOuterDimensionAndAnotherTiled",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{2, 2},
         /*offsets=*/{"0", "0"},
         /*strides=*/{2, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/
         "only the strided dimension 0 can have size > 1"},
        {"CollapseShapeNonContiguous_ZeroStride",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{0, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/
         "Expect constant positive source tile stride. Got: 0"},
        {"CollapseShapeNonContiguous_NegativeStride",
         /*shape=*/{3, 4},
         /*to_shape=*/{12},
         /*tile_sizes=*/{1, 3},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{-1, 1},
         /*upper_bounds=*/{"3", "4"},
         /*expected_error=*/
         "Expect constant positive source tile stride. Got: -1"},
        {"CollapseShapeTrivialTiledDim",
         /*shape=*/{1, 4},
         /*to_shape=*/{4},
         /*tile_sizes=*/{2, 2},
         /*offsets=*/{"ts_0 * tid_0", "ts_1 * tid_1"},
         /*strides=*/{1, 1},
         /*upper_bounds=*/{"1", "4"},
         /*expected_error=*/""},
        {"CollapseShapeWithTrivialTiledDimInGroup",
         /*shape=*/{2, 1, 2},
         /*to_shape=*/{4},
         /*tile_sizes=*/{2, 2, 2},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 2, 1},
         /*upper_bounds=*/{"2", "1", "2"},
         /*expected_error=*/""},
        {"CollapseToSingleElement",
         /*shape=*/{1, 1, 1},
         /*to_shape=*/{1},
         /*tile_sizes=*/{1, 1, 1},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"1", "1", "1"},
         /*expected_error=*/""},
        {"CollapseToSingleElementTiled",
         /*shape=*/{1, 1, 1},
         /*to_shape=*/{1},
         /*tile_sizes=*/{2, 1, 1},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"1", "1", "1"},
         /*expected_error=*/""},
        {"CollapseToScalar",
         /*shape=*/{1, 1, 1},
         /*to_shape=*/{},
         /*tile_sizes=*/{2, 2, 2},
         /*offsets=*/
         {"ts_0 * tid_0", "ts_1 * tid_1", "ts_2 * tid_2"},
         /*strides=*/{1, 1, 1},
         /*upper_bounds=*/{"1", "1", "1"},
         /*expected_error=*/""},
        {"ExpandShapeContiguous_FullTargetInnerDim",
         /*shape=*/{12},
         /*to_shape=*/{3, 4},
         /*tile_sizes=*/{4},
         /*offsets=*/{"ts_0 * tid_0"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/""},
        {"ExpandShapeContiguous_PartialTargetInnerDim",
         /*shape=*/{12},
         /*to_shape=*/{3, 4},
         /*tile_sizes=*/{2},
         /*offsets=*/{"1"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/""},
        {"ExpandShapeContiguous_MultipleTargetInnerDims",
         /*shape=*/{12},
         /*to_shape=*/{3, 4},
         /*tile_sizes=*/{8},
         /*offsets=*/{"4"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/""},
        {"ExpandShapeContiguous_Unsupported_NonBox",
         /*shape=*/{12},
         /*to_shape=*/{3, 4},
         /*tile_sizes=*/{5},
         /*offsets=*/{"0"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/"Multiple dimensions are partially tiled"},
        {"ExpandShapeContiguous_WithUnitDim",
         /*shape=*/{12},
         /*to_shape=*/{3, 1, 4},
         /*tile_sizes=*/{4},
         /*offsets=*/{"ts_0 * tid_0"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/""},
        {"ExpandShapeContiguous_To1DIdentity",
         /*shape=*/{12},
         /*to_shape=*/{1, 12},
         /*tile_sizes=*/{4},
         /*offsets=*/{"4"},
         /*strides=*/{1},
         /*upper_bounds=*/{"12"},
         /*expected_error=*/""},
        {"ExpandSingleElement",
         /*shape=*/{1},
         /*to_shape=*/{1, 1, 1},
         /*tile_sizes=*/{1},
         /*offsets=*/{"ts_0 * tid_0"},
         /*strides=*/{1},
         /*upper_bounds=*/{"1"},
         /*expected_error=*/""},
        {"ExpandSingleTiledElement",
         /*shape=*/{1},
         /*to_shape=*/{1, 1, 1},
         /*tile_sizes=*/{1},
         /*offsets=*/{"ts_0 * tid_0"},
         /*strides=*/{1},
         /*upper_bounds=*/{"1"},
         /*expected_error=*/""},
        {"ExpandShapeNonContiguous_SteppedSource",
         /*shape=*/{128},
         /*to_shape=*/{1, 2, 64},
         /*tile_sizes=*/{2},
         /*offsets=*/{"0"},
         /*strides=*/{64},
         /*upper_bounds=*/{"128"},
         /*expected_error=*/""},
    }),
    [](const ::testing::TestParamInfo<
        ReshapeExamplesTilePropagationTest::ParamType>& info) {
      return info.param.name;
    });

class ReshapeTilePropagationTest : public HloHardwareIndependentTestBase {};

TEST_F(ReshapeTilePropagationTest, UnsupportedReshapeErrorFormat) {
  mlir::MLIRContext mlir_context;
  Shape input_shape = ShapeUtil::MakeShape(F32, {12});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 4});

  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(output_shape, p0));

  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(p0),
                          &mlir_context));
  ASSERT_OK(tiling_space->AssignTileSizes({5}));
  SmallVector<DimTile> input_dim_tiles =
      llvm::to_vector(tiling_space->tiled_roots()[0].dim_tiles());
  input_dim_tiles[0].stride = CreateSymbolicConstant(1, &mlir_context);
  Tile input_tile = Tile(*tiling_space, std::move(input_dim_tiles));
  auto output_tiles =
      PropagateTileToOutput(*tiling_space, *reshape, input_tile, 0);

  ASSERT_FALSE(output_tiles.ok());
  EXPECT_THAT(output_tiles.status(), MatchString(R"(
  UNIMPLEMENTED: Reshape is non-contiguous [12] -> [3, 4], tiling
  offset [v0 * 5], size [5], stride [1], upper bound [12] ->
  offset [(v0 * 5) / 4], size [2], stride [1], upper bound [(v0 * 5) / 4 + 2];
  offset [(v0 * 5) mod 4], size [1], stride [1], upper bound [(v0 * 5) mod 4 + 1]:
  Multiple dimensions are partially tiled
  )"));
}

}  // namespace
}  // namespace xla::gpu::experimental

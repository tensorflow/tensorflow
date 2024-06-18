/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/symbolic_tile.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::testing::ElementsAre;
using ::testing::ExplainMatchResult;
using ::testing::Optional;

MATCHER_P(MatchSymbolicTileString, symbolic_tile_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(symbolic_tile_string, arg.ToString()),
      result_listener);
}

std::vector<int64_t> EvaluateMapAt(AffineMap affine_map,
                                   absl::Span<int64_t const> parameters) {
  CHECK_EQ(affine_map.getNumSymbols(), parameters.size());
  CHECK_EQ(affine_map.getNumDims(), 0);

  SmallVector<AffineExpr> symbol_replacements = llvm::to_vector(
      llvm::map_range(parameters, [affine_map](const int64_t v) -> AffineExpr {
        return mlir::getAffineConstantExpr(v, affine_map.getContext());
      }));

  AffineMap simplified_affine_map =
      mlir::simplifyAffineMap(affine_map.replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symbol_replacements, /*numResultDims=*/0,
          /*numResultSyms=*/0));

  SmallVector<int64_t> results = llvm::to_vector(llvm::map_range(
      simplified_affine_map.getResults(), [](AffineExpr result) -> int64_t {
        return llvm::cast<mlir::AffineConstantExpr>(result).getValue();
      }));

  return std::vector<int64_t>(results.begin(), results.end());
}

using SymbolicTileTest = IndexingTestBase;

TEST_F(SymbolicTileTest, CanPropagateTileFromDotOutputToInputs) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[11, 17, 19] parameter(0)
      p1 = f32[11, 19, 23] parameter(1)
      ROOT dot = f32[11, 17, 23] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
          offset_map: ()[s0, s1, s2] -> (0, 0, 0)
          size_map: ()[s0, s1, s2] -> (s0, s1, 19)
          stride_map: ()[s0, s1, s2] -> (1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughTrivialReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[11, 17, 19] parameter(0)
      ROOT reshape = f32[1, 11, 17, 19] reshape(p0)
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3] -> (0, 0, 0)
        size_map: ()[s0, s1, s2, s3] -> (s1, s2, s3)
        stride_map: ()[s0, s1, s2, s3] -> (1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughNonTrivialMergeReshapeFromOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT reshape = f32[48,4]{1,0} reshape(p0)
    }
  )"));

  // TODO(bchetioui): support expanding one dimension to more than two
  // dimensions and constrain accordingly.
  // TODO(b/334043867): add disjunctions in order to relax some of these
  // constraints. Currently we only support the reshaped tile size to be a
  // multiple of the smaller collapsed axes---we also need to support the case
  // where the tile size is a divisor of the collapsed axis.
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (0, 0, 0, 0)
        size_map: ()[s0, s1] -> (1, (s0 + 5) floordiv 6, s0 - ((s0 - 1) floordiv 6) * 6, s1)
        stride_map: ()[s0, s1] -> (0, 1, 1, 1)
        constraints:
          s0 mod 6 in [0, 0]
      )")));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughNonTrivialSplitReshapeFromOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[192,4]{1,0} parameter(0)
      ROOT reshape = f32[4,8,6,4]{3,2,1,0} reshape(p0)
    }
  )"));

  std::optional<SymbolicTile> symbolic_tile =
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin());

  // Collapsed dimensions force us to create nested conditionals, since the
  // stride of the output corresponds to the stride of the minormost expression
  // along which elements are captured in the composite expression. Hence, the
  // resulting expression is very ugly.
  EXPECT_THAT(symbolic_tile, Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3] -> (0, 0)
        size_map: ()[s0, s1, s2, s3] -> ((s0 * s1) * s2, s3)
        stride_map: ()[s0, s1, s2, s3] ->
          (((-s2 + 7) floordiv 6) * (((-s1 + 9) floordiv 8) *
          ((-((-s0 + 5) floordiv 4) + 1) * 48) +
          (-((-s1 + 9) floordiv 8) + 1) * 6) + -((-s2 + 7) floordiv 6) + 1, 1)
      )")));

  // Capturing elements along dimensions 0, 1, and 2 makes the stride equal to
  // 1.
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {4, 8, 6, 4}),
              ElementsAre(1, 1));
  // Capturing elements along dimension 2 makes the stride equal to 1.
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {1, 1, 6, 4}),
              ElementsAre(1, 1));
  // Capturing elements only along dimension 1 makes the stride equal to
  // the length of dimension 2 (6).
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {1, 8, 1, 4}),
              ElementsAre(6, 1));
  // Capturing elements only along dimension 0 makes the stride equal to the
  // product of the lengths of dimensions 1 and 2 (8 * 6).
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {2, 1, 1, 4}),
              ElementsAre(48, 1));
  // Capturing elements along dimension 0 and dimension 1 makes the stride
  // equal to the length of dimension 2 (6).
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {2, 8, 1, 4}),
              ElementsAre(6, 1));
  // Capturing a single element in the collapsed dimensions makes the stride 0.
  EXPECT_THAT(EvaluateMapAt(symbolic_tile->stride_map(), {1, 1, 1, 4}),
              ElementsAre(0, 1));
}

TEST_F(SymbolicTileTest, FailsToPropagateTileThroughNonTrivialReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[12, 4, 19] parameter(0)
      ROOT reshape = f32[4, 12, 19] reshape(p0)
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughElementwiseOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[150] parameter(0)
      p1 = f32[150] parameter(1)
      ROOT add = f32[150] add(p0, p1)
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0] -> (0)
        size_map: ()[s0] -> (s0)
        stride_map: ()[s0] -> (1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileFromBroadcastOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[150] parameter(0)
      ROOT broadcast = f32[157,150] broadcast(p0), dimensions={1}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (0)
        size_map: ()[s0, s1] -> (s1)
        stride_map: ()[s0, s1] -> (1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileFromReduceOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = f32[125,150] parameter(0)
      c0 = f32[] constant(-inf)
      ROOT reduce = f32[150] reduce(p0, c0), dimensions={0}, to_apply=max
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0] -> (0, 0)
        size_map: ()[s0] -> (125, s0)
        stride_map: ()[s0] -> (1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughReverse) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[179] parameter(0)
      ROOT reverse = f32[179] reverse(p0), dimensions={0}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0] -> (-s0 + 179)
        size_map: ()[s0] -> (s0)
        stride_map: ()[s0] -> (1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileFromSliceOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[120,142] parameter(0)
      ROOT slice = f32[10,21] slice(p0), slice={[40:60:2], [20:104:4]}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (40, 20)
        size_map: ()[s0, s1] -> (s0, s1)
        stride_map: ()[s0, s1] -> (2, 4)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughTranspose) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[21,10] parameter(0)
      ROOT transpose = f32[10,21] transpose(p0), dimensions={1,0}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (0, 0)
        size_map: ()[s0, s1] -> (s1, s0)
        stride_map: ()[s0, s1] -> (1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughConcatenate) {
  // TODO(b/325488844): Add additional concat test cases with constraints.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,5,7] parameter(0)
      p1 = f32[2,11,7] parameter(1)
      p2 = f32[2,17,7] parameter(2)
      ROOT concat = f32[2,33,7] concatenate(p0, p1, p2), dimensions={1}
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2] -> (0, 0, 0)
        size_map: ()[s0, s1, s2] -> (s0, s1, s2)
        stride_map: ()[s0, s1, s2] -> (1, 1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2] -> (0, -5, 0)
        size_map: ()[s0, s1, s2] -> (s0, s1, s2)
        stride_map: ()[s0, s1, s2] -> (1, 1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[2].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2] -> (0, -16, 0)
        size_map: ()[s0, s1, s2] -> (s0, s1, s2)
        stride_map: ()[s0, s1, s2] -> (1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughPadOpWithoutInteriorPadding) {
  // TODO(b/325488844): Add pad tests with defined constraints on tile input.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      input = f32[4, 4] parameter(0)
      padding_value = f32[] parameter(1)
      ROOT pad = f32[8,8] pad(input, padding_value), padding=2_2_0x1_3_0
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (-2, -1)
        size_map: ()[s0, s1] -> (s0, s1)
        stride_map: ()[s0, s1] -> (1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughDynamicSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
  )"));

  ASSERT_EQ(input_indexing.indexing_maps.size(), 4);

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      // s0, s1, s2: tile sizes
      // s3, s4: runtime parameters
      // Note: We don't have s0 in the size map's rhs, because the first dim
      // of the tile size can only be 1. The second offset is optimized to 0,
      // because that is the only possible value.
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3, s4] -> (s3, 0, s4)
        size_map: ()[s0, s1, s2] -> (1, s1, s2)
        stride_map: ()[s0, s1, s2] -> (0, 1, 1)
        rt_vars:
          s3 in [0, 1]
            hlo: %of1 = s32[] parameter(1)
            (d0, d1, d2) -> ()
          s4 in [0, 226]
            hlo: %of3 = s32[] parameter(3)
            (d0, d1, d2) -> ()
      )")));
  for (int i = 1; i <= 3; i++) {
    EXPECT_THAT(
        SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[i].begin()),
        Optional(MatchSymbolicTileString(R"(
        Symbolic tile with
          offset_map: ()[s0, s1, s2] -> ()
          size_map: ()[s0, s1, s2] -> ()
          stride_map: ()[s0, s1, s2] -> ()
        )")));
  }
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughDynamicUpdateSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[20,30] parameter(0)
      %upd = s32[5,10] parameter(1)
      %of1 = s32[] parameter(2)
      %of2 = s32[] parameter(3)
      ROOT %dus = s32[20,30] dynamic-update-slice(
          s32[20,30] %src, s32[5,10] %upd, s32[] %of1, s32[] %of2)
    }
  )"));

  ASSERT_EQ(input_indexing.indexing_maps.size(), 4);

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (0, 0)
        size_map: ()[s0, s1] -> (s0, s1)
        stride_map: ()[s0, s1] -> (1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      // s0, s1: tile sizes
      // s2, s3: runtime parameters
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3] -> (-s2, -s3)
        size_map: ()[s0, s1] -> (s0, s1)
        stride_map: ()[s0, s1] -> (1, 1)
        rt_vars:
          s2 in [0, 15]
            hlo: %of1 = s32[] parameter(2)
            (d0, d1) -> ()
          s3 in [0, 20]
            hlo: %of2 = s32[] parameter(3)
            (d0, d1) -> ()
      )")));
  for (int i = 2; i <= 3; i++) {
    EXPECT_THAT(
        SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[i].begin()),
        Optional(MatchSymbolicTileString(R"(
        Symbolic tile with
          offset_map: ()[s0, s1] -> ()
          size_map: ()[s0, s1] -> ()
          stride_map: ()[s0, s1] -> ()
        )")));
  }
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughGather) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY main {
      operand = f32[33,76,70] parameter(0)
      indices = s32[1806,2] parameter(1)
      ROOT r = f32[1806,7,8,4] gather(operand, indices), offset_dims={1,2,3},
                                 collapsed_slice_dims={}, start_index_map={0,1},
                                 index_vector_dim=1, slice_sizes={7,8,4}
    }
  )"));

  ASSERT_EQ(input_indexing.indexing_maps.size(), 2);

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      // s0, s1, s2, s3: tile sizes
      // s4, s5: runtime parameters
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3, s4, s5] -> (s4, s5, 0)
        size_map: ()[s0, s1, s2, s3] -> (s1, s2, s3)
        stride_map: ()[s0, s1, s2, s3] -> (1, 1, 1)
        rt_vars:
          s4 in [0, 26]
            hlo: %indices = s32[1806,2]{1,0} parameter(1)
            (d0, d1, d2, d3) -> (d0, 0)
          s5 in [0, 68]
            hlo: %indices = s32[1806,2]{1,0} parameter(1)
            (d0, d1, d2, d3) -> (d0, 1)
      )")));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2, s3] -> (0, 0)
        size_map: ()[s0, s1, s2, s3] -> (s0, 2)
        stride_map: ()[s0, s1, s2, s3] -> (1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughSplitReshapeOfReverse) {
  // A split reshape of a reverse creates a negative unit stride atop a
  // floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      reverse = f32[1,8,6,4]{3,2,1,0} reverse(p0), dimensions={1,2}
      ROOT reshape = f32[48,4]{1,0} reshape(reverse)
    }

    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[48,4]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  // TODO(b/331257678): the expected expressions should be simplified.
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] ->
          (0, -((s0 + 5) floordiv 6) + 8, -(s0 - ((s0 - 1) floordiv 6) * 6) + 6, 0)
        size_map: ()[s0, s1] ->
          (1, (s0 + 5) floordiv 6, s0 - ((s0 - 1) floordiv 6) * 6, s1)
        stride_map: ()[s0, s1] -> (0, 1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughSliceOfSplitReshape) {
  // TODO(b/326998704): constraints should allow us to unblock this use case.
  // A slice of a split reshape creates a non-unit stride atop a floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      reshape = f32[48,4]{1,0} reshape(p0)
      ROOT slice = f32[5,2]{1,0} slice(reshape), slice={[18:43:5], [0:4:2]}
    }

    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[5,2]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughMisalignedSliceOfSplitReshape) {
  // TODO(b/326998704): constraints should allow us to unblock part of this use
  // case.
  // TODO(b/331257678): handling correctly cases where offsets don't get
  // simplified away perfectly will allow us to unblock part of this use case.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      reshape = f32[48,4]{1,0} reshape(p0)
      ROOT slice = f32[5,2]{1,0} slice(reshape), slice={[20:45:5], [0:4:2]}
    }

    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[5,2]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughSliceOfSplitReshapeOnTranspose) {
  // TODO(b/326998704): constraints should allow us to unblock this use case.
  // A slice of a split reshape creates a non-unit stride atop a floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,6,8,4]{3,2,1,0} parameter(0)
      transpose = f32[1,8,6,4]{3,2,1,0} transpose(p0), dimensions={0,2,1,3}
      reshape = f32[48,4]{1,0} reshape(transpose)
      ROOT slice = f32[5,2]{1,0} slice(reshape), slice={[18:43:5], [0:4:2]}
    }

    ENTRY e {
      p0 = f32[1,6,8,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[5,2]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughSliceOfSplitReshapeOfReverse) {
  // TODO(b/326998704): constraints should allow us to unblock this use case.
  // A slice of a split reshape of a reverse creates a negative non-unit stride
  // atop a floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      reverse = f32[1,8,6,4]{3,2,1,0} reverse(p0), dimensions={1,2}
      reshape = f32[48,4]{1,0} reshape(reverse)
      ROOT slice = f32[5,2]{1,0} slice(reshape), slice={[18:43:5], [0:4:2]}
    }

    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[5,2]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughReductionOfConcatenation) {
  // TODO(b/330906085): concatenating across a reduction dimension needs to be
  // handled to unblock this.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    max_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT maximum = f32[] maximum(p0, p1)
    }

    computation {
      p0 = f32[10,8]{1,0} parameter(0)
      p1 = f32[20,8]{1,0} parameter(1)
      concatenate = f32[30,8]{1,0} concatenate(p0, p1), dimensions={0}
      neg_inf = f32[] constant(-inf)
      ROOT reduce = f32[8] reduce(concatenate, neg_inf), dimensions={0},
        to_apply=max_computation
    }

    ENTRY e {
      p0 = f32[10,8]{1,0} parameter(0)
      p1 = f32[20,8]{1,0} parameter(1)
      ROOT fusion = f32[8] fusion(p0, p1), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_EQ(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      std::nullopt);
}

TEST_F(SymbolicTileTest, CanCombineCompatibleConstraints) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,8,6,4,8]{4,3,2,1,0} parameter(0)
      ROOT reshape = f32[48,32]{1,0} reshape(p0)
    }
  )"));

  // TODO(b/334043867): add disjunctions in order to relax some of these
  // constraints. Currently we only support the reshaped axis to be a multiple
  // of the smaller collapsed axes.
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1] -> (0, 0, 0, 0, 0)
        size_map: ()[s0, s1] -> (1, (s0 + 5) floordiv 6, s0 - ((s0 - 1) floordiv 6) * 6, (s1 + 7) floordiv 8, s1 - ((s1 - 1) floordiv 8) * 8)
        stride_map: ()[s0, s1] -> (0, 1, 1, 1, 1)
        constraints:
          s0 mod 6 in [0, 0]
          s1 mod 8 in [0, 0]
      )")));
}

TEST_F(SymbolicTileTest,
       DerivesUnsatisfiableConstraintWhenMergingOfConstraintsIsUnsupported) {
  // This is kind of an artificial test case that we could easily support---we
  // assume here that we can't merge two constraints that are the same.
  // Nevertheless, there doesn't seem to be an obvious way to produce other
  // constraints that would trigger this particular failure at the moment. This
  // will change as we support more constraints, disjunctions, etc...
  IndexingMap indexing_map(
      ParseAffineMap("(d0) -> (d0 mod 6, d0 mod 6)", &mlir_context_),
      /*dimensions=*/{DimVar{0, 10}}, /*range_vars=*/{}, /*rt_vars=*/{});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(std::move(indexing_map)),
              Optional(MatchSymbolicTileString(R"(
              Symbolic tile with
              offset_map: ()[s0] -> (0, 0)
              size_map: ()[s0] -> (s0 - ((s0 - 1) floordiv 6) * 6, s0 - ((s0 - 1) floordiv 6) * 6)
              stride_map: ()[s0] -> (1, 1)
              constraints:
                unsatisfiable
              )")));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileWhenPreexistingConstraintsCanBeSimplifiedAway) {
  // The example is from
  // https://github.com/google/paxml/blob/91893818862645f5e9f23b84f530e611551745f6/paxml/contrib/gpu/scripts_gpu/configs.py#L107-L120.
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1, d2)[s0] -> (d0 * 2048 + d1, s0)",
                     &mlir_context_),
      {4, 2048, 50304}, {50304});
  // This constraint is redundant, because it can be derived from the domains of
  // the dimension variables.
  indexing_map.AddConstraint(ParseAffineExpr("d0 * 2048 + d1", &mlir_context_),
                             Interval{0, 8191});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: ()[s0, s1, s2] -> (0, 0)
        size_map: ()[s0, s1, s2] -> (s0 * s1, 50304)
        stride_map: ()[s0, s1, s2] -> (((-s1 + 2049) floordiv 2048) * ((-((-s0 + 5) floordiv 4) + 1) * 2048) + -((-s1 + 2049) floordiv 2048) + 1, 1)
      )")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

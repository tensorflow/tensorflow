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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::testing::ElementsAre;
using ::testing::ExplainMatchResult;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::SizeIs;

using Constraint = ConstraintExpression::Constraint;

MATCHER_P(MatchSymbolicTileString, symbolic_tile_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(symbolic_tile_string, arg.ToString()),
      result_listener);
}

MATCHER_P(MatchConstraintExpressionString, constraint_expression_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(constraint_expression_string, arg.ToString()),
      result_listener);
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
          offset_map: (d0, d1, d2) -> (0, 0, 0)
          size_map: (d0, d1, d2) -> (d0, d1, 19)
          stride_map: (d0, d1, d2) -> (1, 1, 1)
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
        offset_map: (d0, d1, d2, d3) -> (0, 0, 0)
        size_map: (d0, d1, d2, d3) -> (d1, d2, d3)
        stride_map: (d0, d1, d2, d3) -> (1, 1, 1)
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
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1) -> (0, 0, 0, 0)
        size_map: (d0, d1) -> (1, (d0 + 5) floordiv 6, d0 - ((d0 - 1) floordiv 6) * 6, d1)
        stride_map: (d0, d1) -> (0, 1, 1, 1)
        constraints:
          6 mod d0 in [0, 0] || d0 mod 6 in [0, 0]
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
        offset_map: (d0, d1, d2, d3) -> (0, 0)
        size_map: (d0, d1, d2, d3) -> ((d0 * d1) * d2, d3)
        stride_map: (d0, d1, d2, d3) ->
          (((-d2 + 7) floordiv 6) * (((-d1 + 9) floordiv 8) *
          ((-((-d0 + 5) floordiv 4) + 1) * 48) +
          (-((-d1 + 9) floordiv 8) + 1) * 6) + -((-d2 + 7) floordiv 6) + 1, 1)
        constraints: d0 in [1, 1] && d1 in [1, 1] ||
                     d0 in [1, 1] && d2 in [1, 1] ||
                     d0 in [1, 1] && d2 in [6, 6] ||
                     d1 in [1, 1] && d2 in [1, 1] ||
                     d1 in [8, 8] && d2 in [1, 1] ||
                     d1 in [8, 8] && d2 in [6, 6]
      )")));

  // Capturing elements along dimensions 0, 1, and 2 makes the stride equal to
  // 1.
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {4, 8, 6, 4}),
              ElementsAre(1, 1));
  // Capturing elements along dimension 2 makes the stride equal to 1.
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {1, 1, 6, 4}),
              ElementsAre(1, 1));
  // Capturing elements only along dimension 1 makes the stride equal to
  // the length of dimension 2 (6).
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {1, 8, 1, 4}),
              ElementsAre(6, 1));
  // Capturing elements only along dimension 0 makes the stride equal to the
  // product of the lengths of dimensions 1 and 2 (8 * 6).
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {2, 1, 1, 4}),
              ElementsAre(48, 1));
  // Capturing elements along dimension 0 and dimension 1 makes the stride
  // equal to the length of dimension 2 (6).
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {2, 8, 1, 4}),
              ElementsAre(6, 1));
  // Capturing a single element in the collapsed dimensions makes the stride 0.
  EXPECT_THAT(EvaluateAffineMap(symbolic_tile->stride_map(), {1, 1, 1, 4}),
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
        offset_map: (d0) -> (0)
        size_map: (d0) -> (d0)
        stride_map: (d0) -> (1)
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
        offset_map: (d0, d1) -> (0)
        size_map: (d0, d1) -> (d1)
        stride_map: (d0, d1) -> (1)
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
        offset_map: (d0) -> (0, 0)
        size_map: (d0) -> (125, d0)
        stride_map: (d0) -> (1, 1)
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
        offset_map: (d0) -> (-d0 + 179)
        size_map: (d0) -> (d0)
        stride_map: (d0) -> (1)
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
        offset_map: (d0, d1) -> (40, 20)
        size_map: (d0, d1) -> (d0, d1)
        stride_map: (d0, d1) -> (2, 4)
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
        offset_map: (d0, d1) -> (0, 0)
        size_map: (d0, d1) -> (d1, d0)
        stride_map: (d0, d1) -> (1, 1)
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
        offset_map: (d0, d1, d2) -> (0, 0, 0)
        size_map: (d0, d1, d2) -> (d0, d1, d2)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2) -> (0, -5, 0)
        size_map: (d0, d1, d2) -> (d0, d1, d2)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[2].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2) -> (0, -16, 0)
        size_map: (d0, d1, d2) -> (d0, d1, d2)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
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
        offset_map: (d0, d1) -> (-2, -1)
        size_map: (d0, d1) -> (d0, d1)
        stride_map: (d0, d1) -> (1, 1)
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
      // d0, d1, d2: tile sizes
      // s0, s1: runtime parameters
      // Note: We don't have d0 in the size map's rhs, because the first dim
      // of the tile size can only be 1. The second offset is optimized to 0,
      // because that is the only possible value.
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2)[s0, s1, s2] -> (s0, s1, s2)
        size_map: (d0, d1, d2) -> (d0, d1, d2)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
        rt_vars:
          s0 in [0, 1],
          s1 in [0, 0],
          s2 in [0, 226],
      )")));
  for (int i = 1; i <= 3; i++) {
    EXPECT_THAT(
        SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[i].begin()),
        Optional(MatchSymbolicTileString(R"(
        Symbolic tile with
          offset_map: (d0, d1, d2) -> ()
          size_map: (d0, d1, d2) -> ()
          stride_map: (d0, d1, d2) -> ()
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
        offset_map: (d0, d1) -> (0, 0)
        size_map: (d0, d1) -> (d0, d1)
        stride_map: (d0, d1) -> (1, 1)
      )")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      // d0, d1: tile sizes
      // s0, s1: runtime parameters
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1)[s0, s1] -> (-s0, -s1)
        size_map: (d0, d1) -> (d0, d1)
        stride_map: (d0, d1) -> (1, 1)
        rt_vars:
          s0 in [0, 15],
          s1 in [0, 20],
      )")));
  for (int i = 2; i <= 3; i++) {
    EXPECT_THAT(
        SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[i].begin()),
        Optional(MatchSymbolicTileString(R"(
        Symbolic tile with
          offset_map: (d0, d1) -> ()
          size_map: (d0, d1) -> ()
          stride_map: (d0, d1) -> ()
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
      // d0, d1, d2, d3: tile sizes
      // s0, s1: runtime parameters
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2, d3)[s0, s1] -> (s0, s1, 0)
        size_map: (d0, d1, d2, d3) -> (d1, d2, d3)
        stride_map: (d0, d1, d2, d3) -> (1, 1, 1)
        rt_vars:
          s0 in [0, 26],
          s1 in [0, 68],
      )")));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2, d3) -> (0, 0)
        size_map: (d0, d1, d2, d3) -> (d0, 2)
        stride_map: (d0, d1, d2, d3) -> (1, 1)
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
        offset_map: (d0, d1) ->
          (0, -((d0 + 5) floordiv 6) + 8, -(d0 - ((d0 - 1) floordiv 6) * 6) + 6, 0)
        size_map: (d0, d1) ->
          (1, (d0 + 5) floordiv 6, d0 - ((d0 - 1) floordiv 6) * 6, d1)
        stride_map: (d0, d1) -> (0, 1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughSplitReductionOfSplittedAxis) {
  // A split reshape of a reverse creates a sum of strided symbols.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    computation {
      p0 = f32[18] parameter(0)
      bitcast = f32[9,2] bitcast(p0)
      c0 = f32[] constant(0)
      reduce_0 = f32[9] reduce(bitcast, c0), dimensions={1}, to_apply=add
      ROOT reduce_1 = f32[] reduce(reduce_0, c0), dimensions={0}, to_apply=add
    }

    ENTRY e {
      p0 = f32[18] parameter(0)
      ROOT fusion = f32[] fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: () -> (0)
        size_map: () -> (18)
        stride_map: () -> (1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughSummationOfSymbols) {
  // Such an indexing map is representative of a sequence of HLOs containing a
  // bitcast followed by two sequential reductions of the split axis, i.e.
  // something like
  //   p0 = f32[18] parameter(0)
  //   bitcast = f32[9,2] bitcast(p0)
  //   reduce_0 = f32[9] reduce(bitcast), dimensions={1}
  //   reduce_1 = f32[] reduce(reduce_0), dimensions={0}
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("()[s0, s1] -> (s1 * 2 + s0)", &mlir_context_), {},
      {2, 9});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: () -> (0)
        size_map: () -> (18)
        stride_map: () -> (1)
      )")));
}

TEST_F(SymbolicTileTest, CanPropagateTileModAndFloorDiv) {
  // Such an indexing map is representative of HLOs with bitcasts collapsing
  // more than two axes, i.e. something like
  //   p0 = f32[3,5,7]{2,1,0} parameter(0)
  //   bitcast = f32[105]{0} bitcast(p0)
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(
          "(d0) -> (d0 floordiv 35, (d0 floordiv 7) mod 5, d0 mod 7)",
          &mlir_context_),
      {105}, {});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0) -> (0, 0, 0)
        size_map: (d0) -> ((d0 + 34) floordiv 35, (d0 + 6) floordiv 7 - (((d0 + 6) floordiv 7 - 1) floordiv 5) * 5, d0 - ((d0 - 1) floordiv 7) * 7)
        stride_map: (d0) -> (1, 1, 1)
        constraints: ((d0 + 6) floordiv 7) mod 5 in [0, 0] && 7 mod d0 in [0, 0] || ((d0 + 6) floordiv 7) mod 5 in [0, 0] && d0 mod 7 in [0, 0] || 5 mod ((d0 + 6) floordiv 7) in [0, 0] && 7 mod d0 in [0, 0] || 5 mod ((d0 + 6) floordiv 7) in [0, 0] && d0 mod 7 in [0, 0]
      )")));
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughSliceOfSplitReshape) {
  // TODO(b/349487906): constraints should allow us to unblock this use case.
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
  // TODO(b/349487906): constraints should allow us to unblock this use case.
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
  // TODO(b/349487906): constraints should allow us to unblock this use case.
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

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1) -> (0, 0, 0, 0, 0)
        size_map: (d0, d1) -> (1, (d0 + 5) floordiv 6, d0 - ((d0 - 1) floordiv 6) * 6, (d1 + 7) floordiv 8, d1 - ((d1 - 1) floordiv 8) * 8)
        stride_map: (d0, d1) -> (0, 1, 1, 1, 1)
        constraints:
          6 mod d0 in [0, 0] && 8 mod d1 in [0, 0] ||
          6 mod d0 in [0, 0] && d1 mod 8 in [0, 0] ||
          8 mod d1 in [0, 0] && d0 mod 6 in [0, 0] ||
          d0 mod 6 in [0, 0] && d1 mod 8 in [0, 0]
      )")));
}

TEST_F(SymbolicTileTest,
       CanDeriveTileWhenPreexistingConstraintsCanBeSimplifiedAway) {
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
        offset_map: (d0, d1, d2) -> (0, 0)
        size_map: (d0, d1, d2) -> (d0 * d1, 50304)
        stride_map: (d0, d1, d2) -> (((-d1 + 2049) floordiv 2048) * ((-((-d0 + 5) floordiv 4) + 1) * 2048) + -((-d1 + 2049) floordiv 2048) + 1, 1)
        constraints: d0 in [1, 1] || d1 in [1, 1] || d1 in [2048, 2048]
      )")));
}

TEST_F(SymbolicTileTest, CanDeriveTileWhenTheIndexingMapHasSymbolsInASum) {
  // The example is from
  // https://github.com/google/paxml/blob/91893818862645f5e9f23b84f530e611551745f6/paxml/contrib/gpu/scripts_gpu/configs.py#L107-L120.
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1, d2)[s0] -> (d0, d1, d2 * 128 + s0)",
                     &mlir_context_),
      {4, 2048, 393}, {128});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2) -> (0, 0, 0)
        size_map: (d0, d1, d2) -> (d0, d1, d2 * 128)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest, ResultingConstraintsAreSimplifiedAway) {
  // The example is from
  // https://github.com/google/paxml/blob/91893818862645f5e9f23b84f530e611551745f6/paxml/contrib/gpu/scripts_gpu/configs.py#L107-L120.
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1, d2)[s0] -> (d0, d1, d2 * 128 + s0)",
                     &mlir_context_),
      {4, 2048, 393}, {128});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0, d1, d2) -> (0, 0, 0)
        size_map: (d0, d1, d2) -> (d0, d1, d2 * 128)
        stride_map: (d0, d1, d2) -> (1, 1, 1)
      )")));
}

TEST_F(SymbolicTileTest, PointDimensionsAreNotSimplified) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), /*dim_upper_bounds=*/{1},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(SymbolicTile::FromIndexingMap(indexing_map),
              Optional(MatchSymbolicTileString(R"(
      Symbolic tile with
        offset_map: (d0) -> (0)
        size_map: (d0) -> (d0)
        stride_map: (d0) -> (1)
      )")));
}

class ConstraintExpressionTest : public IndexingTestBase {
 public:
  ConstraintExpression::Constraint GetConstraint(const std::string& string_expr,
                                                 int64_t lower, int64_t upper) {
    return {ParseAffineExpr(string_expr, &mlir_context_),
            Interval{lower, upper}};
  }
  ConstraintExpression Simplify(ConstraintExpression constraints) {
    constraints.Simplify();
    return constraints;
  }
};

TEST_F(ConstraintExpressionTest, CheckAlwaysSatisfied) {
  auto always_satisfied = ConstraintExpression::GetAlwaysSatisfied();
  EXPECT_TRUE(always_satisfied.is_satisfiable());
  EXPECT_TRUE(always_satisfied.IsAlwaysSatisfied());
  EXPECT_THAT(always_satisfied,
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_TRUE(always_satisfied.IsSatisfiedBy({}));
  EXPECT_THAT(always_satisfied || GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(always_satisfied && GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("d0 in [0, 0]"));
  EXPECT_THAT(always_satisfied || ConstraintExpression::GetUnsatisfiable(),
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(ConstraintExpression::GetUnsatisfiable() || always_satisfied,
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(Simplify(always_satisfied),
              MatchConstraintExpressionString("always satisfied"));
}

TEST_F(ConstraintExpressionTest, CheckUnsatisfiable) {
  auto unsatisfiable = ConstraintExpression::GetUnsatisfiable();
  EXPECT_FALSE(unsatisfiable.is_satisfiable());
  EXPECT_FALSE(unsatisfiable.IsAlwaysSatisfied());
  EXPECT_THAT(unsatisfiable, MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_FALSE(unsatisfiable.IsSatisfiedBy({}));
  EXPECT_THAT(unsatisfiable || GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("d0 in [0, 0]"));
  EXPECT_THAT(unsatisfiable && GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(unsatisfiable && ConstraintExpression::GetAlwaysSatisfied(),
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(ConstraintExpression::GetAlwaysSatisfied() && unsatisfiable,
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(Simplify(unsatisfiable),
              MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(ConstraintExpressionTest, PrettyPrintingTest) {
  ConstraintExpression constraints =
      GetConstraint("d2", 5, 6) ||
      (GetConstraint("d1", 3, 4) && GetConstraint("d0", 1, 2));
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [1, 2] && d1 in [3, 4] || d2 in [5, 6]"));
}

TEST_F(ConstraintExpressionTest,
       ConjunctionOfConstraintsOnTheSameExpressionAreIntersected) {
  ConstraintExpression constraints{GetConstraint("d0", 0, 5)};
  EXPECT_THAT(constraints, MatchConstraintExpressionString("d0 in [0, 5]"));

  // Constraints are intersected.
  constraints = constraints && GetConstraint("d0", 3, 6);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("d0 in [3, 5]"));

  // Empty intersection results in unsatisfiability.
  constraints = constraints && GetConstraint("d0", 7, 8);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformConjunctionOfConstraintExpressionWithConjointConstraints) {  // NOLINT(whitespace/line_length)
  ConstraintExpression constraints = GetConstraint("d0", 0, 5) &&
                                     GetConstraint("d1", 0, 5) &&
                                     GetConstraint("d2", 0, 5);
  // Constraints can be merged without trouble, and hence the constraint
  // expression is satisfiable.
  EXPECT_TRUE(constraints.is_satisfiable());
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [0, 5] && d1 in [0, 5] && d2 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CorrectlyEliminatesConjunctionFromDisjunctionWhenItBecomesUnsatisfiable) {
  ConstraintExpression constraints =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);
  EXPECT_THAT(constraints,
              MatchConstraintExpressionString("d0 in [0, 5] || d1 in [0, 5]"));

  // `conjunction_1` && `conjunction_3` is an unsatisfiable constraint. Taking
  // the conjunction of the existing constraint expression with `conjunction_3`
  // should therefore evict the unsatisfiable intersection of `conjunction_1`
  // and `conjunction_3` from the disjoint expression.
  constraints = constraints && GetConstraint("d0", 6, 6);

  EXPECT_THAT(constraints,
              MatchConstraintExpressionString("d0 in [6, 6] && d1 in [0, 5]"));

  // But becomes unsatisfiable if we eliminate the last remaining constraint by
  // constructing another unsatisfiable conjunction.
  constraints = constraints && GetConstraint("d0", 7, 7);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformDisjunctionOfConstraintExpressionWithConjointConstraints) {  // NOLINT(whitespace/line_length)
  ConstraintExpression constraints =
      GetConstraint("d0", 0, 5) && GetConstraint("d1", 0, 5);
  constraints = constraints || GetConstraint("d2", 0, 5);
  EXPECT_TRUE(constraints.is_satisfiable());
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [0, 5] && d1 in [0, 5] || d2 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformConjunctionOfConstraintExpressionWithConstraintExpression) {  // NOLINT(whitespace/line_length)
  // Construct the first `ConstraintExpression` to be of the form
  //   a || b.
  ConstraintExpression constraints_1 =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);

  // Construct the second `ConstraintExpression` to be of the form
  //   c || d || e.
  ConstraintExpression constraints_2 = GetConstraint("d2", 0, 5) ||
                                       GetConstraint("d3", 0, 5) ||
                                       GetConstraint("d4", 0, 5);

  // Taking the conjunction of the two `ConstraintExpression`s should result in
  // a `ConstraintExpression` of the form
  //   a && c || a && d || a && e || b && c || b && d || b && e.
  ConstraintExpression result_constraint_expression =
      constraints_1 && constraints_2;

  EXPECT_TRUE(result_constraint_expression.is_satisfiable());
  // There are now six conjunctions in the disjoint expression, as described
  // above.
  EXPECT_THAT(
      result_constraint_expression,
      MatchConstraintExpressionString(
          "d0 in [0, 5] && d2 in [0, 5] || d0 in [0, 5] && d3 in [0, 5] || "
          "d0 in [0, 5] && d4 in [0, 5] || d1 in [0, 5] && d2 in [0, 5] || "
          "d1 in [0, 5] && d3 in [0, 5] || d1 in [0, 5] && d4 in [0, 5]"));

  // Lastly, make sure that the conjunction of an empty `ConstraintExpression`
  // with a non-empty one results in passing the non-empty one through, on both
  // sides.
  ConstraintExpression always_satisfied =
      ConstraintExpression::GetAlwaysSatisfied();
  EXPECT_THAT(always_satisfied && constraints_2,
              MatchConstraintExpressionString(
                  "d2 in [0, 5] || d3 in [0, 5] || d4 in [0, 5]"));
  EXPECT_THAT(constraints_2 && always_satisfied,
              MatchConstraintExpressionString(
                  "d2 in [0, 5] || d3 in [0, 5] || d4 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformDisjunctionOfConstraintExpressionWithConstraintExpression) {  // NOLINT(whitespace/line_length)
  // Construct the first `ConstraintExpression` to be of the form
  //   a || b.
  ConstraintExpression constraints_1 =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);

  // Construct the second `ConstraintExpression` to be of the form
  //   c || d || e.
  ConstraintExpression constraints_2 = GetConstraint("d2", 0, 5) ||
                                       GetConstraint("d3", 0, 5) ||
                                       GetConstraint("d4", 0, 5);

  // Taking the disjunction of the two `ConstraintExpression`s should result in
  // a `ConstraintExpression` of the form
  //   a || b || c || d || e.
  ConstraintExpression result_constraint_expression =
      constraints_1 || constraints_2;

  EXPECT_TRUE(result_constraint_expression.is_satisfiable());
  // There are now five conjunctions in the disjoint expression, as described
  // above.
  EXPECT_THAT(result_constraint_expression,
              MatchConstraintExpressionString(
                  "d0 in [0, 5] || d1 in [0, 5] || d2 in [0, 5] || "
                  "d3 in [0, 5] || d4 in [0, 5]"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAlwaysSatisfiedContraintExpression) {
  ConstraintExpression constraints = GetConstraint("d0", 0, 1) ||
                                     GetConstraint("25", 0, 100) ||
                                     GetConstraint("1", 1, 1);
  constraints.Simplify();
  EXPECT_THAT(constraints, MatchConstraintExpressionString("always satisfied"));
}

TEST_F(ConstraintExpressionTest, CanSimplifyUnsatisfiableContraintExpression) {
  ConstraintExpression constraints =
      GetConstraint("d0", 0, -1) || GetConstraint("1", 2, 3);
  constraints.Simplify();
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAwayAlwaysSatisfiedPartOfConjunction) {
  EXPECT_THAT(Simplify(GetConstraint("d0", 0, 1) && GetConstraint("1", 1, 1) &&
                       GetConstraint("d1", 0, 1) && GetConstraint("2", 2, 3)),
              MatchConstraintExpressionString("d0 in [0, 1] && d1 in [0, 1]"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAwayUnsatisfiablePartOfDisjunction) {
  EXPECT_THAT(Simplify(GetConstraint("d0", 0, 1) ||
                       (GetConstraint("d1", 0, 1) && GetConstraint("1", 0, 0) &&
                        GetConstraint("d2", 0, 1))),
              MatchConstraintExpressionString("d0 in [0, 1]"));
}

TEST_F(ConstraintExpressionTest, SimplifyRemovesRedundantConstraints) {
  Constraint c0 = GetConstraint("d0", 0, 0);
  Constraint c1 = GetConstraint("d1", 1, 1);
  // We could simplify those contraints even further to `d0 in [0, 0]` by
  // checking that one conjunction is a subset of the other, but we don't do
  // that yet.
  EXPECT_THAT(
      Simplify((c0 && c1) || (c1 && c0) || c0 || (c1 && c0) || (c0 && c1)),
      MatchConstraintExpressionString(
          "d0 in [0, 0] || d0 in [0, 0] && d1 in [1, 1]"));
}

TEST_F(ConstraintExpressionTest, ConstraintSatisfactionIsEvaluatedCorrectly) {
  Constraint c0 = GetConstraint("d0 mod 6", 0, 0);
  Constraint c1 = GetConstraint("d1 mod 8", 0, 0);
  Constraint c2 = GetConstraint("d0 mod 13", 0, 0);
  ConstraintExpression constraints = (c0 && c1) || (c1 && c2);

  // Parameters {6, 8} satisfy these constraints.
  std::vector<int64_t> possible_tile_parameters({6, 8});
  EXPECT_TRUE(constraints.IsSatisfiedBy(possible_tile_parameters));

  // Parameters {13, 8} should also satisfy these constraints.
  std::vector<int64_t> other_possible_tile_parameters({13, 8});
  EXPECT_TRUE(constraints.IsSatisfiedBy(other_possible_tile_parameters));

  // However, tile sizes {6, 7} do not satisfy these constraints.
  std::vector<int64_t> impossible_tile_parameters({6, 7});
  EXPECT_FALSE(constraints.IsSatisfiedBy(impossible_tile_parameters));

  // Anything satisfies an always satisfied constraint expression.
  EXPECT_TRUE(ConstraintExpression::GetAlwaysSatisfied().IsSatisfiedBy(
      impossible_tile_parameters));

  // Nothing satisfies an unsatisfiable constraint expression.
  EXPECT_FALSE(ConstraintExpression::GetUnsatisfiable().IsSatisfiedBy(
      possible_tile_parameters));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

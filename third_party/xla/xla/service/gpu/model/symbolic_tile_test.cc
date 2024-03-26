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

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ExplainMatchResult;
using ::testing::Optional;
using ::testing::StrEq;

MATCHER_P3(MatchSymbolicTile, offset_map_string, size_map_string,
           stride_map_string,
           absl::StrCat(negation
                            ? "equals "
                            : "doesn't equal symbolic tile with offset_map_ ",
                        offset_map_string, " and size_map_ ", size_map_string,
                        " and stride_map_ ", stride_map_string)) {
  AffineMapPrinter printer;
  return ExplainMatchResult(StrEq(offset_map_string),
                            printer.ToString(arg.offset_map()),
                            result_listener) &&
         ExplainMatchResult(StrEq(size_map_string),
                            printer.ToString(arg.size_map()),
                            result_listener) &&
         ExplainMatchResult(StrEq(stride_map_string),
                            printer.ToString(arg.stride_map()),
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
      Optional(MatchSymbolicTile("()[s0, s1, s2] -> (0, 0, 0)",
                                 "()[s0, s1, s2] -> (s0, s1, 19)",
                                 "()[s0, s1, s2] -> (1, 1, 1)")));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTile("()[s0, s1, s2] -> (0, 0, 0)",
                                 "()[s0, s1, s2] -> (s0, 19, s2)",
                                 "()[s0, s1, s2] -> (1, 1, 1)")));
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
      Optional(MatchSymbolicTile("()[s0, s1, s2, s3] -> (0, 0, 0)",
                                 "()[s0, s1, s2, s3] -> (s1, s2, s3)",
                                 "()[s0, s1, s2, s3] -> (1, 1, 1)")));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughNonTrivialMergeReshapeFromOutputToInput) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT bitcast = f32[48,4]{1,0} bitcast(p0)
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTile(
          "()[s0, s1] -> (0, 0, 0, 0)",
          "()[s0, s1] -> "
          "(1, (s0 + 5) floordiv 6, s0 - ((s0 - 1) floordiv 6) * 6, s1)",
          "()[s0, s1] -> (0, 1, 1, 1)")));
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
      Optional(MatchSymbolicTile("()[s0] -> (0)", "()[s0] -> (s0)",
                                 "()[s0] -> (1)")));
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
      Optional(MatchSymbolicTile("()[s0, s1] -> (0)", "()[s0, s1] -> (s1)",
                                 "()[s0, s1] -> (1)")));
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
      Optional(MatchSymbolicTile("()[s0] -> (0, 0)", "()[s0] -> (125, s0)",
                                 "()[s0] -> (1, 1)")));
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
      Optional(MatchSymbolicTile("()[s0] -> (-s0 + 179)", "()[s0] -> (s0)",
                                 "()[s0] -> (1)")));
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
      Optional(MatchSymbolicTile("()[s0, s1] -> (40, 20)",
                                 "()[s0, s1] -> (s0, s1)",
                                 "()[s0, s1] -> (2, 4)")));
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
      Optional(MatchSymbolicTile("()[s0, s1] -> (0, 0)",
                                 "()[s0, s1] -> (s1, s0)",
                                 "()[s0, s1] -> (1, 1)")));
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
      Optional(MatchSymbolicTile("()[s0, s1, s2] -> (0, 0, 0)",
                                 "()[s0, s1, s2] -> (s0, s1, s2)",
                                 "()[s0, s1, s2] -> (1, 1, 1)")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTile("()[s0, s1, s2] -> (0, -5, 0)",
                                 "()[s0, s1, s2] -> (s0, s1, s2)",
                                 "()[s0, s1, s2] -> (1, 1, 1)")));
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[2].begin()),
      Optional(MatchSymbolicTile("()[s0, s1, s2] -> (0, -16, 0)",
                                 "()[s0, s1, s2] -> (s0, s1, s2)",
                                 "()[s0, s1, s2] -> (1, 1, 1)")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughPadOpWithoutInteriorPadding) {
  // TODO(b/325488844): Add pad tests with defined constraints on tile input.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[8,8] pad(p0, p1), padding=2_2_0x1_3_0
    }
  )"));

  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTile("()[s0, s1] -> (-2, -1)",
                                 "()[s0, s1] -> (s0, s1)",
                                 "()[s0, s1] -> (1, 1)")));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughSplitReshapeOfReverse) {
  // A split reshape of a reverse creates a negative unit stride atop a
  // floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      reverse = f32[1,8,6,4]{3,2,1,0} reverse(p0), dimensions={1,2}
      ROOT bitcast = f32[48,4]{1,0} bitcast(reverse)
    }

    ENTRY e {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      ROOT fusion = f32[48,4]{1,0} fusion(p0), kind=kLoop, calls=computation
    }
  )"));

  // TODO(b/331257678): the expected expressions should be simplified.
  EXPECT_THAT(
      SymbolicTile::FromIndexingMap(*input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTile(
          "()[s0, s1] -> (0, -((s0 + 5) floordiv 6) + 8, "
          "-(s0 - ((s0 - 1) floordiv 6) * 6) + 6, 0)",
          "()[s0, s1] -> "
          "(1, (s0 + 5) floordiv 6, s0 - ((s0 - 1) floordiv 6) * 6, s1)",
          "()[s0, s1] -> (0, 1, 1, 1)")));
}

TEST_F(SymbolicTileTest,
       FailsGracefullyAtPropagatingTileThroughSliceOfSplitReshape) {
  // TODO(b/326998704): constraints should allow us to unblock this use case.
  // A slice of a split reshape creates a non-unit stride atop a floordiv.
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    computation {
      p0 = f32[1,8,6,4]{3,2,1,0} parameter(0)
      bitcast = f32[48,4]{1,0} bitcast(p0)
      ROOT slice = f32[5,2]{1,0} slice(bitcast), slice={[18:43:5], [0:4:2]}
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
      bitcast = f32[48,4]{1,0} bitcast(p0)
      ROOT slice = f32[5,2]{1,0} slice(bitcast), slice={[20:45:5], [0:4:2]}
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
      bitcast = f32[48,4]{1,0} bitcast(transpose)
      ROOT slice = f32[5,2]{1,0} slice(bitcast), slice={[18:43:5], [0:4:2]}
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
      bitcast = f32[48,4]{1,0} bitcast(reverse)
      ROOT slice = f32[5,2]{1,0} slice(bitcast), slice={[18:43:5], [0:4:2]}
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

}  // namespace
}  // namespace gpu
}  // namespace xla

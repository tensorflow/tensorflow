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

#include "xla/service/gpu/model/indexing_map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class IndexingMapTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter printer_;
};

TEST_F(IndexingMapTest, ComposeWithPermutation) {
  IndexingMap producer{
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1, s0)", &mlir_context_),
      Domain::FromUpperBounds({4, 4}, {2, 2})};

  IndexingMap consumer{ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
                       Domain::FromUpperBounds({4}, {4})};

  auto composed = ComposeIndexingMaps(producer, consumer);
  EXPECT_THAT(
      composed,
      MatchIndexingMap(
          "(d0)[s0, s1, s2] -> (s2, d0, s1, s0)", ElementsAre(MatchRange(0, 3)),
          ElementsAre(MatchRange(0, 1), MatchRange(0, 1), MatchRange(0, 3))));
}

TEST_F(IndexingMapTest, SimplifyConstantDims) {
  IndexingMap indexing_map{ParseAffineMap("(d0) -> (d0)", &mlir_context_),
                           Domain{{Range{5, 5}}, {}}};
  indexing_map.Simplify();
  EXPECT_THAT(printer_.ToString(indexing_map.affine_map),
              HasSubstr("(d0) -> (5)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsIfSmallerThanDivisor) {
  auto serialized_map = "(d0, d1) -> (d0 + d1 floordiv 16, d1 mod 16)";
  IndexingMap indexing_map{ParseAffineMap(serialized_map, &mlir_context_),
                           Domain::FromUpperBounds({8, 16}, {})};
  indexing_map.Simplify();

  EXPECT_THAT(printer_.ToString(indexing_map.affine_map),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithMultipliers) {
  auto serialized_map =
      "(d0, d1, d2) -> ((d0 * 100 + d1 * 10 + d2) floordiv 100, "
      "((d0 * 100 + d1 * 10 + d2) mod 100) floordiv 10, "
      "d2 mod 10)";

  IndexingMap indexing_map{ParseAffineMap(serialized_map, &mlir_context_),
                           Domain::FromUpperBounds({9, 9, 9}, {})};
  indexing_map.Simplify();

  EXPECT_THAT(printer_.ToString(indexing_map.affine_map),
              HasSubstr("(d0, d1, d2) -> (d0, d1, d2)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithDivisibleMultipliers) {
  auto serialized_map =
      "(d0, d1, d2) -> ((d0 * 16 + d1 * 4 + d2) floordiv 8, "
      "(d0 * 16 + d1 * 4 + d2) mod 8)";

  IndexingMap indexing_map{ParseAffineMap(serialized_map, &mlir_context_),
                           Domain::FromUpperBounds({10, 10, 10}, {})};
  indexing_map.Simplify();

  EXPECT_THAT(printer_.ToString(indexing_map.affine_map),
              HasSubstr("(d0, d1, d2) -> (d0 * 2 + (d1 * 4 + d2) floordiv 8, "
                        "(d1 * 4 + d2) mod 8)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithReverse) {
  auto serialized_map =
      "(d0, d1) -> (-((d0 * -11 - d1 + 109) floordiv 11) + 9, "
      "d0 * 11 + d1 + ((d0 * -11 - d1 + 109) floordiv 11) * 11 - 99)";
  IndexingMap indexing_map{ParseAffineMap(serialized_map, &mlir_context_),
                           Domain::FromUpperBounds({8, 9}, {})};
  indexing_map.Simplify();

  EXPECT_THAT(printer_.ToString(indexing_map.affine_map),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

TEST_F(IndexingMapTest, RangeEvaluatorTest) {
  Domain domain({Range{0, 9}, Range{-10, -1}, Range{-1, 2}, Range{0, 0}}, {});
  RangeEvaluator range_evaluator(&domain);
  mlir::AffineExpr d0, d1, d2, d3;
  bindDims(&mlir_context_, d0, d1, d2, d3);

  // d0 is always positive.
  EXPECT_TRUE(range_evaluator.IsAlwaysPositiveOrZero(d0));
  EXPECT_FALSE(range_evaluator.IsAlwaysNegativeOrZero(d0));

  // d1 is always negative.
  EXPECT_FALSE(range_evaluator.IsAlwaysPositiveOrZero(d1));
  EXPECT_TRUE(range_evaluator.IsAlwaysNegativeOrZero(d1));

  // d2 is sometimes positive and sometimes negative.
  EXPECT_FALSE(range_evaluator.IsAlwaysPositiveOrZero(d2));
  EXPECT_FALSE(range_evaluator.IsAlwaysNegativeOrZero(d2));

  // d3 is always 0.
  EXPECT_TRUE(range_evaluator.IsAlwaysPositiveOrZero(d3));
  EXPECT_TRUE(range_evaluator.IsAlwaysNegativeOrZero(d3));
}

// TODO(b/313840171): Simplify `(d1 * 4 + d2) floordiv 8` to `d1 floordiv 2`.

// TODO(b/313840171): Simplify `(d0 * 8 + d1) floordiv 16` to `d0 floordiv 2`.

// TODO(b/313840171): Simplify `((d0 * 8 + d1) mod 16) floordiv 4` to
// `((d0 * 8 + d1) floordiv 4) mod 4` to `(d0 * 2 + d1 floordiv 4) mod 4`.

TEST_F(IndexingMapTest, AffineMapPrinterTest) {
  auto map =
      ParseAffineMap("(d0, d1)[s0, s1] -> (d0 + d1 floordiv 8, s0 + s1 mod 16)",
                     &mlir_context_);
  printer_.SetDimensionName(0, "offset");
  printer_.SetSymbolName(1, "linear_index");
  EXPECT_THAT(printer_.ToString(map),
              HasSubstr("(offset, d1)[s0, linear_index] -> "
                        "(offset + d1 floordiv 8, s0 + linear_index mod 16)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

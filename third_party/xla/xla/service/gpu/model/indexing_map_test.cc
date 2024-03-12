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

#include <optional>

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

class IndexingMapTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter printer_;
};

TEST_F(IndexingMapTest, Evaluation) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1, s0)", &mlir_context_),
      {4, 4}, {2, 2});

  auto results = indexing_map.Evaluate(
      mlir::getAffineConstantExprs({1, 2}, &mlir_context_),
      mlir::getAffineConstantExprs({3, 4}, &mlir_context_));
  EXPECT_THAT(results, ElementsAre(2, 1, 4, 3));

  auto feasible = indexing_map.ConstraintsSatisfied(
      mlir::getAffineConstantExprs({1, 2}, &mlir_context_),
      mlir::getAffineConstantExprs({3, 4}, &mlir_context_));
  EXPECT_TRUE(feasible);

  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 4", &mlir_context_),
                             Range{0, 0});

  auto infeasible = indexing_map.ConstraintsSatisfied(
      mlir::getAffineConstantExprs({1, 2}, &mlir_context_),
      mlir::getAffineConstantExprs({5, 4}, &mlir_context_));
  EXPECT_FALSE(infeasible);
}

TEST_F(IndexingMapTest, Composition_Permutation) {
  IndexingMap producer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1, s0)", &mlir_context_),
      {4, 4}, {2, 2});

  IndexingMap consumer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_), {4}, {4});

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
                          domain:
                          d0 in [0, 3]
                          s0 in [0, 1]
                          s1 in [0, 1]
                          s2 in [0, 3]
                        )"));
}

TEST_F(IndexingMapTest, Composition_RestrictedRange) {
  IndexingMap producer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1, s0)", &mlir_context_),
      {5, 6}, {7, 2});

  IndexingMap consumer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_), {10}, {8});

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
                          domain:
                          d0 in [0, 4]
                          s0 in [0, 6]
                          s1 in [0, 1]
                          s2 in [0, 5]
                        )"));
}

TEST_F(IndexingMapTest, Composition_ProducerAndConsumerHaveConstraints) {
  IndexingMap producer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1, s0)", &mlir_context_),
      {50, 60}, {70, 20});
  producer.AddConstraint(ParseAffineExpr("d0 mod 8", &mlir_context_),
                         Range{0, 0});
  producer.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                         Range{1, 1});

  IndexingMap consumer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_), {10}, {8});
  consumer.AddConstraint(ParseAffineExpr("d0 + s0", &mlir_context_),
                         Range{0, 20});
  consumer.AddConstraint(ParseAffineExpr("s0 mod 4", &mlir_context_),
                         Range{0, 0});

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
                          domain:
                          d0 in [0, 9]
                          s0 in [0, 69]
                          s1 in [0, 19]
                          s2 in [0, 7]
                          d0 + s2 in [0, 20]
                          d0 mod 8 in [0, 0]
                          s0 mod 3 in [1, 1]
                          s2 mod 4 in [0, 0]
                        )"));
  composed.Simplify();
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
                          domain:
                          d0 in [0, 9]
                          s0 in [0, 69]
                          s1 in [0, 19]
                          s2 in [0, 7]
                          d0 mod 8 in [0, 0]
                          s0 mod 3 in [1, 1]
                          s2 mod 4 in [0, 0]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintUsesSymbol) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1)", &mlir_context_),
      {50, 60}, {70, 20});
  // This constraint cannot be removed, because it contains a "used symbol".
  indexing_map.AddConstraint(ParseAffineExpr("s0 + s1", &mlir_context_),
                             Range{1, 100});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                             Range{0, 0});
  indexing_map.RemoveUnusedSymbols();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0, s1] -> (d1, d0, s1)
                          domain:
                          d0 in [0, 49]
                          d1 in [0, 59]
                          s0 in [0, 69]
                          s1 in [0, 19]
                          s0 + s1 in [1, 100]
                          s0 mod 3 in [0, 0]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintUsesOnlyUnusedSymbols) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d1, d0, s1)", &mlir_context_),
      {50, 60}, {70, 20});
  // This constraint can be removed, because it contains only the unused symbol.
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                             Range{0, 0});
  indexing_map.RemoveUnusedSymbols();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0] -> (d1, d0, s0)
                          domain:
                          d0 in [0, 49]
                          d1 in [0, 59]
                          s0 in [0, 19]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintsWithManySymbols) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42)",
                     &mlir_context_),
      {32}, {1, 2, 3, 4, 5});
  indexing_map.AddConstraint(
      ParseAffineExpr("d0 * 4 + s1 + s3", &mlir_context_), Range{24, 459});
  indexing_map.RemoveUnusedSymbols();
  // Symbols s0, s2, s4 will be removed and s1 and s3 will become s0 and s1.
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                              (d0)[s0, s1] -> (d0 * 4 + s0 + s1 - 42)
                              domain:
                              d0 in [0, 31]
                              s0 in [0, 1]
                              s1 in [0, 3]
                              d0 * 4 + s0 + s1 in [24, 459]
                            )"));
}

TEST_F(IndexingMapTest, ConstraintRangeSimplification_Sum) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("(d0 mod 8) + 5", &mlir_context_),
                             Range{50, 54});

  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [0, 99]
                          d0 mod 8 in [45, 49]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_FloorDivPositiveDivisorPositiveBounds) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("d0 floordiv 8", &mlir_context_),
                             Range{5, 11});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [40, 95]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_FloorDivPositiveDivisorNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {Range{0, 99}}, {Range{-99, 99}});

  indexing_map.AddConstraint(ParseAffineExpr("s0 floordiv 3", &mlir_context_),
                             Range{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [-33, -13]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_FloorDivNegativeDivisorNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {Range{0, 99}}, {Range{-99, 99}});

  indexing_map.AddConstraint(ParseAffineExpr("s0 floordiv -3", &mlir_context_),
                             Range{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [15, 35]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_MulPositiveMultiplierPositiveBounds) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("d0 * 8", &mlir_context_),
                             Range{14, 33});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [2, 4]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_MulPositiveMultiplierNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {Range{0, 99}}, {Range{-99, 99}});

  indexing_map.AddConstraint(ParseAffineExpr("s0 * 3", &mlir_context_),
                             Range{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [-3, -2]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintRangeSimplification_MulNegativeMultiplierNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {Range{0, 99}}, {Range{-99, 99}});

  indexing_map.AddConstraint(ParseAffineExpr("s0 * -3", &mlir_context_),
                             Range{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [2, 3]
                        )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ConstantDims) {
  IndexingMap indexing_map = IndexingMap(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {Range{5, 5}}, {});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                  (d0) -> (5)
                                                  domain:
                                                  d0 in [5, 5]
                                                )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_DivsAndModsIfSmallerThanDivisor) {
  auto serialized_map = "(d0, d1) -> (d0 + d1 floordiv 16, d1 mod 16)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {8, 16}, {});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                  (d0, d1) -> (d0, d1)
                                                  domain:
                                                  d0 in [0, 7]
                                                  d1 in [0, 15]
                                                )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsAndModsWithMultipliers) {
  auto serialized_map =
      "(d0, d1, d2) -> ((d0 * 100 + d1 * 10 + d2) floordiv 100, "
      "((d0 * 100 + d1 * 10 + d2) mod 100) floordiv 10, "
      "d2 mod 10)";

  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {9, 9, 9}, {});
  indexing_map.Simplify();

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                  (d0, d1, d2) -> (d0, d1, d2)
                                                  domain:
                                                  d0 in [0, 8]
                                                  d1 in [0, 8]
                                                  d2 in [0, 8]
                                                )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_DivsAndModsWithDivisibleMultipliers) {
  auto serialized_map =
      "(d0, d1, d2) -> ((d0 * 16 + d1 * 4 + d2) floordiv 8, "
      "                 (d0 * 16 + d1 * 4 + d2) mod 8)";

  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {10, 10, 10}, {});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
    (d0, d1, d2) -> (d0 * 2 + (d1 + d2 floordiv 4) floordiv 2,
                     (d1 * 4 + d2) mod 8)
    domain:
    d0 in [0, 9]
    d1 in [0, 9]
    d2 in [0, 9]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsAndModsWithReverse) {
  auto serialized_map =
      "(d0, d1) -> (-((d0 * -11 - d1 + 109) floordiv 11) + 9, "
      "d0 * 11 + d1 + ((d0 * -11 - d1 + 109) floordiv 11) * 11 - 99)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {8, 9}, {});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                 (d0, d1) -> (d0, d1)
                                                 domain:
                                                 d0 in [0, 7]
                                                 d1 in [0, 8]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsInSequence) {
  auto serialized_map =
      "()[s0] -> (s0 - ((s0 floordiv 2) floordiv 7) * 14 + (s0 floordiv 14) * "
      "14)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {1234});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                 ()[s0] -> (s0)
                                                 domain:
                                                 s0 in [0, 1233]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivGcdGreater1) {
  auto serialized_map =
      "()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2 - ((s0 * 2 + s1 floordiv 64) "
      "floordiv 3) * 768 + ((s0 * 128 + s1) floordiv 192) * 768)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {1234, 128, 4});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      ()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2)
      domain:
      s0 in [0, 1233]
      s1 in [0, 127]
      s2 in [0, 3]
    )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ExtractFromMod) {
  auto serialized_map =
      "()[s0, s1, s2, s3] -> ((s0 * 458752 + s1 + s2 * 4 + s3 * 512) mod "
      "20000)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {872, 4, 128, 896});
  indexing_map.Simplify();
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      ()[s0, s1, s2, s3] -> (
        s1 + (s0 * 458752 + s2 * 4 + s3 * 512) mod 20000
      )
      domain:
      s0 in [0, 871]
      s1 in [0, 3]
      s2 in [0, 127]
      s3 in [0, 895]
    )"));
}

TEST_F(IndexingMapTest, RangeEvaluatorTest) {
  RangeEvaluator range_evaluator(
      {Range{0, 9}, Range{-10, -1}, Range{-1, 2}, Range{0, 0}}, {},
      &mlir_context_);
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

TEST(RangeComparisionTest, Comparisons) {
  Range range{12, 64};
  EXPECT_EQ(range > 11, true);
  EXPECT_EQ(range > 12, std::nullopt);
  EXPECT_EQ(range > 65, false);

  EXPECT_EQ(range < 65, true);
  EXPECT_EQ(range < 64, std::nullopt);
  EXPECT_EQ(range < 10, false);

  EXPECT_EQ(range == 11, false);
  EXPECT_EQ(range == 15, std::nullopt);
  EXPECT_EQ(range == 65, false);

  EXPECT_EQ(range != 11, true);
  EXPECT_EQ(range != 15, std::nullopt);
  EXPECT_EQ(range != 65, true);

  EXPECT_EQ(range >= 12, true);
  EXPECT_EQ(range >= 64, std::nullopt);
  EXPECT_EQ(range >= 65, false);

  EXPECT_EQ(range <= 11, false);
  EXPECT_EQ(range <= 64, true);
  EXPECT_EQ(range <= 63, std::nullopt);
  EXPECT_EQ(range <= 65, true);

  Range point{15, 15};
  EXPECT_EQ(point == 15, true);
  EXPECT_EQ(point == 16, false);

  EXPECT_EQ(point != 15, false);
  EXPECT_EQ(point != 16, true);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

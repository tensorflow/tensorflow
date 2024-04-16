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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineMap;
using ::testing::ElementsAre;

class IndexingMapTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter printer_;
};

TEST_F(IndexingMapTest, RTVar) {
  auto zero_dim_map = AffineMap::get(&mlir_context_);
  std::vector<RTVar> rt_vars{RTVar{Interval{0, 2},
                                   /*instr=*/nullptr, zero_dim_map},
                             RTVar({Interval{0, 7},
                                    /*instr=*/nullptr, zero_dim_map})};

  IndexingMap indexing_map(
      ParseAffineMap("(d0, d1)[s0, s1, s2] -> (d1, d0, s0 + s1, s1)",
                     &mlir_context_),
      {DimVar{{0, 99}}, DimVar{{0, 43}}}, {RangeVar{{-99, 99}}},
      std::move(rt_vars));
  printer_.SetSymbolName(0, "range");
  printer_.SetSymbolName(1, "rt_0");
  printer_.SetSymbolName(2, "rt_1");
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0, d1)[range, rt_0, rt_1] -> (d1, d0, range + rt_0, rt_0)
              domain:
              d0 in [0, 99]
              d1 in [0, 43]
              range in [-99, 99]
              rt_0 in [0, 2]
                hlo: NULL
                () -> ()
              rt_1 in [0, 7]
                hlo: NULL
                () -> ()
              )"));
}

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
                             Interval{0, 0});

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

TEST_F(IndexingMapTest, Composition_RestrictedInterval) {
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
                         Interval{0, 0});
  producer.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                         Interval{1, 1});

  IndexingMap consumer = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_), {10}, {8});
  consumer.AddConstraint(ParseAffineExpr("d0 + s0", &mlir_context_),
                         Interval{0, 20});
  consumer.AddConstraint(ParseAffineExpr("s0 mod 4", &mlir_context_),
                         Interval{0, 0});

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
  composed.Simplify(GetIndexingMapForInstruction);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
                          domain:
                          d0 in [0, 8]
                          s0 in [1, 67]
                          s1 in [0, 19]
                          s2 in [0, 4]
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
                             Interval{1, 100});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                             Interval{0, 0});
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
                             Interval{0, 0});
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
      ParseAffineExpr("d0 * 4 + s1 + s3", &mlir_context_), Interval{24, 459});
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

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintsWithRTVars) {
  auto zero_dim_map = AffineMap::get(&mlir_context_);
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42)",
                     &mlir_context_),
      {DimVar{{0, 31}}}, {RangeVar{{0, 0}}, RangeVar{{0, 1}}, RangeVar{{0, 2}}},
      {RTVar{Interval{0, 3},
             /*instr=*/nullptr, zero_dim_map},
       RTVar{Interval{0, 4},
             /*instr=*/nullptr, zero_dim_map}});
  indexing_map.AddConstraint(
      ParseAffineExpr("d0 * 4 + s1 + s3", &mlir_context_), Interval{24, 459});
  indexing_map.RemoveUnusedSymbols();
  // Symbols s0, s2, s4 will be removed and s1 and s3 will become s0 and s1.
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                              (d0)[s0, s1] -> (d0 * 4 + s0 + s1 - 42)
                              domain:
                              d0 in [0, 31]
                              s0 in [0, 1]
                              s1 in [0, 3]
                                hlo: NULL
                                () -> ()
                              d0 * 4 + s0 + s1 in [24, 459]
                            )"));
}

TEST_F(IndexingMapTest, ConstraintIntervalSimplification_Sum) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("(d0 mod 8) + 5", &mlir_context_),
                             Interval{50, 54});

  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [0, 99]
                          d0 mod 8 in [45, 49]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivPositiveDivisorPositiveBounds) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("d0 floordiv 8", &mlir_context_),
                             Interval{5, 11});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [40, 95]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivPositiveDivisorNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {DimVar{{0, 99}}}, {RangeVar{{-99, 99}}}, /*rt_vars=*/{});

  indexing_map.AddConstraint(ParseAffineExpr("s0 floordiv 3", &mlir_context_),
                             Interval{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [-33, -13]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivNegativeDivisorNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {DimVar{{0, 99}}}, {RangeVar{{-99, 99}}}, /*rt_vars=*/{});

  indexing_map.AddConstraint(ParseAffineExpr("s0 floordiv -3", &mlir_context_),
                             Interval{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [15, 35]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulPositiveMultiplierPositiveBounds) {
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_), {100}, {});

  indexing_map.AddConstraint(ParseAffineExpr("d0 * 8", &mlir_context_),
                             Interval{14, 33});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0) -> (d0)
                          domain:
                          d0 in [2, 4]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulPositiveMultiplierNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {DimVar{{0, 99}}}, {RangeVar{{-99, 99}}}, /*rt_vars=*/{});

  indexing_map.AddConstraint(ParseAffineExpr("s0 * 3", &mlir_context_),
                             Interval{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [-3, -2]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulNegativeMultiplierNegativeBounds) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0)[s0] -> (d0)", &mlir_context_),
                  {DimVar{{0, 99}}}, {RangeVar{{-99, 99}}}, /*rt_vars=*/{});

  indexing_map.AddConstraint(ParseAffineExpr("s0 * -3", &mlir_context_),
                             Interval{-11, -5});
  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0] -> (d0)
                          domain:
                          d0 in [0, 99]
                          s0 in [2, 3]
                        )"));
}

TEST_F(IndexingMapTest, ConstraintMerge_Mod) {
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0, s1] -> (d0, s1, s0)", &mlir_context_),
      {DimVar{{0, 4}}}, {RangeVar{{-21, -1}}, RangeVar{{0, 10}}},
      /*rt_vars=*/{});
  indexing_map.AddConstraint(ParseAffineExpr("d0 mod 3", &mlir_context_),
                             Interval{0, 0});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 2", &mlir_context_),
                             Interval{0, 0});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                             Interval{0, 0});
  indexing_map.AddConstraint(ParseAffineExpr("s1 mod 5", &mlir_context_),
                             Interval{1, 1});
  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(), MatchIndexingString(R"(
                          (d0)[s0, s1] -> (d0, s1, s0)
                          domain:
                          d0 in [0, 3]
                          s0 in [-18, -6]
                          s1 in [1, 6]
                          d0 mod 3 in [0, 0]
                          s0 mod 6 in [0, 0]
                          s1 mod 5 in [1, 1]
                        )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ConstantDims) {
  IndexingMap indexing_map =
      IndexingMap(ParseAffineMap("(d0) -> (d0)", &mlir_context_),
                  {DimVar{{5, 5}}}, /*range_vars=*/{}, /*rt_vars=*/{});
  indexing_map.Simplify(GetIndexingMapForInstruction);
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
  indexing_map.Simplify(GetIndexingMapForInstruction);
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
  indexing_map.Simplify(GetIndexingMapForInstruction);

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
  indexing_map.Simplify(GetIndexingMapForInstruction);
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
  indexing_map.Simplify(GetIndexingMapForInstruction);
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
                                                 (d0, d1) -> (d0, d1)
                                                 domain:
                                                 d0 in [0, 7]
                                                 d1 in [0, 8]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape) {
  auto serialized_map =
      "()[s0] -> ((s0 * 128) mod 715 + ((s0 * 128) floordiv 715) * 715)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {128});
  indexing_map.Simplify(GetIndexingMapForInstruction);
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      ()[s0] -> (s0 * 128)
      domain: s0 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape_Regression) {
  // We have s0 * 128 in the mod, but s0 * 64 in the floordiv *.
  auto serialized_map =
      "()[s0] -> ((s0 * 128) mod 715 + ((s0 * 64) floordiv 715) * 715)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {128});
  indexing_map.Simplify(GetIndexingMapForInstruction);
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      ()[s0] -> ((s0 * 128) mod 715 + ((s0 * 64) floordiv 715) * 715)
      domain: s0 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsInSequence) {
  auto serialized_map =
      "()[s0] -> (s0 - ((s0 floordiv 2) floordiv 7) * 14 + (s0 floordiv 14) * "
      "14)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {1234});
  indexing_map.Simplify(GetIndexingMapForInstruction);
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
  indexing_map.Simplify(GetIndexingMapForInstruction);
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
  indexing_map.Simplify(GetIndexingMapForInstruction);
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

TEST_F(IndexingMapTest,
       AffineMapSimplification_ExtractFromDiv_NegativeMultiplier) {
  auto serialized_map =
      "()[s0, s1] -> ((s0 * 16 - (s1 floordiv 4) floordiv 2 + (s1 floordiv 8) "
      "* 2) floordiv 4)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {}, {2, 128});
  indexing_map.Simplify(GetIndexingMapForInstruction);
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      ()[s0, s1] -> (
        s0 * 4 + s1 floordiv 32
      )
      domain:
      s0 in [0, 1]
      s1 in [0, 127]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_Simple) {
  auto serialized_map = "(d0)[s0, s1, s2] -> (s2, d0, s1, s0 floordiv 6)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {4}, {7, 2, 6});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 6", &mlir_context_),
                             Interval{0, 0});

  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
      domain:
        d0 in [0, 3]
        s0 in [0, 1]
        s1 in [0, 1]
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_WithShift) {
  auto serialized_map = "(d0)[s0, s1, s2] -> (s2, d0, s1, s0)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {4}, {42, 2, 6});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 6", &mlir_context_),
                             Interval{3, 3});

  // [BEFORE] Allowed values for s0: 3, 9, 15, ..., 39 = (6 * 6 + 3)
  // [AFTER] Allowed values for s0: 0, 1, 2, ..., 6
  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0 * 6 + 3)
      domain:
        d0 in [0, 3]
        s0 in [0, 6]
        s1 in [0, 1]
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_TwoModConstraints) {
  auto serialized_map = "(d0)[s0, s1, s2] -> (s2, d0, s1, s0 floordiv 6)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {4}, {7, 2, 6});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 2", &mlir_context_),
                             Interval{0, 0});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 3", &mlir_context_),
                             Interval{0, 0});

  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0)
      domain:
        d0 in [0, 3]
        s0 in [0, 1]
        s1 in [0, 1]
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_RescaledSymbolInOtherConstraint) {
  auto serialized_map = "(d0)[s0, s1, s2] -> (s2, d0, s1, s0)";
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap(serialized_map, &mlir_context_), {4}, {10, 2, 6});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 6", &mlir_context_),
                             Interval{3, 3});
  indexing_map.AddConstraint(ParseAffineExpr("s0 * s2", &mlir_context_),
                             Interval{0, 28});

  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0 * 6 + 3)
      domain:
        d0 in [0, 3]
        s0 in [0, 1]
        s1 in [0, 1]
        s2 in [0, 5]
        (s0 * 6 + 3) * s2 in [0, 28]
    )"));
}

TEST_F(IndexingMapTest, RangeEvaluatorTest) {
  RangeEvaluator range_evaluator(
      {Interval{0, 9}, Interval{-10, -1}, Interval{-1, 2}, Interval{0, 0}}, {},
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

TEST(IntervalComparisionTest, Comparisons) {
  Interval interval{12, 64};
  EXPECT_EQ(interval > 11, true);
  EXPECT_EQ(interval > 12, std::nullopt);
  EXPECT_EQ(interval > 65, false);

  EXPECT_EQ(interval < 65, true);
  EXPECT_EQ(interval < 64, std::nullopt);
  EXPECT_EQ(interval < 10, false);

  EXPECT_EQ(interval == 11, false);
  EXPECT_EQ(interval == 15, std::nullopt);
  EXPECT_EQ(interval == 65, false);

  EXPECT_EQ(interval != 11, true);
  EXPECT_EQ(interval != 15, std::nullopt);
  EXPECT_EQ(interval != 65, true);

  EXPECT_EQ(interval >= 12, true);
  EXPECT_EQ(interval >= 64, std::nullopt);
  EXPECT_EQ(interval >= 65, false);

  EXPECT_EQ(interval <= 11, false);
  EXPECT_EQ(interval <= 64, true);
  EXPECT_EQ(interval <= 63, std::nullopt);
  EXPECT_EQ(interval <= 65, true);

  Interval point{15, 15};
  EXPECT_EQ(point == 15, true);
  EXPECT_EQ(point == 16, false);

  EXPECT_EQ(point != 15, false);
  EXPECT_EQ(point != 16, true);
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_ScalarConstant) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %constant = s64[] constant(42)
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("()[s0] -> (s0)", &mlir_context_),
      /*dimensions=*/{},
      /*range_vars=*/{},
      {RTVar{Interval{42, 42},
             hlo_module.value()->entry_computation()->root_instruction(),
             AffineMap::get(0, 0, {}, &mlir_context_)}});

  EXPECT_TRUE(indexing_map.Simplify(GetIndexingMapForInstruction));

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              () -> (42)
              domain:
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_StaticIndexIntoTensorConstant) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %constant = s64[2, 4]{1,0} constant({{1, 2, 3, 4}, {11, 12, 13, 14}})
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("()[s0] -> (s0)", &mlir_context_),
      /*dimensions=*/{},
      /*range_vars=*/{},
      {RTVar{Interval{1, 14},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("() -> (1,2)", &mlir_context_)}});

  EXPECT_TRUE(indexing_map.Simplify(GetIndexingMapForInstruction));

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              () -> (13)
              domain:
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_NonFoldableTensor) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %constant = s64[2, 4]{1,0} constant({{1, 2, 3, 4}, {11, 12, 13, 14}})
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (s0)", &mlir_context_),
      /*dimensions=*/{},
      /*range_vars=*/{},
      {RTVar{Interval{1, 14},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (1, d0)", &mlir_context_)}});

  EXPECT_FALSE(indexing_map.Simplify(GetIndexingMapForInstruction));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_Iota) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %iota = s64[10, 10]{1,0} iota(), iota_dimension=0
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 255}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 9},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 7)", &mlir_context_)}});

  EXPECT_TRUE(indexing_map.Simplify(GetIndexingMapForInstruction));

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, d0)
              domain:
              d0 in [0, 255]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_IotaAsConstant) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %iota = s64[10, 10]{1,0} iota(), iota_dimension=1
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 255}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 9},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 7)", &mlir_context_)}});

  EXPECT_TRUE(indexing_map.Simplify(GetIndexingMapForInstruction));

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, 7)
              domain:
              d0 in [0, 255]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_ConstraintsGetUpdated) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        ROOT %iota = s64[10, 10]{1,0} iota(), iota_dimension=0
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 255}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 9},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 7)", &mlir_context_)}});
  indexing_map.AddConstraint(ParseAffineExpr("s0 mod 2", &mlir_context_),
                             Interval{0, 0});

  EXPECT_TRUE(indexing_map.Simplify(GetIndexingMapForInstruction));

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, d0)
              domain:
              d0 in [0, 254]
              d0 mod 2 in [0, 0]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_Broadcast) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %iota = s64[12]{0} iota(), iota_dimension=0
        ROOT %broadcast = s64[32, 12]{1,0} broadcast(s64[12]{0} %iota), dimensions={1}
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // (d0, 11): d0 maps into the broadcasted dimension, so it doesn't matter
  // and 11 maps to 11 in iota.
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 31}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 11},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 11)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, 11)
              domain:
              d0 in [0, 31]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_ChainedNoncomputeOps) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %iota = s64[12]{0} iota(), iota_dimension=0
        %reverse = s64[12]{0} reverse(s64[12]{0} %iota), dimensions={0}
        %reshape = s64[3,4]{1,0} reshape(s64[12]{0} %reverse)
        ROOT %broadcast = s64[36,3,4]{2,1,0} broadcast(s64[3,4]{1,0} %reshape), dimensions={1,2}
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // - Iota: [0, 1, ,,,, 11]
  // - Reverse: [11, 10, ..., 0]
  // - Reshape: [[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]
  // - Coordinates: (d0 floordiv 12, 3)
  // - y-coordinate=3 means we index into [8, 4, 0]
  // - x-coordinate=(d0 floordiv 12) means our constant looks like this:
  //   [8, ..., 8, 4, ..., 4, 0, ..., 0]
  // - Hence our final expression: (d0 floordiv 12) * -4 + 8
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 35}},
      /*range_vars=*/{},
      {RTVar{
          Interval{0, 11},
          hlo_module.value()->entry_computation()->root_instruction(),
          ParseAffineMap("(d0) -> (d0, d0 floordiv 12, 3)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, (d0 floordiv 12) * -4 + 8)
              domain:
              d0 in [0, 35]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_PartialRTVarRemoval) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %constant = s64[12]{0} constant({...})
        ROOT %broadcast = s64[24,12]{1,0} broadcast(s64[12]{0} %constant), dimensions={1}
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // (d0, d0 floordiv 2): d0 maps into the broadcasted dimension, so it can't be
  // removed, but d0 floordiv 2 doesn't yield an affine expression so we need to
  // keep the RTVar, but can optimize it by removing the broadcast.
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 23}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 512},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, d0 floordiv 2)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0)[s0] -> (d0, s0)
              domain:
              d0 in [0, 23]
              s0 in [0, 512]
                hlo: %constant = s64[12]{0} constant({...})
                (d0) -> (d0 floordiv 2)
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_Add) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %constant = s64[] constant(42)
        %broadcast = s64[12,13,24]{2,1,0} broadcast(s64[] %constant), dimensions={}
        %iota = s64[12,13,24]{2,1,0} iota(), iota_dimension=2
        ROOT %add = s64[12,13,24]{2,1,0} add(s64[12,13,24]{2,1,0} %broadcast, s64[12,13,24]{2,1,0} %iota)
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // The iota dimension is the last dimension in (d0, 7, 2 * d0), hence this
  // composes to 42 + 2 * d0
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 11}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 11},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 7, 2 * d0)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, d0 * 2 + 42)
              domain:
              d0 in [0, 11]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_Multiply) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %iota0 = s64[12,12]{1,0} iota(), iota_dimension=0
        %iota1 = s64[12]{0} iota(), iota_dimension=0
        %broadcast = s64[12,12]{1,0} broadcast(s64[12]{0} %iota1), dimensions={1}
        %multiply = s64[12,12]{1,0} multiply(s64[12,12]{1,0} %iota0, s64[12,12]{1,0} %broadcast)
        ROOT %reverse = s64[12,12]{1,0} reverse(s64[12,12]{1,0} %multiply), dimensions={0}
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // Iota0: [[0, ..., 0], [1, ..., 1], ..., [11, ..., 11]]
  // Iota1: [0, ..., 11]
  // Broadcast1: [[0, 1, ..., 11], [0, 1, ..., 11], ..., [0, 1, ..., 11]]
  // Mul: [[0, .., 0], [0, 1, ..., 11], [0, 2, ..., 22], ..., [0, 11, ..., 121]]
  // Reverse: [[0, 11, ..., 121], [0, 10, ..., 110], ..., [0, ..., 0]]
  // Therefore (d0, d0) evaluates to: (11 - d0) * d0.
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 11}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 11},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, d0)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0) -> (d0, (-d0 + 11) * d0)
              domain:
              d0 in [0, 11]
              )"));
}

TEST_F(IndexingMapTest, ReplaceConstantRTVars_PartiallyOptimizableAdd) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> hlo_module =
      ParseAndReturnVerifiedModule(R"hlo(
      HloModule m

      ENTRY e {
        %constant = s64[12]{0} constant({...})
        %broadcast = s64[12,13,24]{2,1,0} broadcast(s64[12]{0} %constant), dimensions={0}
        %iota = s64[12,13,24]{2,1,0} iota(), iota_dimension=2
        ROOT %add = s64[12,13,24]{2,1,0} add(s64[12,13,24]{2,1,0} %broadcast, s64[12,13,24]{2,1,0} %iota)
      }
    )hlo");

  ASSERT_TRUE(hlo_module.ok());

  // The iota dimension is the last dimension in (d0, 7, 2 * d0), the constant
  // only depends on the first dimension. The constant consists of some
  // arbitrary values that cannot be represent as an affine expression, hence
  // the RTVar remains in-place.
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0] -> (d0, s0)", &mlir_context_),
      /*dimensions=*/{{0, 11}},
      /*range_vars=*/{},
      {RTVar{Interval{0, 11},
             hlo_module.value()->entry_computation()->root_instruction(),
             ParseAffineMap("(d0) -> (d0, 7, 2 * d0)", &mlir_context_)}});

  indexing_map.Simplify(GetIndexingMapForInstruction);

  EXPECT_THAT(indexing_map.ToString(printer_), MatchIndexingString(R"(
              (d0)[s0] -> (d0, d0 * 2 + s0)
              domain:
              d0 in [0, 11]
              s0 in [0, 11]
                hlo: %constant = s64[12]{0} constant({...})
                (d0) -> (d0)
              )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

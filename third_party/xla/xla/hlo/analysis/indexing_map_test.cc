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

#include "xla/hlo/analysis/indexing_map.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::mlir::AffineMap;
using ::testing::AnyOf;
using ::testing::ElementsAre;

class IndexingMapTest : public HloHardwareIndependentTestBase {
 public:
  IndexingMap Parse(absl::string_view indexing_map_str) {
    auto indexing_map = ParseIndexingMap(indexing_map_str, &mlir_context_);
    EXPECT_TRUE(indexing_map.has_value());
    return *indexing_map;
  }

  mlir::MLIRContext mlir_context_;
};

std::vector<bool> ConvertToSTL(const llvm::SmallBitVector& bit_vector) {
  std::vector<bool> result;
  result.reserve(bit_vector.size());
  for (int i = 0; i < bit_vector.size(); ++i) {
    result.push_back(bit_vector[i]);
  }
  return result;
}

TEST_F(IndexingMapTest, VariableKind) {
  EXPECT_EQ(ToVariableType("default"), VariableKind::kDefault);
  EXPECT_EQ(ToVariableType("th_x"), VariableKind::kThreadX);
  EXPECT_EQ(ToVariableType("th_y"), VariableKind::kThreadY);
  EXPECT_EQ(ToVariableType("th_z"), VariableKind::kThreadZ);
  EXPECT_EQ(ToVariableType("bl_x"), VariableKind::kBlockX);
  EXPECT_EQ(ToVariableType("bl_y"), VariableKind::kBlockY);
  EXPECT_EQ(ToVariableType("bl_z"), VariableKind::kBlockZ);
  EXPECT_EQ(ToVariableType("warp"), VariableKind::kWarp);
  EXPECT_EQ(ToVariableType("th_w"), VariableKind::kWarpThread);

  EXPECT_EQ(ToVariableName(VariableKind::kDefault), "default");
  EXPECT_EQ(ToVariableName(VariableKind::kThreadX), "th_x");
  EXPECT_EQ(ToVariableName(VariableKind::kThreadY), "th_y");
  EXPECT_EQ(ToVariableName(VariableKind::kThreadZ), "th_z");
  EXPECT_EQ(ToVariableName(VariableKind::kBlockX), "bl_x");
  EXPECT_EQ(ToVariableName(VariableKind::kBlockY), "bl_y");
  EXPECT_EQ(ToVariableName(VariableKind::kBlockZ), "bl_z");
  EXPECT_EQ(ToVariableName(VariableKind::kWarp), "warp");
  EXPECT_EQ(ToVariableName(VariableKind::kWarpThread), "th_w");
}

TEST_F(IndexingMapTest, VerifyDimensions) {
  auto indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_),
      /*dim_upper_bounds=*/{10, 10}, /*symbol_upper_bounds=*/{});

  std::stringstream ss;
  EXPECT_FALSE(indexing_map.Verify(ss));
  EXPECT_EQ(ss.str(),
            "dim size must match the number of dimensions in the affine map");
}

TEST_F(IndexingMapTest, VerifySymbols) {
  auto indexing_map = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0)", &mlir_context_),
      /*dim_upper_bounds=*/{10}, /*symbol_upper_bounds=*/{10});

  std::stringstream ss;
  EXPECT_FALSE(indexing_map.Verify(ss));
  EXPECT_EQ(ss.str(),
            "range vars size + rt var size must match the number of symbols in "
            "the affine map");
}

TEST_F(IndexingMapTest, RTVar) {
  IndexingMap indexing_map(
      ParseAffineMap("(d0, d1)[range, rt0, rt1] -> (d1, d0, range + rt0, rt1)",
                     &mlir_context_),
      {IndexingMap::Variable{0, 99, "d0"}, IndexingMap::Variable{0, 43, "d1"}},
      {IndexingMap::Variable{-99, 99, "range"}},
      {IndexingMap::Variable{Interval{0, 2}},
       IndexingMap::Variable({Interval{0, 7}})});
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                (d0, d1)[range]{rt0, rt1} -> (d1, d0, range + rt0, rt1),
                domain:
                d0 in [0, 99],
                d1 in [0, 43],
                range in [-99, 99],
                rt0 in [0, 2],
                rt1 in [0, 7]
              )"));
}

TEST_F(IndexingMapTest, Evaluation) {
  IndexingMap indexing_map = Parse(R"(
     (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
     domain:
     d0 in [0, 3],
     d1 in [0, 3],
     s0 in [0, 1],
     s1 in [0, 1]
  )");
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
  IndexingMap producer = Parse(R"(
     (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
     domain:
     d0 in [0, 3],
     d1 in [0, 3],
     s0 in [0, 1],
     s1 in [0, 1]
  )");
  IndexingMap consumer = Parse(R"(
     (d0)[s0] -> (d0, s0),
     domain:
     d0 in [0, 3],
     s0 in [0, 3]
  )");
  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
                          domain:
                          d0 in [0, 3],
                          s0 in [0, 1],
                          s1 in [0, 1],
                          s2 in [0, 3]
                        )"));
}

TEST_F(IndexingMapTest, Composition_RestrictedInterval) {
  IndexingMap producer = Parse(R"(
     (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
     domain:
     d0 in [0, 4],
     d1 in [0, 5],
     s0 in [0, 6],
     s1 in [0, 1]
  )");

  IndexingMap consumer = Parse(R"(
     (d0)[s0] -> (d0, s0),
     domain:
     d0 in [0, 9],
     s0 in [0, 7]
  )");

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
                          domain:
                          d0 in [0, 4],
                          s0 in [0, 6],
                          s1 in [0, 1],
                          s2 in [0, 5]
                        )"));
}

TEST_F(IndexingMapTest, Composition_ProducerAndConsumerHaveConstraints) {
  IndexingMap producer = Parse(R"(
     (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
     domain:
     d0 in [0, 49],
     d1 in [0, 59],
     s0 in [0, 69],
     s1 in [0, 19],
     d0 mod 8 in [0, 0],
     s0 mod 3 in [1, 1]
  )");

  IndexingMap consumer = Parse(R"(
     (d0)[s0] -> (d0, s0),
     domain:
     d0 in [0, 9],
     s0 in [0, 7],
     d0 + s0 in [0, 20],
     s0 mod 4 in [0, 0]
  )");

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
                          domain:
                          d0 in [0, 9],
                          s0 in [0, 69],
                          s1 in [0, 19],
                          s2 in [0, 7],
                          d0 + s2 in [0, 20],
                          d0 mod 8 in [0, 0],
                          s0 mod 3 in [1, 1],
                          s2 mod 4 in [0, 0]
                        )"));
  EXPECT_TRUE(composed.Simplify());
  EXPECT_THAT(composed, MatchIndexingMap(R"(
                          (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
                          domain:
                          d0 in [0, 8],
                          s0 in [1, 67],
                          s1 in [0, 19],
                          s2 in [0, 4],
                          d0 mod 8 in [0, 0],
                          s0 mod 3 in [1, 1],
                          s2 mod 4 in [0, 0]
                        )"));
}

TEST_F(IndexingMapTest, Composition_RTVar) {
  std::vector<IndexingMap::Variable> rt_vars{
      IndexingMap::Variable{Interval{0, 0}},
      IndexingMap::Variable({Interval{0, 1}}),
      IndexingMap::Variable({Interval{0, 226}})};

  IndexingMap producer(
      ParseAffineMap(
          "(d0, d1, d2)[rt0, rt1, rt2] -> (d0 + rt0, d1 + rt1, d2 + rt2)",
          &mlir_context_),
      {IndexingMap::Variable{{0, 0}}, IndexingMap::Variable{{0, 1}},
       IndexingMap::Variable{{0, 226}}},
      {}, std::move(rt_vars));

  IndexingMap consumer(
      ParseAffineMap("(d0, d1)[s] -> (0, d1, s)", &mlir_context_),
      {IndexingMap::Variable{0, 0}, IndexingMap::Variable{0, 1}},
      {IndexingMap::Variable{0, 31, "s"}}, {});

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(ToString(composed), MatchIndexingString(R"(
    (d0, d1)[s]{rt0, rt1, rt2} -> (rt0, d1 + rt1, s + rt2),
    domain:
    d0 in [0, 0],
    d1 in [0, 1],
    s in [0, 31],
    rt0 in [0, 0],
    rt1 in [0, 1],
    rt2 in [0, 226]
  )"));
}

TEST_F(IndexingMapTest, Composition_OnlyRTVars) {
  IndexingMap producer(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d0 + s0, d1 + 4 * s1)",
                     &mlir_context_),
      {IndexingMap::Variable{0, 24}, IndexingMap::Variable{0, 15}}, {},
      {IndexingMap::Variable{Interval{0, 2}, "ps_0"},
       IndexingMap::Variable{Interval{0, 1}, "ps_1"}});

  std::vector<IndexingMap::Variable> consumer_rt_vars;
  IndexingMap consumer(
      ParseAffineMap("(d0, d1)[s0, s1] -> (d0 + 2 * s0, d1 + 3 * s1)",
                     &mlir_context_),
      {IndexingMap::Variable{0, 24}, IndexingMap::Variable{0, 15}}, {},
      {IndexingMap::Variable{Interval{0, 25}, "cs_0"},
       IndexingMap::Variable{Interval{0, 16}, "cs_1"}});

  auto composed = ComposeIndexingMaps(consumer, producer);
  EXPECT_THAT(ToString(composed), MatchIndexingString(R"(
    (d0, d1){ps_0, ps_1, cs_0, cs_1} ->
      (d0 + cs_0 * 2 + ps_0, d1 + cs_1 * 3 + ps_1 * 4),
    domain:
    d0 in [0, 24],
    d1 in [0, 15],
    ps_0 in [0, 2],
    ps_1 in [0, 1],
    cs_0 in [0, 25],
    cs_1 in [0, 16],
    d0 + cs_0 * 2 in [0, 24],
    d1 + cs_1 * 3 in [0, 15]
  )"));
}

TEST_F(IndexingMapTest, RemoveUnusedVars_ConstraintUsesDim) {
  // This constraint cannot be removed, because it contains a dimension.
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (d1, s0, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    d0 + s0 in [1, 100],
    s0 mod 3 in [0, 0]
  )");
  indexing_map.RemoveUnusedVars();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0, s1] -> (d1, s0, s1),
                          domain:
                          d0 in [0, 49],
                          d1 in [0, 59],
                          s0 in [0, 69],
                          s1 in [0, 19],
                          d0 + s0 in [1, 100],
                          s0 mod 3 in [0, 0]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedVars_ConstraintUsesUnusedDim) {
  // This constraint can be removed, because it contains only the unused dim.
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (s0, d1, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    d0 mod 3 in [0, 0]
  )");
  indexing_map.RemoveUnusedVars();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0)[s0, s1] -> (s0, d0, s1),
                          domain:
                          d0 in [0, 59],
                          s0 in [0, 69],
                          s1 in [0, 19]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintUsesOnlyUnusedSym) {
  // This constraint can be removed, because it contains only the unused symbol.
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (d0, d1, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    s0 mod 3 in [0, 0]
  )");
  indexing_map.RemoveUnusedSymbols();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0] -> (d0, d1, s0),
                          domain:
                          d0 in [0, 49],
                          d1 in [0, 59],
                          s0 in [0, 19]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedVars_ConstraintsWithManyDims) {
  auto indexing_map = Parse(R"(
    (d0, d1, d2, d3, d4)[s0, s1, s2] -> (s0 * 4 + d1 + d3 - 42),
    domain:
    d0 in [0, 0],
    d1 in [0, 1],
    d2 in [0, 2],
    d3 in [0, 3],
    d4 in [0, 4],
    s0 in [0, 31],
    s1 in [0, 63],
    s2 in [0, 95],
    s0 * 4 + d1 + d3 in [24, 459],
    s0 + s2 in [0, 512]
  )");
  // dimensions d0, d2, d4 and symbol s1 will be removed.
  auto unused_vars = indexing_map.RemoveUnusedVars();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                              (d0, d1)[s0, s1] -> (d0 + s0 * 4 + d1 - 42),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 3],
                              s0 in [0, 31],
                              s1 in [0, 95],
                              d0 + s0 * 4 + d1 in [24, 459],
                              s0 + s1 in [0, 512]
                            )"));
  EXPECT_THAT(ConvertToSTL(unused_vars),
              ::testing::ElementsAreArray(
                  {true, false, true, false, true, false, true, false}));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintUsesSymbol) {
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (d1, d0, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    s0 + s1 in [1, 100],
    s0 mod 3 in [0, 0]
  )");
  // This constraint cannot be removed, because it contains a "used symbol".
  indexing_map.RemoveUnusedSymbols();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0, s1] -> (d1, d0, s1),
                          domain:
                          d0 in [0, 49],
                          d1 in [0, 59],
                          s0 in [0, 69],
                          s1 in [0, 19],
                          s0 + s1 in [1, 100],
                          s0 mod 3 in [0, 0]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintUsesOnlyUnusedSymbols) {
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (d1, d0, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    s0 mod 3 in [0, 0]
  )");
  // This constraint can be removed, because it contains only the unused symbol.
  indexing_map.RemoveUnusedSymbols();
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0, d1)[s0] -> (d1, d0, s0),
                          domain:
                          d0 in [0, 49],
                          d1 in [0, 59],
                          s0 in [0, 19]
                        )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintIsAConstantWithinRange) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, 49],
    0 in [-10, 5]
  )");
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                          (d0) -> (d0),
                          domain:
                          d0 in [0, 49]
                        )"));
}

TEST_F(IndexingMapTest, KnownEmpty_CreatingIndexingMapWithInfeasibleRange) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, -2]
  )");
  EXPECT_THAT(indexing_map, MatchIndexingMap("KNOWN EMPTY"));
}

TEST_F(IndexingMapTest, KnownEmpty_AddingConstraintOutOfRange) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, 49],
    0 in [10, 15]
  )");
  // Addition of this constraint makes the domain empty.
  EXPECT_THAT(indexing_map, MatchIndexingMap("KNOWN EMPTY"));
}

TEST_F(IndexingMapTest, KnownEmpty_Composition) {
  auto indexing_map = Parse("(d0) -> (d0), domain: d0 in [0, 49]");
  auto known_empty = Parse("(d0) -> (d0), domain: d0 in [0, -1]");
  EXPECT_THAT(known_empty, MatchIndexingMap("KNOWN EMPTY"));
  EXPECT_THAT(indexing_map * known_empty, MatchIndexingMap("KNOWN EMPTY"));
  EXPECT_THAT(known_empty * indexing_map, MatchIndexingMap("KNOWN EMPTY"));
  EXPECT_EQ((indexing_map * known_empty).GetAffineMap().getNumResults(), 1);
  EXPECT_EQ((known_empty * indexing_map).GetAffineMap().getNumResults(), 1);
}

TEST_F(IndexingMapTest,
       KnownEmpty_AddingConstraintOutOfRangeAfterSimplification) {
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (d1, d0, s1),
    domain:
    d0 in [0, 49],
    d1 in [0, 59],
    s0 in [0, 69],
    s1 in [0, 19],
    s1 floordiv 20 in [2, 2]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(indexing_map, MatchIndexingMap("KNOWN EMPTY"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintsWithManySymbols) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42),
    domain:
    d0 in [0, 31],
    s0 in [0, 0],
    s1 in [0, 1],
    s2 in [0, 2],
    s3 in [0, 3],
    s4 in [0, 4],
    d0 * 4 + s1 + s3 in [24, 459]
  )");
  indexing_map.RemoveUnusedSymbols();
  // Symbols s0, s2, s4 will be removed and s1 and s3 will become s0 and s1.
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                              (d0)[s0, s1] -> (d0 * 4 + s0 + s1 - 42),
                              domain:
                              d0 in [0, 31],
                              s0 in [0, 1],
                              s1 in [0, 3],
                              d0 * 4 + s0 + s1 in [24, 459]
                            )"));
}

TEST_F(IndexingMapTest, RemoveUnusedSymbols_ConstraintsWithRTVars) {
  IndexingMap indexing_map(
      ParseAffineMap("(d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42)",
                     &mlir_context_),
      {IndexingMap::Variable{{0, 31}}},
      {IndexingMap::Variable{{0, 0}}, IndexingMap::Variable{{0, 1}},
       IndexingMap::Variable{{0, 2}}},
      {IndexingMap::Variable{Interval{0, 3}},
       IndexingMap::Variable{Interval{0, 4}}});
  indexing_map.AddConstraint(
      ParseAffineExpr("d0 * 4 + s1 + s3", &mlir_context_), Interval{24, 459});
  indexing_map.RemoveUnusedSymbols();
  // Symbols s0, s2, s4 will be removed and s1 and s3 will become s0 and s1.
  EXPECT_THAT(indexing_map, MatchIndexingMap(R"(
                              (d0)[s0]{rt0} -> (d0 * 4 + s0 + rt0 - 42),
                              domain:
                              d0 in [0, 31],
                              s0 in [0, 1],
                              rt0 in [0, 3],
                              d0 * 4 + s0 + rt0 in [24, 459]
                            )"));
};

TEST_F(IndexingMapTest, ConvertSymbolsToDimensions) {
  IndexingMap indexing_map(
      ParseAffineMap(
          "(d0)[s0, s1, s2, s3] -> (d0 * 4 + s0 + s1 + 2 * s2 + 3 * s3 - 42)",
          &mlir_context_),
      {IndexingMap::Variable{{0, 31}}},
      {IndexingMap::Variable{{0, 0}}, IndexingMap::Variable{{0, 1}}},
      {IndexingMap::Variable{Interval{0, 3}},
       IndexingMap::Variable{Interval{0, 4}}});
  indexing_map.AddConstraint(
      ParseAffineExpr("d0 * 4 + s0 + 2 * s2", &mlir_context_),
      Interval{24, 459});
  EXPECT_THAT(indexing_map.ConvertSymbolsToDimensions(), MatchIndexingMap(R"(
      (d0, d1, d2, d3, d4) -> (d0 * 4 + d1 + d2 + d3 * 2 + d4 * 3 - 42),
      domain:
      d0 in [0, 31],
      d1 in [0, 0],
      d2 in [0, 1],
      d3 in [0, 3],
      d4 in [0, 4],
      d0 * 4 + d1 + d3 * 2 in [24, 459]
    )"));
}

TEST_F(IndexingMapTest, ConstraintIntervalSimplification_Sum) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, 99],
    d0 mod 8 + 5 in [50, 54]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0) -> (d0),
                          domain:
                          d0 in [0, 99],
                          d0 mod 8 in [45, 49]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_Sum_IndependentOfSymbol) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1] -> (d0 * 6 + s0 * 3 + s1),
    domain:
    d0 in [0, 1999],
    s0 in [0, 1],
    s1 in [0, 2],
    d0 * 6 + s0 * 3 + s1 in [0, 599]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0, s1] -> (d0 * 6 + s0 * 3 + s1),
                          domain:
                          d0 in [0, 99],
                          s0 in [0, 1],
                          s1 in [0, 2]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_Sum_NotIndependentOfSymbol) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1] -> (d0 * 6 + s0 * 3 + s1),
    domain:
    d0 in [0, 1999],
    s0 in [0, 1],
    s1 in [0, 2],
    d0 * 6 + s0 * 3 + s1 in [0, 598]
  )");
  EXPECT_FALSE(indexing_map.Simplify());
}

TEST_F(IndexingMapTest, ConstraintIntervalSimplification_Sum_GcdGreaterOne) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0 * 6 + s0 * 3),
    domain:
    d0 in [0, 1999],
    s0 in [0, 1],
    d0 * 6 + s0 * 3 in [0, 599]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0] -> (d0 * 6 + s0 * 3),
                          domain:
                          d0 in [0, 99],
                          s0 in [0, 1]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivPositiveDivisorPositiveBounds) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, 99],
    d0 floordiv 8 in [5, 11]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0) -> (d0),
                          domain:
                          d0 in [40, 95]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivPositiveDivisorNegativeBounds) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0),
    domain:
    d0 in [0, 99],
    s0 in [-99, 99],
    s0 floordiv 3 in [-11, -5]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0] -> (d0),
                          domain:
                          d0 in [0, 99],
                          s0 in [-33, -13]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_FloorDivNegativeDivisorNegativeBounds) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0),
    domain:
    d0 in [0, 99],
    s0 in [-99, 99],
    s0 floordiv -3 in [-11, -5]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0] -> (d0),
                          domain:
                          d0 in [0, 99],
                          s0 in [15, 35]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulPositiveMultiplierPositiveBounds) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [0, 99],
    d0 * 8 in [14, 33]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0) -> (d0),
                          domain:
                          d0 in [2, 4]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulPositiveMultiplierNegativeBounds) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0),
    domain:
    d0 in [0, 99],
    s0 in [-99, 99],
    s0 * 3 in [-11, -5]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0] -> (d0),
                          domain:
                          d0 in [0, 99],
                          s0 in [-3, -2]
                        )"));
}

TEST_F(IndexingMapTest,
       ConstraintIntervalSimplification_MulNegativeMultiplierNegativeBounds) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0),
    domain:
    d0 in [0, 99],
    s0 in [-99, 99],
    s0 * -3 in [-11, -5]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0] -> (d0),
                          domain:
                          d0 in [0, 99],
                          s0 in [2, 3]
                        )"));
}

TEST_F(IndexingMapTest, ConstraintMerge_Mod) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1] -> (d0, s1, s0),
    domain:
    d0 in [0, 3],
    s0 in [-21, -2],
    s1 in [0, 10],
    d0 mod 3 in [0, 0],
    s0 mod 2 in [0, 0],
    s0 mod 3 in [0, 0],
    s1 mod 5 in [1, 1]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                          (d0)[s0, s1] -> (d0, s1, s0),
                          domain:
                          d0 in [0, 3],
                          s0 in [-18, -6],
                          s1 in [1, 6],
                          d0 mod 3 in [0, 0],
                          s0 mod 6 in [0, 0],
                          s1 mod 5 in [1, 1]
                        )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ConstantDims) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0),
    domain:
    d0 in [5, 5]
  )");
  EXPECT_FALSE(
      indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve));
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                  (d0) -> (5),
                                                  domain:
                                                  d0 in [5, 5]
                                                )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SumOrderRegression) {
  // This is a regression test for a bug where we didn't canonicalize the order
  // of summands correctly, leading to `Simplify` not being idempotent.
  auto indexing_map = Parse(R"(
    (d0, d1)[s0, s1] -> (((((d0 + (d0 mod 3)) floordiv 3)
      + (s0 + ((s0 + s0) mod 3))) + (((d0 + s0) mod 3) + 0))),
    domain:
    d0 in [0, 9],
    d1 in [0, 19],
    s0 in [0, 29],
    s1 in [0, 39]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_FALSE(indexing_map.Simplify());
}

TEST_F(IndexingMapTest, AffineMapSimplification_SumOrderRegression2) {
  // This is a regression test for a bug where we didn't simplify the affine
  // expression fully after a single iteration.
  auto indexing_map = Parse(R"(
    (d0)[s0] -> ((((s0 + d0) + d0) floordiv 2)),
    domain:
    d0 in [0, 9],
    s0 in [0, 19]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_FALSE(indexing_map.Simplify());
}

TEST_F(IndexingMapTest, AffineMapSimplification_FloorDivRegression) {
  auto indexing_map = Parse(R"(
    (d0, d1) -> (((d0 floordiv 3) * 3 + d1 floordiv 2) floordiv 6),
    domain:
    d0 in [0, 11],
    d1 in [0, 5]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 (d0, d1) -> (d0 floordiv 6),
                                                 domain:
                                                 d0 in [0, 11],
                                                 d1 in [0, 5]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ModIsSub) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0 mod 42),
    domain:
    d0 in [53, 71]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 (d0) -> (d0 - 42),
                                                 domain:
                                                 d0 in [53, 71]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ModIsAdd) {
  auto indexing_map = Parse(R"(
    (d0) -> (d0 mod 5),
    domain:
    d0 in [-5, -1]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 (d0) -> (d0 + 5),
                                                 domain:
                                                 d0 in [-5, -1]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_ModIsNotAdd) {
  auto indexing_map1 = Parse("(d0) -> (d0 mod 5), domain: d0 in [-4, 0]");
  EXPECT_FALSE(indexing_map1.Simplify());
  auto indexing_map2 = Parse("(d0) -> (d0 mod 5), domain: d0 in [-6, -1]");
  EXPECT_FALSE(indexing_map2.Simplify());
}

TEST_F(IndexingMapTest, AffineMapSimplification_SubIsMod) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0 - (s0 floordiv 3) * 3 + s0),
    domain:
    d0 in [0, 1],
    s0 in [0, 3]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 (d0)[s0] -> (d0 + s0 mod 3),
                                                 domain:
                                                 d0 in [0, 1],
                                                 s0 in [0, 3]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SubIsModMultiplied) {
  auto indexing_map = Parse(R"(
    (d0)[s0] -> (d0 - (s0 floordiv 3) * 12 + s0 * 7),
    domain:
    d0 in [0, 1],
    s0 in [0, 3]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                (d0)[s0] -> (d0 + (s0 mod 3) * 4 + s0 * 3),
                domain:
                d0 in [0, 1],
                s0 in [0, 3]
              )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SubIsModSum) {
  auto indexing_map = Parse(R"(
    (d0)[s0] ->  (1 + d0 - ((s0 + 1) floordiv 3) * 3 + s0),
    domain:
    d0 in [0, 1],
    s0 in [0, 3]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                (d0)[s0] -> (d0 + (s0 + 1) mod 3),
                domain:
                d0 in [0, 1],
                s0 in [0, 3]
              )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_DivsAndModsIfSmallerThanDivisor) {
  auto indexing_map = Parse(R"(
    (d0, d1) -> (d0 + d1 floordiv 16, d1 mod 16),
    domain:
    d0 in [0, 7],
    d1 in [0, 15]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                  (d0, d1) -> (d0, d1),
                                                  domain:
                                                  d0 in [0, 7],
                                                  d1 in [0, 15]
                                                )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsAndModsWithMultipliers) {
  auto indexing_map = Parse(R"(
    (d0, d1, d2) -> ((d0 * 100 + d1 * 10 + d2) floordiv 100,
                     ((d0 * 100 + d1 * 10 + d2) mod 100) floordiv 10,
                     d2 mod 10),
    domain:
    d0 in [0, 8],
    d1 in [0, 8],
    d2 in [0, 8]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                  (d0, d1, d2) -> (d0, d1, d2),
                                                  domain:
                                                  d0 in [0, 8],
                                                  d1 in [0, 8],
                                                  d2 in [0, 8]
                                                )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_DivsAndModsWithDivisibleMultipliers) {
  auto indexing_map = Parse(R"(
    (d0, d1, d2) -> ((d0 * 16 + d1 * 4 + d2) floordiv 8,
                     (d0 * 16 + d1 * 4 + d2) mod 8),
    domain:
    d0 in [0, 9],
    d1 in [0, 9],
    d2 in [0, 9]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
    (d0, d1, d2) -> (d0 * 2 + (d1 * 4 + d2) floordiv 8,
                     (d1 * 4 + d2) mod 8),
    domain:
    d0 in [0, 9],
    d1 in [0, 9],
    d2 in [0, 9]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsAndModsWithReverse) {
  auto indexing_map = Parse(R"(
    (d0, d1) -> (-((d0 * -11 - d1 + 109) floordiv 11) + 9,
                 d0 * 11 + d1 + ((d0 * -11 - d1 + 109) floordiv 11) * 11 - 99),
    domain:
    d0 in [0, 7],
    d1 in [0, 8]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 (d0, d1) -> (d0, d1),
                                                 domain:
                                                 d0 in [0, 7],
                                                 d1 in [0, 8]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape) {
  auto indexing_map = Parse(R"(
    ()[s0] -> ((s0 * 128) mod 715 + ((s0 * 128) floordiv 715) * 715),
    domain:
    s0 in [0, 127]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0] -> (s0 * 128),
      domain:
      s0 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape2) {
  auto indexing_map = Parse(R"(
    (d0, d1) -> ((d0 mod 8) * 128 + d1 + (d0 floordiv 8) * 1024),
    domain:
    d0 in [0, 1023],
    d1 in [0, 127]
  )");
  ;
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0, d1) -> (d0 * 128 + d1),
      domain:
      d0 in [0, 1023],
      d1 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape3) {
  auto indexing_map = Parse(R"(
    (d0, d1) -> (((d1 * 2 + d0 floordiv 64) mod 3) * 256 + (d0 mod 64) * 4
      + ((d1 * 128 + d0) floordiv 192) * 768),
    domain:
    d0 in [0, 127],
    d1 in [0, 3071]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0, d1) -> (d0 * 4 + d1 * 512),
      domain:
      d0 in [0, 127],
      d1 in [0, 3071]
  )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_ModWithNegativeMultiplerDoesNotGetSimplified) {
  auto indexing_map = Parse(R"(
    (d0) -> ((-d0) mod 2),
    domain:
    d0 in [0, 127]
  )");
  EXPECT_FALSE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0) -> ((-d0) mod 2),
      domain:
      d0 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyBitcastAndBack) {
  // `d0 floordiv 1536` is the result of simplifying this:
  // `((d0 * 2 + d1 floordiv 64) floordiv 3) floordiv 1024`.
  // This test verifies that we can still simplify the map after the
  // simplification of the floordiv.
  auto indexing_map = Parse(R"(
    (d0, d1) -> ((d0 floordiv 1536) * 786432
      + (((d0 * 2 + d1 floordiv 64) floordiv 3) mod 1024) * 768
      + ((d0 * 2 + d1 floordiv 64) mod 3) * 256 + (d1 mod 64) * 4),
    domain:
    d0 in [0, 3071],
    d1 in [0, 127]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0, d1) -> (d0 * 512 + d1 * 4),
      domain:
      d0 in [0, 3071],
      d1 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_SimplifyReshape_Regression) {
  // We have s0 * 128 in the mod, but s0 * 64 in the floordiv *.
  auto indexing_map = Parse(R"(
    ()[s0] -> ((s0 * 128) mod 715 + ((s0 * 64) floordiv 715) * 715),
    domain:
    s0 in [0, 127]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0] -> (((s0 * 64) floordiv 715) * 715 + (s0 * 128) mod 715),
      domain:
      s0 in [0, 127]
  )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivsInSequence) {
  auto indexing_map = Parse(R"(
    ()[s0] -> (s0 - ((s0 floordiv 2) floordiv 7) * 14 + (s0 floordiv 14) * 14),
    domain:
    s0 in [0, 1233]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
                                                 ()[s0] -> (s0),
                                                 domain:
                                                 s0 in [0, 1233]
                                               )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivDiv) {
  auto indexing_map = Parse(R"(
    ()[s0, s1] -> ((s0 * 2 + s1 floordiv 64) floordiv 3),
    domain:
    s0 in [0, 1233],
    s1 in [0, 127]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0, s1] -> ((s0 * 128 + s1) floordiv 192),
      domain:
      s0 in [0, 1233],
      s1 in [0, 127]
    )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivSumConstant) {
  auto indexing_map = Parse(R"(
    ()[s0] -> ((s0 * 6 + 9) floordiv 18),
    domain:
    s0 in [0, 1233]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0] -> ((s0 * 2 + 3) floordiv 6),
      domain:
      s0 in [0, 1233]
    )"));
}

TEST_F(IndexingMapTest, AffineMapSimplification_DivSumDiv) {
  auto indexing_map = Parse(R"(
    ()[s0, s1] -> ((s0 floordiv 3 + s1 floordiv 3) floordiv 6),
    domain:
    s0 in [0, 1233],
    s1 in [0, 127]
  )");
  // The rewrite tested in AffineMapSimplification_DivDiv must not trigger here.
  EXPECT_FALSE(indexing_map.Simplify());
}

TEST_F(IndexingMapTest, AffineMapSimplification_NegativeDiv) {
  // (s0 floordiv 2) floordiv -7 is not s0 floordiv -14:
  // 15 // 2 // -7 = -1
  // 15 // -14 = -2
  auto indexing_map = Parse(R"(
    ()[s0] -> ((s0 floordiv 2) floordiv -7),
    domain:
    s0 in [0, 1233]
  )");
  EXPECT_FALSE(indexing_map.Simplify());
}

TEST_F(IndexingMapTest, AffineMapSimplification_ExtractFromMod) {
  auto indexing_map = Parse(R"(
    ()[s0, s1, s2, s3] -> ((s0 * 458752 + s1 + s2 * 4 + s3 * 512) mod 20000),
    domain:
    s0 in [0, 871],
    s1 in [0, 3],
    s2 in [0, 127],
    s3 in [0, 895]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0, s1, s2, s3] -> (
        ((s0 * 114688 + s3 * 128 + s2) mod 5000) * 4 + s1
      ),
      domain:
      s0 in [0, 871],
      s1 in [0, 3],
      s2 in [0, 127],
      s3 in [0, 895]
    )"));
}

TEST_F(IndexingMapTest,
       AffineMapSimplification_ExtractFromDiv_NegativeMultiplier) {
  auto indexing_map = Parse(R"(
    ()[s0, s1] -> ((s0 * 16 - (s1 floordiv 4) floordiv 2 + (s1 floordiv 8) * 2)
      floordiv 4),
    domain:
    s0 in [0, 1],
    s1 in [0, 127]
  )");
  EXPECT_TRUE(indexing_map.Simplify());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      ()[s0, s1] -> (
        s0 * 4 + s1 floordiv 32
      ),
      domain:
      s0 in [0, 1],
      s1 in [0, 127]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_Simple) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s1, s0 floordiv 6),
    domain:
    d0 in [0, 3],
    s0 in [0, 6],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 mod 6 in [0, 0]
  )");
  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
      domain:
        d0 in [0, 3],
        s0 in [0, 1],
        s1 in [0, 1],
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_WithShift) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
    domain:
    d0 in [0, 3],
    s0 in [0, 41],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 mod 6 in [3, 3]
  )");
  // [BEFORE] Allowed values for s0: 3, 9, 15, ..., 39 = (6 * 6 + 3)
  // [AFTER] Allowed values for s0: 0, 1, 2, ..., 6
  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0 * 6 + 3),
      domain:
        d0 in [0, 3],
        s0 in [0, 6],
        s1 in [0, 1],
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_TwoModConstraints) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s1, s0 floordiv 6),
    domain:
    d0 in [0, 3],
    s0 in [0, 7],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 mod 2 in [0, 0],
    s0 mod 3 in [0, 0]
  )");
  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
      domain:
        d0 in [0, 3],
        s0 in [0, 1],
        s1 in [0, 1],
        s2 in [0, 5]
    )"));
}

TEST_F(IndexingMapTest, RescaleSymbols_RescaledSymbolInOtherNonModConstraint) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
    domain:
    d0 in [0, 3],
    s0 in [0, 9],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 * s2 in [0, 28],
    s0 mod 6 in [3, 3]
  )");
  EXPECT_TRUE(indexing_map.RescaleSymbols());
  EXPECT_THAT(ToString(indexing_map), MatchIndexingString(R"(
      (d0)[s0, s1, s2] -> (s2, d0, s1, s0 * 6 + 3),
      domain:
        d0 in [0, 3],
        s0 in [0, 1],
        s1 in [0, 1],
        s2 in [0, 5],
        (s0 * 6 + 3) * s2 in [0, 28]
    )"));
}

TEST_F(IndexingMapTest,
       RescaleSymbols_TwoModConstraintsForTheSameSymbolWhichCannotBeMerged) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s1, s0),
    domain:
    d0 in [0, 3],
    s0 in [0, 99],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 mod 6 in [3, 3],
    s0 mod 7 in [5, 5]
  )");
  EXPECT_TRUE(indexing_map.RescaleSymbols());

  const mlir::AffineExpr result3 = indexing_map.GetAffineMap().getResult(3);
  ASSERT_THAT(indexing_map.GetConstraints(), ::testing::SizeIs(1));
  const mlir::AffineExpr constraint_expr =
      indexing_map.GetConstraints().begin()->first;
  const Interval constraint_interval =
      indexing_map.GetConstraints().begin()->second;

  // TODO(b/347240603): This case is not yet fully supported, because the
  // resulting indexing map depends on the hashmap iteration order, so it can
  // have different values randomly. Also the range of s0 can depend on the
  // iteration order and how many times we simplify. Maybe this case is not so
  // important for now.
  EXPECT_THAT(
      std::make_tuple(result3, constraint_expr, constraint_interval),
      AnyOf(
          std::make_tuple(ParseAffineExpr("s0 * 6 + 3", &mlir_context_),
                          ParseAffineExpr("(s0 * 6 + 3) mod 7", &mlir_context_),
                          Interval{5, 5}),
          std::make_tuple(ParseAffineExpr("s0 * 7 + 5", &mlir_context_),
                          ParseAffineExpr("(s0 * 7 + 5) mod 6", &mlir_context_),
                          Interval{3, 3})));
}

TEST_F(IndexingMapTest, RescaleSymbolsKeepsHashmapConsistent) {
  auto indexing_map = Parse(R"(
    (d0)[s0, s1, s2] -> (s2, d0, s0, s0 floordiv 6),
    domain:
    d0 in [0, 3],
    s0 in [0, 6],
    s1 in [0, 1],
    s2 in [0, 5],
    s0 mod 6 in [0, 0],
    s0 * s1 in [0, 100]
  )");
  EXPECT_TRUE(indexing_map.RescaleSymbols());

  for (auto& [expr, interval] : indexing_map.GetConstraints()) {
    EXPECT_TRUE(indexing_map.GetConstraints().contains(expr))
        << "Don't modify the *keys* of the hashmap.";
  }
}

TEST_F(IndexingMapTest, RangeEvaluatorTest) {
  auto indexing_map = Parse(R"(
    (d0, d1, d2, d3)[] -> (0),
    domain:
    d0 in [0, 9],
    d1 in [-10, -1],
    d2 in [-1, 2],
    d3 in [0, 0]
  )");
  RangeEvaluator range_evaluator(indexing_map, &mlir_context_);
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

TEST(IntervalComparisonTest, PointComparisons) {
  Interval interval{12, 64};
  auto point = [](int64_t n) { return Interval{n, n}; };
  EXPECT_EQ(interval.Gt(point(11)), true);
  EXPECT_EQ(interval.Gt(point(12)), std::nullopt);
  EXPECT_EQ(interval.Gt(point(65)), false);

  EXPECT_EQ(interval.Lt(point(65)), true);
  EXPECT_EQ(interval.Lt(point(64)), std::nullopt);
  EXPECT_EQ(interval.Lt(point(10)), false);

  EXPECT_EQ(interval.Eq(point(11)), false);
  EXPECT_EQ(interval.Eq(point(12)), std::nullopt);
  EXPECT_EQ(interval.Eq(point(15)), std::nullopt);
  EXPECT_EQ(interval.Eq(point(65)), false);

  EXPECT_EQ(interval.Ne(point(11)), true);
  EXPECT_EQ(interval.Ne(point(15)), std::nullopt);
  EXPECT_EQ(interval.Ne(point(65)), true);

  EXPECT_EQ(interval.Ge(point(12)), true);
  EXPECT_EQ(interval.Ge(point(64)), std::nullopt);
  EXPECT_EQ(interval.Ge(point(65)), false);

  EXPECT_EQ(interval.Le(point(11)), false);
  EXPECT_EQ(interval.Le(point(64)), true);
  EXPECT_EQ(interval.Le(point(63)), std::nullopt);
  EXPECT_EQ(interval.Le(point(65)), true);

  EXPECT_EQ(point(15).Eq(point(15)), true);
  EXPECT_EQ(point(15).Eq(point(16)), false);

  EXPECT_EQ(point(15).Ne(point(15)), false);
  EXPECT_EQ(point(15).Ne(point(16)), true);
}

TEST(IntervalComparisonTest, RangeComparisons) {
  Interval interval{12, 64};
  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };
  EXPECT_EQ(interval.Gt(range(-10, 11)), true);
  EXPECT_EQ(interval.Gt(range(-10, 12)), std::nullopt);
  EXPECT_EQ(interval.Gt(interval), std::nullopt);
  EXPECT_EQ(interval.Gt(range(10, 20)), std::nullopt);
  EXPECT_EQ(interval.Gt(range(50, 60)), std::nullopt);
  EXPECT_EQ(interval.Gt(range(64, 100)), false);
  EXPECT_EQ(interval.Gt(range(65, 100)), false);

  EXPECT_EQ(interval.Lt(range(65, 100)), true);
  EXPECT_EQ(interval.Lt(range(64, 100)), std::nullopt);
  EXPECT_EQ(interval.Lt(interval), std::nullopt);
  EXPECT_EQ(interval.Lt(range(50, 60)), std::nullopt);
  EXPECT_EQ(interval.Lt(range(10, 20)), std::nullopt);
  EXPECT_EQ(interval.Lt(range(-10, 12)), false);
  EXPECT_EQ(interval.Lt(range(-10, 11)), false);

  EXPECT_EQ(interval.Eq(interval), std::nullopt);
  EXPECT_EQ(interval.Eq(range(65, 100)), false);
  EXPECT_EQ(interval.Eq(range(0, 11)), false);
}

MATCHER_P(IntervalIs, interval, "") {
  std::pair<int64_t, int64_t> arg_pair{arg.lower, arg.upper};
  return ::testing::ExplainMatchResult(
      ::testing::Pair(interval.lower, interval.upper), arg_pair,
      result_listener);
}

TEST(IntervalMathTest, Addition) {
  Interval a{12, 64};
  Interval b{-100, 120};
  Interval sum{12 - 100, 64 + 120};
  EXPECT_THAT(a + b, IntervalIs(sum));
}

TEST(IntervalMathTest, AdditionSaturating) {
  Interval a{12, 64};
  Interval b{-100, 120};
  Interval c{100, std::numeric_limits<int64_t>::max() - 80};
  Interval any{std::numeric_limits<int64_t>::min(),
               std::numeric_limits<int64_t>::max()};
  Interval positive{0, std::numeric_limits<int64_t>::max()};
  Interval negative{std::numeric_limits<int64_t>::min(), 0};
  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };

  EXPECT_THAT(positive + negative, IntervalIs(any));
  EXPECT_THAT(any + any, IntervalIs(any));
  EXPECT_THAT(b + any, IntervalIs(any));

  EXPECT_THAT(c + any, IntervalIs(any));
  EXPECT_THAT(c + positive,
              IntervalIs(range(100, std::numeric_limits<int64_t>::max())));
  Interval c_plus_negative{negative.lower, c.upper};
  EXPECT_THAT(c + negative, IntervalIs(c_plus_negative));

  Interval a_plus_c{112, std::numeric_limits<int64_t>::max() - 16};
  EXPECT_THAT(a + c, IntervalIs(a_plus_c));
  Interval b_plus_c{0, std::numeric_limits<int64_t>::max()};
  EXPECT_THAT(b + c, IntervalIs(b_plus_c));
}

TEST(IntervalMathTest, Multiplication) {
  Interval pos{10, 100};
  Interval neg{-10, -1};
  Interval both_small{-5, 6};
  Interval both_large{-20, 1000};

  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };
  EXPECT_THAT(pos * neg, IntervalIs(range(-1000, -10)));
  EXPECT_THAT(pos * both_small, IntervalIs(range(-500, 600)));
  EXPECT_THAT(pos * both_large, IntervalIs(range(-2000, 100000)));
  EXPECT_THAT(neg * both_small, IntervalIs(range(-60, 50)));
  EXPECT_THAT(neg * both_large, IntervalIs(range(-10000, 200)));
  EXPECT_THAT(both_small * both_large, IntervalIs(range(-5000, 6000)));
}

TEST(IntervalMathTest, MultiplicationSaturating) {
  Interval any{std::numeric_limits<int64_t>::min(),
               std::numeric_limits<int64_t>::max()};
  Interval bit33{42, std::numeric_limits<uint32_t>::max()};
  Interval bit33_sq{42 * 42, std::numeric_limits<int64_t>::max()};
  EXPECT_THAT(bit33 * bit33, IntervalIs(bit33_sq));
  EXPECT_THAT(any * any, IntervalIs(any));

  Interval greater_41{42, std::numeric_limits<int64_t>::max()};
  Interval neg_one{-1, -1};
  Interval less_neg_41{std::numeric_limits<int64_t>::min(), -42};
  EXPECT_THAT(greater_41 * neg_one, IntervalIs(less_neg_41));
  EXPECT_THAT(less_neg_41 * neg_one, IntervalIs(greater_41));
  EXPECT_THAT(any * neg_one, IntervalIs(any));
}

template <typename T>
void ExpectSupportsAbslHashAndEqAndNe(absl::Span<const T> values) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(values));

  // C++20 compilers automatically generate != from ==, but XLA has to work with
  // C++17, so we test that we explicitly implemented !=. Otherwise it could
  // happen that some compilers can compile XLA, and some can't.
  for (const T& a : values) {
    for (const T& b : values) {
      EXPECT_EQ(a != b, !(a == b));
    }
  }
}

TEST_F(IndexingMapTest, IntervalSupportsAbslHashAndEqAndNe) {
  ExpectSupportsAbslHashAndEqAndNe<Interval>(
      {Interval{1, 1}, Interval{0, 1}, Interval{1, 2}});
}

TEST_F(IndexingMapTest, IntervalSupportsLlvmStyleHashingAndEqAndNe) {
  auto check_consistent = [](const Interval& a, const Interval& b) {
    if (a == b) {
      EXPECT_EQ(hash_value(a), hash_value(b));
    }
    if (hash_value(a) != hash_value(b)) {
      EXPECT_NE(a, b);
    }
    // Some LLVM containers use "!=".
    EXPECT_EQ(a != b, !(a == b));
  };

  std::vector<Interval> intervals = {Interval{1, 1}, Interval{0, 1},
                                     Interval{1, 2}};
  for (const auto& a : intervals) {
    for (const auto& b : intervals) {
      check_consistent(a, b);
    }
  }
}

TEST_F(IndexingMapTest, DimVarSupportsAbslHashAndEqAndNe) {
  ExpectSupportsAbslHashAndEqAndNe<IndexingMap::Variable>(
      {IndexingMap::Variable{1, 1}, IndexingMap::Variable{0, 1},
       IndexingMap::Variable{1, 2}});
}

TEST_F(IndexingMapTest, RangeVarSupportsAbslHashAndEqAndNe) {
  ExpectSupportsAbslHashAndEqAndNe<IndexingMap::Variable>(
      {IndexingMap::Variable{1, 1}, IndexingMap::Variable{0, 1},
       IndexingMap::Variable{1, 2}});
}

TEST_F(IndexingMapTest, RTVarSupportsAbslHashAndEqAndNe) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(R"(
                            HloModule m
                            ENTRY e {
                              ROOT %constant = s64[] constant(42)
                            }
                          )"));
  ASSERT_NE(hlo_module, nullptr);

  ExpectSupportsAbslHashAndEqAndNe<IndexingMap::Variable>(
      {IndexingMap::Variable{Interval{1, 1}},
       IndexingMap::Variable{Interval{1, 2}},
       IndexingMap::Variable{Interval{1, 2}},
       IndexingMap::Variable{Interval{1, 2}}});
}

TEST_F(IndexingMapTest, IndexingMapSupportsAbslHashAndEqAndNe) {
  ExpectSupportsAbslHashAndEqAndNe<IndexingMap>(
      {Parse(R"(
        (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
        domain:
        d0 in [0, 49],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79]
       )"),
       Parse(R"(
        (d0, d1)[s0, s1] -> (d1 * 2, d0, s1, s0),
        domain:
        d0 in [0, 49],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79]
       )"),
       Parse(R"(
        (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
        domain:
        d0 in [0, 50],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79]
       )"),
       Parse(R"(
        (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
        domain:
        d0 in [0, 49],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79]
       )"),
       Parse(R"(
        (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
        domain:
        d0 in [0, 49],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79],
        d0 mod 8 in [0, 0],
        d0 mod 16 in [0, 0]
       )"),
       Parse(R"(
        (d0, d1)[s0, s1] -> (d1, d0, s1, s0),
        domain:
        d0 in [0, 49],
        d1 in [0, 59],
        s0 in [0, 69],
        s1 in [0, 79],
        d0 mod 8 in [0, 0],
        d0 mod 32 in [0, 0]
      )"),
       IndexingMap(
           ParseAffineMap("(d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42)",
                          &mlir_context_),
           {IndexingMap::Variable{{0, 31}}},
           {IndexingMap::Variable{{0, 0}}, IndexingMap::Variable{{0, 1}},
            IndexingMap::Variable{{0, 2}}},
           {IndexingMap::Variable{Interval{0, 3}},
            IndexingMap::Variable{Interval{0, 4}}}),
       IndexingMap(
           ParseAffineMap("(d0)[s0, s1, s2, s3, s4] -> (d0 * 4 + s1 + s3 - 42)",
                          &mlir_context_),
           {IndexingMap::Variable{{0, 31}}},
           {IndexingMap::Variable{{0, 0}}, IndexingMap::Variable{{0, 1}},
            IndexingMap::Variable{{0, 2}}},
           {IndexingMap::Variable{Interval{0, 3}},
            IndexingMap::Variable{Interval{0, 5}}})});
}

TEST_F(IndexingMapTest, ConvertRangeVariablesToDimensions) {
  IndexingMap indexing_map = Parse(R"(
     (d0, d1)[to_convert_0, range, to_convert_1]
       -> (d1, d0, range + to_convert_1, to_convert_0),
     domain:
     d0 in [0, 3],
     d1 in [0, 3],
     to_convert_0 in [0, 2],
     range in [0, 1],
     to_convert_1 in [0, 3]
  )");
  EXPECT_THAT(ConvertRangeVariablesToDimensions(indexing_map, {0, 2}),
              MatchIndexingMap(R"(
     (d0, d1, to_convert_0, to_convert_1)[range]
       -> (d1, d0, to_convert_1 + range, to_convert_0),
     domain:
     d0 in [0, 3],
     d1 in [0, 3],
     to_convert_0 in [0, 2],
     to_convert_1 in [0, 3],
     range in [0, 1]
  )"));
}

}  // namespace
}  // namespace xla

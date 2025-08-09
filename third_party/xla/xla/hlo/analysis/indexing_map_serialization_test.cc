/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/analysis/indexing_map_serialization.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class IndexingMapSerializationTest : public HloHardwareIndependentTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  void ParseAndCheck(absl::string_view indexing_map_str) {
    auto indexing_map = ParseIndexingMap(indexing_map_str, &mlir_context_);
    ASSERT_TRUE(indexing_map.has_value());
    EXPECT_THAT(ToString(*indexing_map), MatchIndexingString(indexing_map_str));
  }
};

TEST_F(IndexingMapSerializationTest, EmptyMap) { ParseAndCheck("() -> ()"); }

TEST_F(IndexingMapSerializationTest, UndefinedMap) {
  EXPECT_THAT(ToString(IndexingMap::GetUndefined()),
              MatchIndexingString("UNDEFINED"));
}

TEST_F(IndexingMapSerializationTest, KnownEmptyMap) {
  auto map =
      ParseIndexingMap("(d0) -> (), domain: d0 in [1, 0]", &mlir_context_);
  EXPECT_TRUE(map->IsKnownEmpty());
  EXPECT_THAT(ToString(*map), MatchIndexingString("KNOWN EMPTY"));
}

TEST_F(IndexingMapSerializationTest, DimsOnly) {
  ParseAndCheck(R"(
    (d0, d1) -> (d0 mod 2 + d1),
    domain:
    d0 in [0, 3],
    d1 in [-4, 4]
  )");
}

TEST_F(IndexingMapSerializationTest, SymbolsOnly) {
  ParseAndCheck(R"(
    ()[s0, s1] -> (s0 floordiv s1),
    domain:
    s0 in [0, 3],
    s1 in [0, 4]
  )");
}

TEST_F(IndexingMapSerializationTest, RuntimeOnly) {
  ParseAndCheck(R"(
    (){r0, r1} -> (r0),
    domain:
    r0 in [1, 1],
    r1 in [0, 1]
  )");
}

TEST_F(IndexingMapSerializationTest, DimsAndRuntime) {
  ParseAndCheck(R"(
    (d0){r0, r1} -> (d0 floordiv r0 + r1),
    domain:
    d0 in [0, 3],
    r0 in [1, 1],
    r1 in [0, 1]
  )");
}

TEST_F(IndexingMapSerializationTest, DimsAndSymbols) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2] -> (s2, d0 + d1, s1, s0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3]
  )");
}

TEST_F(IndexingMapSerializationTest, DimsAndRanges) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2] -> (s2, d0 + d1, s1, s0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3]
  )");
}

TEST_F(IndexingMapSerializationTest, DimsRangesAndRuntime) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2]{r0} -> (s2, d0 + d1, s1, s0 * r0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3],
    r0 in [0, 4]
  )");
}

TEST_F(IndexingMapSerializationTest, DimRangesRuntimeAndConstraints) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2]{r0} -> (s2, d0 + d1, s1, s0, r0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3],
    r0 in [0, 100],
    (r0 + 1) mod 5 in [1, 1],
    d0 mod 4 in [0, 0],
    d1 + s0 in [0, 45]
  )");
}

TEST_F(IndexingMapSerializationTest, Int64Bounds) {
  ParseAndCheck(R"(
    (bl_x) -> (bl_x floordiv 100663296),
    domain:
    bl_x in [0, 2415919103]
  )");
}

TEST_F(IndexingMapSerializationTest, AffineExprsWithParens) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1] -> ((d0 + d0 mod 3) floordiv 3
      + s0 + (s0 * 2) mod 3 + (d0 + s0) mod 3),
    domain:
    d0 in [0, 9],
    d1 in [0, 19],
    s0 in [0, 29],
    s1 in [0, 39]
  )");
}

// This test will be updated when the printing uses types of variables.
TEST_F(IndexingMapSerializationTest, CustomNames) {
  ParseAndCheck(R"(
    (th_x, bl_x)[s0, vector_elem, s2]{gpu}
      -> (s2, th_x * gpu + bl_x, vector_elem, s0),
    domain:
    th_x in [0, 3],
    bl_x in [0, 4],
    s0 in [0, 1],
    vector_elem in [0, 1],
    s2 in [0, 3],
    gpu in [0, 1],
    (th_x * gpu) mod 4 in [0, 0],
    bl_x + s0 in [0, 45]
  )");
}

TEST_F(IndexingMapSerializationTest, AffineMapPrinterTest) {
  mlir::AffineExpr d0, d1, s0, s1, r0, r1;
  mlir::bindDims(&mlir_context_, d0, d1);
  mlir::bindSymbols(&mlir_context_, s0, s1, r0, r1);

  // (d0, d1)[s0, s1]{r0, r1} ->
  //   (d0 + d1 floordiv 8 - r0 * 64, s0 + s1 mod 16 + r1).
  auto map = mlir::AffineMap::get(
      2, 4, {d0 + d1.floorDiv(8) - r0 * 64, s0 + s1 % 16 + r1}, &mlir_context_);
  EXPECT_THAT(ToString(map, {"offset", "d1"}, {"s0", "linear_index"},
                       {"gpu_index", "r1"}),
              HasSubstr("(offset, d1)[s0, linear_index]{gpu_index, r1} -> "
                        "(offset + d1 floordiv 8 - gpu_index * 64, s0 + "
                        "linear_index mod 16 + r1)"));
  EXPECT_THAT(ToString(map),
              HasSubstr("(d0, d1)[s0, s1, s2, s3] -> "
                        "(d0 + d1 floordiv 8 - s2 * 64, s0 + s1 mod 16 + s3)"));
}

}  // namespace
}  // namespace xla

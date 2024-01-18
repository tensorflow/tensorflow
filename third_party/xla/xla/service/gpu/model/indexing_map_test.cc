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
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::bindDims;
using ::mlir::bindSymbols;
using ::mlir::getAffineDimExpr;
using ::testing::HasSubstr;

class IndexingMapTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter printer_;
};

TEST_F(IndexingMapTest, SimplifyConstantDims) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);

  auto map = AffineMap::get(1, 0, d0, &mlir_context_);
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 5, 5);

  EXPECT_THAT(printer_.ToString(simplifier.Simplify(map)),
              HasSubstr("(d0) -> (5)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsIfSmallerThanDivisor) {
  AffineExpr d0, d1;
  bindDims(&mlir_context_, d0, d1);

  // (d0, d1) -> (d0 + d1 floordiv 16, d1 mod 16).
  auto map =
      AffineMap::get(2, 0, {d0 + d1.floorDiv(16), d1 % 16}, &mlir_context_);

  // d0 in [0, 8) and d1 in [0, 16).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 7);
  simplifier.SetInclusiveBounds(d1, 0, 15);

  EXPECT_THAT(printer_.ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithMultipliers) {
  AffineExpr d0, d1, d2;
  bindDims(&mlir_context_, d0, d1, d2);

  //  (d0, d1, d2) -> ((d0 * 100 + d1 * 10 + d2) floordiv 100,
  //                  "((d0 * 100 + d1 * 10 + d2) mod 100) floordiv 10,
  //                    d2 mod 10)"
  AffineExpr weighted_sum = d0 * 100 + d1 * 10 + d2;
  auto map = AffineMap::get(
      3, 0,
      {weighted_sum.floorDiv(100), (weighted_sum % 100).floorDiv(10), d2 % 10},
      &mlir_context_);

  // d_i in [0, 10).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 9);
  simplifier.SetInclusiveBounds(d1, 0, 9);
  simplifier.SetInclusiveBounds(d2, 0, 9);

  EXPECT_THAT(printer_.ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1, d2) -> (d0, d1, d2)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithDivisibleMultipliers) {
  AffineExpr d0, d1, d2;
  bindDims(&mlir_context_, d0, d1, d2);

  // (d0, d1, d2) -> ((d0 * 16 + d1 * 4 + d2) floordiv 8, "
  //                  (d0 * 16 + d1 * 4 + d2) mod 8)
  AffineExpr weighted_sum = d0 * 16 + d1 * 4 + d2;
  auto map = AffineMap::get(3, 0, {weighted_sum.floorDiv(8), weighted_sum % 8},
                            &mlir_context_);

  // d_0 in [0, 10).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 1);
  simplifier.SetInclusiveBounds(d1, 0, 3);
  simplifier.SetInclusiveBounds(d2, 0, 3);

  EXPECT_THAT(printer_.ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1, d2) -> (d0 * 2 + (d1 * 4 + d2) floordiv 8, "
                        "(d1 * 4 + d2) mod 8)"));
}

TEST_F(IndexingMapTest, SimplifyDivsAndModsWithReverse) {
  AffineExpr d0, d1;
  bindDims(&mlir_context_, d0, d1);

  // (d0, d1) -> (-((d0 * -11 - d1 + 109) floordiv 11) + 9,
  //              d0 * 11 + d1 + ((d0 * -11 - d1 + 109) floordiv 11) * 11 - 99).
  AffineExpr weighted_sum = (-11 * d0 - d1 + 109).floorDiv(11);
  auto map = AffineMap::get(
      2, 0, {9 - weighted_sum, 11 * d0 + d1 + weighted_sum * 11 - 99},
      &mlir_context_);

  // d0 in [0, 10) and d1 in [0, 11).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 9);
  simplifier.SetInclusiveBounds(d1, 0, 10);

  EXPECT_THAT(printer_.ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

TEST_F(IndexingMapTest, AffineExprSignExtraction) {
  AffineExpr d0, d1, d2, d3;
  bindDims(&mlir_context_, d0, d1, d2, d3);
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 9);
  simplifier.SetInclusiveBounds(d1, -10, -1);
  simplifier.SetInclusiveBounds(d2, -1, 1);
  simplifier.SetInclusiveBounds(d3, 0, 0);

  // d0 is always positive.
  EXPECT_TRUE(simplifier.IsAlwaysPositiveOrZero(d0));
  EXPECT_FALSE(simplifier.IsAlwaysNegativeOrZero(d0));

  // d1 is always negative.
  EXPECT_FALSE(simplifier.IsAlwaysPositiveOrZero(d1));
  EXPECT_TRUE(simplifier.IsAlwaysNegativeOrZero(d1));

  // d2 is sometimes positive and sometimes negative.
  EXPECT_FALSE(simplifier.IsAlwaysPositiveOrZero(d2));
  EXPECT_FALSE(simplifier.IsAlwaysNegativeOrZero(d2));

  // d3 is always 0.
  EXPECT_TRUE(simplifier.IsAlwaysPositiveOrZero(d3));
  EXPECT_TRUE(simplifier.IsAlwaysNegativeOrZero(d3));
}

// TODO(b/313840171): Simplify `(d1 * 4 + d2) floordiv 8` to `d1 floordiv 2`.

// TODO(b/313840171): Simplify `(d0 * 8 + d1) floordiv 16` to `d0 floordiv 2`.

// TODO(b/313840171): Simplify `((d0 * 8 + d1) mod 16) floordiv 4` to
// `((d0 * 8 + d1) floordiv 4) mod 4` to `(d0 * 2 + d1 floordiv 4) mod 4`.

TEST_F(IndexingMapTest, AffineMapPrinterTest) {
  AffineExpr d0, d1, s0, s1;
  bindDims(&mlir_context_, d0, d1);
  bindSymbols(&mlir_context_, s0, s1);

  // (d0, d1)[s0, s1] -> (d0 + d1 floordiv 8, s0 + s1 mod 16).
  auto map =
      AffineMap::get(2, 2, {d0 + d1.floorDiv(8), s0 + s1 % 16}, &mlir_context_);

  printer_.SetDimensionName(0, "offset");
  printer_.SetSymbolName(1, "linear_index");
  EXPECT_THAT(printer_.ToString(map),
              HasSubstr("(offset, d1)[s0, linear_index] -> "
                        "(offset + d1 floordiv 8, s0 + linear_index mod 16)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

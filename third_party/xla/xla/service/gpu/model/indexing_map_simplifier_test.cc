/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/indexing_map_simplifier.h"

#include <gmock/gmock.h>
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::getAffineBinaryOpExpr;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::testing::HasSubstr;

class IndexingMapSimplifierTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

TEST_F(IndexingMapSimplifierTest, SimplifyConstantDims) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);

  auto map = AffineMap::get(1, 0, d0, &mlir_context_);
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 5, 5);

  EXPECT_THAT(ToString(simplifier.Simplify(map)), HasSubstr("(d0) -> (5)"));
}

TEST_F(IndexingMapSimplifierTest, SimplifyDivsAndModsIfSmallerThanDivisor) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);
  AffineExpr d1 = getAffineDimExpr(1, &mlir_context_);
  AffineExpr c16 = getAffineConstantExpr(16, &mlir_context_);

  // (d0, d1) -> (d0 + d1 floordiv 16, d1 mod 16).
  AffineExpr result0 = getAffineBinaryOpExpr(
      AffineExprKind::Add, d0,
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, d1, c16));
  AffineExpr result1 = getAffineBinaryOpExpr(AffineExprKind::Mod, d1, c16);
  auto map = AffineMap::get(2, 0, {result0, result1}, &mlir_context_);

  // d0 in [0, 8) and d1 in [0, 16).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 7);
  simplifier.SetInclusiveBounds(d1, 0, 15);

  EXPECT_THAT(ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

TEST_F(IndexingMapSimplifierTest, SimplifyDivsAndModsWithMultipliers) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);
  AffineExpr d1 = getAffineDimExpr(1, &mlir_context_);
  AffineExpr d2 = getAffineDimExpr(2, &mlir_context_);
  AffineExpr c10 = getAffineConstantExpr(10, &mlir_context_);
  AffineExpr c100 = getAffineConstantExpr(100, &mlir_context_);

  //  (d0, d1, d2) -> ((d0 * 100 + d1 * 10 + d2) floordiv 100,
  //                  "((d0 * 100 + d1 * 10 + d2) mod 100) floordiv 10,
  //                    d2 mod 10)"
  AffineExpr weighted_sum =
      getAffineBinaryOpExpr(AffineExprKind::Mul, d0, c100);
  weighted_sum = getAffineBinaryOpExpr(
      AffineExprKind::Add, weighted_sum,
      getAffineBinaryOpExpr(AffineExprKind::Mul, d1, c10));
  weighted_sum = getAffineBinaryOpExpr(AffineExprKind::Add, weighted_sum, d2);

  AffineExpr result0 =
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, weighted_sum, c100);
  AffineExpr result1 = getAffineBinaryOpExpr(
      AffineExprKind::FloorDiv,
      getAffineBinaryOpExpr(AffineExprKind::Mod, weighted_sum, c100), c10);
  AffineExpr result2 = getAffineBinaryOpExpr(AffineExprKind::Mod, d2, c10);

  auto map = AffineMap::get(3, 0, {result0, result1, result2}, &mlir_context_);

  // d_i in [0, 10).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 9);
  simplifier.SetInclusiveBounds(d1, 0, 9);
  simplifier.SetInclusiveBounds(d2, 0, 9);

  EXPECT_THAT(ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1, d2) -> (d0, d1, d2)"));
}

TEST_F(IndexingMapSimplifierTest, SimplifyDivsAndModsWithDivisibleMultipliers) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);
  AffineExpr d1 = getAffineDimExpr(1, &mlir_context_);
  AffineExpr d2 = getAffineDimExpr(2, &mlir_context_);
  AffineExpr c4 = getAffineConstantExpr(4, &mlir_context_);
  AffineExpr c8 = getAffineConstantExpr(8, &mlir_context_);
  AffineExpr c16 = getAffineConstantExpr(16, &mlir_context_);

  // (d0, d1, d2) -> ((d0 * 16 + d1 * 4 + d2) floordiv 8, "
  //                  (d0 * 16 + d1 * 4 + d2) mod 8)
  AffineExpr weighted_sum = getAffineBinaryOpExpr(AffineExprKind::Mul, d0, c16);
  weighted_sum =
      getAffineBinaryOpExpr(AffineExprKind::Add, weighted_sum,
                            getAffineBinaryOpExpr(AffineExprKind::Mul, d1, c4));
  weighted_sum = getAffineBinaryOpExpr(AffineExprKind::Add, weighted_sum, d2);

  AffineExpr result0 =
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, weighted_sum, c8);
  AffineExpr result1 =
      getAffineBinaryOpExpr(AffineExprKind::Mod, weighted_sum, c8);

  auto map = AffineMap::get(3, 0, {result0, result1}, &mlir_context_);

  // d_0 in [0, 10).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 1);
  simplifier.SetInclusiveBounds(d1, 0, 3);
  simplifier.SetInclusiveBounds(d2, 0, 3);

  EXPECT_THAT(ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1, d2) -> (d0 * 2 + (d1 * 4 + d2) floordiv 8, "
                        "(d1 * 4 + d2) mod 8)"));
}

TEST_F(IndexingMapSimplifierTest, SimplifyDivsAndModsWithReverse) {
  AffineExpr d0 = getAffineDimExpr(0, &mlir_context_);
  AffineExpr d1 = getAffineDimExpr(1, &mlir_context_);
  AffineExpr mc1 = getAffineConstantExpr(-1, &mlir_context_);
  AffineExpr c9 = getAffineConstantExpr(9, &mlir_context_);
  AffineExpr c11 = getAffineConstantExpr(11, &mlir_context_);
  AffineExpr mc11 = getAffineConstantExpr(-11, &mlir_context_);
  AffineExpr mc99 = getAffineConstantExpr(-99, &mlir_context_);
  AffineExpr c109 = getAffineConstantExpr(109, &mlir_context_);

  // (d0, d1) -> (-((d0 * -11 - d1 + 109) floordiv 11) + 9,
  //              d0 * 11 + d1 + ((d0 * -11 - d1 + 109) floordiv 11) * 11 - 99).
  AffineExpr weighted_sum =
      getAffineBinaryOpExpr(AffineExprKind::Mul, d0, mc11);
  weighted_sum = getAffineBinaryOpExpr(
      AffineExprKind::Add, weighted_sum,
      getAffineBinaryOpExpr(AffineExprKind::Mul, d1, mc1));
  weighted_sum = getAffineBinaryOpExpr(AffineExprKind::Add, weighted_sum, c109);
  weighted_sum =
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, weighted_sum, c11);

  AffineExpr result0 = getAffineBinaryOpExpr(
      AffineExprKind::Add, c9,
      getAffineBinaryOpExpr(AffineExprKind::Mul, mc1, weighted_sum));
  AffineExpr result1 = getAffineBinaryOpExpr(
      AffineExprKind::Add,
      getAffineBinaryOpExpr(AffineExprKind::Add,
                            getAffineBinaryOpExpr(AffineExprKind::Mul, c11, d0),
                            d1),
      getAffineBinaryOpExpr(
          AffineExprKind::Add,
          getAffineBinaryOpExpr(AffineExprKind::Mul, c11, weighted_sum), mc99));

  auto map = AffineMap::get(2, 0, {result0, result1}, &mlir_context_);

  // d0 in [0, 10) and d1 in [0, 11).
  IndexingMapSimplifier simplifier(&mlir_context_);
  simplifier.SetInclusiveBounds(d0, 0, 9);
  simplifier.SetInclusiveBounds(d1, 0, 10);

  EXPECT_THAT(ToString(simplifier.Simplify(map)),
              HasSubstr("(d0, d1) -> (d0, d1)"));
}

// TODO(b/313840171): Simplify `(d1 * 4 + d2) floordiv 8` to `d1 floordiv 2`.

// TODO(b/313840171): Simplify `(d0 * 8 + d1) floordiv 16` to `d0 floordiv 2`.

// TODO(b/313840171): Simplify `((d0 * 8 + d1) mod 16) floordiv 4` to
// `((d0 * 8 + d1) floordiv 4) mod 4` to `(d0 * 2 + d1 floordiv 4) mod 4`.

}  // namespace
}  // namespace gpu
}  // namespace xla

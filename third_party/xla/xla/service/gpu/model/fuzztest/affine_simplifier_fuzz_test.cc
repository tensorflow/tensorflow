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
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/log/check.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/fuzztest/affine_grammar.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_test_utils.h"

namespace xla {
namespace gpu {
namespace {

IndexingMap GetMap(std::string input, int64_t d0_min, int64_t d0_size,
                   int64_t d1_min, int64_t d1_size, int64_t s0_min,
                   int64_t s0_size, int64_t s1_min, int64_t s1_size) {
  static mlir::MLIRContext* context = new mlir::MLIRContext();
  mlir::AffineMap affine_map = xla::gpu::ParseAffineMap(input, context);
  CHECK_EQ(affine_map.getNumResults(), 1);

  // Set the sizes of unused variables to 1.
  if (!affine_map.isFunctionOfSymbol(0)) {
    s0_size = 1;
  }
  if (!affine_map.isFunctionOfSymbol(1)) {
    s1_size = 1;
  }
  if (!affine_map.isFunctionOfDim(0)) {
    d0_size = 1;
  }
  if (!affine_map.isFunctionOfDim(1)) {
    d1_size = 1;
  }

  Interval s0_interval = {s0_min, s0_min + s0_size - 1};
  Interval s1_interval = {s1_min, s1_min + s1_size - 1};
  Interval d0_interval = {d0_min, d0_min + d0_size - 1};
  Interval d1_interval = {d1_min, d1_min + d1_size - 1};

  return IndexingMap(affine_map, {{d0_interval}, {d1_interval}},
                     {{s0_interval}, {s1_interval}}, {});
}

void TestCorrectness(std::string input, int64_t d0_min, int64_t d0_size,
                     int64_t d1_min, int64_t d1_size, int64_t s0_min,
                     int64_t s0_size, int64_t s1_min, int64_t s1_size) {
  // Verifies that the simplified map produces the same results as the original
  // map at every point in its domain.
  IndexingMap map = GetMap(input, d0_min, d0_size, d1_min, d1_size, s0_min,
                           s0_size, s1_min, s1_size);
  IndexingMap map_simplified = map;
  map_simplified.Simplify();

  mlir::AffineExpr original = map.GetAffineMap().getResult(0);
  mlir::AffineExpr simplified = map_simplified.GetAffineMap().getResult(0);

  EXPECT_OK(VerifyExprsAreIdentical(
      original, simplified, map.GetDimensionBounds(), map.GetSymbolBounds()));
}

void TestIdempotency(std::string input, int64_t d0_min, int64_t d0_size,
                     int64_t d1_min, int64_t d1_size, int64_t s0_min,
                     int64_t s0_size, int64_t s1_min, int64_t s1_size) {
  // Verifies that Simplify(Simplify(map)) == Simplify(map).
  IndexingMap map = GetMap(input, d0_min, d0_size, d1_min, d1_size, s0_min,
                           s0_size, s1_min, s1_size);
  if (map.Simplify()) {
    auto before_simplification = map.GetAffineMap();
    EXPECT_FALSE(map.Simplify());
    EXPECT_EQ(before_simplification, map.GetAffineMap())
        << AffineMapPrinter().ToString(before_simplification);
  }
}

int64_t Cost(mlir::AffineExpr expr) {
  switch (expr.getKind()) {
    case mlir::AffineExprKind::Constant:
      return 0;
    case mlir::AffineExprKind::DimId:
      return 1;
    case mlir::AffineExprKind::SymbolId:
      return 1;
    case mlir::AffineExprKind::Add:
      return 2 + Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getLHS()) +
             Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getRHS());
    case mlir::AffineExprKind::Mul:
      return 4 + Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getLHS()) +
             Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getRHS());
    case mlir::AffineExprKind::FloorDiv:
    case mlir::AffineExprKind::CeilDiv:
    case mlir::AffineExprKind::Mod:
      return 10 + Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getLHS()) +
             Cost(mlir::cast<mlir::AffineBinaryOpExpr>(expr).getRHS());
  }
}

void TestNoAdditionalSimplificationFromMlir(std::string input, int64_t d0_min,
                                            int64_t d0_size, int64_t d1_min,
                                            int64_t d1_size, int64_t s0_min,
                                            int64_t s0_size, int64_t s1_min,
                                            int64_t s1_size) {
  // Checks that no additional simplification is done by the MLIR simplifier, as
  // measured by the above cost function.
  // We don't check that simplify(mlir-simplify(simplify(map)) == simplify(map)
  // because the MLIR simplifier sometimes makes expressions more complicated,
  // which we then don't clean up completely.
  IndexingMap map = GetMap(input, d0_min, d0_size, d1_min, d1_size, s0_min,
                           s0_size, s1_min, s1_size);
  if (map.Simplify()) {
    auto simplified = mlir::simplifyAffineMap(map.GetAffineMap());
    IndexingMap new_map(simplified, map.GetDimVars(), map.GetRangeVars(), {});
    new_map.Simplify();
    EXPECT_LE(Cost(map.GetAffineMap().getResult(0)),
              Cost(new_map.GetAffineMap().getResult(0)))
        << "unexpected additional simplification from "
        << AffineMapPrinter().ToString(map.GetAffineMap()) << " to "
        << AffineMapPrinter().ToString(new_map.GetAffineMap()) << " via "
        << AffineMapPrinter().ToString(simplified);
  }
}

auto AffineDomain() {
  // The ranges are chosen to include entirely negative, entirely positive and
  // mixed domains (but mostly positive ones).
  return fuzztest::TupleOf(
      fuzztest::InAffineGrammar(), fuzztest::InRange<int64_t>(-10, 100),
      fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
      fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
      fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
      fuzztest::InRange<int64_t>(0, 10));
}

TEST(AffineSimplifierFuzzTest,
     TestNoAdditionalSimplificationFromMlirRegression) {
  TestNoAdditionalSimplificationFromMlir(
      "(d0, d1)[s0, s1] -> ((s1 + ((d0 + (s0 mod 2)) + d1)))", 0, 2, 0, 1, 2, 2,
      0, 1);
}

FUZZ_TEST(AffineSimplifierFuzzTest, TestCorrectness)
    .WithDomains(AffineDomain());
FUZZ_TEST(AffineSimplifierFuzzTest, TestIdempotency)
    .WithDomains(AffineDomain());
FUZZ_TEST(AffineSimplifierFuzzTest, TestNoAdditionalSimplificationFromMlir)
    .WithDomains(AffineDomain());

}  // namespace
}  // namespace gpu
}  // namespace xla

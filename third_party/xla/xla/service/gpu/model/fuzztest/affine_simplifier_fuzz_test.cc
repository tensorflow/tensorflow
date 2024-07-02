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
#include <functional>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/fuzztest/affine_grammar.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_test_utils.h"

namespace xla {
namespace gpu {
namespace {

// Evaluates the given affine expression with the given dimensions and symbols.
// This intentionally does not use `replaceDimsAndSymbols`, since we're testing
// simplification here (which includes the constant folding that
// replaceDimsAndSymbols would do).
// Returns `nullopt` for undefined behavior.
std::optional<int64_t> Evaluate(mlir::AffineExpr expr,
                                const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& syms) {
  if (auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = Evaluate(binary.getLHS(), dims, syms);
    auto rhs = Evaluate(binary.getRHS(), dims, syms);
    if (!lhs || !rhs) return std::nullopt;
    switch (binary.getKind()) {
      case mlir::AffineExprKind::Add:
        return *lhs + *rhs;
      case mlir::AffineExprKind::Mul:
        return *lhs * *rhs;
      case mlir::AffineExprKind::FloorDiv:
        return rhs == 0
                   ? std::nullopt
                   : std::make_optional(llvm::divideFloorSigned(*lhs, *rhs));
      case mlir::AffineExprKind::Mod:
        return rhs <= 0 ? std::nullopt
                        : std::make_optional(llvm::mod(*lhs, *rhs));
      default:
        LOG(FATAL) << "Unsupported binary op";
    }
  }
  if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    return syms[sym.getPosition()];
  }
  if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    return dims[dim.getPosition()];
  }
  return mlir::dyn_cast<mlir::AffineConstantExpr>(expr).getValue();
}

void TestCorrectness(std::string input, int64_t d0_min, int64_t d0_size,
                     int64_t d1_min, int64_t d1_size, int64_t s0_min,
                     int64_t s0_size, int64_t s1_min, int64_t s1_size) {
  // Verifies that the simplified map produces the same results as the original
  // map at every point in its domain.
  static mlir::MLIRContext* context = new mlir::MLIRContext();
  mlir::AffineMap affine_map = xla::gpu::ParseAffineMap(input, context);
  CHECK_EQ(affine_map.getNumResults(), 1);
  IndexingMap map(
      affine_map,
      {{d0_min, d0_min + d0_size - 1}, {d1_min, d1_min + d1_size - 1}},
      {{s0_min, s0_min + s0_size - 1}, {s1_min, s1_min + s1_size - 1}}, {});

  IndexingMap map_simplified = map;
  map_simplified.Simplify();

  mlir::AffineExpr original = map.GetAffineMap().getResult(0);
  mlir::AffineExpr simplified = map_simplified.GetAffineMap().getResult(0);

  std::vector<int64_t> dims(map.GetDimensionCount());
  std::vector<int64_t> syms(map.GetSymbolCount());
  std::function<bool(int64_t dim, int64_t sym)> test_equality;

  // Sets induction_var to each value in the given range and checks that the
  // simplified map and the original map produce the same result.
  auto test_range = [&](int64_t next_dim, int64_t next_sym, Interval range,
                        int64_t& induction_var) {
    for (int64_t i = range.lower; i <= range.upper; ++i) {
      induction_var = i;
      if (!test_equality(next_dim, next_sym)) return false;
    }
    return true;
  };

  // Enumerates all possible values for the dimensions starting at `dim` and
  // `sym` and checks equality of the simplified map and the original map.
  test_equality = [&](int64_t dim, int64_t sym) {
    if (dim < dims.size()) {
      Interval range = map.GetAffineMap().isFunctionOfDim(dim)
                           ? map.GetDimensionBound(dim)
                           : Interval{0, 0};
      return test_range(dim + 1, sym, range, dims[dim]);
    }

    if (sym < syms.size()) {
      Interval range = map.GetAffineMap().isFunctionOfSymbol(sym)
                           ? map.GetSymbolBound(sym)
                           : Interval{0, 0};
      return test_range(dim, sym + 1, range, syms[sym]);
    }

    std::optional<int64_t> original_val = Evaluate(original, dims, syms);
    std::optional<int64_t> simplified_val = Evaluate(simplified, dims, syms);

    EXPECT_EQ(original_val, simplified_val)
        << "in simplified map "
        << AffineMapPrinter().ToString(map_simplified.GetAffineMap()) << " at ("
        << absl::StrJoin(dims, ",") << ")[" << absl::StrJoin(syms, ",") << "]";
    return original_val == simplified_val;
  };
  EXPECT_TRUE(test_equality(0, 0));
}

void TestIdempotency(std::string input, int64_t d0_min, int64_t d0_size,
                     int64_t d1_min, int64_t d1_size, int64_t s0_min,
                     int64_t s0_size, int64_t s1_min, int64_t s1_size) {
  // Test that verifies that Simplify(Simplify(map)) == Simplify(map).
  static mlir::MLIRContext* context = new mlir::MLIRContext();
  mlir::AffineMap affine_map = xla::gpu::ParseAffineMap(input, context);
  IndexingMap map(
      affine_map,
      {{d0_min, d0_min + d0_size - 1}, {d1_min, d1_min + d1_size - 1}},
      {{s0_min, s0_min + s0_size - 1}, {s1_min, s1_min + s1_size - 1}}, {});

  if (map.Simplify()) {
    auto before_simplification = map.GetAffineMap();
    EXPECT_FALSE(map.Simplify());
    EXPECT_EQ(before_simplification, map.GetAffineMap())
        << AffineMapPrinter().ToString(before_simplification);
  }
}

// The ranges are chosen to include entirely negative, entirely positive and
// mixed domains (but mostly positive ones).
FUZZ_TEST(AffineSimplifierFuzzTest, TestCorrectness)
    .WithDomains(
        fuzztest::InAffineGrammar(), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10));

FUZZ_TEST(AffineSimplifierFuzzTest, TestIdempotency)
    .WithDomains(
        fuzztest::InAffineGrammar(), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10), fuzztest::InRange<int64_t>(-10, 100),
        fuzztest::InRange<int64_t>(0, 10));

}  // namespace
}  // namespace gpu
}  // namespace xla

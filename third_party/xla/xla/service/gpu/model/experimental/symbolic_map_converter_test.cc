/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/experimental/symbolic_map_converter.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineMap;
using ::mlir::MLIRContext;

// TODO: b/433693782 - This code is duplicated from indexing_test_utils. Remove
// this function as soon as symbolic_map_converter is not needed anymore. If
// not, we should refactor it to a common test library.
// Helper function to parse an AffineMap from a string.
AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                         MLIRContext* context) {
  std::string full_affine_map_string =
      absl::StrCat("affine_map<", serialized_affine_map, ">");
  return mlir::cast<mlir::AffineMapAttr>(
             mlir::parseAttribute(full_affine_map_string, context))
      .getValue();
}

class SymbolicMapConverterTest : public ::testing::Test {
 public:
  SymbolicMapConverterTest() : symbolic_expr_context_(&mlir_context_) {}

  MLIRContext mlir_context_;
  SymbolicExprContext symbolic_expr_context_;
};

TEST_F(SymbolicMapConverterTest, AffineToSymbolicRoundTrip) {
  AffineMap affine_map = ParseAffineMap(
      "(d0, d1)[s0, s1] -> (d0 + s1 * 2, d1 - s0, d0 floordiv 3, d1 mod 4)",
      &mlir_context_);

  SymbolicMap symbolic_map =
      AffineMapToSymbolicMap(affine_map, &symbolic_expr_context_);

  EXPECT_EQ(symbolic_map.GetNumResults(), 4);

  AffineMap round_trip_map =
      SymbolicMapToAffineMap(symbolic_map, &mlir_context_);
  EXPECT_EQ(affine_map, round_trip_map);
}

TEST_F(SymbolicMapConverterTest, SymbolicToAffineFailure) {
  SymbolicExpr d0 = symbolic_expr_context_.CreateVariable(0);
  SymbolicExpr c1 = symbolic_expr_context_.CreateConstant(1);
  // kMax is not representable in AffineExpr.
  SymbolicExpr max_expr = d0.max(c1);

  AffineMap affine_map = SymbolicMapToAffineMap(
      SymbolicMap::Get(&symbolic_expr_context_, 1, 0, {max_expr}),
      &mlir_context_);
  EXPECT_FALSE(affine_map);
}

TEST_F(SymbolicMapConverterTest, SymbolicToAffineNestedFailure) {
  SymbolicExpr d0 = symbolic_expr_context_.CreateVariable(0);
  SymbolicExpr c1 = symbolic_expr_context_.CreateConstant(1);
  SymbolicExpr c2 = symbolic_expr_context_.CreateConstant(2);

  // d0 + max(c1, c2). max is not representable in AffineExpr.
  SymbolicExpr nested_max_expr = d0 + c1.max(c2);

  // This should not crash and should return a null AffineMap.
  AffineMap affine_map = SymbolicMapToAffineMap(
      SymbolicMap::Get(&symbolic_expr_context_, 1, 0, {nested_max_expr}),
      &mlir_context_);
  EXPECT_FALSE(affine_map);
}

TEST_F(SymbolicMapConverterTest, AffineExprsToSymbolicExprs) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlir_context_);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &mlir_context_);
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, &mlir_context_);
  mlir::AffineExpr c1 = mlir::getAffineConstantExpr(1, &mlir_context_);
  llvm::SmallVector<mlir::AffineExpr> affine_exprs = {d0, d1, s0, c1};
  llvm::SmallVector<SymbolicExpr> symbolic_exprs = AffineExprsToSymbolicExprs(
      affine_exprs, &symbolic_expr_context_, /*num_dims=*/2);
  EXPECT_EQ(symbolic_exprs.size(), 4);
  EXPECT_EQ(symbolic_exprs[0], symbolic_expr_context_.CreateVariable(0));
  EXPECT_EQ(symbolic_exprs[1], symbolic_expr_context_.CreateVariable(1));
  EXPECT_EQ(symbolic_exprs[2], symbolic_expr_context_.CreateVariable(2));
  EXPECT_EQ(symbolic_exprs[3], symbolic_expr_context_.CreateConstant(1));
}

TEST_F(SymbolicMapConverterTest,
       ConvertAffineConstraintsToSymbolicConstraints) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlir_context_);
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, &mlir_context_);
  mlir::AffineExpr c1 = mlir::getAffineConstantExpr(1, &mlir_context_);

  llvm::MapVector<mlir::AffineExpr, Interval> affine_constraints;
  affine_constraints[d0 + s0] = {0, 127};
  affine_constraints[s0 * 2] = {0, 63};
  affine_constraints[d0 - c1] = {10, 20};

  llvm::MapVector<SymbolicExpr, Interval> symbolic_constraints =
      ConvertAffineConstraintsToSymbolicConstraints(
          affine_constraints, &symbolic_expr_context_, /*num_dims=*/1);

  SymbolicExpr sym_d0 = symbolic_expr_context_.CreateVariable(0);
  SymbolicExpr sym_s0 = symbolic_expr_context_.CreateVariable(1);
  SymbolicExpr sym_c1 = symbolic_expr_context_.CreateConstant(1);

  EXPECT_EQ(symbolic_constraints.size(), 3);
  EXPECT_EQ(symbolic_constraints[sym_d0 + sym_s0], (Interval{0, 127}));
  EXPECT_EQ(symbolic_constraints[sym_s0 * 2], (Interval{0, 63}));
  EXPECT_EQ(symbolic_constraints[sym_d0 - sym_c1], (Interval{10, 20}));
}

TEST_F(SymbolicMapConverterTest, ConvertAffineToSymbolicExpr) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlir_context_);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &mlir_context_);
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, &mlir_context_);
  mlir::AffineExpr c1 = mlir::getAffineConstantExpr(1, &mlir_context_);
  mlir::AffineExpr c2 = mlir::getAffineConstantExpr(2, &mlir_context_);
  mlir::AffineExpr c3 = mlir::getAffineConstantExpr(3, &mlir_context_);

  mlir::AffineExpr affine_expr =
      mlir::getAffineBinaryOpExpr(
          mlir::AffineExprKind::Mod,
          mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv,
                                      d0 * c2 + s0 - c1, c2),
          c3) +
      d1;  // ((d0 * 2 + s0 - 1) floordiv 2) mod 3 + d1

  SymbolicExpr exp_d0 = symbolic_expr_context_.CreateVariable(0);
  SymbolicExpr exp_d1 = symbolic_expr_context_.CreateVariable(1);
  SymbolicExpr exp_s0 = symbolic_expr_context_.CreateVariable(2);
  SymbolicExpr exp_c1 = symbolic_expr_context_.CreateConstant(1);
  SymbolicExpr exp_c2 = symbolic_expr_context_.CreateConstant(2);
  SymbolicExpr exp_c3 = symbolic_expr_context_.CreateConstant(3);

  SymbolicExpr expected_symbolic_expr =
      ((exp_d0 * exp_c2 + exp_s0 - exp_c1) / exp_c2) % exp_c3 + exp_d1;

  EXPECT_EQ(AffineExprToSymbolicExpr(affine_expr, &symbolic_expr_context_,
                                     /*num_dims=*/2),
            expected_symbolic_expr);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

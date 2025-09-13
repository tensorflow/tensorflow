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

#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineMap;
using ::mlir::MLIRContext;

TEST(SymbolicMapConverterTest, AffineToSymbolicRoundTrip) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  AffineMap affine_map = ParseAffineMap(
      "(d0, d1)[s0, s1] -> (d0 + s1 * 2, d1 - s0, d0 floordiv 3, d1 mod 4)",
      &mlir_context);

  SymbolicMap symbolic_map =
      AffineMapToSymbolicMap(affine_map, &symbolic_context);

  EXPECT_EQ(symbolic_map.GetNumResults(), 4);

  AffineMap round_trip_map =
      SymbolicMapToAffineMap(symbolic_map, &mlir_context);
  EXPECT_EQ(affine_map, round_trip_map);
}

TEST(SymbolicMapConverterTest, SymbolicToAffineFailure) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  SymbolicExpr d0 = symbolic_context.CreateVariable(0);
  SymbolicExpr c1 = symbolic_context.CreateConstant(1);
  // kMax is not representable in AffineExpr.
  SymbolicExpr max_expr = d0.max(c1);

  AffineMap affine_map = SymbolicMapToAffineMap(
      SymbolicMap::Get(&symbolic_context, 1, 0, {max_expr}), &mlir_context);
  EXPECT_FALSE(affine_map);
}

TEST(SymbolicMapConverterTest, SymbolicToAffineNestedFailure) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  SymbolicExpr d0 = symbolic_context.CreateVariable(0);
  SymbolicExpr c1 = symbolic_context.CreateConstant(1);
  SymbolicExpr c2 = symbolic_context.CreateConstant(2);

  // d0 + max(c1, c2). max is not representable in AffineExpr.
  SymbolicExpr nested_max_expr = d0 + c1.max(c2);

  // This should not crash and should return a null AffineMap.
  AffineMap affine_map = SymbolicMapToAffineMap(
      SymbolicMap::Get(&symbolic_context, 1, 0, {nested_max_expr}),
      &mlir_context);
  EXPECT_FALSE(affine_map);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

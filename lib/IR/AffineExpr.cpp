//===- AffineExpr.cpp - MLIR Affine Expr Classes --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/STLExtras.h"

using namespace mlir;

/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool AffineExpr::isSymbolic() const {
  switch (getKind()) {
  case Kind::Constant:
    return true;
  case Kind::DimId:
    return false;
  case Kind::SymbolId:
    return true;

  case Kind::Add:
  case Kind::Sub:
  case Kind::Mul:
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod: {
    auto expr = cast<AffineBinaryOpExpr>(this);
    return expr->getLHS()->isSymbolic() && expr->getRHS()->isSymbolic();
  }
  }
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExpr::isPureAffine() const {
  switch (getKind()) {
  case Kind::SymbolId:
  case Kind::DimId:
  case Kind::Constant:
    return true;
  case Kind::Add:
  case Kind::Sub: {
    auto op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine();
  }

  case Kind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine() &&
           (isa<AffineConstantExpr>(op->getLHS()) ||
            isa<AffineConstantExpr>(op->getRHS()));
  }
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod: {
    auto op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() &&
           isa<AffineConstantExpr>(op->getRHS());
  }
  }
}

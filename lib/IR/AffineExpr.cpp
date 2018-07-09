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
  case Kind::Mod:
    return cast<AffineBinaryOpExpr>(this)->isSymbolic();
  }
}

bool AffineExpr::isPureAffine() const {
  switch (getKind()) {
  case Kind::SymbolId:
    return cast<AffineSymbolExpr>(this)->isPureAffine();
  case Kind::DimId:
    return cast<AffineDimExpr>(this)->isPureAffine();
  case Kind::Constant:
    return cast<AffineConstantExpr>(this)->isPureAffine();
  case Kind::Add:
    return cast<AffineAddExpr>(this)->isPureAffine();
  case Kind::Sub:
    return cast<AffineSubExpr>(this)->isPureAffine();
  case Kind::Mul:
    return cast<AffineMulExpr>(this)->isPureAffine();
  case Kind::FloorDiv:
    return cast<AffineFloorDivExpr>(this)->isPureAffine();
  case Kind::CeilDiv:
    return cast<AffineCeilDivExpr>(this)->isPureAffine();
  case Kind::Mod:
    return cast<AffineModExpr>(this)->isPureAffine();
  }
}

bool AffineMulExpr::isPureAffine() const {
  return lhsOperand->isPureAffine() && rhsOperand->isPureAffine() &&
    (isa<AffineConstantExpr>(lhsOperand) ||
     isa<AffineConstantExpr>(rhsOperand));
}

bool AffineFloorDivExpr::isPureAffine() const {
  return lhsOperand->isPureAffine() && isa<AffineConstantExpr>(rhsOperand);
}

bool AffineCeilDivExpr::isPureAffine() const {
  return lhsOperand->isPureAffine() && isa<AffineConstantExpr>(rhsOperand);
}

bool AffineModExpr::isPureAffine() const {
  return lhsOperand->isPureAffine() && isa<AffineConstantExpr>(rhsOperand);
}

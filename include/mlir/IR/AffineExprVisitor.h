//===- AffineExprVisitor.h - MLIR AffineExpr Visitor Class ------*- C++ -*-===//
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
//
// This file defines the AffineExpr visitor class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINE_EXPR_VISITOR_H
#define MLIR_IR_AFFINE_EXPR_VISITOR_H

#include "mlir/IR/AffineExpr.h"

namespace mlir {

/// Base class for AffineExpr visitors.
///
/// AffineExpr visitors are used when you want to perform different actions
/// for different kinds of AffineExprs without having to use lots of casts
/// and a big switch statement.
///
/// To define your own visitor, inherit from this class, specifying your
/// new type for the 'SubClass' template parameter, and "override" visitXXX
/// functions in your class. This class is defined in terms of statically
/// resolved overloading, not virtual functions.
///
/// For example, here is a visitor that counts the number of for AffineDimExprs
/// in an AffineExpr.
///
///  /// Declare the class.  Note that we derive from AffineExprVisitor
///  /// instantiated with our new subclasses_ type.
///
///  struct DimExprCounter : public AffineExprVisitor<DimExprCounter> {
///    unsigned numDimExprs;
///    DimExprCounter() : numDimExprs(0) {}
///    void visitAffineDimExpr(AffineDimExpr *expr) { ++numDimExprs; }
///  };
///
///  And this class would be used like this:
///    DimExprCounter dec;
///    dec.visit(affineExpr);
///    numDimExprs = dec.numDimExprs;
///
/// AffineExprVisitor provides visit methods for the following binary affine
/// op expressions:
/// AffineBinaryAddOpExpr, AffineBinaryMulOpExpr, AffineBinaryModOpExpr,
/// AffineBinaryFloorDivOpExpr, AffineBinaryCeilDivOpExpr.
/// Note that default implementations of these methods will call the general
/// AffineBinaryOpExpr method.
///
/// In addition, visit methods are provided for the following affine
//  expressions: AffineConstantExpr, AffineDimExpr, and AffineSymbolExpr.
///
/// Note that if you don't implement visitXXX for some affine expression type,
/// the visitXXX method for Statement superclass will be invoked.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead. Defining and using a AffineExprVisitor is
/// just as efficient as having your own switch statement over the statement
/// opcode.

template <typename SubClass> class AffineExprVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the AffineExprVisitor
  // that you use to visit affine expressions...
public:
  // Function to visit an AffineExpr.
  void visit(AffineExpr *expr) {
    static_assert(std::is_base_of<AffineExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of AffineExprVisitor");
    switch (expr->getKind()) {
    case AffineExpr::Kind::Add: {
      auto *binOpExpr = cast<AffineBinaryOpExpr>(expr);
      visitAffineBinaryOpOperands(binOpExpr);
      return static_cast<SubClass *>(this)->visitAffineBinaryAddOpExpr(
          binOpExpr);
    }
    case AffineExpr::Kind::Mul: {
      auto *binOpExpr = cast<AffineBinaryOpExpr>(expr);
      visitAffineBinaryOpOperands(binOpExpr);
      return static_cast<SubClass *>(this)->visitAffineBinaryMulOpExpr(
          binOpExpr);
    }
    case AffineExpr::Kind::Mod: {
      auto *binOpExpr = cast<AffineBinaryOpExpr>(expr);
      visitAffineBinaryOpOperands(binOpExpr);
      return static_cast<SubClass *>(this)->visitAffineBinaryModOpExpr(
          binOpExpr);
    }
    case AffineExpr::Kind::FloorDiv: {
      auto *binOpExpr = cast<AffineBinaryOpExpr>(expr);
      visitAffineBinaryOpOperands(binOpExpr);
      return static_cast<SubClass *>(this)->visitAffineBinaryFloorDivOpExpr(
          binOpExpr);
    }
    case AffineExpr::Kind::CeilDiv: {
      auto *binOpExpr = cast<AffineBinaryOpExpr>(expr);
      visitAffineBinaryOpOperands(binOpExpr);
      return static_cast<SubClass *>(this)->visitAffineBinaryCeilDivOpExpr(
          binOpExpr);
    }
    case AffineExpr::Kind::Constant:
      return static_cast<SubClass *>(this)->visitAffineConstantExpr(
          cast<AffineConstantExpr>(expr));
    case AffineExpr::Kind::DimId:
      return static_cast<SubClass *>(this)->visitAffineDimExpr(
          cast<AffineDimExpr>(expr));
    case AffineExpr::Kind::SymbolId:
      return static_cast<SubClass *>(this)->visitAffineSymbolExpr(
          cast<AffineSymbolExpr>(expr));
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular statement type.
  // The default behavior is to generalize the statement type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // Default visit methods. Note that the default op-specific binary op visit
  // methods call the general visitAffineBinaryOpExpr visit method.
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr *expr) {}
  void visitAffineBinaryAddOpExpr(AffineBinaryOpExpr *expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitAffineBinaryMulOpExpr(AffineBinaryOpExpr *expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitAffineBinaryModOpExpr(AffineBinaryOpExpr *expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitAffineBinaryFloorDivOpExpr(AffineBinaryOpExpr *expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitAffineBinaryCeilDivOpExpr(AffineBinaryOpExpr *expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitAffineConstantExpr(AffineConstantExpr *expr) {}
  void visitAffineDimExpr(AffineDimExpr *expr) {}
  void visitAffineSymbolExpr(AffineSymbolExpr *expr) {}

private:
  void visitAffineBinaryOpOperands(AffineBinaryOpExpr *expr) {
    visit(expr->getLHS());
    visit(expr->getRHS());
  }
};

} // end namespace mlir

#endif // MLIR_IR_AFFINE_EXPR_VISITOR_H

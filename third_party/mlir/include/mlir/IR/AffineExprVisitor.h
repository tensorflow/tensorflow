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

/// Base class for AffineExpr visitors/walkers.
///
/// AffineExpr visitors are used when you want to perform different actions
/// for different kinds of AffineExprs without having to use lots of casts
/// and a big switch instruction.
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
///    void visitDimExpr(AffineDimExpr expr) { ++numDimExprs; }
///  };
///
///  And this class would be used like this:
///    DimExprCounter dec;
///    dec.visit(affineExpr);
///    numDimExprs = dec.numDimExprs;
///
/// AffineExprVisitor provides visit methods for the following binary affine
/// op expressions:
/// AffineBinaryAddOpExpr, AffineBinaryMulOpExpr,
/// AffineBinaryModOpExpr, AffineBinaryFloorDivOpExpr,
/// AffineBinaryCeilDivOpExpr. Note that default implementations of these
/// methods will call the general AffineBinaryOpExpr method.
///
/// In addition, visit methods are provided for the following affine
//  expressions: AffineConstantExpr, AffineDimExpr, and
//  AffineSymbolExpr.
///
/// Note that if you don't implement visitXXX for some affine expression type,
/// the visitXXX method for Instruction superclass will be invoked.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead. Defining and using a AffineExprVisitor is
/// just as efficient as having your own switch instruction over the instruction
/// opcode.

template <typename SubClass, typename RetTy = void> class AffineExprVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the AffineExprVisitor
  // that you use to visit affine expressions...
public:
  // Function to walk an AffineExpr (in post order).
  RetTy walkPostOrder(AffineExpr expr) {
    static_assert(std::is_base_of<AffineExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of AffineExprVisitor");
    switch (expr.getKind()) {
    case AffineExprKind::Add: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case AffineExprKind::Mul: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case AffineExprKind::Mod: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case AffineExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case AffineExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case AffineExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<AffineConstantExpr>());
    case AffineExprKind::DimId:
      return static_cast<SubClass *>(this)->visitDimExpr(
          expr.cast<AffineDimExpr>());
    case AffineExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<AffineSymbolExpr>());
    }
  }

  // Function to visit an AffineExpr.
  RetTy visit(AffineExpr expr) {
    static_assert(std::is_base_of<AffineExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of AffineExprVisitor");
    switch (expr.getKind()) {
    case AffineExprKind::Add: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case AffineExprKind::Mul: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case AffineExprKind::Mod: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case AffineExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case AffineExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case AffineExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<AffineConstantExpr>());
    case AffineExprKind::DimId:
      return static_cast<SubClass *>(this)->visitDimExpr(
          expr.cast<AffineDimExpr>());
    case AffineExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<AffineSymbolExpr>());
    }
    llvm_unreachable("Unknown AffineExpr");
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // Default visit methods. Note that the default op-specific binary op visit
  // methods call the general visitAffineBinaryOpExpr visit method.
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {}
  void visitAddExpr(AffineBinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitMulExpr(AffineBinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitModExpr(AffineBinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitAffineBinaryOpExpr(expr);
  }
  void visitConstantExpr(AffineConstantExpr expr) {}
  void visitDimExpr(AffineDimExpr expr) {}
  void visitSymbolExpr(AffineSymbolExpr expr) {}

private:
  // Walk the operands - each operand is itself walked in post order.
  void walkOperandsPostOrder(AffineBinaryOpExpr expr) {
    walkPostOrder(expr.getLHS());
    walkPostOrder(expr.getRHS());
  }
};

// This class is used to flatten a pure affine expression (AffineExpr,
// which is in a tree form) into a sum of products (w.r.t constants) when
// possible, and in that process simplifying the expression. For a modulo,
// floordiv, or a ceildiv expression, an additional identifier, called a local
// identifier, is introduced to rewrite the expression as a sum of product
// affine expression. Each local identifier is always and by construction a
// floordiv of a pure add/mul affine function of dimensional, symbolic, and
// other local identifiers, in a non-mutually recursive way. Hence, every local
// identifier can ultimately always be recovered as an affine function of
// dimensional and symbolic identifiers (involving floordiv's); note however
// that by AffineExpr construction, some floordiv combinations are converted to
// mod's. The result of the flattening is a flattened expression and a set of
// constraints involving just the local variables.
//
// d2 + (d0 + d1) floordiv 4  is flattened to d2 + q where 'q' is the local
// variable introduced, with localVarCst containing 4*q <= d0 + d1 <= 4*q + 3.
//
// The simplification performed includes the accumulation of contributions for
// each dimensional and symbolic identifier together, the simplification of
// floordiv/ceildiv/mod expressions and other simplifications that in turn
// happen as a result. A simplification that this flattening naturally performs
// is of simplifying the numerator and denominator of floordiv/ceildiv, and
// folding a modulo expression to a zero, if possible. Three examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
// (d0 - d0 mod 4 + 4) mod 4             simplified to     0
// (3*d0 + 2*d1 + d0) floordiv 2 + d1    simplified to     2*d0 + 2*d1
//
// The way the flattening works for the second example is as follows: d0 % 4 is
// replaced by d0 - 4*q with q being introduced: the expression then simplifies
// to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to
// zero. Note that an affine expression may not always be expressible purely as
// a sum of products involving just the original dimensional and symbolic
// identifiers due to the presence of modulo/floordiv/ceildiv expressions that
// may not be eliminated after simplification; in such cases, the final
// expression can be reconstructed by replacing the local identifiers with their
// corresponding explicit form stored in 'localExprs' (note that each of the
// explicit forms itself would have been simplified).
//
// The expression walk method here performs a linear time post order walk that
// performs the above simplifications through visit methods, with partial
// results being stored in 'operandExprStack'. When a parent expr is visited,
// the flattened expressions corresponding to its two operands would already be
// on the stack - the parent expression looks at the two flattened expressions
// and combines the two. It pops off the operand expressions and pushes the
// combined result (although this is done in-place on its LHS operand expr).
// When the walk is completed, the flattened form of the top-level expression
// would be left on the stack.
//
// A flattener can be repeatedly used for multiple affine expressions that bind
// to the same operands, for example, for all result expressions of an
// AffineMap or AffineValueMap. In such cases, using it for multiple expressions
// is more efficient than creating a new flattener for each expression since
// common identical div and mod expressions appearing across different
// expressions are mapped to the same local identifier (same column position in
// 'localVarCst').
class SimpleAffineExprFlattener
    : public AffineExprVisitor<SimpleAffineExprFlattener> {
public:
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 8>> operandExprStack;

  unsigned numDims;
  unsigned numSymbols;

  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv's.
  unsigned numLocals;

  // AffineExpr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions can be readily used to reconstruct the AffineExpr
  // (tree) form. Note that these expressions themselves would have been
  // simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4
  // will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1)
  // ceildiv 2 would be the local expression stored for q.
  SmallVector<AffineExpr, 4> localExprs;

  SimpleAffineExprFlattener(unsigned numDims, unsigned numSymbols);

  virtual ~SimpleAffineExprFlattener() = default;

  // Visitor method overrides.
  void visitMulExpr(AffineBinaryOpExpr expr);
  void visitAddExpr(AffineBinaryOpExpr expr);
  void visitDimExpr(AffineDimExpr expr);
  void visitSymbolExpr(AffineSymbolExpr expr);
  void visitConstantExpr(AffineConstantExpr expr);
  void visitCeilDivExpr(AffineBinaryOpExpr expr);
  void visitFloorDivExpr(AffineBinaryOpExpr expr);

  //
  // t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
  //
  // A mod expression "expr mod c" is thus flattened by introducing a new local
  // variable q (= expr floordiv c), such that expr mod c is replaced with
  // 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
  void visitModExpr(AffineBinaryOpExpr expr);

protected:
  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // dividend and with respect to a positive constant divisor. localExpr is the
  // simplified tree expression (AffineExpr) corresponding to the quantifier.
  virtual void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                                  AffineExpr localExpr);

private:
  // t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
  // A floordiv is thus flattened by introducing a new local variable q, and
  // replacing that expression with 'q' while adding the constraints
  // c * q <= expr <= c * q + c - 1 to localVarCst (done by
  // FlatAffineConstraints::addLocalFloorDiv).
  //
  // A ceildiv is similarly flattened:
  // t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
  void visitDivExpr(AffineBinaryOpExpr expr, bool isCeil);

  int findLocalId(AffineExpr localExpr);

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }
  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};

} // end namespace mlir

#endif // MLIR_IR_AFFINE_EXPR_VISITOR_H

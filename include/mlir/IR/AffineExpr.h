//===- AffineMap.h - MLIR Affine Map Class ----------------------*- C++ -*-===//
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
// An affine expression is an affine combination of dimension identifiers and
// symbols, including ceildiv/floordiv/mod by a constant integer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINE_EXPR_H
#define MLIR_IR_AFFINE_EXPR_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class MLIRContext;

/// A one-dimensional affine expression.
/// AffineExpression's are immutable (like Type's)
class AffineExpr {
 public:
  enum class Kind {
    // Add.
    Add,
    // Mul.
    Mul,
    // Mod.
    Mod,
    // Floordiv
    FloorDiv,
    // Ceildiv
    CeilDiv,

    /// This is a marker for the last affine binary op. The range of binary op's
    /// is expected to be this element and earlier.
    LAST_AFFINE_BINARY_OP = CeilDiv,

    // Unary op negation
    Neg,

    // Constant integer.
    Constant,
    // Dimensional identifier.
    DimId,
    // Symbolic identifier.
    SymbolId,
  };

  /// Return the classification for this type.
  Kind getKind() const { return kind; }

  ~AffineExpr() = default;

  void print(raw_ostream &os) const;
  void dump() const;

 protected:
  explicit AffineExpr(Kind kind) : kind(kind) {}

 private:
  /// Classification of the subclass
  const Kind kind;
};

/// Binary affine expression.
class AffineBinaryOpExpr : public AffineExpr {
 public:
  static AffineBinaryOpExpr *get(Kind kind, AffineExpr *lhsOperand,
                                 AffineExpr *rhsOperand, MLIRContext *context);

  AffineExpr *getLeftOperand() const { return lhsOperand; }
  AffineExpr *getRightOperand() const { return rhsOperand; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() <= Kind::LAST_AFFINE_BINARY_OP;
  }

 protected:
  explicit AffineBinaryOpExpr(Kind kind, AffineExpr *lhsOperand,
                              AffineExpr *rhsOperand)
      : AffineExpr(kind), lhsOperand(lhsOperand), rhsOperand(rhsOperand) {}

  AffineExpr *const lhsOperand;
  AffineExpr *const rhsOperand;
};

/// Binary affine add expression.
class AffineAddExpr : public AffineBinaryOpExpr {
 public:
  static AffineAddExpr *get(AffineExpr *lhsOperand, AffineExpr *rhsOperand,
                            MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Add;
  }

 private:
  explicit AffineAddExpr(AffineExpr *lhsOperand, AffineExpr *rhsOperand)
      : AffineBinaryOpExpr(Kind::Add, lhsOperand, rhsOperand) {}
};

/// Binary affine mul expression.
class AffineMulExpr : public AffineBinaryOpExpr {
 public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Mul;
  }

 private:
  explicit AffineMulExpr(AffineExpr *lhsOperand, AffineExpr *rhsOperand)
      : AffineBinaryOpExpr(Kind::Mul, lhsOperand, rhsOperand) {}
};

/// Binary affine mod expression.
class AffineModExpr : public AffineBinaryOpExpr {
 public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Mod;
  }

 private:
  explicit AffineModExpr(AffineExpr *lhsOperand, AffineExpr *rhsOperand)
      : AffineBinaryOpExpr(Kind::Mod, lhsOperand, rhsOperand) {}
};

/// Binary affine floordiv expression.
class AffineFloorDivExpr : public AffineBinaryOpExpr {
 public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::FloorDiv;
  }

 private:
  explicit AffineFloorDivExpr(AffineExpr *lhsOperand, AffineExpr *rhsOperand)
      : AffineBinaryOpExpr(Kind::FloorDiv, lhsOperand, rhsOperand) {}
};

/// Binary affine ceildiv expression.
class AffineCeilDivExpr : public AffineBinaryOpExpr {
 public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::CeilDiv;
  }

 private:
  explicit AffineCeilDivExpr(AffineExpr *lhsOperand, AffineExpr *rhsOperand)
      : AffineBinaryOpExpr(Kind::CeilDiv, lhsOperand, rhsOperand) {}
};

/// Unary affine expression.
class AffineUnaryOpExpr : public AffineExpr {
 public:
  static AffineUnaryOpExpr *get(const AffineExpr &operand,
                                MLIRContext *context);

  static AffineUnaryOpExpr *get(const AffineExpr &operand);
  AffineExpr *getOperand() const { return operand; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Neg;
  }

 private:
  explicit AffineUnaryOpExpr(Kind kind, AffineExpr *operand)
      : AffineExpr(kind), operand(operand) {}

  AffineExpr *operand;
};

/// A argument identifier appearing in an affine expression
class AffineDimExpr : public AffineExpr {
 public:
  static AffineDimExpr *get(unsigned position, MLIRContext *context);

  unsigned getPosition() const { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::DimId;
  }

 private:
  explicit AffineDimExpr(unsigned position)
      : AffineExpr(Kind::DimId), position(position) {}

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// A symbolic identifier appearing in an affine expression
class AffineSymbolExpr : public AffineExpr {
 public:
  static AffineSymbolExpr *get(unsigned position, MLIRContext *context);

  unsigned getPosition() const { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::SymbolId;
  }

 private:
  explicit AffineSymbolExpr(unsigned position)
      : AffineExpr(Kind::SymbolId), position(position) {}

  /// Position of this identifier in the symbol list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
class AffineConstantExpr : public AffineExpr {
 public:
  static AffineConstantExpr *get(int64_t constant, MLIRContext *context);

  int64_t getValue() const { return constant; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Constant;
  }

 private:
  explicit AffineConstantExpr(int64_t constant)
      : AffineExpr(Kind::Constant), constant(constant) {}

  // The constant.
  int64_t constant;
};

} // end namespace mlir

#endif  // MLIR_IR_AFFINE_EXPR_H

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
     Add,
     Sub,
     Mul,
     Mod,
     FloorDiv,
     CeilDiv,

     /// This is a marker for the last affine binary op. The range of binary
     /// op's is expected to be this element and earlier.
     LAST_AFFINE_BINARY_OP = CeilDiv,

     // Constant integer.
     Constant,
     // Dimensional identifier.
     DimId,
     // Symbolic identifier.
     SymbolId,
   };

   /// Return the classification for this type.
   Kind getKind() const { return kind; }

   void print(raw_ostream &os) const;
   void dump() const;

   /// Returns true if this expression is made out of only symbols and
   /// constants (no dimensional identifiers).
   bool isSymbolic() const;

   /// Returns true if this is a pure affine expression, i.e., multiplication,
   /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
   bool isPureAffine() const;

 protected:
  explicit AffineExpr(Kind kind) : kind(kind) {}

 private:
  AffineExpr(const AffineExpr&) = delete;
  void operator=(const AffineExpr&) = delete;

  /// Classification of the subclass
  const Kind kind;
};

inline raw_ostream &operator<<(raw_ostream &os, const AffineExpr &expr) {
  expr.print(os);
  return os;
}

/// Binary affine expression.
class AffineBinaryOpExpr : public AffineExpr {
 public:
   AffineExpr *getLHS() const { return lhs; }
   AffineExpr *getRHS() const { return rhs; }

   /// Methods for support type inquiry through isa, cast, and dyn_cast.
   static bool classof(const AffineExpr *expr) {
     return expr->getKind() <= Kind::LAST_AFFINE_BINARY_OP;
  }

 protected:
   static AffineBinaryOpExpr *get(Kind kind, AffineExpr *lhs, AffineExpr *rhs,
                                  MLIRContext *context);

   explicit AffineBinaryOpExpr(Kind kind, AffineExpr *lhs, AffineExpr *rhs)
       : AffineExpr(kind), lhs(lhs), rhs(rhs) {}

   AffineExpr *const lhs;
   AffineExpr *const rhs;
};

/// Binary affine add expression.
class AffineAddExpr : public AffineBinaryOpExpr {
 public:
   static AffineExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                          MLIRContext *context);

   /// Methods for support type inquiry through isa, cast, and dyn_cast.
   static bool classof(const AffineExpr *expr) {
     return expr->getKind() == Kind::Add;
  }
  void print(raw_ostream &os) const;

private:
  /// Simplify the addition of two affine expressions.
  static AffineExpr *simplify(AffineExpr *lhs, AffineExpr *rhs,
                              MLIRContext *context);

  explicit AffineAddExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::Add, lhs, rhs) {}
};

/// Binary affine subtract expression.
class AffineSubExpr : public AffineBinaryOpExpr {
public:
  static AffineSubExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                            MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Sub;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineSubExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::Sub, lhs, rhs) {}
};

/// Binary affine multiplication expression.
class AffineMulExpr : public AffineBinaryOpExpr {
public:
  static AffineMulExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                            MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Mul;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineMulExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::Mul, lhs, rhs) {}
};

/// Binary affine modulo operation expression.
class AffineModExpr : public AffineBinaryOpExpr {
public:
  static AffineModExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                            MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::Mod;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineModExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::Mod, lhs, rhs) {}
};

/// Binary affine floordiv expression.
class AffineFloorDivExpr : public AffineBinaryOpExpr {
 public:
   static AffineFloorDivExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                                  MLIRContext *context);

   /// Methods for support type inquiry through isa, cast, and dyn_cast.
   static bool classof(const AffineExpr *expr) {
     return expr->getKind() == Kind::FloorDiv;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineFloorDivExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::FloorDiv, lhs, rhs) {}
};

/// Binary affine ceildiv expression.
class AffineCeilDivExpr : public AffineBinaryOpExpr {
public:
  static AffineCeilDivExpr *get(AffineExpr *lhs, AffineExpr *rhs,
                                MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::CeilDiv;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineCeilDivExpr(AffineExpr *lhs, AffineExpr *rhs)
      : AffineBinaryOpExpr(Kind::CeilDiv, lhs, rhs) {}
};

/// A dimensional identifier appearing in an affine expression.
///
/// This is a POD type of int size; so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class AffineDimExpr : public AffineExpr {
public:
  static AffineDimExpr *get(unsigned position, MLIRContext *context);

  unsigned getPosition() const { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::DimId;
  }
  void print(raw_ostream &os) const;

private:
  explicit AffineDimExpr(unsigned position)
      : AffineExpr(Kind::DimId), position(position) {}

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// A symbolic identifier appearing in an affine expression.
//
/// This is a POD type of int size, so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class AffineSymbolExpr : public AffineExpr {
 public:
  static AffineSymbolExpr *get(unsigned position, MLIRContext *context);

  unsigned getPosition() const { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExpr *expr) {
    return expr->getKind() == Kind::SymbolId;
  }
  void print(raw_ostream &os) const;

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
  void print(raw_ostream &os) const;

private:
  explicit AffineConstantExpr(int64_t constant)
      : AffineExpr(Kind::Constant), constant(constant) {}

  // The constant.
  int64_t constant;
};

} // end namespace mlir

#endif  // MLIR_IR_AFFINE_EXPR_H

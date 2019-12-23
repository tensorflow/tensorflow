//===- SDBM.h - MLIR SDBM declaration ---------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A striped difference-bound matrix (SDBM) is a set in Z^N (or R^N) defined
// as {(x_1, ... x_n) | f(x_1, ... x_n) >= 0} where f is an SDBM expression.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SDBM_SDBM_H
#define MLIR_DIALECT_SDBM_SDBM_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {

class MLIRContext;
class SDBMDialect;
class SDBMExpr;
class SDBMTermExpr;

/// A utility class for SDBM to represent an integer with potentially infinite
/// positive value. This uses the largest value of int64_t to represent infinity
/// and redefines the arithmetic operators so that the infinity "saturates":
///   inf + x = inf,
///   inf - x = inf.
/// If a sum of two finite values reaches the largest value of int64_t, the
/// behavior of IntInfty is undefined (in practice, it asserts), similarly to
/// regular signed integer overflow.
class IntInfty {
public:
  constexpr static int64_t infty = std::numeric_limits<int64_t>::max();

  /*implicit*/ IntInfty(int64_t v) : value(v) {}

  IntInfty &operator=(int64_t v) {
    value = v;
    return *this;
  }

  static IntInfty infinity() { return IntInfty(infty); }

  int64_t getValue() const { return value; }
  explicit operator int64_t() const { return value; }

  bool isFinite() { return value != infty; }

private:
  int64_t value;
};

inline IntInfty operator+(IntInfty lhs, IntInfty rhs) {
  if (!lhs.isFinite() || !rhs.isFinite())
    return IntInfty::infty;

  // Check for overflows, treating the sum of two values adding up to INT_MAX as
  // overflow.  Convert values to unsigned to get an extra bit and avoid the
  // undefined behavior of signed integer overflows.
  assert((lhs.getValue() <= 0 || rhs.getValue() <= 0 ||
          static_cast<uint64_t>(lhs.getValue()) +
                  static_cast<uint64_t>(rhs.getValue()) <
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) &&
         "IntInfty overflow");
  // Check for underflows by converting values to unsigned to avoid undefined
  // behavior of signed integers perform the addition (bitwise result is same
  // because numbers are required to be two's complement in C++) and check if
  // the sign bit remains negative.
  assert((lhs.getValue() >= 0 || rhs.getValue() >= 0 ||
          ((static_cast<uint64_t>(lhs.getValue()) +
            static_cast<uint64_t>(rhs.getValue())) >>
           63) == 1) &&
         "IntInfty underflow");

  return lhs.getValue() + rhs.getValue();
}

inline bool operator<(IntInfty lhs, IntInfty rhs) {
  return lhs.getValue() < rhs.getValue();
}

inline bool operator<=(IntInfty lhs, IntInfty rhs) {
  return lhs.getValue() <= rhs.getValue();
}

inline bool operator==(IntInfty lhs, IntInfty rhs) {
  return lhs.getValue() == rhs.getValue();
}

inline bool operator!=(IntInfty lhs, IntInfty rhs) { return !(lhs == rhs); }

/// Striped difference-bound matrix is a representation of an integer set bound
/// by a system of SDBMExprs interpreted as inequalities "expr <= 0".
class SDBM {
public:
  /// Obtain an SDBM from a list of SDBM expressions treated as inequalities and
  /// equalities with zero.
  static SDBM get(ArrayRef<SDBMExpr> inequalities,
                  ArrayRef<SDBMExpr> equalities);

  void getSDBMExpressions(SDBMDialect *dialect,
                          SmallVectorImpl<SDBMExpr> &inequalities,
                          SmallVectorImpl<SDBMExpr> &equalities);

  void print(raw_ostream &os);
  void dump();

  IntInfty operator()(int i, int j) { return at(i, j); }

private:
  /// Get the given element of the difference bounds matrix.  First index
  /// corresponds to the negative term of the difference, second index
  /// corresponds to the positive term of the difference.
  IntInfty &at(int i, int j) { return matrix[i * getNumVariables() + j]; }

  /// Populate `inequalities` and `equalities` based on the values at(row,col)
  /// and at(col,row) of the DBM.  Depending on the values being finite and
  /// being subsumed by stripe expressions, this may or may not add elements to
  /// the lists of equalities and inequalities.
  void convertDBMElement(unsigned row, unsigned col, SDBMTermExpr rowExpr,
                         SDBMTermExpr colExpr,
                         SmallVectorImpl<SDBMExpr> &inequalities,
                         SmallVectorImpl<SDBMExpr> &equalities);

  /// Populate `inequalities` based on the value at(pos,pos) of the DBM. Only
  /// adds new inequalities if the inequality is not trivially true.
  void convertDBMDiagonalElement(unsigned pos, SDBMTermExpr expr,
                                 SmallVectorImpl<SDBMExpr> &inequalities);

  /// Get the total number of elements in the matrix.
  unsigned getNumVariables() const {
    return 1 + numDims + numSymbols + numTemporaries;
  }

  /// Get the position in the matrix that corresponds to the given dimension.
  unsigned getDimPosition(unsigned position) const { return 1 + position; }

  /// Get the position in the matrix that corresponds to the given symbol.
  unsigned getSymbolPosition(unsigned position) const {
    return 1 + numDims + position;
  }

  /// Get the position in the matrix that corresponds to the given temporary.
  unsigned getTemporaryPosition(unsigned position) const {
    return 1 + numDims + numSymbols + position;
  }

  /// Number of dimensions in the system,
  unsigned numDims;
  /// Number of symbols in the system.
  unsigned numSymbols;
  /// Number of temporary variables in the system.
  unsigned numTemporaries;

  /// Difference bounds matrix, stored as a linearized row-major vector.
  /// Each value in this matrix corresponds to an inequality
  ///
  ///   v@col - v@row <= at(row, col)
  ///
  /// where v@col and v@row are the variables that correspond to the linearized
  /// position in the matrix.  The positions correspond to
  ///
  ///   - constant 0 (producing constraints v@col <= X and -v@row <= Y);
  ///   - SDBM expression dimensions (d0, d1, ...);
  ///   - SDBM expression symbols (s0, s1, ...);
  ///   - temporary variables (t0, t1, ...).
  ///
  /// Temporary variables are introduced to represent expressions that are not
  /// trivially a difference between two variables.  For example, if one side of
  /// a difference expression is itself a stripe expression, it will be replaced
  /// with a temporary variable assigned equal to this expression.
  ///
  /// Infinite entries in the matrix correspond correspond to an absence of a
  /// constraint:
  ///
  ///   v@col - v@row <= infinity
  ///
  /// is trivially true.  Negated values at symmetric positions in the matrix
  /// allow one to couple two inequalities into a single equality.
  std::vector<IntInfty> matrix;

  /// The mapping between the indices of variables in the DBM and the stripe
  /// expressions they are equal to.  These expressions are stored as they
  /// appeared when constructing an SDBM from a SDBMExprs, in particular no
  /// temporaries can appear in these expressions.  This removes the need to
  /// iteratively substitute definitions of the temporaries in the reverse
  /// conversion.
  DenseMap<unsigned, SDBMExpr> stripeToPoint;
};

} // namespace mlir

#endif // MLIR_DIALECT_SDBM_SDBM_H

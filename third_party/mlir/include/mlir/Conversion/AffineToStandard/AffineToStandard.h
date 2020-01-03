//===- AffineToStandard.h - Convert Affine to Standard dialect --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
#define MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineExpr;
class AffineForOp;
class Location;
struct LogicalResult;
class MLIRContext;
class OpBuilder;
class RewritePattern;
class Value;

// Owning list of rewriting patterns.
class OwningRewritePatternList;

/// Emit code that computes the given affine expression using standard
/// arithmetic operations applied to the provided dimension and symbol values.
Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                       ArrayRef<Value> dimValues, ArrayRef<Value> symbolValues);

/// Collect a set of patterns to convert from the Affine dialect to the Standard
/// dialect, in particular convert structured affine control flow into CFG
/// branch-based control flow.
void populateAffineToStdConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx);

/// Emit code that computes the lower bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineLowerBound(AffineForOp op, OpBuilder &builder);

/// Emit code that computes the upper bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineUpperBound(AffineForOp op, OpBuilder &builder);
} // namespace mlir

#endif // MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H

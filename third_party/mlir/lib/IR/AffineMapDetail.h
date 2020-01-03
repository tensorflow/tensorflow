//===- AffineMapDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of AffineMap.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINEMAPDETAIL_H_
#define AFFINEMAPDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct AffineMapStorage {
  unsigned numDims;
  unsigned numSymbols;

  /// The affine expressions for this (multi-dimensional) map.
  /// TODO: use trailing objects for this.
  ArrayRef<AffineExpr> results;

  MLIRContext *context;
};

} // end namespace detail
} // end namespace mlir

#endif // AFFINEMAPDETAIL_H_

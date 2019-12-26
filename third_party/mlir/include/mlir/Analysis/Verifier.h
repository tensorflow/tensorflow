//===- Verifier.h - Verifier analysis for MLIR structures -------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_VERIFIER_H
#define MLIR_ANALYSIS_VERIFIER_H

namespace mlir {
struct LogicalResult;
class Operation;

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs, on this operation and any nested operations. On error, this
/// reports the error through the MLIRContext and returns failure.
LogicalResult verify(Operation *op);
} //  end namespace mlir

#endif

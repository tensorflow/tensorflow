//===- FxpMathOps.cpp - Op implementation for FxpMathOps ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::fxpmath;

#define GET_OP_CLASSES
#include "mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc"

FxpMathOpsDialect::FxpMathOpsDialect(MLIRContext *context)
    : Dialect(/*name=*/"fxpmath", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc"
      >();
}

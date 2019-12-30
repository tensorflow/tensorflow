//===- DialectRegistration.cpp - Register FxpMathOps dialect --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"

using namespace mlir;
using namespace mlir::fxpmath;

// Static initialization for the fxpmath ops dialect registration.
static mlir::DialectRegistration<FxpMathOpsDialect> FxpMathOps;

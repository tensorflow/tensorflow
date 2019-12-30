//===- LinalgRegistration.cpp - Register the linalg dialect statically ----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"

using namespace mlir;
using namespace mlir::linalg;

// Static initialization for LinalgOps dialect registration.
static DialectRegistration<LinalgDialect> LinalgOps;

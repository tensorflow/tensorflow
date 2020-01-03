//===- DialectRegistration.cpp - Register standard Op dialect -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Ops.h"
using namespace mlir;

// Static initialization for standard op dialect registration.
static DialectRegistration<StandardOpsDialect> StandardOps;

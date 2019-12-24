//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTDIALECT_H
#define MLIR_TESTDIALECT_H

#include "mlir/Analysis/CallInterfaces.h"
#include "mlir/Analysis/InferTypeOpInterface.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "TestOpEnums.h.inc"

namespace mlir {

class TestDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  TestDialect(MLIRContext *context);

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName() { return "test"; }

  LogicalResult verifyOperationAttribute(Operation *op,
                                         NamedAttribute namedAttr) override;
  LogicalResult verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                         unsigned argIndex,
                                         NamedAttribute namedAttr) override;
  LogicalResult verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                            unsigned resultIndex,
                                            NamedAttribute namedAttr) override;
};

#define GET_OP_CLASSES
#include "TestOps.h.inc"

} // end namespace mlir

#endif // MLIR_TESTDIALECT_H

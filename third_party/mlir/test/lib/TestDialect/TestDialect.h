//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
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

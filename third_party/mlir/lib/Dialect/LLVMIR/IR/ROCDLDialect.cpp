//===- ROCDLDialect.cpp - ROCDL IR Ops and Dialect registration -----------===//
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
// This file defines the types and operation details for the ROCDL IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The ROCDL dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace ROCDL {

//===----------------------------------------------------------------------===//
// Printing/parsing for ROCDL ops
//===----------------------------------------------------------------------===//

static void printROCDLOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << " ";
  p.printOperands(op->getOperands());
  if (op->getNumResults() > 0)
    interleaveComma(op->getResultTypes(), p << " : ");
}

// <operation> ::= `rocdl.XYZ` : type
static ParseResult parseROCDLOp(OpAsmParser &parser, OperationState &result) {
  Type type;
  return failure(parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.addTypeToList(type, result.types));
}

//===----------------------------------------------------------------------===//
// ROCDLDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO(herhut): This should be the llvm.rocdl dialect once this is supported.
ROCDLDialect::ROCDLDialect(MLIRContext *context) : Dialect("rocdl", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
      >();

  // Support unknown operations because not all ROCDL operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"

static DialectRegistration<ROCDLDialect> rocdlDialect;

} // namespace ROCDL
} // namespace mlir

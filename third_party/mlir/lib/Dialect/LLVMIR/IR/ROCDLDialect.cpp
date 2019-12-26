//===- ROCDLDialect.cpp - ROCDL IR Ops and Dialect registration -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

using namespace mlir;
using namespace ROCDL;

//===----------------------------------------------------------------------===//
// Printing/parsing for ROCDL ops
//===----------------------------------------------------------------------===//

static void printROCDLOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << " " << op->getOperands();
  if (op->getNumResults() > 0)
    p << " : " << op->getResultTypes();
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

namespace mlir {
namespace ROCDL {
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
} // namespace ROCDL
} // namespace mlir

static DialectRegistration<ROCDLDialect> rocdlDialect;

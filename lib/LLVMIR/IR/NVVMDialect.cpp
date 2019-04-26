//===- NVVMDialect.cpp - NVVM IR Ops and Dialect registration -------------===//
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
// This file defines the types and operation details for the NVVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The NVVM dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/LLVMIR/NVVMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMSpecialRegisterOp(OpAsmPrinter *p, Operation *op) {
  *p << op->getName() << " : ";
  if (op->getNumResults() == 1) {
    *p << op->getResult(0)->getType();
  } else {
    *p << "###invalid type###";
  }
}

// <operation> ::= `llvm.nvvm.XYZ` : type
static bool parseNVVMSpecialRegisterOp(OpAsmParser *parser,
                                       OperationState *result) {
  Type type;
  if (parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  result->addTypes(type);
  return false;
}

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO(herhut): This should be the llvm.nvvm dialect once this is supported.
NVVMDialect::NVVMDialect(MLIRContext *context) : Dialect("nvvm", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/LLVMIR/NVVMOps.cpp.inc"
      >();

  // Support unknown operations because not all NVVM operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/LLVMIR/NVVMOps.cpp.inc"

static DialectRegistration<NVVMDialect> nvvmDialect;

} // namespace NVVM
} // namespace mlir

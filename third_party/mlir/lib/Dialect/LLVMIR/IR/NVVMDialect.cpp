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

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

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
namespace NVVM {

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMIntrinsicOp(OpAsmPrinter *p, Operation *op) {
  *p << op->getName() << " ";
  p->printOperands(op->getOperands());
  if (op->getNumResults() > 0)
    interleaveComma(op->getResultTypes(), *p << " : ");
}

// <operation> ::= `llvm.nvvm.XYZ` : type
static ParseResult parseNVVMSpecialRegisterOp(OpAsmParser *parser,
                                              OperationState *result) {
  Type type;
  if (parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return failure();

  result->addTypes(type);
  return success();
}

static LLVM::LLVMDialect *getLlvmDialect(OpAsmParser *parser) {
  return parser->getBuilder()
      .getContext()
      ->getRegisteredDialect<LLVM::LLVMDialect>();
}

// <operation> ::=
//     `llvm.nvvm.shfl.sync.bfly %dst, %val, %offset, %clamp_and_mask`
//     : result_type
static ParseResult parseNVVMShflSyncBflyOp(OpAsmParser *parser,
                                           OperationState *result) {
  auto llvmDialect = getLlvmDialect(parser);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);

  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  return failure(parser->parseOperandList(ops) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->addTypeToList(type, result->types) ||
                 parser->resolveOperands(ops, {int32Ty, type, int32Ty, int32Ty},
                                         parser->getNameLoc(),
                                         result->operands));
}

// <operation> ::= `llvm.nvvm.vote.ballot.sync %mask, %pred` : result_type
static ParseResult parseNVVMVoteBallotOp(OpAsmParser *parser,
                                         OperationState *result) {
  auto llvmDialect = getLlvmDialect(parser);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto int1Ty = LLVM::LLVMType::getInt1Ty(llvmDialect);

  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  return failure(parser->parseOperandList(ops) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->addTypeToList(type, result->types) ||
                 parser->resolveOperands(ops, {int32Ty, int1Ty},
                                         parser->getNameLoc(),
                                         result->operands));
}

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO(herhut): This should be the llvm.nvvm dialect once this is supported.
NVVMDialect::NVVMDialect(MLIRContext *context) : Dialect("nvvm", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
      >();

  // Support unknown operations because not all NVVM operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"

static DialectRegistration<NVVMDialect> nvvmDialect;

} // namespace NVVM
} // namespace mlir

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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace NVVM;

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMIntrinsicOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << " " << op->getOperands();
  if (op->getNumResults() > 0)
    p << " : " << op->getResultTypes();
}

// <operation> ::= `llvm.nvvm.XYZ` : type
static ParseResult parseNVVMSpecialRegisterOp(OpAsmParser &parser,
                                              OperationState &result) {
  Type type;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  result.addTypes(type);
  return success();
}

static LLVM::LLVMDialect *getLlvmDialect(OpAsmParser &parser) {
  return parser.getBuilder()
      .getContext()
      ->getRegisteredDialect<LLVM::LLVMDialect>();
}

// <operation> ::=
//     `llvm.nvvm.shfl.sync.bfly %dst, %val, %offset, %clamp_and_mask`
//      ({return_value_and_is_valid})? : result_type
static ParseResult parseNVVMShflSyncBflyOp(OpAsmParser &parser,
                                           OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type resultType;
  if (parser.parseOperandList(ops) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType) ||
      parser.addTypeToList(resultType, result.types))
    return failure();

  auto type = resultType.cast<LLVM::LLVMType>();
  for (auto &attr : result.attributes) {
    if (attr.first != "return_value_and_is_valid")
      continue;
    if (type.isStructTy() && type.getStructNumElements() > 0)
      type = type.getStructElementType(0);
    break;
  }

  auto int32Ty = LLVM::LLVMType::getInt32Ty(getLlvmDialect(parser));
  return parser.resolveOperands(ops, {int32Ty, type, int32Ty, int32Ty},
                                parser.getNameLoc(), result.operands);
}

// <operation> ::= `llvm.nvvm.vote.ballot.sync %mask, %pred` : result_type
static ParseResult parseNVVMVoteBallotOp(OpAsmParser &parser,
                                         OperationState &result) {
  auto llvmDialect = getLlvmDialect(parser);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto int1Ty = LLVM::LLVMType::getInt1Ty(llvmDialect);

  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.addTypeToList(type, result.types) ||
                 parser.resolveOperands(ops, {int32Ty, int1Ty},
                                        parser.getNameLoc(), result.operands));
}

// <operation> ::= `llvm.nvvm.mma.sync %lhs... %rhs... %acc...`
//                 : signature_type
static ParseResult parseNVVMMmaOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 12> ops;
  Type type;
  llvm::SMLoc typeLoc;
  if (parser.parseOperandList(ops) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&typeLoc) || parser.parseType(type)) {
    return failure();
  }

  auto signature = type.dyn_cast<FunctionType>();
  if (!signature) {
    return parser.emitError(
        typeLoc, "expected the type to be the full list of input and output");
  }

  if (signature.getNumResults() != 1) {
    return parser.emitError(typeLoc, "expected single result");
  }

  return failure(parser.addTypeToList(signature.getResult(0), result.types) ||
                 parser.resolveOperands(ops, signature.getInputs(),
                                        parser.getNameLoc(), result.operands));
}

static void printNVVMMmaOp(OpAsmPrinter &p, MmaOp &op) {
  p << op.getOperationName() << " " << op.getOperands();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : "
    << FunctionType::get(llvm::to_vector<12>(op.getOperandTypes()),
                         op.getType(), op.getContext());
}

static LogicalResult verify(MmaOp op) {
  auto dialect = op.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto f16Ty = LLVM::LLVMType::getHalfTy(dialect);
  auto f16x2Ty = LLVM::LLVMType::getVectorTy(f16Ty, 2);
  auto f32Ty = LLVM::LLVMType::getFloatTy(dialect);
  auto f16x2x4StructTy = LLVM::LLVMType::getStructTy(
      dialect, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f32x8StructTy = LLVM::LLVMType::getStructTy(
      dialect, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type, 12> operand_types(op.getOperandTypes().begin(),
                                      op.getOperandTypes().end());
  if (operand_types != SmallVector<Type, 8>(8, f16x2Ty) &&
      operand_types != SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                             f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                             f32Ty, f32Ty, f32Ty}) {
    return op.emitOpError(
        "expected operands to be 4 <halfx2>s followed by either "
        "4 <halfx2>s or 8 floats");
  }
  if (op.getType() != f32x8StructTy && op.getType() != f16x2x4StructTy) {
    return op.emitOpError("expected result type to be a struct of either 4 "
                          "<halfx2>s or 8 floats");
  }

  auto alayout = op.getAttrOfType<StringAttr>("alayout");
  auto blayout = op.getAttrOfType<StringAttr>("blayout");

  if (!(alayout && blayout) ||
      !(alayout.getValue() == "row" || alayout.getValue() == "col") ||
      !(blayout.getValue() == "row" || blayout.getValue() == "col")) {
    return op.emitOpError(
        "alayout and blayout attributes must be set to either "
        "\"row\" or \"col\"");
  }

  if (operand_types == SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                             f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                             f32Ty, f32Ty, f32Ty} &&
      op.getType() == f32x8StructTy && alayout.getValue() == "row" &&
      blayout.getValue() == "row") {
    return success();
  }
  return op.emitOpError("unimplemented mma.sync variant");
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

namespace mlir {
namespace NVVM {
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
} // namespace NVVM
} // namespace mlir

static DialectRegistration<NVVMDialect> nvvmDialect;

//===- LinalgOps.cpp - Implementation of the linalg operations ------------===//
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
// This file implements a the Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linalg/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"

using namespace mlir;

//////////////////////////////////////////////////////////////////////////////
// BaseViewOp
//////////////////////////////////////////////////////////////////////////////
void mlir::BaseViewOp::build(Builder *b, OperationState *result, Value *buffer,
                             ArrayRef<Value *> indexings) {
  BufferType bufferType = buffer->getType().cast<BufferType>();
  result->addOperands({buffer});
  result->addOperands(indexings);
  assert(
      std::none_of(indexings.begin(), indexings.end(),
                   [](Value *v) { return !v->getType().isa<RangeType>(); }) &&
      "linalg.base_view takes only arguments of type linalg.range");

  Type elementType = bufferType.getElementType();
  result->addTypes(
      {ViewType::get(b->getContext(), elementType, indexings.size())});
}

LogicalResult mlir::BaseViewOp::verify() {
  if (llvm::empty(getOperands()))
    return emitOpError(
        "requires at least a buffer operand followed by indexings");
  auto bufferType = getOperand(0)->getType().dyn_cast<BufferType>();
  if (!bufferType)
    return emitOpError("first operand must be of BufferType");
  unsigned index = 0;
  for (auto indexing : getIndexings()) {
    if (!indexing->getType().isa<RangeType>()) {
      return emitOpError(Twine(index) + "^th index must be of range type");
    }
    ++index;
  }
  if (getViewType().getRank() != index)
    return emitOpError(
        "the rank of the base view must be the number of its indexings");
  return success();
}

bool mlir::BaseViewOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType bufferInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexingsInfo;
  Type type;
  if (parser->parseOperand(bufferInfo) ||
      parser->parseOperandList(indexingsInfo, -1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  ViewType viewType = type.dyn_cast<ViewType>();
  if (!viewType)
    return parser->emitError(parser->getNameLoc(), "view type expected");
  if (viewType.getRank() != indexingsInfo.size())
    return parser->emitError(parser->getNameLoc(),
                             "expected" + Twine(viewType.getRank()) +
                                 " range indexings");
  return parser->resolveOperand(
             bufferInfo,
             BufferType::get(type.getContext(), viewType.getElementType()),
             result->operands) ||
         (!indexingsInfo.empty() &&
          parser->resolveOperands(indexingsInfo,
                                  RangeType::get(type.getContext()),
                                  result->operands)) ||
         parser->addTypeToList(viewType, result->types);
}

// A BaseViewOp prints as:
//
// ```{.mlir}
//   linalg.base_view %0[%1, %2] : !linalg.view<?x?xf32>
// ```
//
// Where %0 is an ssa-value holding a buffer, %1 and %2 are ssa-value each
// holding a range.
void mlir::BaseViewOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getSupportingBuffer() << "[";
  interleave(
      getIndexings().begin(), getIndexings().end(),
      [&](mlir::Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << "] : " << getType();
}

//////////////////////////////////////////////////////////////////////////////
// BufferAllocOp
//////////////////////////////////////////////////////////////////////////////
void mlir::BufferAllocOp::build(Builder *b, OperationState *result, Type type,
                                Value *size) {
  result->addOperands({size});
  result->addTypes(type);
}

mlir::LogicalResult mlir::BufferAllocOp::verify() {
  if (!size() || !size()->getType().isa<IntegerType>() ||
      !size()->getType().cast<IntegerType>().isInteger(64))
    return emitOpError("first operand should be of type i64");
  if (!VectorType::isValidElementType(getElementType()) &&
      !getElementType().isa<VectorType>())
    return emitOpError("unsupported buffer element type");
  return mlir::success();
}

// A BufferAllocOp prints as:
//
// ```{.mlir}
//   linalg.alloc %0 : !linalg.buffer<f32>
// ```
void mlir::BufferAllocOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *size() << " : " << getType();
}

bool mlir::BufferAllocOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType sizeInfo;
  BufferType bufferType;
  auto int64Ty = parser->getBuilder().getIntegerType(64);
  if (parser->parseOperand(sizeInfo) || parser->parseColonType(bufferType))
    return true;
  if (bufferType.getElementType() != parser->getBuilder().getF32Type())
    return parser->emitError(
        parser->getNameLoc(),
        "Only buffer<f32> supported until mlir::Parser pieces are exposed");
  return parser->resolveOperands(sizeInfo, int64Ty, result->operands) ||
         parser->addTypeToList(bufferType, result->types);
}

//////////////////////////////////////////////////////////////////////////////
// BufferDeallocOp
//////////////////////////////////////////////////////////////////////////////
void mlir::BufferDeallocOp::build(Builder *b, OperationState *result,
                                  Value *buffer) {
  result->addOperands({buffer});
}

mlir::LogicalResult mlir::BufferDeallocOp::verify() {
  if (!getBuffer()->getType())
    return emitOpError("first operand should be of type buffer");
  return mlir::success();
}

// A BufferDeallocOp prints as:
//
// ```{.mlir}
//   linalg.dealloc %0 : !linalg.buffer<f32>
// ```
void mlir::BufferDeallocOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getBuffer() << " : " << getBufferType();
}

bool mlir::BufferDeallocOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType sizeInfo;
  BufferType bufferType;
  return parser->parseOperand(sizeInfo) || parser->parseColonType(bufferType) ||
         parser->resolveOperands(sizeInfo, bufferType, result->operands);
}

//////////////////////////////////////////////////////////////////////////////
// RangeOp
//////////////////////////////////////////////////////////////////////////////
void mlir::RangeOp::build(Builder *b, OperationState *result, Value *min,
                          Value *max, Value *step) {
  result->addOperands({min, max, step});
  result->addTypes({RangeType::get(b->getContext())});
}

// Verification is simply that a RangeOp takes 3 index ssa-value.
mlir::LogicalResult mlir::RangeOp::verify() {
  if (!min() || !min()->getType().isa<IndexType>())
    return emitOpError("first operand should be of type index");
  if (!max() || !max()->getType().isa<IndexType>())
    return emitOpError("second operand should be of type index");
  if (!step() || !step()->getType().isa<IndexType>())
    return emitOpError("third operand should be of type index");
  return mlir::success();
}

// A RangeOp prints as:
//
// ```{.mlir}
//   linalg.range %0:%1:%2 : !linalg.range
// ```
void mlir::RangeOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *min() << ":" << *max() << ":" << *step()
     << " : " << getType();
}

bool mlir::RangeOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> rangeInfo(3);
  RangeType type;
  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(rangeInfo[0]) || parser->parseColon() ||
         parser->parseOperand(rangeInfo[1]) || parser->parseColon() ||
         parser->parseOperand(rangeInfo[2]) || parser->parseColonType(type) ||
         parser->resolveOperands(rangeInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type, result->types);
}

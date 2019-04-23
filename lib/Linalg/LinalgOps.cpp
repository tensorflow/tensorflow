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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"

using namespace mlir;

//////////////////////////////////////////////////////////////////////////////
// BufferAllocOp
//////////////////////////////////////////////////////////////////////////////
void mlir::BufferAllocOp::build(Builder *b, OperationState *result, Type type,
                                Value *size) {
  result->addOperands({size});
  result->addTypes(type);
}

mlir::LogicalResult mlir::BufferAllocOp::verify() {
  if (!size() || !size()->getType().isa<IndexType>())
    return emitOpError("first operand should be of type index");
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
  auto indexTy = parser->getBuilder().getIndexType();
  if (parser->parseOperand(sizeInfo) || parser->parseColonType(bufferType))
    return true;
  if (bufferType.getElementType() != parser->getBuilder().getF32Type())
    return parser->emitError(
        parser->getNameLoc(),
        "Only buffer<f32> supported until mlir::Parser pieces are exposed");
  return parser->resolveOperands(sizeInfo, indexTy, result->operands) ||
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

//////////////////////////////////////////////////////////////////////////////
// SliceOp
//////////////////////////////////////////////////////////////////////////////
void mlir::SliceOp::build(Builder *b, OperationState *result, Value *base,
                          ArrayRef<Value *> indexings) {
  result->addOperands({base});
  result->addOperands(indexings);

  ViewType viewType = base->getType().cast<ViewType>();
  unsigned rank = viewType.getRank();
  for (auto *i : indexings)
    if (!i->getType().isa<RangeType>())
      rank--;
  Type elementType = viewType.getElementType();
  result->addTypes(
      {ViewType::get(b->getContext(), elementType, indexings.size())});
}

LogicalResult mlir::SliceOp::verify() {
  if (llvm::empty(getOperands()))
    return emitOpError(
        "requires at least a view operand followed by 'rank' indices");
  if (!getOperand(0)->getDefiningOp()->isa<ViewOp>())
    return emitOpError(
        "requires at least a view operand followed by 'rank' indices");

  auto viewOp = getOperand(0)->getDefiningOp()->dyn_cast<ViewOp>();
  if (!viewOp)
    return emitOpError("first operand must come from a ViewOp");
  unsigned rank = getBaseViewRank();
  if (llvm::size(getIndexings()) != rank) {
    return emitOpError("requires at least a view operand followed by " +
                       Twine(rank) + " indexings");
  }
  unsigned index = 0;
  for (auto indexing : getIndexings()) {
    if (!indexing->getType().isa<RangeType>() &&
        !indexing->getType().isa<IndexType>()) {
      return emitOpError(Twine(index) +
                         "^th index must be of range or index type");
    }
    if (indexing->getType().isa<IndexType>())
      --rank;
    ++index;
  }
  if (getRank() != rank) {
    return emitOpError("the rank of the view must be the number of its range "
                       "indices: " +
                       Twine(rank));
  }
  return success();
}

bool mlir::SliceOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType baseInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexingsInfo;
  SmallVector<Type, 8> types;
  if (parser->parseOperand(baseInfo) ||
      parser->parseOperandList(indexingsInfo, -1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(types))
    return true;

  if (types.size() != 2 + indexingsInfo.size())
    return parser->emitError(parser->getNameLoc(),
                             "unexpected number of types ");
  ViewType baseViewType = types[0].dyn_cast<ViewType>();
  if (!baseViewType)
    return parser->emitError(parser->getNameLoc(),
                             "view type expected for first type");
  if (indexingsInfo.size() != baseViewType.getRank())
    return parser->emitError(parser->getNameLoc(),
                             "expected " + Twine(baseViewType.getRank()) +
                                 " indexings");
  ViewType viewType = types.back().dyn_cast<ViewType>();
  if (!viewType)
    return parser->emitError(parser->getNameLoc(), "view type expected");

  ArrayRef<Type> indexingTypes =
      ArrayRef<Type>(types).drop_front(1).drop_back(1);
  if (indexingTypes.size() != baseViewType.getRank())
    return parser->emitError(parser->getNameLoc(),
                             "expected " + Twine(baseViewType.getRank()) +
                                 " indexing types");
  return parser->resolveOperand(baseInfo, baseViewType, result->operands) ||
         (!indexingsInfo.empty() &&
          parser->resolveOperands(indexingsInfo, indexingTypes,
                                  indexingsInfo.front().location,
                                  result->operands)) ||
         parser->addTypeToList(viewType, result->types);
}

// A SliceOp prints as:
//
// ```{.mlir}
//   linalg.slice %0[%1, %2] :
//     !linalg.view<?x?xf32>, [indexing-types], !linalg.view<?x?xf32>
// ```
//
// Where %0 is an ssa-value holding a view created from a buffer, %1 and %2 are
// ssa-value each holding a range.
void mlir::SliceOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getBaseView() << "[";
  interleave(
      getIndexings().begin(), getIndexings().end(),
      [&](mlir::Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << "] : " << getBaseViewType();
  for (auto indexing : getIndexings()) {
    *p << ", " << indexing->getType();
  }
  *p << ", " << getType();
}

ViewOp mlir::SliceOp::getBaseViewOp() {
  return getOperand(0)->getDefiningOp()->cast<ViewOp>();
}

ViewType mlir::SliceOp::getBaseViewType() {
  return getBaseViewOp().getType().cast<ViewType>();
}

SmallVector<Value *, 8> mlir::SliceOp::getRanges() {
  llvm::SmallVector<Value *, 8> res;
  for (auto *operand : getIndexings()) {
    if (!operand->getType().isa<IndexType>()) {
      res.push_back(operand);
    }
  }
  return res;
}

//////////////////////////////////////////////////////////////////////////////
// ViewOp
//////////////////////////////////////////////////////////////////////////////
void mlir::ViewOp::build(Builder *b, OperationState *result, Value *buffer,
                         ArrayRef<Value *> indexings) {
  BufferType bufferType = buffer->getType().cast<BufferType>();
  result->addOperands({buffer});
  result->addOperands(indexings);
  assert(
      std::none_of(indexings.begin(), indexings.end(),
                   [](Value *v) { return !v->getType().isa<RangeType>(); }) &&
      "linalg.view takes only arguments of type linalg.range");

  Type elementType = bufferType.getElementType();
  result->addTypes(
      {ViewType::get(b->getContext(), elementType, indexings.size())});
}

LogicalResult mlir::ViewOp::verify() {
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
        "the rank of the view must be the number of its indexings");
  return success();
}

bool mlir::ViewOp::parse(OpAsmParser *parser, OperationState *result) {
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

// A ViewOp prints as:
//
// ```{.mlir}
//   linalg.view %0[%1, %2] : !linalg.view<?x?xf32>
// ```
//
// Where %0 is an ssa-value holding a buffer, %1 and %2 are ssa-value each
// holding a range.
void mlir::ViewOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getSupportingBuffer() << "[";
  interleave(
      getIndexings().begin(), getIndexings().end(),
      [&](mlir::Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << "] : " << getType();
}

namespace mlir {
namespace impl {

// A LinalgLibraryOp prints as:
//
// ```{.mlir}
//   concrete_op_name (ssa-inputs, ssa-outputs) : view-types
// ```
//
// for example:
//
// ```
//   linalg.matmul(%0, %1, %2) :
//     !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
// ```
//
// Where %0, %1 and %2 are ssa-values of type ViewType.
void printLinalgLibraryOp(mlir::OpAsmPrinter *p, Operation *op) {
  assert(op->getAbstractOperation() && "unregistered operation");
  *p << op->getName().getStringRef() << "(";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](mlir::Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << ") : ";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](mlir::Value *v) { *p << v->getType(); }, [&]() { *p << ", "; });
}

bool parseLinalgLibraryOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return parser->parseOperandList(ops, -1, OpAsmParser::Delimiter::Paren) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonTypeList(types) ||
         parser->resolveOperands(ops, types, parser->getNameLoc(),
                                 result->operands);
}
} // namespace impl

#define GET_OP_CLASSES
#include "mlir/Linalg/LinalgOps.cpp.inc"

} // namespace mlir

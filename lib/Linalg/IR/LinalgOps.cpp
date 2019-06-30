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

#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Linalg/Utils/Utils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

////////////////////////////////////////////////////////////////////////////////
// ForOp.
////////////////////////////////////////////////////////////////////////////////
// Check that if a "block" has a terminator, it is an `TerminatorOp`.
static LogicalResult checkHasTerminator(OpState &op, Block &block) {
  if (block.empty() || isa<TerminatorOp>(block.back()))
    return success();

  return op.emitOpError("expects regions to end with '" +
                        TerminatorOp::getOperationName() + "'")
             .attachNote()
         << "in custom textual format, the absence of terminator implies '"
         << TerminatorOp::getOperationName() << "'";
}

// Insert `linalg.terminator` at the end of the ForOp only region's only block
// if it does not have a terminator already.  If a new `linalg.terminator` is
// inserted, the location is specified by `loc`. If the region is empty, insert
// a new block first.
static void ensureTerminator(Region &region, Builder &builder, Location loc) {
  impl::ensureRegionTerminator<TerminatorOp>(region, builder, loc);
}

void mlir::linalg::ForOp::build(Builder *builder, OperationState *result,
                                Value *lb, Value *ub, Value *step) {
  result->addOperands({lb, ub, step});
  Region *bodyRegion = result->addRegion();
  Block *body = new Block();
  body->addArgument(IndexType::get(builder->getContext()));
  bodyRegion->push_back(body);
  ensureTerminator(*bodyRegion, *builder, result->location);
}

LogicalResult mlir::linalg::ForOp::verify() {
  if (!getLowerBound()->getType().isa<IndexType>())
    return emitOpError("lower bound operand must be an index");
  if (!getUpperBound()->getType().isa<IndexType>())
    return emitOpError("upper bound operand must be an index");
  if (!getStep()->getType().dyn_cast<IndexType>())
    return emitOpError("step operand must be an index");
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(getStep()->getDefiningOp()))
    if (cst.getValue() <= 0)
      return emitOpError("constant step operand must be positive");

  if (std::next(getOperation()->getRegions().begin()) !=
      getOperation()->getRegions().end())
    return emitOpError("operation expected to have exactly one region");

  auto &bodyRegion = getOperation()->getRegion(0);
  // The body region must contain a single basic block.
  if (bodyRegion.empty() || std::next(bodyRegion.begin()) != bodyRegion.end())
    return emitOpError("expected body region to have a single block");
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (body->getNumArguments() != 1 ||
      !body->getArgument(0)->getType().isIndex())
    return emitOpError("expected body to have a single index argument for "
                       "the induction variable");
  if (failed(checkHasTerminator(*this, *body)))
    return failure();
  return success();
}

void mlir::linalg::ForOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getInductionVar() << " = "
     << *getLowerBound() << " to " << *getUpperBound() << " step "
     << *getStep();
  p->printRegion(getRegion(),
                 /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/false);
  p->printOptionalAttrDict(getAttrs());
}

ParseResult mlir::linalg::ForOp::parse(OpAsmParser *parser,
                                       OperationState *result) {
  auto &builder = parser->getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser->parseRegionArgument(inductionVariable) || parser->parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser->parseOperand(lb) ||
      parser->resolveOperand(lb, indexType, result->operands) ||
      parser->parseKeyword("to") || parser->parseOperand(ub) ||
      parser->resolveOperand(ub, indexType, result->operands) ||
      parser->parseKeyword("step") || parser->parseOperand(step) ||
      parser->resolveOperand(step, indexType, result->operands))
    return failure();

  // Parse the body region.
  Region *body = result->addRegion();
  if (parser->parseRegion(*body, inductionVariable, indexType))
    return failure();

  ensureTerminator(*body, builder, result->location);

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes))
    return failure();

  return success();
}

mlir::linalg::ForOp mlir::linalg::getForInductionVarOwner(Value *val) {
  auto *ivArg = dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return ForOp();
  assert(ivArg->getOwner() && "unlinked block argument");
  auto *containingInst = ivArg->getOwner()->getContainingOp();
  return dyn_cast_or_null<ForOp>(containingInst);
}

////////////////////////////////////////////////////////////////////////////////
// LoadOp.
////////////////////////////////////////////////////////////////////////////////
void mlir::linalg::LoadOp::build(Builder *b, OperationState *result,
                                 Value *view, ArrayRef<Value *> indices) {
  auto viewType = view->getType().cast<ViewType>();
  result->addOperands(view);
  result->addOperands(indices);
  result->addTypes(viewType.getElementType());
}

// A LoadOp prints as:
//
// ```{.mlir}
//    %0 = linalg.load %V[%c0] : !linalg.view<?xf32>
// ```
void mlir::linalg::LoadOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getView() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getViewType();
}

ParseResult mlir::linalg::LoadOp::parse(OpAsmParser *parser,
                                        OperationState *result) {
  OpAsmParser::OperandType viewInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  ViewType type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(viewInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type) ||
      parser->resolveOperand(viewInfo, type, result->operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
      parser->addTypeToList(type.getElementType(), result->types));
}

LogicalResult mlir::linalg::LoadOp::verify() {
  if (getNumOperands() == 0)
    return emitOpError("expected a view to load from");

  auto viewType = getView()->getType().dyn_cast<ViewType>();
  if (!viewType)
    return emitOpError("first operand must be a view");

  if (getType() != viewType.getElementType())
    return emitOpError("result type must match element type of the view");

  if (getRank() != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for load");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to load must have 'index' type");

  return success();
}

//////////////////////////////////////////////////////////////////////////////
// RangeOp
//////////////////////////////////////////////////////////////////////////////
void mlir::linalg::RangeOp::build(Builder *b, OperationState *result,
                                  Value *min, Value *max, Value *step) {
  result->addOperands({min, max, step});
  result->addTypes({RangeType::get(b->getContext())});
}

// Verification is simply that a RangeOp takes 3 index ssa-value.
LogicalResult mlir::linalg::RangeOp::verify() {
  if (!min() || !min()->getType().isa<IndexType>())
    return emitOpError("first operand should be of type index");
  if (!max() || !max()->getType().isa<IndexType>())
    return emitOpError("second operand should be of type index");
  if (!step() || !step()->getType().isa<IndexType>())
    return emitOpError("third operand should be of type index");
  return success();
}

// A RangeOp prints as:
//
// ```{.mlir}
//   linalg.range %0:%1:%2 : !linalg.range
// ```
void mlir::linalg::RangeOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *min() << ":" << *max() << ":" << *step()
     << " : " << getType();
}

ParseResult mlir::linalg::RangeOp::parse(OpAsmParser *parser,
                                         OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> rangeInfo(3);
  RangeType type;
  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(rangeInfo[0]) || parser->parseColon() ||
      parser->parseOperand(rangeInfo[1]) || parser->parseColon() ||
      parser->parseOperand(rangeInfo[2]) || parser->parseColonType(type) ||
      parser->resolveOperands(rangeInfo, affineIntTy, result->operands) ||
      parser->addTypeToList(type, result->types));
}

//////////////////////////////////////////////////////////////////////////////
// SliceOp
//////////////////////////////////////////////////////////////////////////////
void mlir::linalg::SliceOp::build(Builder *b, OperationState *result,
                                  Value *base, ArrayRef<Value *> indexings) {
  result->addOperands({base});
  result->addOperands(indexings);

  ViewType viewType = base->getType().cast<ViewType>();
  unsigned rank = viewType.getRank();
  for (auto *i : indexings)
    if (!i->getType().isa<RangeType>())
      rank--;
  Type elementType = viewType.getElementType();
  result->addTypes({ViewType::get(b->getContext(), elementType, rank)});
}

LogicalResult mlir::linalg::SliceOp::verify() {
  if (llvm::empty(getOperands()))
    return emitOpError(
        "requires at least a view operand followed by 'rank' indices");
  unsigned rank = getBaseViewRank();
  if (llvm::size(getIndexings()) != rank) {
    return emitOpError("requires at least a view operand followed by ")
           << rank << " indexings";
  }
  unsigned index = 0;
  for (auto indexing : getIndexings()) {
    if (!indexing->getType().isa<RangeType>() &&
        !indexing->getType().isa<IndexType>()) {
      return emitOpError() << index
                           << "^th index must be of range or index type";
    }
    if (indexing->getType().isa<IndexType>())
      --rank;
    ++index;
  }
  if (getRank() != rank) {
    return emitOpError()
           << "the rank of the view must be the number of its range indices ("
           << rank << ") but got: " << getRank();
  }
  return success();
}

ParseResult mlir::linalg::SliceOp::parse(OpAsmParser *parser,
                                         OperationState *result) {
  OpAsmParser::OperandType baseInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexingsInfo;
  SmallVector<Type, 8> types;
  if (parser->parseOperand(baseInfo) ||
      parser->parseOperandList(indexingsInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(types))
    return failure();

  if (types.size() != 2 + indexingsInfo.size())
    return parser->emitError(parser->getNameLoc(),
                             "unexpected number of types ");
  ViewType baseViewType = types[0].dyn_cast<ViewType>();
  if (!baseViewType)
    return parser->emitError(parser->getNameLoc(),
                             "view type expected for first type");
  if (indexingsInfo.size() != baseViewType.getRank())
    return parser->emitError(parser->getNameLoc(), "expected ")
           << baseViewType.getRank() << " indexings";
  ViewType viewType = types.back().dyn_cast<ViewType>();
  if (!viewType)
    return parser->emitError(parser->getNameLoc(), "view type expected");

  ArrayRef<Type> indexingTypes =
      ArrayRef<Type>(types).drop_front(1).drop_back(1);
  if (indexingTypes.size() != baseViewType.getRank())
    return parser->emitError(parser->getNameLoc(), "expected ")
           << baseViewType.getRank() << " indexing types";
  return failure(
      parser->resolveOperand(baseInfo, baseViewType, result->operands) ||
      (!indexingsInfo.empty() &&
       parser->resolveOperands(indexingsInfo, indexingTypes,
                               indexingsInfo.front().location,
                               result->operands)) ||
      parser->addTypeToList(viewType, result->types));
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
void mlir::linalg::SliceOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getBaseView() << "[";
  interleave(
      getIndexings().begin(), getIndexings().end(), [p](Value *v) { *p << *v; },
      [p]() { *p << ", "; });
  *p << "] : " << getBaseViewType();
  for (auto indexing : getIndexings()) {
    *p << ", " << indexing->getType();
  }
  *p << ", " << getType();
}

ViewOp mlir::linalg::SliceOp::getBaseViewOp() {
  return cast<ViewOp>(getOperand(0)->getDefiningOp());
}

ViewType mlir::linalg::SliceOp::getBaseViewType() {
  return getOperand(0)->getType().cast<ViewType>();
}

SmallVector<Value *, 8> mlir::linalg::SliceOp::getRanges() {
  llvm::SmallVector<Value *, 8> res;
  for (auto *operand : getIndexings()) {
    if (!operand->getType().isa<IndexType>()) {
      res.push_back(operand);
    }
  }
  return res;
}

////////////////////////////////////////////////////////////////////////////////
// StoreOp.
////////////////////////////////////////////////////////////////////////////////
void mlir::linalg::StoreOp::build(Builder *b, OperationState *result,
                                  Value *valueToStore, Value *view,
                                  ArrayRef<Value *> indices) {
  result->addOperands(valueToStore);
  result->addOperands(view);
  result->addOperands(indices);
}

// A StoreOp prints as:
//
// ```{.mlir}
//    linalg.store %f, %V[%c0] : !linalg.view<?xf32>
// ```
void mlir::linalg::StoreOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getValueToStore();
  *p << ", " << *getView() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getViewType();
}

ParseResult mlir::linalg::StoreOp::parse(OpAsmParser *parser,
                                         OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType viewInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  ViewType viewType;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(storeValueInfo) || parser->parseComma() ||
      parser->parseOperand(viewInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(viewType) ||
      parser->resolveOperand(storeValueInfo, viewType.getElementType(),
                             result->operands) ||
      parser->resolveOperand(viewInfo, viewType, result->operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, result->operands));
}

LogicalResult mlir::linalg::StoreOp::verify() {
  if (getNumOperands() < 2)
    return emitOpError("expected a value to store and a view");

  // Second operand is a memref type.
  auto viewType = getView()->getType().dyn_cast<ViewType>();
  if (!viewType)
    return emitOpError("second operand must be a view");

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != viewType.getElementType())
    return emitOpError("first operand must have same element type as the view");

  if (getNumOperands() != 2 + viewType.getRank())
    return emitOpError("store index operand count not equal to view rank");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to store must have 'index' type");

  return success();
}

//////////////////////////////////////////////////////////////////////////////
// ViewOp
//////////////////////////////////////////////////////////////////////////////
void mlir::linalg::ViewOp::build(Builder *b, OperationState *result,
                                 Value *buffer, ArrayRef<Value *> indexings) {
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

LogicalResult mlir::linalg::ViewOp::verify() {
  if (llvm::empty(getOperands()))
    return emitOpError(
        "requires at least a buffer operand followed by indexings");
  auto bufferType = getOperand(0)->getType().dyn_cast<BufferType>();
  if (!bufferType)
    return emitOpError("first operand must be of BufferType");
  unsigned index = 0;
  for (auto indexing : getIndexings()) {
    if (!indexing->getType().isa<RangeType>()) {
      return emitOpError() << index << "^th index must be of range type";
    }
    ++index;
  }
  if (getViewType().getRank() != index)
    return emitOpError()
           << "the rank of the view must be the number of its indexings";
  return success();
}

ParseResult mlir::linalg::ViewOp::parse(OpAsmParser *parser,
                                        OperationState *result) {
  OpAsmParser::OperandType bufferInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexingsInfo;
  Type type;
  if (parser->parseOperand(bufferInfo) ||
      parser->parseOperandList(indexingsInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return failure();

  ViewType viewType = type.dyn_cast<ViewType>();
  if (!viewType)
    return parser->emitError(parser->getNameLoc(), "view type expected");
  if (viewType.getRank() != indexingsInfo.size())
    return parser->emitError(parser->getNameLoc(), "expected")
           << viewType.getRank() << " range indexings";
  return failure(
      parser->resolveOperand(
          bufferInfo,
          BufferType::get(type.getContext(), viewType.getElementType()),
          result->operands) ||
      (!indexingsInfo.empty() &&
       parser->resolveOperands(indexingsInfo, RangeType::get(type.getContext()),
                               result->operands)) ||
      parser->addTypeToList(viewType, result->types));
}

// A ViewOp prints as:
//
// ```{.mlir}
//   linalg.view %0[%1, %2] : !linalg.view<?x?xf32>
// ```
//
// Where %0 is an ssa-value holding a buffer, %1 and %2 are ssa-value each
// holding a range.
void mlir::linalg::ViewOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getSupportingBuffer() << "[";
  interleave(
      getIndexings().begin(), getIndexings().end(), [&](Value *v) { *p << *v; },
      [&]() { *p << ", "; });
  *p << "] : " << getType();
}

///////////////////// Operations defined with Tablegen /////////////////////////
// For such operations that do not correspond to library calls (i.e. defined in
// LinalgOps.td), we define an overloaded `print` function and a
// parse`className` function.

static void print(OpAsmPrinter *p, BufferAllocOp op) {
  *p << op.getOperationName() << " ";
  if (!llvm::empty(op.size()))
    *p << *op.getOperand(0);
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getBufferType();
}

static ParseResult parseBufferAllocOp(OpAsmParser *parser,
                                      OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 1> sizeInfo;
  BufferType bufferType;
  auto indexTy = parser->getBuilder().getIndexType();
  if (parser->parseOperandList(sizeInfo) || parser->parseColonType(bufferType))
    return failure();
  if (sizeInfo.empty())
    return parser->addTypeToList(bufferType, result->types);
  return failure(parser->resolveOperands(sizeInfo, indexTy, result->operands) ||
                 parser->addTypeToList(bufferType, result->types));
}

static LogicalResult verify(BufferAllocOp op) {
  if (!op.getBufferType().hasConstantSize()) {
    if (llvm::size(op.size()) != 1 ||
        !op.getOperand(0)->getType().isa<IndexType>())
      return op.emitOpError(
          "one operand of type index expected for dynamic buffer");
  } else { // op.getBufferType().hasConstantSize()
    if (!llvm::empty(op.size()))
      return op.emitOpError("unexpected static buffer operand");
    if (op.getBufferType().getBufferSize().getValue() <= 0)
      return op.emitOpError("expected nonnegative static buffer size");
  }
  if (!VectorType::isValidElementType(op.getElementType()) &&
      !op.getElementType().isa<VectorType>())
    return op.emitOpError("unsupported buffer element type");
  return success();
}

static void print(OpAsmPrinter *p, BufferDeallocOp op) {
  *p << op.getOperationName() << " " << *op.buffer();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getBufferType();
}

static ParseResult parseBufferDeallocOp(OpAsmParser *parser,
                                        OperationState *result) {
  OpAsmParser::OperandType bufferInfo;
  BufferType bufferType;
  if (parser->parseOperand(bufferInfo) || parser->parseColonType(bufferType))
    return failure();
  return parser->resolveOperands(bufferInfo, bufferType, result->operands);
}

static void print(OpAsmPrinter *p, BufferSizeOp op) {
  *p << op.getOperationName() << " " << *op.getOperand();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getOperand()->getType();
}

static ParseResult parseBufferSizeOp(OpAsmParser *parser,
                                     OperationState *result) {
  OpAsmParser::OperandType op;
  Type type;
  return failure(parser->parseOperand(op) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperand(op, type, result->operands) ||
                 parser->addTypeToList(parser->getBuilder().getIndexType(),
                                       result->types));
}

static void print(OpAsmPrinter *p, linalg::DimOp op) {
  *p << op.getOperationName() << " " << *op.getOperand() << ", "
     << op.getIndex();
  p->printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"index"});
  *p << " : " << op.getOperand()->getType();
}

static ParseResult parseDimOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type type;
  Type indexType = parser->getBuilder().getIndexType();
  return failure(parser->parseOperand(operandInfo) || parser->parseComma() ||
                 parser->parseAttribute(indexAttr, indexType, "index",
                                        result->attributes) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperand(operandInfo, type, result->operands) ||
                 parser->addTypeToList(indexType, result->types));
}

static void print(OpAsmPrinter *p, RangeIntersectOp op) {
  *p << op.getOperationName() << " " << *op.getOperand(0) << ", "
     << *op.getOperand(1);
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getOperand(0)->getType();
}

static ParseResult parseRangeIntersectOp(OpAsmParser *parser,
                                         OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  return failure(parser->parseOperandList(ops) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperands(ops, type, result->operands) ||
                 parser->addTypeToList(type, result->types));
}

static void print(OpAsmPrinter *p, SubViewOp op) {
  *p << op.getOperationName() << " " << *op.getOperand(0) << "[";
  auto ranges = op.getRanges();
  interleaveComma(ranges, *p, [&p](const SubViewOp::Range &i) {
    *p << *i.min << ", " << *i.max << ", " << *i.step;
  });
  *p << "]";
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getViewType();
}

static ParseResult parseSubViewOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType inputView, resultView;
  Type viewType;
  if (parser->parseOperand(inputView))
    return failure();

  SmallVector<OpAsmParser::OperandType, 12> ops;
  // TODO(ntv) evolve parsing from
  //    linalg.subview %0[%1, %2, %3, %4, %5, %6]
  // to something resembling
  //    linalg.subview %0[%1:%2:%3][%4:%5:%6]
  if (parser->parseOperandList(ops, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(viewType))
    return failure();

  auto indexTy = parser->getBuilder().getIndexType();
  return failure(
      parser->resolveOperand(inputView, viewType, result->operands) ||
      parser->resolveOperands(ops, indexTy, result->operands) ||
      parser->addTypeToList(viewType, result->types));
}

/////// Operations corresponding to library calls defined with Tablegen ////////
// For such operations correspond to library calls (i.e. defined in
// LinalgLibraryOps.td), we define an overloaded `print` function and a
// parse`className` function.

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
static void printLinalgLibraryOp(OpAsmPrinter *p, Operation *op) {
  assert(op->getAbstractOperation() && "unregistered operation");
  *p << op->getName().getStringRef() << "(";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << ")";
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : ";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value *v) { *p << v->getType(); }, [&]() { *p << ", "; });
}

static ParseResult parseLinalgLibraryOp(OpAsmParser *parser,
                                        OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(parser->parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonTypeList(types) ||
                 parser->resolveOperands(ops, types, parser->getNameLoc(),
                                         result->operands));
}

static LogicalResult verify(FillOp op) {
  auto viewType = op.getOutputViewType(0);
  auto fillType = op.getValue()->getType();
  if (viewType.getElementType() != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
  return success();
}

static LogicalResult verify(CopyOp op) {
  auto outputViewType = op.getOutputViewType(0);
  auto inputViewType = op.getInputViewType(0);
  if (inputViewType.getElementType() != outputViewType.getElementType())
    return op.emitOpError("expects views of the same type");
  if (inputViewType.getRank() != outputViewType.getRank())
    return op.emitOpError("expects views of the same rank");
  auto rank = op.getNumParallelLoops();
  auto inputPermutationMap = op.inputPermutation();
  if (inputPermutationMap) {
    if (inputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional input_permutation map of rank ")
             << rank;
    if (!inputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional input_permutation map to be a permutation");
  }
  auto outputPermutationMap = op.outputPermutation();
  if (outputPermutationMap) {
    if (outputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional output_permutation map of rank ")
             << rank;
    if (!outputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional output_permutation map to be a permutation");
  }
  if (rank == 0 && inputPermutationMap)
    return op.emitOpError("expected no input permutation when rank == 0");
  if (rank == 0 && outputPermutationMap)
    return op.emitOpError("expected no output permutation when rank == 0");
  return success();
}

static LogicalResult
verifyStrideOrDilation(ConvOp op, ArrayRef<Attribute> attrs, bool isStride) {
  auto strideOrDilation = isStride ? "stride" : "dilation";
  if (attrs.size() != op.getNumWindowLoops())
    return op.emitOpError("expects num ")
           << strideOrDilation
           << "s equal to number of window dimensions: " << attrs.size()
           << " vs " << op.getNumWindowLoops();
  return success();
}

static LogicalResult verify(ConvOp op) {
  auto oType = op.output()->getType().cast<ViewType>();
  auto fType = op.filter()->getType().cast<ViewType>();
  auto iType = op.input()->getType().cast<ViewType>();
  if (oType.getElementType() != iType.getElementType() ||
      oType.getElementType() != fType.getElementType())
    return op.emitOpError("expects view elemental types to match");
  if (oType.getRank() != iType.getRank() || oType.getRank() != fType.getRank())
    return op.emitOpError("expects view ranks to match");
  if (auto strides = op.strides()) {
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

llvm::raw_ostream &mlir::linalg::operator<<(llvm::raw_ostream &os,
                                            SubViewOp::Range &range) {
  return os << "range " << *range.min << ":" << *range.max << ":"
            << *range.step;
}

namespace mlir {
namespace linalg {

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgLibraryOps.cpp.inc"

} // namespace linalg
} // namespace mlir

static AffineMap extractOrIdentityMap(llvm::Optional<AffineMap> maybeMap,
                                      unsigned rank, MLIRContext *context) {
  if (maybeMap)
    return maybeMap.getValue();
  if (rank == 0)
    return AffineMap();
  return AffineMap::getMultiDimIdentityMap(rank, context);
}

// Returns `num` AffineDimExpr dimensions at positions [curIdx, curIdx + num)
// and increments `curIdx` to `curIdx + num`.
static SmallVector<AffineExpr, 4>
makeAffineDimExprs(unsigned num, unsigned &curIdx, MLIRContext *context) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    res.push_back(getAffineDimExpr(curIdx++, context));
  return res;
}

static SmallVector<AffineExpr, 4>
weightedConvInputIndex(ConvOp op, ArrayRef<AffineExpr> a,
                       ArrayRef<AffineExpr> b) {
  assert(a.size() == b.size());
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size());
  for (unsigned i = 0, e = a.size(); i < e; ++i) {
    res.push_back(op.getStride(i) * a[i] + op.getDilation(i) * b[i]);
  }
  return res;
}

static SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                         ArrayRef<AffineExpr> b) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size() + b.size());
  res.assign(a.begin(), a.end());
  res.append(b.begin(), b.end());
  return res;
}

static SmallVector<ValueHandle, 8>
foldedAffineApplies(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value *> vals, OperationFolder &folder) {
  assert(map.getNumSymbols() == 0);
  assert(map.getNumInputs() == vals.size());
  SmallVector<ValueHandle, 8> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, 0, e);
    SmallVector<Value *, 4> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(
        ValueHandle(folder.create<AffineApplyOp>(b, loc, exprMap, operands)));
  }
  return res;
}

// Note: both functions below would completely disappear with a simple tensor
// kernel language.
//
// Ideally this should all be Tablegen'd but there is no good story for
// AffineMap for now.
SmallVector<AffineMap, 4> mlir::linalg::loopToOperandRangesMaps(Operation *op) {
  MLIRContext *context = op->getContext();
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    // I(input_perm(ivs)) -> O(output_perm(ivs))
    auto maybeInputMap = copyOp.inputPermutation();
    auto maybeOutputMap = copyOp.outputPermutation();
    unsigned inputRank = copyOp.getInputViewType(0).getRank();
    unsigned outputRank = copyOp.getOutputViewType(0).getRank();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(maybeInputMap, inputRank, context),
        extractOrIdentityMap(maybeOutputMap, outputRank, context)};
  }
  if (auto fillOp = dyn_cast<FillOp>(op)) {
    // filling_value -> O(ivs)
    unsigned rank = fillOp.getNumParallelLoops();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(llvm::None, rank, context)};
  }
  auto i = getAffineDimExpr(0, context);
  auto j = getAffineDimExpr(1, context);
  auto k = getAffineDimExpr(2, context);
  if (isa<DotOp>(op))
    // A(r_i) * B(r_i) -> C()
    return SmallVector<AffineMap, 4>{AffineMap::get(1, 0, {i}),
                                     AffineMap::get(1, 0, {i}), AffineMap()};
  if (isa<MatvecOp>(op))
    //   A(i, r_j) * B(r_j) -> C(i)
    return SmallVector<AffineMap, 4>{AffineMap::get(2, 0, {i, j}),
                                     AffineMap::get(2, 0, {j}),
                                     AffineMap::get(2, 0, {i})};
  if (isa<MatmulOp>(op))
    //   A(i, r_k) * B(r_k, j) -> C(i, j)
    return SmallVector<AffineMap, 4>{AffineMap::get(3, 0, {i, k}),
                                     AffineMap::get(3, 0, {k, j}),
                                     AffineMap::get(3, 0, {i, j})};
  if (auto convOp = dyn_cast<ConvOp>(op)) {
    //   F(z0, ..., zN-1, q, k) * I(b, x0 + z0, ..., xN-1 + zN-1, q) ->
    //     O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`.
    auto nWin = convOp.getNumWindowLoops();
    assert(nWin > 0 && "expected at least one window dimension");
    unsigned idx = 0;
    // In the following, AffineDimExprs are indexed in loop order:
    //   [ b, xs, k,           q,                     zs]
    //    parallels     non-window reductions     windows
    //
    // Parallel dims are exactly the dimensions indexing `output`:
    //     output[b, x[0], ..., x[N-1], k]; i.e.
    //  * batch dimensions (bs with #bs = 1 for now)
    //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
    //  * output filter dimensions (ks with #ks = 1 for now)
    auto bs = makeAffineDimExprs(convOp.getNumBatchDimensions(), idx, context);
    auto xs = makeAffineDimExprs(nWin, idx, context);
    auto ks = makeAffineDimExprs(convOp.getNumOutputFeatureDimensions(), idx,
                                 context);
    // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
    auto qs =
        makeAffineDimExprs(convOp.getNumInputFeatureDimensions(), idx, context);
    // Window reduction dims: sum_{z[0], ..., z[N-1], q}
    auto zs = makeAffineDimExprs(nWin, idx, context);
    // Construct the weighedSum expression.
    auto ws = weightedConvInputIndex(convOp, xs, zs);
    return SmallVector<AffineMap, 4>{
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks)),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0], ..., x[N-1]*s[N-1] + d[N-1]*z[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs)),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks))};
  }
  llvm_unreachable("Missing loopToOperandRangesMaps for op");
}

static SmallVector<Value *, 4> permuteIvs(ArrayRef<Value *> ivs,
                                          Optional<AffineMap> permutation,
                                          OperationFolder &state) {
  return permutation ? applyMapToValues(ScopedContext::getBuilder(),
                                        ScopedContext::getLocation(),
                                        permutation.getValue(), ivs, state)
                     : SmallVector<Value *, 4>(ivs.begin(), ivs.end());
}

// Ideally this should all be Tablegen'd but there is no good story for op
// expansion directly in MLIR for now.
void mlir::linalg::emitScalarImplementation(
    llvm::ArrayRef<Value *> parallelIvs, llvm::ArrayRef<Value *> reductionIvs,
    llvm::ArrayRef<Value *> windowIvs, LinalgOp &linalgOp,
    OperationFolder &folder) {
  using linalg_load = ValueBuilder<linalg::LoadOp>;
  using linalg_store = OperationBuilder<linalg::StoreOp>;
  using IndexedValue = TemplatedIndexedValue<linalg_load, linalg_store>;
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::op::operator==;
  using edsc::intrinsics::select;

  auto nPar = parallelIvs.size();
  auto nRed = reductionIvs.size();
  auto nWin = windowIvs.size();
  SmallVector<Value *, 8> allIvs;
  allIvs.reserve(nPar + nRed + nWin);
  allIvs.assign(parallelIvs.begin(), parallelIvs.end());
  allIvs.append(reductionIvs.begin(), reductionIvs.end());
  allIvs.append(windowIvs.begin(), windowIvs.end());

  // Default OpBuilder supports 0-D case (no loops).
  OpBuilder b(linalgOp.getOperation());
  auto nLoops = nPar + nRed + nWin;
  if (nLoops > 0) {
    auto innermostLoop = linalg::getForInductionVarOwner(allIvs.back());
    // accounts for linalg.terminator in loop.
    b = innermostLoop.getBodyBuilder();
  }

  auto loc = linalgOp.getLoc();
  ScopedContext scope(b, loc);
  auto *op = linalgOp.getOperation();
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    OperationFolder state;
    auto inputIvs = permuteIvs(parallelIvs, copyOp.inputPermutation(), state);
    auto outputIvs = permuteIvs(parallelIvs, copyOp.outputPermutation(), state);
    SmallVector<IndexHandle, 8> iivs(inputIvs.begin(), inputIvs.end());
    SmallVector<IndexHandle, 8> oivs(outputIvs.begin(), outputIvs.end());
    // clang-format off
    IndexedValue O(copyOp.getOutput(0)), I(copyOp.getInput(0));
    nLoops > 0 ?
        O(oivs) = I(iivs) :
        O() = I();
    // clang-format on
    return;
  }
  if (auto fillOp = dyn_cast<FillOp>(op)) {
    SmallVector<IndexHandle, 8> ivs(parallelIvs.begin(), parallelIvs.end());
    // clang-format off
    IndexedValue O(fillOp.getOutput(0));
    nLoops > 0 ?
        O(ivs) = ValueHandle(fillOp.getValue()) :
        O() = ValueHandle(fillOp.getValue());
    // clang-format on
    return;
  }
  if (auto dotOp = dyn_cast<DotOp>(op)) {
    IndexHandle r_i(reductionIvs[0]);
    IndexedValue A(dotOp.getInput(0)), B(dotOp.getInput(1)),
        C(dotOp.getOutput(0));
    C() = C() + A(r_i) * B(r_i);
    return;
  }
  if (auto matvecOp = dyn_cast<MatvecOp>(op)) {
    IndexHandle i(parallelIvs[0]), r_j(reductionIvs[0]);
    IndexedValue A(matvecOp.getInput(0)), B(matvecOp.getInput(1)),
        C(matvecOp.getOutput(0));
    C(i) = C(i) + A(i, r_j) * B(r_j);
    return;
  }
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    IndexHandle i(parallelIvs[0]), j(parallelIvs[1]), r_k(reductionIvs[0]);
    IndexedValue A(matmulOp.getInput(0)), B(matmulOp.getInput(1)),
        C(matmulOp.getOutput(0));
    C(i, j) = C(i, j) + A(i, r_k) * B(r_k, j);
    return;
  }
  if (auto convOp = dyn_cast<ConvOp>(op)) {
    auto maps = loopToOperandRangesMaps(op);
    SmallVector<ValueHandle, 8> fIdx(
        foldedAffineApplies(b, loc, maps[0], allIvs, folder));
    SmallVector<ValueHandle, 8> imIdx(
        foldedAffineApplies(b, loc, maps[1], allIvs, folder));
    SmallVector<ValueHandle, 8> oIdx(
        foldedAffineApplies(b, loc, maps[2], allIvs, folder));
    IndexedValue F(convOp.filter()), I(convOp.input()), O(convOp.output());
    O(oIdx) += F(fIdx) * I(imIdx);
    return;
  }
  llvm_unreachable("Missing emitScalarImplementation for op");
}

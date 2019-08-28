//===- VectorOps.cpp - MLIR Super Vectorizer Operations -------------------===//
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
// This file implements convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
// VectorOpsDialect
//===----------------------------------------------------------------------===//

mlir::vector::VectorOpsDialect::VectorOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<VectorTransferReadOp, VectorTransferWriteOp,
                VectorTypeCastOp>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, ExtractElementOp op) {
  *p << op.getOperationName() << " " << *op.vector() << op.position();
  p->printOptionalAttrDict(op.getAttrs(), {"position"});
  *p << " : " << op.vector()->getType();
}

static ParseResult parseExtractElementOp(OpAsmParser *parser,
                                         OperationState *result) {
  llvm::SMLoc attributeLoc, typeLoc;
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType vector;
  Type type;
  Attribute attr;
  if (parser->parseOperand(vector) ||
      parser->getCurrentLocation(&attributeLoc) ||
      parser->parseAttribute(attr, "position", attrs) ||
      parser->parseOptionalAttributeDict(attrs) ||
      parser->getCurrentLocation(&typeLoc) || parser->parseColonType(type))
    return failure();

  auto vectorType = type.dyn_cast<VectorType>();
  if (!vectorType)
    return parser->emitError(typeLoc, "expected vector type");

  auto positionAttr = attr.dyn_cast<ArrayAttr>();
  if (!positionAttr ||
      static_cast<int64_t>(positionAttr.size()) > vectorType.getRank())
    return parser->emitError(
        attributeLoc,
        "expected position attribute of rank smaller than vector");

  Type resType =
      (static_cast<int64_t>(positionAttr.size()) == vectorType.getRank())
          ? vectorType.getElementType()
          : VectorType::get(
                vectorType.getShape().drop_front(positionAttr.size()),
                vectorType.getElementType());

  result->attributes = attrs;
  return failure(parser->resolveOperand(vector, type, result->operands) ||
                 parser->addTypeToList(resType, result->types));
}

static LogicalResult verify(ExtractElementOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  if (positionAttr.size() > static_cast<unsigned>(op.getVectorType().getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than vector");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() > op.getVectorType().getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a positive integer smaller than the corresponding "
                "vector dimension";
  }
  return success();
}
//===----------------------------------------------------------------------===//
// OuterProductOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, OuterProductOp op) {
  *p << op.getOperationName() << " " << *op.lhs() << ", " << *op.rhs();
  if (llvm::size(op.acc()) > 0)
    *p << ", " << **op.acc().begin();
  *p << " : " << op.lhs()->getType() << ", " << op.rhs()->getType();
}

static ParseResult parseOuterProductOp(OpAsmParser *parser,
                                       OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> operandsInfo;
  Type tLHS, tRHS;
  if (parser->parseOperandList(operandsInfo) || parser->parseColonType(tLHS) ||
      parser->parseComma() || parser->parseType(tRHS))
    return failure();
  if (operandsInfo.size() < 2)
    return parser->emitError(parser->getNameLoc(),
                             "expected at least 2 operands");
  VectorType vLHS = tLHS.dyn_cast<VectorType>();
  VectorType vRHS = tRHS.dyn_cast<VectorType>();
  if (!vLHS || !vRHS)
    return parser->emitError(parser->getNameLoc(), "expected 2 vector types");
  VectorType resType = VectorType::get({vLHS.getDimSize(0), vRHS.getDimSize(0)},
                                       vLHS.getElementType());
  return failure(
      parser->resolveOperand(operandsInfo[0], tLHS, result->operands) ||
      parser->resolveOperand(operandsInfo[1], tRHS, result->operands) ||
      (operandsInfo.size() > 2 &&
       parser->resolveOperand(operandsInfo[2], resType, result->operands)) ||
      parser->addTypeToList(resType, result->types));
}

static LogicalResult verify(OuterProductOp op) {
  VectorType vLHS = op.getOperandVectorTypeLHS(),
             vRHS = op.getOperandVectorTypeRHS(),
             vACC = op.getOperandVectorTypeACC(), vRES = op.getVectorType();
  if (vLHS.getRank() != 1)
    return op.emitOpError("expected 1-d vector for operand #1");
  if (vRHS.getRank() != 1)
    return op.emitOpError("expected 1-d vector for operand #2");
  if (vRES.getRank() != 2)
    return op.emitOpError("expected 2-d vector result");
  if (vLHS.getDimSize(0) != vRES.getDimSize(0))
    return op.emitOpError("expected #1 operand dim to match result dim #1");
  if (vRHS.getDimSize(0) != vRES.getDimSize(1))
    return op.emitOpError("expected #2 operand dim to match result dim #2");
  if (vACC && vACC != vRES)
    return op.emitOpError("expected operand #3 of same type as result type");
  return success();
}

//===----------------------------------------------------------------------===//
// VectorTransferReadOp
//===----------------------------------------------------------------------===//
template <typename EmitFun>
static LogicalResult verifyPermutationMap(AffineMap permutationMap,
                                          EmitFun emitOpError) {
  SmallVector<bool, 8> seen(permutationMap.getNumInputs(), false);
  for (auto expr : permutationMap.getResults()) {
    auto dim = expr.dyn_cast<AffineDimExpr>();
    auto zero = expr.dyn_cast<AffineConstantExpr>();
    if (zero) {
      if (zero.getValue() != 0) {
        return emitOpError(
            "requires a projected permutation_map (at most one dim or the zero "
            "constant can appear in each result)");
      }
      continue;
    }
    if (!dim) {
      return emitOpError("requires a projected permutation_map (at most one "
                         "dim or the zero constant can appear in each result)");
    }
    if (seen[dim.getPosition()]) {
      return emitOpError(
          "requires a permutation_map that is a permutation (found one dim "
          "used more than once)");
    }
    seen[dim.getPosition()] = true;
  }
  return success();
}

void VectorTransferReadOp::build(Builder *builder, OperationState *result,
                                 VectorType vectorType, Value *srcMemRef,
                                 ArrayRef<Value *> srcIndices,
                                 AffineMap permutationMap,
                                 Optional<Value *> paddingValue) {
  result->addOperands(srcMemRef);
  result->addOperands(srcIndices);
  if (paddingValue) {
    result->addOperands({*paddingValue});
  }
  result->addAttribute(getPermutationMapAttrName(),
                       builder->getAffineMapAttr(permutationMap));
  result->addTypes(vectorType);
}

auto VectorTransferReadOp::getIndices() -> operand_range {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

Optional<Value *> VectorTransferReadOp::getPaddingValue() {
  auto memRefRank = getMemRefType().getRank();
  if (getNumOperands() <= Offsets::FirstIndexOffset + memRefRank) {
    return None;
  }
  return Optional<Value *>(getOperand(Offsets::FirstIndexOffset + memRefRank));
}

AffineMap VectorTransferReadOp::getPermutationMap() {
  return getAttrOfType<AffineMapAttr>(getPermutationMapAttrName()).getValue();
}

void VectorTransferReadOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " ";
  p->printOperand(getMemRef());
  *p << "[";
  p->printOperands(getIndices());
  *p << "]";
  auto optionalPaddingValue = getPaddingValue();
  if (optionalPaddingValue) {
    *p << ", (";
    p->printOperand(*optionalPaddingValue);
    *p << ")";
  }
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getMemRefType();
  *p << ", " << getResultType();
}

ParseResult VectorTransferReadOp::parse(OpAsmParser *parser,
                                        OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  SmallVector<OpAsmParser::OperandType, 8> paddingInfo;
  SmallVector<Type, 2> types;

  // Parsing with support for optional paddingValue.
  if (parser->parseOperand(memrefInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseTrailingOperandList(paddingInfo,
                                       OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(types))
    return failure();

  // Resolution.
  if (types.size() != 2)
    return parser->emitError(parser->getNameLoc(), "expected 2 types");
  MemRefType memrefType = types[0].dyn_cast<MemRefType>();
  if (!memrefType)
    return parser->emitError(parser->getNameLoc(), "memRef type expected");
  VectorType vectorType = types[1].dyn_cast<VectorType>();
  if (!vectorType)
    return parser->emitError(parser->getNameLoc(), "vector type expected");

  // Extract optional paddingValue.
  // At this point, indexInfo may contain the optional paddingValue, pop it
  // out.
  if (static_cast<int64_t>(indexInfo.size()) != memrefType.getRank())
    return parser->emitError(parser->getNameLoc(),
                             "expected " + Twine(memrefType.getRank()) +
                                 " indices to the memref");
  if (paddingInfo.size() > 1)
    return parser->emitError(parser->getNameLoc(),
                             "expected at most one padding value");
  Type paddingType;
  bool hasOptionalPaddingValue = !paddingInfo.empty();
  if (hasOptionalPaddingValue) {
    paddingType = vectorType.getElementType();
  }
  auto indexType = parser->getBuilder().getIndexType();
  return failure(
      parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
      parser->resolveOperands(indexInfo, indexType, result->operands) ||
      (hasOptionalPaddingValue &&
       parser->resolveOperand(paddingInfo[0], paddingType, result->operands)) ||
      parser->addTypeToList(vectorType, result->types));
}

LogicalResult VectorTransferReadOp::verify() {
  // Consistency of memref type in function type.
  if (llvm::empty(getOperands())) {
    return emitOpError(
        "requires at least a memref operand followed by 'rank' indices");
  }
  if (!getMemRef()->getType().isa<MemRefType>()) {
    return emitOpError("requires a memref as first operand");
  }
  // Consistency of vector type in function type.
  if (!getResult()->getType().isa<VectorType>()) {
    return emitOpError("should have a vector result type in function type: "
                       "memref_type<...xelemental_type>, vector_type");
  }
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = getMemRefType();
  VectorType vectorType = getResultType();
  if (memrefType.getElementType() != vectorType.getElementType())
    return emitOpError(
        "requires memref and vector types of the same elemental type");
  // Consistency of number of input types.
  auto optionalPaddingValue = getPaddingValue();
  unsigned expectedNumOperands = Offsets::FirstIndexOffset +
                                 memrefType.getRank() +
                                 (optionalPaddingValue ? 1 : 0);
  // Checks on the actual operands and their types.
  if (getNumOperands() != expectedNumOperands) {
    return emitOpError("expects ")
           << expectedNumOperands << " operands (of which "
           << memrefType.getRank() << " indices)";
  }
  // Consistency of padding value with vector type.
  if (optionalPaddingValue) {
    auto paddingValue = *optionalPaddingValue;
    auto elementalType = paddingValue->getType();
    if (!VectorType::isValidElementType(elementalType)) {
      return emitOpError("requires valid padding vector elemental type");
    }
    if (elementalType != vectorType.getElementType()) {
      return emitOpError(
          "requires formal padding and vector of the same elemental type");
    }
  }
  // Consistency of indices types.
  unsigned numIndices = 0;
  for (auto *idx : getIndices()) {
    if (!idx->getType().isIndex()) {
      return emitOpError(
          "index to vector.transfer_read must have 'index' type");
    }
    ++numIndices;
  }
  if (numIndices != memrefType.getRank()) {
    return emitOpError("requires at least a memref operand followed by ")
           << memrefType.getRank() << " indices";
  }

  // Consistency of AffineMap attribute.
  if (!getAttrOfType<AffineMapAttr>(getPermutationMapAttrName())) {
    return emitOpError("requires an AffineMapAttr named 'permutation_map'");
  }
  auto permutationMap = getPermutationMap();
  if (permutationMap.getNumSymbols() != 0) {
    return emitOpError("requires a permutation_map without symbols");
  }
  if (permutationMap.getNumInputs() != memrefType.getRank()) {
    return emitOpError("requires a permutation_map with input dims of the "
                       "same rank as the memref type");
  }
  if (permutationMap.getNumResults() != vectorType.getRank()) {
    return emitOpError("requires a permutation_map with result dims of the "
                       "same rank as the vector type (")
           << permutationMap.getNumResults() << " vs " << vectorType.getRank();
  }
  return verifyPermutationMap(permutationMap,
                              [this](Twine t) { return emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// VectorTransferWriteOp
//===----------------------------------------------------------------------===//
void VectorTransferWriteOp::build(Builder *builder, OperationState *result,
                                  Value *srcVector, Value *dstMemRef,
                                  ArrayRef<Value *> dstIndices,
                                  AffineMap permutationMap) {
  result->addOperands({srcVector, dstMemRef});
  result->addOperands(dstIndices);
  result->addAttribute(getPermutationMapAttrName(),
                       builder->getAffineMapAttr(permutationMap));
}

auto VectorTransferWriteOp::getIndices() -> operand_range {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

AffineMap VectorTransferWriteOp::getPermutationMap() {
  return getAttrOfType<AffineMapAttr>(getPermutationMapAttrName()).getValue();
}

void VectorTransferWriteOp::print(OpAsmPrinter *p) {
  *p << getOperationName();
  *p << " " << *getVector();
  *p << ", " << *getMemRef();
  *p << "[";
  p->printOperands(getIndices());
  *p << "]";
  p->printOptionalAttrDict(getAttrs());
  *p << " : ";
  p->printType(getVectorType());
  *p << ", ";
  p->printType(getMemRefType());
}

ParseResult VectorTransferWriteOp::parse(OpAsmParser *parser,
                                         OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  SmallVector<Type, 2> types;
  auto indexType = parser->getBuilder().getIndexType();
  if (parser->parseOperand(storeValueInfo) || parser->parseComma() ||
      parser->parseOperand(memrefInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(types))
    return failure();

  if (types.size() != 2)
    return parser->emitError(parser->getNameLoc(), "expected 2 types");
  VectorType vectorType = types[Offsets::VectorOffset].dyn_cast<VectorType>();
  if (!vectorType)
    return parser->emitError(parser->getNameLoc(), "vector type expected");
  MemRefType memrefType = types[Offsets::MemRefOffset].dyn_cast<MemRefType>();
  if (!memrefType)
    return parser->emitError(parser->getNameLoc(), "memRef type expected");

  return failure(
      parser->resolveOperands(storeValueInfo, vectorType, result->operands) ||
      parser->resolveOperands(memrefInfo, memrefType, result->operands) ||
      parser->resolveOperands(indexInfo, indexType, result->operands));
}

LogicalResult VectorTransferWriteOp::verify() {
  // Consistency of memref type in function type.
  if (llvm::empty(getOperands())) {
    return emitOpError(
        "requires at least a memref operand followed by 'rank' indices");
  }
  if (!getMemRef()->getType().isa<MemRefType>()) {
    return emitOpError("requires a memref first operand");
  }
  // Consistency of vector type in function type.
  if (!getVector()->getType().isa<VectorType>()) {
    return emitOpError("should have a vector input type in function type: "
                       "(vector_type, memref_type [, elemental_type]) -> ()");
  }
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = getMemRefType();
  VectorType vectorType = getVectorType();
  if (memrefType.getElementType() != vectorType.getElementType())
    return emitOpError(
        "requires memref and vector types of the same elemental type");
  // Consistency of number of input types.
  unsigned expectedNumOperands =
      Offsets::FirstIndexOffset + memrefType.getRank();
  // Checks on the actual operands and their types.
  if (getNumOperands() != expectedNumOperands) {
    return emitOpError() << "expects " << expectedNumOperands
                         << " operands (of which " << memrefType.getRank()
                         << " indices)";
  }
  // Consistency of indices types.
  unsigned numIndices = 0;
  for (auto *idx : getIndices()) {
    if (!idx->getType().isIndex()) {
      return emitOpError(
          "index to vector.transfer_write must have 'index' type");
    }
    numIndices++;
  }
  if (numIndices != memrefType.getRank()) {
    return emitOpError("requires at least a memref operand followed by ")
           << memrefType.getRank() << " indices";
  }

  // Consistency of AffineMap attribute.
  if (!getAttrOfType<AffineMapAttr>(getPermutationMapAttrName())) {
    return emitOpError("requires an AffineMapAttr named 'permutation_map'");
  }
  auto permutationMap = getPermutationMap();
  if (permutationMap.getNumSymbols() != 0) {
    return emitOpError("requires a permutation_map without symbols");
  }
  if (permutationMap.getNumInputs() != memrefType.getRank()) {
    return emitOpError("requires a permutation_map with input dims of the "
                       "same rank as the memref type");
  }
  if (permutationMap.getNumResults() != vectorType.getRank()) {
    return emitOpError("requires a permutation_map with result dims of the "
                       "same rank as the vector type (")
           << permutationMap.getNumResults() << " vs " << vectorType.getRank();
  }
  return verifyPermutationMap(permutationMap,
                              [this](Twine t) { return emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// VectorTypeCastOp
//===----------------------------------------------------------------------===//
void VectorTypeCastOp::build(Builder *builder, OperationState *result,
                             Value *srcVector, Type dstType) {
  result->addOperands(srcVector);
  result->addTypes(dstType);
}

ParseResult VectorTypeCastOp::parse(OpAsmParser *parser,
                                    OperationState *result) {
  OpAsmParser::OperandType operand;
  Type srcType, dstType;
  return failure(parser->parseOperand(operand) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(srcType) || parser->parseComma() ||
                 parser->parseType(dstType) ||
                 parser->addTypeToList(dstType, result->types) ||
                 parser->resolveOperand(operand, srcType, result->operands));
}

void VectorTypeCastOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << ' ' << *getOperand() << " : "
     << getOperand()->getType() << ", " << getType();
}

LogicalResult VectorTypeCastOp::verify() {
  auto dstMemrefType = getType().dyn_cast<MemRefType>();
  if (!dstMemrefType)
    return emitOpError("expects target type to be a memref type");
  auto dstVectorType = dstMemrefType.getElementType().dyn_cast<VectorType>();
  if (!dstVectorType)
    return emitOpError(
        "expects vector as an element of the target memref type");
  if (!dstMemrefType.hasStaticShape())
    return emitOpError("does not support dynamic shapes");

  if (!getOperand()->getType().isa<MemRefType>())
    return emitOpError("expects source type to be a memref type");

  return success();
}

namespace mlir {

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"

} // namespace mlir

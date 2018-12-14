//===- SuperVectorOps.cpp - MLIR Super Vectorizer Operations---------------===//
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

#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// SuperVectorOpsDialect
//===----------------------------------------------------------------------===//

SuperVectorOpsDialect::SuperVectorOpsDialect(MLIRContext *context)
    : Dialect(/*opPrefix=*/"", context) {
  addOperations<VectorTransferReadOp, VectorTransferWriteOp>();
}

//===----------------------------------------------------------------------===//
// VectorTransferReadOp
//===----------------------------------------------------------------------===//
template <typename EmitFun>
static bool verifyPermutationMap(AffineMap permutationMap,
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
  return false;
}

void VectorTransferReadOp::build(Builder *builder, OperationState *result,
                                 VectorType vectorType, SSAValue *srcMemRef,
                                 ArrayRef<SSAValue *> srcIndices,
                                 AffineMap permutationMap,
                                 Optional<SSAValue *> paddingValue) {
  result->addOperands(srcMemRef);
  result->addOperands(srcIndices);
  if (paddingValue) {
    result->addOperands({*paddingValue});
  }
  result->addAttribute(getPermutationMapAttrName(),
                       builder->getAffineMapAttr(permutationMap));
  result->addTypes(vectorType);
}

llvm::iterator_range<Operation::operand_iterator>
VectorTransferReadOp::getIndices() {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

llvm::iterator_range<Operation::const_operand_iterator>
VectorTransferReadOp::getIndices() const {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

Optional<SSAValue *> VectorTransferReadOp::getPaddingValue() {
  auto memRefRank = getMemRefType().getRank();
  if (getNumOperands() <= Offsets::FirstIndexOffset + memRefRank) {
    return None;
  }
  return Optional<SSAValue *>(
      getOperand(Offsets::FirstIndexOffset + memRefRank));
}

Optional<const SSAValue *> VectorTransferReadOp::getPaddingValue() const {
  auto memRefRank = getMemRefType().getRank();
  if (getNumOperands() <= Offsets::FirstIndexOffset + memRefRank) {
    return None;
  }
  return Optional<const SSAValue *>(
      getOperand(Offsets::FirstIndexOffset + memRefRank));
}

AffineMap VectorTransferReadOp::getPermutationMap() const {
  return getAttrOfType<AffineMapAttr>(getPermutationMapAttrName()).getValue();
}

void VectorTransferReadOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << " ";
  p->printOperand(getMemRef());
  *p << ", ";
  p->printOperands(getIndices());
  auto optionalPaddingValue = getPaddingValue();
  if (optionalPaddingValue) {
    *p << ", ";
    p->printOperand(*optionalPaddingValue);
  }
  p->printOptionalAttrDict(getAttrs());
  // Construct the FunctionType and print it.
  llvm::SmallVector<Type, 8> inputs{getMemRefType()};
  // Must have at least one actual index, see verify.
  const SSAValue *firstIndex = *(getIndices().begin());
  Type indexType = firstIndex->getType();
  inputs.append(getMemRefType().getRank(), indexType);
  if (optionalPaddingValue) {
    inputs.push_back((*optionalPaddingValue)->getType());
  }
  *p << " : "
     << FunctionType::get(inputs, {getResultType()}, indexType.getContext());
}

bool VectorTransferReadOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 8> parsedOperands;
  Type type;

  // Parsing with support for optional paddingValue.
  auto fail = parser->parseOperandList(parsedOperands) ||
              parser->parseOptionalAttributeDict(result->attributes) ||
              parser->parseColonType(type);
  if (fail) {
    return true;
  }

  // Resolution.
  auto funType = type.dyn_cast<FunctionType>();
  if (!funType)
    return parser->emitError(parser->getNameLoc(), "Function type expected");
  if (funType.getNumInputs() < 1)
    return parser->emitError(parser->getNameLoc(),
                             "Function type expects at least one input");
  MemRefType memrefType =
      funType.getInput(Offsets::MemRefOffset).dyn_cast<MemRefType>();
  if (!memrefType)
    return parser->emitError(parser->getNameLoc(),
                             "MemRef type expected for first input");
  if (funType.getNumResults() < 1)
    return parser->emitError(parser->getNameLoc(),
                             "Function type expects exactly one vector result");
  VectorType vectorType = funType.getResult(0).dyn_cast<VectorType>();
  if (!vectorType)
    return parser->emitError(parser->getNameLoc(),
                             "Vector type expected for first result");
  if (parsedOperands.size() != funType.getNumInputs())
    return parser->emitError(parser->getNameLoc(),
                             "requires " + Twine(funType.getNumInputs()) +
                                 " operands");

  // Extract optional paddingValue.
  OpAsmParser::OperandType memrefInfo = parsedOperands[0];
  // At this point, indexInfo may contain the optional paddingValue, pop it out.
  SmallVector<OpAsmParser::OperandType, 8> indexInfo{
      parsedOperands.begin() + Offsets::FirstIndexOffset, parsedOperands.end()};
  Type paddingType;
  OpAsmParser::OperandType paddingValue;
  bool hasPaddingValue = indexInfo.size() > memrefType.getRank();
  unsigned expectedNumOperands = Offsets::FirstIndexOffset +
                                 memrefType.getRank() +
                                 (hasPaddingValue ? 1 : 0);
  if (hasPaddingValue) {
    paddingType = funType.getInputs().back();
    paddingValue = indexInfo.pop_back_val();
  }
  if (funType.getNumInputs() != expectedNumOperands)
    return parser->emitError(
        parser->getNameLoc(),
        "requires actual number of operands to match function type");

  auto indexType = parser->getBuilder().getIndexType();
  return parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, indexType, result->operands) ||
         (hasPaddingValue && parser->resolveOperand(paddingValue, paddingType,
                                                    result->operands)) ||
         parser->addTypeToList(vectorType, result->types);
}

bool VectorTransferReadOp::verify() const {
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
                       "(memref_type [, elemental_type]) -> vector_type");
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
    return emitOpError("expects " + Twine(expectedNumOperands) +
                       " operands to match the types");
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
          "index to vector_transfer_read must have 'index' type");
    }
    ++numIndices;
  }
  if (numIndices != memrefType.getRank()) {
    return emitOpError("requires at least a memref operand followed by " +
                       Twine(memrefType.getRank()) + " indices");
  }

  // Consistency of AffineMap attribute.
  if (!getAttrOfType<AffineMapAttr>(getPermutationMapAttrName())) {
    return emitOpError("requires an AffineMapAttr named 'permutation_map'");
  }
  auto permutationMap = getPermutationMap();
  if (!permutationMap.getRangeSizes().empty()) {
    return emitOpError("requires an unbounded permutation_map");
  }
  if (permutationMap.getNumSymbols() != 0) {
    return emitOpError("requires a permutation_map without symbols");
  }
  if (permutationMap.getNumInputs() != memrefType.getRank()) {
    return emitOpError("requires a permutation_map with input dims of the "
                       "same rank as the memref type");
  }
  if (permutationMap.getNumResults() != vectorType.getRank()) {
    return emitOpError("requires a permutation_map with result dims of the "
                       "same rank as the vector type (" +
                       Twine(permutationMap.getNumResults()) + " vs " +
                       Twine(vectorType.getRank()));
  }
  return verifyPermutationMap(permutationMap,
                              [this](Twine t) { return emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// VectorTransferWriteOp
//===----------------------------------------------------------------------===//
void VectorTransferWriteOp::build(Builder *builder, OperationState *result,
                                  SSAValue *srcVector, SSAValue *dstMemRef,
                                  ArrayRef<SSAValue *> dstIndices,
                                  AffineMap permutationMap) {
  result->addOperands({srcVector, dstMemRef});
  result->addOperands(dstIndices);
  result->addAttribute(getPermutationMapAttrName(),
                       builder->getAffineMapAttr(permutationMap));
}

llvm::iterator_range<Operation::operand_iterator>
VectorTransferWriteOp::getIndices() {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

llvm::iterator_range<Operation::const_operand_iterator>
VectorTransferWriteOp::getIndices() const {
  auto begin = getOperation()->operand_begin() + Offsets::FirstIndexOffset;
  auto end = begin + getMemRefType().getRank();
  return {begin, end};
}

AffineMap VectorTransferWriteOp::getPermutationMap() const {
  return getAttrOfType<AffineMapAttr>(getPermutationMapAttrName()).getValue();
}

void VectorTransferWriteOp::print(OpAsmPrinter *p) const {
  *p << getOperationName();
  *p << " " << *getVector();
  *p << ", " << *getMemRef();
  *p << ", ";
  p->printOperands(getIndices());
  p->printOptionalAttrDict(getAttrs());
  Type indexType = (*getIndices().begin())->getType();
  *p << " : ";
  p->printType(getVectorType());
  *p << ", ";
  p->printType(getMemRefType());
  for (unsigned r = 0, n = getMemRefType().getRank(); r < n; ++r) {
    *p << ", ";
    p->printType(indexType);
  }
}

bool VectorTransferWriteOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 8> parsedOperands;
  SmallVector<Type, 8> types;

  // Parsing with support for optional paddingValue.
  auto fail = parser->parseOperandList(parsedOperands) ||
              parser->parseOptionalAttributeDict(result->attributes) ||
              parser->parseColonTypeList(types);
  if (fail) {
    return true;
  }

  // Resolution.
  if (parsedOperands.size() != types.size())
    return parser->emitError(
        parser->getNameLoc(),
        "requires number of operands and input types to match");
  if (parsedOperands.size() < Offsets::FirstIndexOffset)
    return parser->emitError(parser->getNameLoc(),
                             "requires at least vector and memref operands");
  VectorType vectorType = types[Offsets::VectorOffset].dyn_cast<VectorType>();
  if (!vectorType)
    return parser->emitError(parser->getNameLoc(),
                             "Vector type expected for first input type");
  MemRefType memrefType = types[Offsets::MemRefOffset].dyn_cast<MemRefType>();
  if (!memrefType)
    return parser->emitError(parser->getNameLoc(),
                             "MemRef type expected for second input type");

  unsigned expectedNumOperands =
      Offsets::FirstIndexOffset + memrefType.getRank();
  if (parsedOperands.size() != expectedNumOperands)
    return parser->emitError(parser->getNameLoc(),
                             "requires " + Twine(expectedNumOperands) +
                                 " operands");

  OpAsmParser::OperandType vectorInfo = parsedOperands[Offsets::VectorOffset];
  OpAsmParser::OperandType memrefInfo = parsedOperands[Offsets::MemRefOffset];
  SmallVector<OpAsmParser::OperandType, 8> indexInfo{
      parsedOperands.begin() + Offsets::FirstIndexOffset, parsedOperands.end()};
  auto indexType = parser->getBuilder().getIndexType();
  return parser->resolveOperand(vectorInfo, vectorType, result->operands) ||
         parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, indexType, result->operands);
}

bool VectorTransferWriteOp::verify() const {
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
    return emitOpError("expects " + Twine(expectedNumOperands) +
                       " operands to match the types");
  }
  // Consistency of indices types.
  unsigned numIndices = 0;
  for (auto *idx : getIndices()) {
    if (!idx->getType().isIndex()) {
      return emitOpError(
          "index to vector_transfer_write must have 'index' type");
    }
    numIndices++;
  }
  if (numIndices != memrefType.getRank()) {
    return emitOpError("requires at least a memref operand followed by " +
                       Twine(memrefType.getRank()) + " indices");
  }

  // Consistency of AffineMap attribute.
  if (!getAttrOfType<AffineMapAttr>(getPermutationMapAttrName())) {
    return emitOpError("requires an AffineMapAttr named 'permutation_map'");
  }
  auto permutationMap = getPermutationMap();
  if (!permutationMap.getRangeSizes().empty()) {
    return emitOpError("requires an unbounded permutation_map");
  }
  if (permutationMap.getNumSymbols() != 0) {
    return emitOpError("requires a permutation_map without symbols");
  }
  if (permutationMap.getNumInputs() != memrefType.getRank()) {
    return emitOpError("requires a permutation_map with input dims of the "
                       "same rank as the memref type");
  }
  if (permutationMap.getNumResults() != vectorType.getRank()) {
    return emitOpError("requires a permutation_map with result dims of the "
                       "same rank as the vector type (" +
                       Twine(permutationMap.getNumResults()) + " vs " +
                       Twine(vectorType.getRank()));
  }
  return verifyPermutationMap(permutationMap,
                              [this](Twine t) { return emitOpError(t); });
}

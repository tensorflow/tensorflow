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
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ExtractElementOp op) {
  p << op.getOperationName() << " " << *op.vector() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.vector()->getType();
}

static ParseResult parseExtractElementOp(OpAsmParser &parser,
                                         OperationState &result) {
  llvm::SMLoc attributeLoc, typeLoc;
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType vector;
  Type type;
  Attribute attr;
  if (parser.parseOperand(vector) || parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(attr, "position", attrs) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.getCurrentLocation(&typeLoc) || parser.parseColonType(type))
    return failure();

  auto vectorType = type.dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typeLoc, "expected vector type");

  auto positionAttr = attr.dyn_cast<ArrayAttr>();
  if (!positionAttr ||
      static_cast<int64_t>(positionAttr.size()) > vectorType.getRank())
    return parser.emitError(
        attributeLoc,
        "expected position attribute of rank smaller than vector");

  Type resType =
      (static_cast<int64_t>(positionAttr.size()) == vectorType.getRank())
          ? vectorType.getElementType()
          : VectorType::get(
                vectorType.getShape().drop_front(positionAttr.size()),
                vectorType.getElementType());

  result.attributes = attrs;
  return failure(parser.resolveOperand(vector, type, result.operands) ||
                 parser.addTypeToList(resType, result.types));
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

static void print(OpAsmPrinter &p, OuterProductOp op) {
  p << op.getOperationName() << " " << *op.lhs() << ", " << *op.rhs();
  if (llvm::size(op.acc()) > 0)
    p << ", " << **op.acc().begin();
  p << " : " << op.lhs()->getType() << ", " << op.rhs()->getType();
}

static ParseResult parseOuterProductOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> operandsInfo;
  Type tLHS, tRHS;
  if (parser.parseOperandList(operandsInfo) || parser.parseColonType(tLHS) ||
      parser.parseComma() || parser.parseType(tRHS))
    return failure();
  if (operandsInfo.size() < 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected at least 2 operands");
  VectorType vLHS = tLHS.dyn_cast<VectorType>();
  VectorType vRHS = tRHS.dyn_cast<VectorType>();
  if (!vLHS || !vRHS)
    return parser.emitError(parser.getNameLoc(), "expected 2 vector types");
  VectorType resType = VectorType::get({vLHS.getDimSize(0), vRHS.getDimSize(0)},
                                       vLHS.getElementType());
  return failure(
      parser.resolveOperand(operandsInfo[0], tLHS, result.operands) ||
      parser.resolveOperand(operandsInfo[1], tRHS, result.operands) ||
      (operandsInfo.size() > 2 &&
       parser.resolveOperand(operandsInfo[2], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types));
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

static void print(OpAsmPrinter &p, VectorTransferReadOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.memref());
  p << "[";
  p.printOperands(op.indices());
  p << "], ";
  p.printOperand(op.padding());
  p << " ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getMemRefType();
  p << ", " << op.getVectorType();
}

ParseResult parseVectorTransferReadOp(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  OpAsmParser::OperandType paddingInfo;
  SmallVector<Type, 2> types;
  // Parsing with support for optional paddingValue.
  if (parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(paddingInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "two types required");
  auto indexType = parser.getBuilder().getIndexType();
  MemRefType memRefType = types[0].dyn_cast<MemRefType>();
  if (!memRefType)
    return parser.emitError(typesLoc, "memref type required"), failure();
  Type vectorType = types[1];
  return failure(
      parser.resolveOperand(memrefInfo, memRefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands) ||
      parser.resolveOperand(paddingInfo, memRefType.getElementType(),
                            result.operands) ||
      parser.addTypeToList(vectorType, result.types));
}

static LogicalResult verify(VectorTransferReadOp op) {
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = op.getMemRefType();
  VectorType vectorType = op.getVectorType();
  if (memrefType.getElementType() != vectorType.getElementType())
    return op.emitOpError(
        "requires memref and vector types of the same elemental type");
  auto elementalType = op.padding()->getType();
  if (!VectorType::isValidElementType(elementalType))
    return op.emitOpError("requires valid padding vector elemental type");
  if (elementalType != vectorType.getElementType())
    return op.emitOpError(
        "requires formal padding and vector of the same elemental type");
  if (llvm::size(op.indices()) != memrefType.getRank())
    return op.emitOpError("requires ") << memrefType.getRank() << " indices";
  auto permutationMap = op.permutation_map();
  if (permutationMap.getNumSymbols() != 0)
    return op.emitOpError("requires permutation_map without symbols");
  if (permutationMap.getNumInputs() != memrefType.getRank())
    return op.emitOpError("requires a permutation_map with input dims of the "
                          "same rank as the memref type");
  if (permutationMap.getNumResults() != vectorType.getRank())
    return op.emitOpError("requires a permutation_map with result dims of the "
                          "same rank as the vector type");
  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// VectorTransferWriteOp
//===----------------------------------------------------------------------===//
static void print(OpAsmPrinter &p, VectorTransferWriteOp op) {
  p << op.getOperationName() << " " << *op.vector() << ", " << *op.memref();
  p << "[";
  p.printOperands(op.indices());
  p << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.getVectorType());
  p << ", ";
  p.printType(op.getMemRefType());
}

ParseResult parseVectorTransferWriteOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  SmallVector<Type, 2> types;
  if (parser.parseOperand(storeValueInfo) || parser.parseComma() ||
      parser.parseOperand(memRefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "two types required");
  auto indexType = parser.getBuilder().getIndexType();
  Type vectorType = types[0], memRefType = types[1];
  return failure(
      parser.resolveOperand(storeValueInfo, vectorType, result.operands) ||
      parser.resolveOperand(memRefInfo, memRefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands));
}

static LogicalResult verify(VectorTransferWriteOp op) {
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = op.getMemRefType();
  VectorType vectorType = op.getVectorType();
  if (memrefType.getElementType() != vectorType.getElementType())
    return op.emitOpError(
        "requires memref and vector types of the same elemental type");
  if (llvm::size(op.indices()) != memrefType.getRank())
    return op.emitOpError("requires ") << memrefType.getRank() << " indices";

  // Consistency of AffineMap attribute.
  auto permutationMap = op.permutation_map();
  if (permutationMap.getNumSymbols() != 0)
    return op.emitOpError("requires a symbol-less permutation_map");
  if (permutationMap.getNumInputs() != memrefType.getRank())
    return op.emitOpError("requires a permutation_map with input dims of the "
                          "same rank as the memref type: ")
           << permutationMap.getNumInputs() << " vs " << memrefType;
  if (permutationMap.getNumResults() != vectorType.getRank())
    return op.emitOpError("requires a permutation_map with result dims of the "
                          "same rank as the vector type.")
           << permutationMap.getNumResults() << " vs " << vectorType;
  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// VectorTypeCastOp
//===----------------------------------------------------------------------===//

static MemRefType inferVectorTypeCastResultType(MemRefType t) {
  return MemRefType::get({}, VectorType::get(t.getShape(), t.getElementType()));
}

void VectorTypeCastOp::build(Builder *builder, OperationState &result,
                             Value *source) {
  result.addOperands(source);
  result.addTypes(
      inferVectorTypeCastResultType(source->getType().cast<MemRefType>()));
}

static void print(OpAsmPrinter &p, VectorTypeCastOp &op) {
  auto type = op.getOperand()->getType().cast<MemRefType>();
  p << op.getOperationName() << ' ' << *op.memref() << " : " << type << " to "
    << inferVectorTypeCastResultType(type);
}

static LogicalResult verify(VectorTypeCastOp &op) {
  auto resultType = inferVectorTypeCastResultType(op.getMemRefType());
  if (op.getResultMemRefType() != resultType)
    return op.emitOpError("expects result type to be: ") << resultType;
  return success();
}

namespace mlir {

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"

} // namespace mlir

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
// VectorContractionOp
//===----------------------------------------------------------------------===//

static ParseResult parseVectorContractionOp(OpAsmParser &parser,
                                            OperationState &result) {
  OpAsmParser::OperandType lhsInfo;
  OpAsmParser::OperandType rhsInfo;
  OpAsmParser::OperandType accInfo;
  SmallVector<OpAsmParser::OperandType, 2> masksInfo;
  SmallVector<Type, 2> types;
  Type resultVectorType;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsInfo) || parser.parseComma() ||
      parser.parseOperand(rhsInfo) || parser.parseComma() ||
      parser.parseOperand(accInfo) ||
      parser.parseTrailingOperandList(masksInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.parseKeywordType("into", resultVectorType) ||
      parser.resolveOperand(lhsInfo, types[0], result.operands) ||
      parser.resolveOperand(rhsInfo, types[1], result.operands) ||
      parser.resolveOperand(accInfo, resultVectorType, result.operands) ||
      parser.addTypeToList(resultVectorType, result.types))
    return failure();

  if (masksInfo.empty())
    return success();
  if (masksInfo.size() != 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected zero or exactly 2 vector mask operands");
  auto indexType = parser.getBuilder().getIndexType();
  auto lhsType = types[0].cast<VectorType>();
  auto rhsType = types[1].cast<VectorType>();
  SmallVector<Type, 2> maskTypes;
  SmallVector<Type, 4> lhsMaskElementTypes(lhsType.getRank(), indexType);
  maskTypes.push_back(
      TupleType::get(lhsMaskElementTypes, parser.getBuilder().getContext()));
  SmallVector<Type, 4> rhsMaskElementTypes(rhsType.getRank(), indexType);
  maskTypes.push_back(
      TupleType::get(rhsMaskElementTypes, parser.getBuilder().getContext()));
  if (parser.resolveOperands(masksInfo, maskTypes, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, VectorContractionOp op) {
  p << op.getOperationName() << " " << *op.lhs() << ", " << *op.rhs();
  p << ", " << *op.acc();
  if (llvm::size(op.masks()) == 2) {
    p << ", " << **op.masks().begin();
    p << ", " << **(op.masks().begin() + 1);
  }
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.lhs()->getType() << ", " << op.rhs()->getType() << " into "
    << op.getResultType();
}

static bool verifyDimMap(VectorType lhsType, VectorType rhsType,
                         const std::vector<std::pair<int64_t, int64_t>> &map) {
  for (auto &dimPair : map) {
    if (dimPair.first < 0 || dimPair.first >= lhsType.getRank() ||
        dimPair.second < 0 || dimPair.second >= rhsType.getRank() ||
        lhsType.getDimSize(dimPair.first) != rhsType.getDimSize(dimPair.second))
      return false;
  }
  return true;
}

static bool verifyOutputShape(
    VectorType lhsType, VectorType rhsType, VectorType accType,
    VectorType resType,
    const std::vector<std::pair<int64_t, int64_t>> &contractingDimMap,
    const std::vector<std::pair<int64_t, int64_t>> &batchDimMap) {
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }
  DenseSet<int64_t> rhsBatchDimSet;
  for (auto &dimPair : batchDimMap)
    rhsBatchDimSet.insert(dimPair.second);

  // Add free and batch dimensions from 'lhsType' to 'expectedResultDims'.
  SmallVector<int64_t, 4> expectedResultDims;
  for (int64_t i = 0, e = lhsType.getRank(); i < e; ++i) {
    if (lhsContractingDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(lhsType.getDimSize(i));
  }

  // Add free dimensions from 'rhsType' to 'expectedResultDims'.
  for (int64_t i = 0, e = rhsType.getRank(); i < e; ++i) {
    if (rhsContractingDimSet.count(i) > 0 || rhsBatchDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(rhsType.getDimSize(i));
  }

  // Verify dimension from 'resType' against 'expectedResultDims'.
  if (resType.getShape().size() != expectedResultDims.size() ||
      accType.getShape().size() != expectedResultDims.size())
    return false;
  for (int64_t i = 0, e = resType.getRank(); i < e; ++i) {
    if (resType.getDimSize(i) != expectedResultDims[i] ||
        accType.getDimSize(i) != expectedResultDims[i])
      return false;
  }
  return true;
}

static LogicalResult verify(VectorContractionOp op) {
  auto lhsType = op.getLhsType();
  auto rhsType = op.getRhsType();
  auto accType = op.getAccType();
  auto resType = op.getResultType();
  auto contractingDimMap = op.getContractingDimMap();
  auto batchDimMap = op.getBatchDimMap();

  // Verify at least one contracting dimension pair was specified.
  if (contractingDimMap.empty())
    return op.emitOpError("expected at least one contracting dimension pair");

  // Verify contracting dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, contractingDimMap))
    return op.emitOpError("invalid contracting dimension map");

  // Verify batch dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, batchDimMap))
    return op.emitOpError("invalid batch dimension map");

  // Verify 'accType' and 'resType' shape.
  if (!verifyOutputShape(lhsType, rhsType, accType, resType, contractingDimMap,
                         batchDimMap))
    return op.emitOpError("invalid accumulator/result vector shape");

  // Verify that either two vector masks are set or none are set.
  auto lhsMaskType = op.getLHSVectorMaskType();
  auto rhsMaskType = op.getRHSVectorMaskType();
  if ((lhsMaskType && !rhsMaskType) || (!lhsMaskType && rhsMaskType))
    return op.emitOpError("invalid number of vector masks specified");
  if (lhsMaskType && rhsMaskType) {
    // Verify tuple element size is != rank.
    if (lhsMaskType.getTypes().size() != lhsType.getShape().size() ||
        rhsMaskType.getTypes().size() != rhsType.getShape().size())
      return op.emitOpError("invalid number of vector mask elements");
    // Verify all tuple elements are index type.
    for (auto eltType : lhsMaskType.getTypes()) {
      if (!eltType.isa<IndexType>())
        return op.emitOpError("vector mask element must have index type");
    }
  }
  return success();
}

static std::vector<std::pair<int64_t, int64_t>> getDimMap(Attribute attr) {
  std::vector<std::pair<int64_t, int64_t>> dimMap;
  auto dimPairs = attr.dyn_cast_or_null<ArrayAttr>();
  if (!dimPairs)
    return dimMap;
  for (auto dimPairAttr : dimPairs) {
    auto dimPair = dimPairAttr.cast<ArrayAttr>();
    assert(dimPair.size() == 2);
    auto lhsDim = dimPair.begin()->cast<IntegerAttr>().getInt();
    auto rhsDim = std::prev(dimPair.end())->cast<IntegerAttr>().getInt();
    dimMap.push_back({lhsDim, rhsDim});
  }
  return dimMap;
}

std::vector<std::pair<int64_t, int64_t>>
VectorContractionOp::getContractingDimMap() {
  return getDimMap(getAttr(getContractingDimMapAttrName()));
}

std::vector<std::pair<int64_t, int64_t>> VectorContractionOp::getBatchDimMap() {
  return getDimMap(getAttr(getBatchDimMapAttrName()));
}

//===----------------------------------------------------------------------===//
// VectorExtractElementOp
//===----------------------------------------------------------------------===//

static Type inferExtractOpResultType(VectorType vectorType,
                                     ArrayAttr position) {
  if (static_cast<int64_t>(position.size()) == vectorType.getRank())
    return vectorType.getElementType();
  return VectorType::get(vectorType.getShape().drop_front(position.size()),
                         vectorType.getElementType());
}

void VectorExtractElementOp::build(Builder *builder, OperationState &result,
                                   Value *source, ArrayRef<int32_t> position) {
  result.addOperands(source);
  auto positionAttr = builder->getI32ArrayAttr(position);
  result.addTypes(inferExtractOpResultType(source->getType().cast<VectorType>(),
                                           positionAttr));
  result.addAttribute(getPositionAttrName(), positionAttr);
}

static void print(OpAsmPrinter &p, VectorExtractElementOp op) {
  p << op.getOperationName() << " " << *op.vector() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.vector()->getType();
}

static ParseResult parseVectorExtractElementOp(OpAsmParser &parser,
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
        "expected position attribute of rank smaller than vector rank");

  Type resType = inferExtractOpResultType(vectorType, positionAttr);
  result.attributes = attrs;
  return failure(parser.resolveOperand(vector, type, result.operands) ||
                 parser.addTypeToList(resType, result.types));
}

static LogicalResult verify(VectorExtractElementOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  if (positionAttr.size() > static_cast<unsigned>(op.getVectorType().getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than vector rank");
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
// VectorStridedSliceOp
//===----------------------------------------------------------------------===//

static Type inferVectorExtractRangeOpResultType(VectorType vectorType,
                                                ArrayAttr offsets,
                                                ArrayAttr sizes,
                                                ArrayAttr strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(sizes.getValue()[idx].cast<IntegerAttr>().getInt());
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType());
}

void VectorStridedSliceOp::build(Builder *builder, OperationState &result,
                                 Value *source, ArrayRef<int64_t> offsets,
                                 ArrayRef<int64_t> sizes,
                                 ArrayRef<int64_t> strides) {
  result.addOperands(source);
  auto offsetsAttr = builder->getI64ArrayAttr(offsets);
  auto sizesAttr = builder->getI64ArrayAttr(sizes);
  auto stridesAttr = builder->getI64ArrayAttr(strides);
  result.addTypes(
      inferVectorExtractRangeOpResultType(source->getType().cast<VectorType>(),
                                          offsetsAttr, sizesAttr, stridesAttr));
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static void print(OpAsmPrinter &p, VectorStridedSliceOp op) {
  p << op.getOperationName() << " " << *op.vector();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector()->getType() << " to " << op.getResult()->getType();
}

static ParseResult parseVectorStridedSliceOp(OpAsmParser &parser,
                                             OperationState &result) {
  llvm::SMLoc attributeLoc, typeLoc;
  OpAsmParser::OperandType vector;
  VectorType vectorType, resultVectorType;
  return failure(parser.parseOperand(vector) ||
                 parser.getCurrentLocation(&attributeLoc) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.getCurrentLocation(&typeLoc) ||
                 parser.parseColonType(vectorType) ||
                 parser.parseKeywordType("to", resultVectorType) ||
                 parser.resolveOperand(vector, vectorType, result.operands) ||
                 parser.addTypeToList(resultVectorType, result.types));
}

// TODO(ntv) Should be moved to Tablegen Confined attributes.
static bool isIntegerArrayAttrSmallerThanShape(VectorStridedSliceOp op,
                                               ArrayAttr arrayAttr,
                                               ShapedType shape,
                                               StringRef attrName) {
  if (arrayAttr.size() > static_cast<unsigned>(shape.getRank())) {
    op.emitOpError("expected ")
        << attrName << " attribute of rank smaller than vector rank";
    return false;
  }
  return true;
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
static bool isIntegerArrayAttrConfinedToRange(VectorStridedSliceOp op,
                                              ArrayAttr arrayAttr, int64_t min,
                                              int64_t max, StringRef attrName,
                                              bool halfOpen = true) {
  for (auto attr : arrayAttr) {
    auto val = attr.cast<IntegerAttr>().getInt();
    auto upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper) {
      op.emitOpError("expected ")
          << attrName << " to be confined to [" << min << ", " << upper << ")";
      return false;
    }
  }
  return true;
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
static bool
isIntegerArrayAttrConfinedToShape(VectorStridedSliceOp op, ArrayAttr arrayAttr,
                                  ShapedType shape, StringRef attrName,
                                  bool halfOpen = true, int64_t min = 0) {
  assert(arrayAttr.size() <= static_cast<unsigned>(shape.getRank()));
  for (auto it : llvm::zip(arrayAttr, shape.getShape())) {
    auto val = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<1>(it);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max) {
      op.emitOpError("expected ")
          << attrName << " to be confined to [" << min << ", " << max << ")";
      return false;
    }
  }
  return true;
}

// Returns true if all integers in `arrayAttr` are in the interval [min, max}.
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
static bool isSumOfIntegerArrayAttrConfinedToShape(
    VectorStridedSliceOp op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    ShapedType shape, StringRef attrName1, StringRef attrName2,
    bool halfOpen = true, int64_t min = 1) {
  assert(arrayAttr1.size() <= static_cast<unsigned>(shape.getRank()));
  assert(arrayAttr2.size() <= static_cast<unsigned>(shape.getRank()));
  for (auto it : llvm::zip(arrayAttr1, arrayAttr2, shape.getShape())) {
    auto val1 = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto val2 = std::get<1>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max) {
      op.emitOpError("expected sum(")
          << attrName1 << ", " << attrName2 << ") to be confined to [" << min
          << ", " << max << ")";
      return false;
    }
  }
  return true;
}

static LogicalResult verify(VectorStridedSliceOp op) {
  auto type = op.getVectorType();
  auto offsets = op.offsets();
  auto sizes = op.sizes();
  auto strides = op.strides();
  if (offsets.size() != sizes.size() || offsets.size() != strides.size()) {
    op.emitOpError(
        "expected offsets, sizes and strides attributes of same size");
    return failure();
  }

  auto offName = VectorStridedSliceOp::getOffsetsAttrName();
  auto sizesName = VectorStridedSliceOp::getSizesAttrName();
  auto stridesName = VectorStridedSliceOp::getStridesAttrName();
  if (!isIntegerArrayAttrSmallerThanShape(op, offsets, type, offName) ||
      !isIntegerArrayAttrSmallerThanShape(op, sizes, type, sizesName) ||
      !isIntegerArrayAttrSmallerThanShape(op, strides, type, stridesName) ||
      !isIntegerArrayAttrConfinedToShape(op, offsets, type, offName) ||
      !isIntegerArrayAttrConfinedToShape(op, sizes, type, sizesName,
                                         /*halfOpen=*/false, /*min=*/1) ||
      !isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                         /*halfOpen=*/false) ||
      !isSumOfIntegerArrayAttrConfinedToShape(op, offsets, sizes, type, offName,
                                              sizesName, /*halfOpen=*/false))
    return failure();

  auto resultType = inferVectorExtractRangeOpResultType(
      op.getVectorType(), op.offsets(), op.sizes(), op.strides());
  if (op.getResult()->getType() != resultType) {
    op.emitOpError("expected result type to be ") << resultType;
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VectorOuterProductOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, VectorOuterProductOp op) {
  p << op.getOperationName() << " " << *op.lhs() << ", " << *op.rhs();
  if (llvm::size(op.acc()) > 0)
    p << ", " << **op.acc().begin();
  p << " : " << op.lhs()->getType() << ", " << op.rhs()->getType();
}

static ParseResult parseVectorOuterProductOp(OpAsmParser &parser,
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

static LogicalResult verify(VectorOuterProductOp op) {
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

//===----------------------------------------------------------------------===//
// VectorIndexTupleOp
//===----------------------------------------------------------------------===//

ParseResult parseVectorIndexTupleOp(OpAsmParser &parser,
                                    OperationState &result) {
  auto indexType = parser.getBuilder().getIndexType();
  Type resultType;
  SmallVector<OpAsmParser::OperandType, 4> operandInfo;
  return failure(
      parser.parseOperandList(operandInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType) ||
      parser.resolveOperands(operandInfo, indexType, result.operands) ||
      parser.addTypeToList(resultType, result.types));
}

static void print(OpAsmPrinter &p, VectorIndexTupleOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.operands());
  p << " : " << op.getResult()->getType();
}

static LogicalResult verify(VectorIndexTupleOp &op) {
  for (auto operand : op.getOperands())
    if (!operand->getType().isa<IndexType>())
      return op.emitOpError("all operands must be of index type");
  return success();
}

namespace mlir {

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"

} // namespace mlir

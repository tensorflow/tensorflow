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
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringSet.h"

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
// ContractionOp
//===----------------------------------------------------------------------===//

static ParseResult parseContractionOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType lhsInfo;
  OpAsmParser::OperandType rhsInfo;
  OpAsmParser::OperandType accInfo;
  SmallVector<OpAsmParser::OperandType, 2> masksInfo;
  SmallVector<Type, 2> types;
  Type resultVectorType;
  auto loc = parser.getCurrentLocation();
  DictionaryAttr dictAttr;
  // TODO(andydavis, ntv) Unify linalg op attribute parsing.
  if (parser.parseAttribute(dictAttr, "_", result.attributes) ||
      parser.parseOperand(lhsInfo) || parser.parseComma() ||
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
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());
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

static void print(OpAsmPrinter &p, ContractionOp op) {
  // TODO(andydavis, ntv) Unify printing code with linalg ops.
  auto attrNames = op.getTraitAttrNames();
  llvm::StringSet<> traitAttrsSet;
  traitAttrsSet.insert(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : op.getAttrs()) {
    if (traitAttrsSet.count(attr.first.strref()) > 0)
      attrs.push_back(attr);
  }
  auto dictAttr = DictionaryAttr::get(attrs, op.getContext());
  p << op.getOperationName() << " " << dictAttr << " " << *op.lhs() << ", ";
  p << *op.rhs() << ", " << *op.acc();
  if (llvm::size(op.masks()) == 2) {
    p << ", " << **op.masks().begin();
    p << ", " << **(op.masks().begin() + 1);
  }
  p.printOptionalAttrDict(op.getAttrs(), attrNames);
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

static LogicalResult verify(ContractionOp op) {
  auto lhsType = op.getLhsType();
  auto rhsType = op.getRhsType();
  auto accType = op.getAccType();
  auto resType = op.getResultType();

  // Verify that an indexing map was specified for each vector operand.
  if (op.indexing_maps().size() != 3)
    return op.emitOpError("expected an indexing map for each vector operand");

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = op.iterator_types().getValue().size();
  for (auto it : llvm::enumerate(op.indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    if (map.getNumSymbols() != 0)
      return op.emitOpError("expected indexing map ")
             << index << " to have no symbols";
    if (map.getNumDims() != numIterators)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " number of inputs";
    auto operandType = op.getOperand(index)->getType().cast<VectorType>();
    unsigned rank = operandType.getShape().size();
    if (map.getNumResults() != rank)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << rank << " number of outputs";
    if (!map.isProjectedPermutation())
      return op.emitOpError("expected indexing map ")
             << index << " to be a projected permutation of its inputs";
  }

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

SmallVector<StringRef, 2> ContractionOp::getTraitAttrNames() {
  return SmallVector<StringRef, 2>{"indexing_maps", "iterator_types"};
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

static std::vector<std::pair<int64_t, int64_t>>
getDimMap(ArrayRef<AffineMap> indexingMaps, ArrayAttr iteratorTypes,
          StringRef targetIteratorTypeName, MLIRContext *context) {
  std::vector<std::pair<int64_t, int64_t>> dimMap;
  for (auto it : llvm::enumerate(iteratorTypes)) {
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName != targetIteratorTypeName)
      continue;
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), context);
    int64_t lhsDim = getResultIndex(indexingMaps[0], targetExpr);
    int64_t rhsDim = getResultIndex(indexingMaps[1], targetExpr);
    if (lhsDim >= 0 && rhsDim >= 0)
      dimMap.push_back({lhsDim, rhsDim});
  }
  return dimMap;
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getContractingDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getReductionIteratorTypeName(), getContext());
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getBatchDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getParallelIteratorTypeName(), getContext());
}

SmallVector<AffineMap, 4> ContractionOp::getIndexingMaps() {
  SmallVector<AffineMap, 4> res;
  auto mapAttrs = indexing_maps().getValue();
  res.reserve(mapAttrs.size());
  for (auto mapAttr : mapAttrs)
    res.push_back(mapAttr.cast<AffineMapAttr>().getValue());
  return res;
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static Type inferExtractElementOpResultType(VectorType vectorType,
                                            ArrayAttr position) {
  if (static_cast<int64_t>(position.size()) == vectorType.getRank())
    return vectorType.getElementType();
  return VectorType::get(vectorType.getShape().drop_front(position.size()),
                         vectorType.getElementType());
}

void ExtractElementOp::build(Builder *builder, OperationState &result,
                             Value *source, ArrayRef<int32_t> position) {
  result.addOperands(source);
  auto positionAttr = builder->getI32ArrayAttr(position);
  result.addTypes(inferExtractElementOpResultType(
      source->getType().cast<VectorType>(), positionAttr));
  result.addAttribute(getPositionAttrName(), positionAttr);
}

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
        "expected position attribute of rank smaller than vector rank");

  Type resType = inferExtractElementOpResultType(vectorType, positionAttr);
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
        "expected position attribute of rank smaller than vector rank");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() > op.getVectorType().getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, BroadcastOp op) {
  p << op.getOperationName() << " " << *op.source();
  p << " : " << op.getSourceType();
  p << " to " << op.getVectorType();
}

static LogicalResult verify(BroadcastOp op) {
  VectorType srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  VectorType dstVectorType = op.getVectorType();
  // Scalar to vector broadcast is always valid. A vector
  // to vector broadcast needs some additional checking.
  if (srcVectorType) {
    const int64_t srcRank = srcVectorType.getRank();
    const int64_t dstRank = dstVectorType.getRank();
    // TODO(ajcbik): implement proper rank testing for broadcast;
    // this is just a temporary placeholder check.
    if (srcRank > dstRank) {
      return op.emitOpError("source rank higher than destination rank");
    }
  }
  return success();
}

static ParseResult parseBroadcastOp(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::OperandType source;
  Type sourceType;
  VectorType vectorType;
  return failure(parser.parseOperand(source) ||
                 parser.parseColonType(sourceType) ||
                 parser.parseKeywordType("to", vectorType) ||
                 parser.resolveOperand(source, sourceType, result.operands) ||
                 parser.addTypeToList(vectorType, result.types));
}

//===----------------------------------------------------------------------===//
// InsertElementOp
//===----------------------------------------------------------------------===//

void InsertElementOp::build(Builder *builder, OperationState &result,
                            Value *source, Value *dest,
                            ArrayRef<int32_t> position) {
  result.addOperands({source, dest});
  auto positionAttr = builder->getI32ArrayAttr(position);
  result.addTypes(dest->getType());
  result.addAttribute(getPositionAttrName(), positionAttr);
}

static void print(OpAsmPrinter &p, InsertElementOp op) {
  p << op.getOperationName() << " " << *op.source() << ", " << *op.dest()
    << op.position();
  p.printOptionalAttrDict(op.getAttrs(),
                          {InsertElementOp::getPositionAttrName()});
  p << " : " << op.getSourceType();
  p << " into " << op.getDestVectorType();
}

static ParseResult parseInsertElementOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType source, dest;
  Type sourceType;
  VectorType destType;
  Attribute attr;
  return failure(parser.parseOperand(source) || parser.parseComma() ||
                 parser.parseOperand(dest) ||
                 parser.parseAttribute(attr,
                                       InsertElementOp::getPositionAttrName(),
                                       result.attributes) ||
                 parser.parseOptionalAttrDict(attrs) ||
                 parser.parseColonType(sourceType) ||
                 parser.parseKeywordType("into", destType) ||
                 parser.resolveOperand(source, sourceType, result.operands) ||
                 parser.resolveOperand(dest, destType, result.operands) ||
                 parser.addTypeToList(destType, result.types));
}

static LogicalResult verify(InsertElementOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  auto destVectorType = op.getDestVectorType();
  if (positionAttr.size() > static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than dest vector rank");
  auto srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  if (srcVectorType &&
      (static_cast<unsigned>(srcVectorType.getRank()) + positionAttr.size() !=
       static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError("expected position attribute rank + source rank to "
                          "match dest vector rank");
  else if (!srcVectorType && (positionAttr.size() !=
                              static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError(
        "expected position attribute rank to match the dest vector rank");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() > destVectorType.getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "dest vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InsertStridedSliceOp
//===----------------------------------------------------------------------===//

void InsertStridedSliceOp::build(Builder *builder, OperationState &result,
                                 Value *source, Value *dest,
                                 ArrayRef<int64_t> offsets,
                                 ArrayRef<int64_t> strides) {
  result.addOperands({source, dest});
  auto offsetsAttr = builder->getI64ArrayAttr(offsets);
  auto stridesAttr = builder->getI64ArrayAttr(strides);
  result.addTypes(dest->getType());
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static void print(OpAsmPrinter &p, InsertStridedSliceOp op) {
  p << op.getOperationName() << " " << *op.source() << ", " << *op.dest()
    << " ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getSourceVectorType() << " into " << op.getDestVectorType();
}

static ParseResult parseInsertStridedSliceOp(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::OperandType source, dest;
  VectorType sourceVectorType, destVectorType;
  return failure(
      parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(dest) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceVectorType) ||
      parser.parseKeywordType("into", destVectorType) ||
      parser.resolveOperand(source, sourceVectorType, result.operands) ||
      parser.resolveOperand(dest, destVectorType, result.operands) ||
      parser.addTypeToList(destVectorType, result.types));
}

// TODO(ntv) Should be moved to Tablegen Confined attributes.
template <typename OpType>
LogicalResult isIntegerArrayAttrSmallerThanShape(OpType op, ArrayAttr arrayAttr,
                                                 ArrayRef<int64_t> shape,
                                                 StringRef attrName) {
  if (arrayAttr.size() > shape.size())
    return op.emitOpError("expected ")
           << attrName << " attribute of rank smaller than vector rank";
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
LogicalResult isIntegerArrayAttrConfinedToRange(OpType op, ArrayAttr arrayAttr,
                                                int64_t min, int64_t max,
                                                StringRef attrName,
                                                bool halfOpen = true) {
  for (auto attr : arrayAttr) {
    auto val = attr.cast<IntegerAttr>().getInt();
    auto upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper)
      return op.emitOpError("expected ") << attrName << " to be confined to ["
                                         << min << ", " << upper << ")";
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
LogicalResult
isIntegerArrayAttrConfinedToShape(OpType op, ArrayAttr arrayAttr,
                                  ArrayRef<int64_t> shape, StringRef attrName,
                                  bool halfOpen = true, int64_t min = 0) {
  assert(arrayAttr.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr, shape)) {
    auto val = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<1>(it);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max)
      return op.emitOpError("expected ")
             << attrName << " dimension " << index << " to be confined to ["
             << min << ", " << max << ")";
    ++index;
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the interval [min, max}.
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
LogicalResult isSumOfIntegerArrayAttrConfinedToShape(
    OpType op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    ArrayRef<int64_t> shape, StringRef attrName1, StringRef attrName2,
    bool halfOpen = true, int64_t min = 1) {
  assert(arrayAttr1.size() <= shape.size());
  assert(arrayAttr2.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr1, arrayAttr2, shape)) {
    auto val1 = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto val2 = std::get<1>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max)
      return op.emitOpError("expected sum(")
             << attrName1 << ", " << attrName2 << ") dimension " << index
             << " to be confined to [" << min << ", " << max << ")";
    ++index;
  }
  return success();
}

static ArrayAttr makeI64ArrayAttr(ArrayRef<int64_t> values,
                                  MLIRContext *context) {
  auto attrs = functional::map(
      [context](int64_t v) -> Attribute {
        return IntegerAttr::get(IntegerType::get(64, context), APInt(64, v));
      },
      values);
  return ArrayAttr::get(attrs, context);
}

static LogicalResult verify(InsertStridedSliceOp op) {
  auto sourceVectorType = op.getSourceVectorType();
  auto destVectorType = op.getDestVectorType();
  auto offsets = op.offsets();
  auto strides = op.strides();
  if (offsets.size() != static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected offsets of same size as destination vector rank");
  if (strides.size() != static_cast<unsigned>(sourceVectorType.getRank()))
    return op.emitOpError(
        "expected strides of same size as source vector rank");
  if (sourceVectorType.getRank() > destVectorType.getRank())
    return op.emitOpError(
        "expected source rank to be smaller than destination rank");

  auto sourceShape = sourceVectorType.getShape();
  auto destShape = destVectorType.getShape();
  SmallVector<int64_t, 4> sourceShapeAsDestShape(
      destShape.size() - sourceShape.size(), 0);
  sourceShapeAsDestShape.append(sourceShape.begin(), sourceShape.end());
  auto offName = InsertStridedSliceOp::getOffsetsAttrName();
  auto stridesName = InsertStridedSliceOp::getStridesAttrName();
  if (failed(
          isIntegerArrayAttrConfinedToShape(op, offsets, destShape, offName)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(
          op, offsets,
          makeI64ArrayAttr(sourceShapeAsDestShape, op.getContext()), destShape,
          offName, "source vector shape",
          /*halfOpen=*/false, /*min=*/1)))
    return failure();

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
// StridedSliceOp
//===----------------------------------------------------------------------===//

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          ArrayAttr offsets, ArrayAttr sizes,
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

void StridedSliceOp::build(Builder *builder, OperationState &result,
                           Value *source, ArrayRef<int64_t> offsets,
                           ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides) {
  result.addOperands(source);
  auto offsetsAttr = builder->getI64ArrayAttr(offsets);
  auto sizesAttr = builder->getI64ArrayAttr(sizes);
  auto stridesAttr = builder->getI64ArrayAttr(strides);
  result.addTypes(
      inferStridedSliceOpResultType(source->getType().cast<VectorType>(),
                                    offsetsAttr, sizesAttr, stridesAttr));
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static void print(OpAsmPrinter &p, StridedSliceOp op) {
  p << op.getOperationName() << " " << *op.vector();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector()->getType() << " to " << op.getResult()->getType();
}

static ParseResult parseStridedSliceOp(OpAsmParser &parser,
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

static LogicalResult verify(StridedSliceOp op) {
  auto type = op.getVectorType();
  auto offsets = op.offsets();
  auto sizes = op.sizes();
  auto strides = op.strides();
  if (offsets.size() != sizes.size() || offsets.size() != strides.size()) {
    op.emitOpError(
        "expected offsets, sizes and strides attributes of same size");
    return failure();
  }

  auto shape = type.getShape();
  auto offName = StridedSliceOp::getOffsetsAttrName();
  auto sizesName = StridedSliceOp::getSizesAttrName();
  auto stridesName = StridedSliceOp::getStridesAttrName();
  if (failed(isIntegerArrayAttrSmallerThanShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, sizes, shape, sizesName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, strides, shape,
                                                stridesName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, sizes, shape, sizesName,
                                               /*halfOpen=*/false,
                                               /*min=*/1)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(op, offsets, sizes, shape,
                                                    offName, sizesName,
                                                    /*halfOpen=*/false)))
    return failure();

  auto resultType = inferStridedSliceOpResultType(
      op.getVectorType(), op.offsets(), op.sizes(), op.strides());
  if (op.getResult()->getType() != resultType) {
    op.emitOpError("expected result type to be ") << resultType;
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransferReadOp
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

static void print(OpAsmPrinter &p, TransferReadOp op) {
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

ParseResult parseTransferReadOp(OpAsmParser &parser, OperationState &result) {
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

static LogicalResult verify(TransferReadOp op) {
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
// TransferWriteOp
//===----------------------------------------------------------------------===//
static void print(OpAsmPrinter &p, TransferWriteOp op) {
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

ParseResult parseTransferWriteOp(OpAsmParser &parser, OperationState &result) {
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

static LogicalResult verify(TransferWriteOp op) {
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
// TypeCastOp
//===----------------------------------------------------------------------===//

static MemRefType inferVectorTypeCastResultType(MemRefType t) {
  return MemRefType::get({}, VectorType::get(t.getShape(), t.getElementType()));
}

void TypeCastOp::build(Builder *builder, OperationState &result,
                       Value *source) {
  result.addOperands(source);
  result.addTypes(
      inferVectorTypeCastResultType(source->getType().cast<MemRefType>()));
}

static void print(OpAsmPrinter &p, TypeCastOp &op) {
  auto type = op.getOperand()->getType().cast<MemRefType>();
  p << op.getOperationName() << ' ' << *op.memref() << " : " << type << " to "
    << inferVectorTypeCastResultType(type);
}

static LogicalResult verify(TypeCastOp &op) {
  auto resultType = inferVectorTypeCastResultType(op.getMemRefType());
  if (op.getResultMemRefType() != resultType)
    return op.emitOpError("expects result type to be: ") << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// IndexTupleOp
//===----------------------------------------------------------------------===//

ParseResult parseIndexTupleOp(OpAsmParser &parser, OperationState &result) {
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

static void print(OpAsmPrinter &p, IndexTupleOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.operands());
  p << " : " << op.getResult()->getType();
}

static LogicalResult verify(IndexTupleOp &op) {
  for (auto operand : op.getOperands())
    if (!operand->getType().isa<IndexType>())
      return op.emitOpError("all operands must be of index type");
  return success();
}

namespace mlir {
namespace vector {

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"

} // namespace vector
} // namespace mlir

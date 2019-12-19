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
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
// VectorOpsDialect
//===----------------------------------------------------------------------===//

VectorOpsDialect::VectorOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *VectorOpsDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}

IntegerType vector::getVectorSubscriptType(Builder &builder) {
  return builder.getIntegerType(32);
}

ArrayAttr vector::getVectorSubscriptAttr(Builder &builder,
                                         ArrayRef<int32_t> values) {
  return builder.getI32ArrayAttr(values);
}

//===----------------------------------------------------------------------===//
// ContractionOp
//===----------------------------------------------------------------------===//

void vector::ContractionOp::build(Builder *builder, OperationState &result,
                                  Value *lhs, Value *rhs, Value *acc,
                                  ArrayAttr indexingMaps,
                                  ArrayAttr iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc->getType());
  result.addAttribute(getIndexingMapsAttrName(), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(), iteratorTypes);
}

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
  auto lhsType = types[0].cast<VectorType>();
  auto rhsType = types[1].cast<VectorType>();
  auto maskElementType = parser.getBuilder().getI1Type();
  SmallVector<Type, 2> maskTypes;
  maskTypes.push_back(VectorType::get(lhsType.getShape(), maskElementType));
  maskTypes.push_back(VectorType::get(rhsType.getShape(), maskElementType));
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
  for (auto attr : op.getAttrs())
    if (traitAttrsSet.count(attr.first.strref()) > 0)
      attrs.push_back(attr);

  auto dictAttr = DictionaryAttr::get(attrs, op.getContext());
  p << op.getOperationName() << " " << dictAttr << " " << *op.lhs() << ", ";
  p << *op.rhs() << ", " << *op.acc();
  if (op.masks().size() == 2)
    p << ", " << op.masks();

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
    // Verify mask rank == argument rank.
    if (lhsMaskType.getShape().size() != lhsType.getShape().size() ||
        rhsMaskType.getShape().size() != rhsType.getShape().size())
      return op.emitOpError("invalid vector mask rank");
  }
  return success();
}

ArrayRef<StringRef> ContractionOp::getTraitAttrNames() {
  static constexpr StringLiteral names[2] = {getIndexingMapsAttrName(),
                                             getIteratorTypesAttrName()};
  ArrayRef<StringLiteral> res{names};
  return ArrayRef<StringRef>{res.begin(), res.end()};
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

void ContractionOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resShape = getResultType().getShape();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  SmallVector<int64_t, 2> iterationShape;
  for (auto it : llvm::enumerate(iterator_types())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName == getReductionIteratorTypeName()) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      assert(lhsDimIndex >= 0);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
      continue;
    }
    // Get parallel dimension size from result shape.
    int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
    assert(resDimIndex >= 0);
    iterationBounds.push_back(resShape[resDimIndex]);
  }
}

void ContractionOp::getIterationIndexMap(
    std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap) {
  unsigned numMaps = indexing_maps().getValue().size();
  iterationIndexMap.resize(numMaps);
  for (auto it : llvm::enumerate(indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
      auto dim = map.getResult(i).cast<AffineDimExpr>();
      iterationIndexMap[index][dim.getPosition()] = i;
    }
  }
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

static void print(OpAsmPrinter &p, vector::ExtractElementOp op) {
  p << op.getOperationName() << " " << *op.vector() << "[" << *op.position()
    << " : " << op.position()->getType() << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector()->getType();
}

static ParseResult parseExtractElementOp(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::OperandType vector, position;
  Type positionType;
  VectorType vectorType;
  if (parser.parseOperand(vector) || parser.parseLSquare() ||
      parser.parseOperand(position) || parser.parseColonType(positionType) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(vectorType))
    return failure();
  Type resultType = vectorType.getElementType();
  return failure(
      parser.resolveOperand(vector, vectorType, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands) ||
      parser.addTypeToList(resultType, result.types));
}

static LogicalResult verify(vector::ExtractElementOp op) {
  VectorType vectorType = op.getVectorType();
  if (vectorType.getRank() != 1)
    return op.emitOpError("expected 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

static Type inferExtractOpResultType(VectorType vectorType,
                                     ArrayAttr position) {
  if (static_cast<int64_t>(position.size()) == vectorType.getRank())
    return vectorType.getElementType();
  return VectorType::get(vectorType.getShape().drop_front(position.size()),
                         vectorType.getElementType());
}

void vector::ExtractOp::build(Builder *builder, OperationState &result,
                              Value *source, ArrayRef<int32_t> position) {
  result.addOperands(source);
  auto positionAttr = getVectorSubscriptAttr(*builder, position);
  result.addTypes(inferExtractOpResultType(source->getType().cast<VectorType>(),
                                           positionAttr));
  result.addAttribute(getPositionAttrName(), positionAttr);
}

static void print(OpAsmPrinter &p, vector::ExtractOp op) {
  p << op.getOperationName() << " " << *op.vector() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.vector()->getType();
}

static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
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

static LogicalResult verify(vector::ExtractOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  if (positionAttr.size() > static_cast<unsigned>(op.getVectorType().getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than vector rank");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() >= op.getVectorType().getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractSlicesOp
//===----------------------------------------------------------------------===//

void ExtractSlicesOp::build(Builder *builder, OperationState &result,
                            TupleType tupleType, Value *vector,
                            ArrayRef<int64_t> sizes,
                            ArrayRef<int64_t> strides) {
  result.addOperands(vector);
  auto sizesAttr = builder->getI64ArrayAttr(sizes);
  auto stridesAttr = builder->getI64ArrayAttr(strides);
  result.addTypes(tupleType);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static ParseResult parseExtractSlicesOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  ArrayAttr sizesAttr;
  StringRef sizesAttrName = ExtractSlicesOp::getSizesAttrName();
  ArrayAttr stridesAttr;
  StringRef stridesAttrName = ExtractSlicesOp::getStridesAttrName();
  VectorType vectorType;
  TupleType resultTupleType;
  return failure(
      parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(sizesAttr, sizesAttrName, result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(stridesAttr, stridesAttrName, result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(vectorType) ||
      parser.parseKeywordType("into", resultTupleType) ||
      parser.resolveOperand(operandInfo, vectorType, result.operands) ||
      parser.addTypeToList(resultTupleType, result.types));
}

static void print(OpAsmPrinter &p, ExtractSlicesOp op) {
  p << op.getOperationName() << ' ' << *op.vector() << ", ";
  p << op.sizes() << ", " << op.strides();
  p.printOptionalAttrDict(
      op.getAttrs(),
      /*elidedAttrs=*/{ExtractSlicesOp::getSizesAttrName(),
                       ExtractSlicesOp::getStridesAttrName()});
  p << " : " << op.vector()->getType();
  p << " into " << op.getResultTupleType();
}

static LogicalResult
isValidExtractOrInsertSlicesType(Operation *op, VectorType vectorType,
                                 TupleType tupleType, ArrayRef<int64_t> sizes,
                                 ArrayRef<int64_t> strides) {
  // Check for non-unit strides.
  // TODO(b/144845578) Support non-1 strides.
  if (llvm::any_of(strides, [](int64_t s) { return s != 1; }))
    return op->emitError("requires unit strides");
  // Check that 'vectorType' rank matches rank of tuple element vectors.
  unsigned rank = vectorType.getRank();
  auto is_vector_type_of_rank = [&](Type t) {
    return t.isa<VectorType>() && t.cast<VectorType>().getRank() == rank;
  };
  if (!llvm::all_of(tupleType.getTypes(), is_vector_type_of_rank))
    return op->emitError("requires vector tuple elements of rank ") << rank;
  // Check that 'sizes' and 'strides' are of size == 'rank'.
  if (sizes.size() != rank || strides.size() != rank)
    return op->emitError("requires sizes and strides of rank ") << rank;

  // Compute the number of slices in each dimension.
  // TODO(andydavis) Move this into a slice generation helper function.
  auto shape = vectorType.getShape();
  SmallVector<int64_t, 4> dimSliceCounts(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimSliceCounts[i] = ceilDiv(shape[i], sizes[i]);
  // Compute the strides between slices in each dimension.
  SmallVector<int64_t, 4> sliceStrides(rank);
  sliceStrides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    sliceStrides[i] = sliceStrides[i + 1] * dimSliceCounts[i + 1];

  // Generate each slice shape based on 'sizes', 'strides' and 'vectorType',
  // and varify that the same matches the corresponding tuple element 'i'.
  for (int64_t i = 0, e = tupleType.size(); i < e; ++i) {
    // De-linearize w.r.t. 'sliceStrides'.
    SmallVector<int64_t, 4> vectorOffsets(rank);
    int64_t linearIndex = i;
    for (unsigned j = 0; j < rank; ++j) {
      vectorOffsets[j] = linearIndex / sliceStrides[j];
      linearIndex %= sliceStrides[j];
    }
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = mlir::functional::zipMap(
        [](int64_t v1, int64_t v2) { return v1 * v2; }, vectorOffsets, sizes);
    // Initialize 'sliceSizes' to target 'sizes'
    SmallVector<int64_t, 4> sliceSizes(sizes.begin(), sizes.end());
    for (unsigned j = 0; j < rank; ++j) {
      // Based on 'offsets' and 'shape' clip some dim sizes for partial tiles.
      sliceSizes[j] = std::min(sliceSizes[j], shape[j] - offsets[j]);
    }
    // Create slice VectorType type.
    auto sliceVectorType =
        VectorType::get(sliceSizes, vectorType.getElementType());
    // Verify that 'sliceVectorType' matches tupleType.getTypes(i)
    if (sliceVectorType != tupleType.getType(i))
      return op->emitError("invalid tuple element type ") << sliceVectorType;
  }
  return success();
}

static LogicalResult verify(ExtractSlicesOp op) {
  SmallVector<int64_t, 4> sizes;
  op.getSizes(sizes);
  SmallVector<int64_t, 4> strides;
  op.getStrides(strides);
  return isValidExtractOrInsertSlicesType(
      op.getOperation(), op.getSourceVectorType(), op.getResultTupleType(),
      sizes, strides);
}

static void populateFromInt64AttrArray(ArrayAttr arrayAttr,
                                       SmallVectorImpl<int64_t> &results) {
  for (auto attr : arrayAttr)
    results.push_back(attr.cast<IntegerAttr>().getInt());
}

void ExtractSlicesOp::getSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(sizes(), results);
}

void ExtractSlicesOp::getStrides(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(strides(), results);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, BroadcastOp op) {
  p << op.getOperationName() << " " << *op.source() << " : "
    << op.getSourceType() << " to " << op.getVectorType();
}

static LogicalResult verify(BroadcastOp op) {
  VectorType srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  VectorType dstVectorType = op.getVectorType();
  // Scalar to vector broadcast is always valid. A vector
  // to vector broadcast needs some additional checking.
  if (srcVectorType) {
    int64_t srcRank = srcVectorType.getRank();
    int64_t dstRank = dstVectorType.getRank();
    if (srcRank > dstRank)
      return op.emitOpError("source rank higher than destination rank");
    // Source has an exact match or singleton value for all trailing dimensions
    // (all leading dimensions are simply duplicated).
    int64_t lead = dstRank - srcRank;
    for (int64_t r = 0; r < srcRank; ++r) {
      int64_t srcDim = srcVectorType.getDimSize(r);
      int64_t dstDim = dstVectorType.getDimSize(lead + r);
      if (srcDim != 1 && srcDim != dstDim)
        return op.emitOpError("dimension mismatch (")
               << srcDim << " vs. " << dstDim << ")";
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
// ShuffleOp
//===----------------------------------------------------------------------===//

void ShuffleOp::build(Builder *builder, OperationState &result, Value *v1,
                      Value *v2, ArrayRef<int32_t> mask) {
  result.addOperands({v1, v2});
  auto maskAttr = getVectorSubscriptAttr(*builder, mask);
  result.addTypes(v1->getType());
  result.addAttribute(getMaskAttrName(), maskAttr);
}

static void print(OpAsmPrinter &p, ShuffleOp op) {
  p << op.getOperationName() << " " << *op.v1() << ", " << *op.v2() << " "
    << op.mask();
  p.printOptionalAttrDict(op.getAttrs(), {ShuffleOp::getMaskAttrName()});
  p << " : " << op.v1()->getType() << ", " << op.v2()->getType();
}

static LogicalResult verify(ShuffleOp op) {
  VectorType resultType = op.getVectorType();
  VectorType v1Type = op.getV1VectorType();
  VectorType v2Type = op.getV2VectorType();
  // Verify ranks.
  int64_t resRank = resultType.getRank();
  int64_t v1Rank = v1Type.getRank();
  int64_t v2Rank = v2Type.getRank();
  if (resRank != v1Rank || v1Rank != v2Rank)
    return op.emitOpError("rank mismatch");
  // Verify all but leading dimension sizes.
  for (int64_t r = 1; r < v1Rank; ++r) {
    int64_t resDim = resultType.getDimSize(r);
    int64_t v1Dim = v1Type.getDimSize(r);
    int64_t v2Dim = v2Type.getDimSize(r);
    if (resDim != v1Dim || v1Dim != v2Dim)
      return op.emitOpError("dimension mismatch");
  }
  // Verify mask length.
  auto maskAttr = op.mask().getValue();
  int64_t maskLength = maskAttr.size();
  if (maskLength != resultType.getDimSize(0))
    return op.emitOpError("mask length mismatch");
  // Verify all indices.
  int64_t indexSize = v1Type.getDimSize(0) + v2Type.getDimSize(0);
  for (auto en : llvm::enumerate(maskAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 || attr.getInt() >= indexSize)
      return op.emitOpError("mask index #")
             << (en.index() + 1) << " out of range";
  }
  return success();
}

static ParseResult parseShuffleOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType v1, v2;
  Attribute attr;
  VectorType v1Type, v2Type;
  if (parser.parseOperand(v1) || parser.parseComma() ||
      parser.parseOperand(v2) ||
      parser.parseAttribute(attr, ShuffleOp::getMaskAttrName(),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(v1Type) || parser.parseComma() ||
      parser.parseType(v2Type) ||
      parser.resolveOperand(v1, v1Type, result.operands) ||
      parser.resolveOperand(v2, v2Type, result.operands))
    return failure();
  // Construct resulting type: leading dimension matches mask length,
  // all trailing dimensions match the operands.
  auto maskAttr = attr.dyn_cast<ArrayAttr>();
  if (!maskAttr)
    return parser.emitError(parser.getNameLoc(), "missing mask attribute");
  int64_t maskLength = maskAttr.size();
  if (maskLength <= 0)
    return parser.emitError(parser.getNameLoc(), "invalid mask length");
  int64_t v1Rank = v1Type.getRank();
  SmallVector<int64_t, 4> shape;
  shape.reserve(v1Rank);
  shape.push_back(maskLength);
  for (int64_t r = 1; r < v1Rank; ++r)
    shape.push_back(v1Type.getDimSize(r));
  VectorType resType = VectorType::get(shape, v1Type.getElementType());
  parser.addTypeToList(resType, result.types);
  return success();
}

//===----------------------------------------------------------------------===//
// InsertElementOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, InsertElementOp op) {
  p << op.getOperationName() << " " << *op.source() << ", " << *op.dest() << "["
    << *op.position() << " : " << op.position()->getType() << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.dest()->getType();
}

static ParseResult parseInsertElementOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType source, dest, position;
  Type positionType;
  VectorType destType;
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(dest) || parser.parseLSquare() ||
      parser.parseOperand(position) || parser.parseColonType(positionType) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(destType))
    return failure();
  Type sourceType = destType.getElementType();
  return failure(
      parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(dest, destType, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands) ||
      parser.addTypeToList(destType, result.types));
}

static LogicalResult verify(InsertElementOp op) {
  auto dstVectorType = op.getDestVectorType();
  if (dstVectorType.getRank() != 1)
    return op.emitOpError("expected 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::build(Builder *builder, OperationState &result, Value *source,
                     Value *dest, ArrayRef<int32_t> position) {
  result.addOperands({source, dest});
  auto positionAttr = getVectorSubscriptAttr(*builder, position);
  result.addTypes(dest->getType());
  result.addAttribute(getPositionAttrName(), positionAttr);
}

static void print(OpAsmPrinter &p, InsertOp op) {
  p << op.getOperationName() << " " << *op.source() << ", " << *op.dest()
    << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {InsertOp::getPositionAttrName()});
  p << " : " << op.getSourceType() << " into " << op.getDestVectorType();
}

static ParseResult parseInsertOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType source, dest;
  Type sourceType;
  VectorType destType;
  Attribute attr;
  return failure(parser.parseOperand(source) || parser.parseComma() ||
                 parser.parseOperand(dest) ||
                 parser.parseAttribute(attr, InsertOp::getPositionAttrName(),
                                       result.attributes) ||
                 parser.parseOptionalAttrDict(attrs) ||
                 parser.parseColonType(sourceType) ||
                 parser.parseKeywordType("into", destType) ||
                 parser.resolveOperand(source, sourceType, result.operands) ||
                 parser.resolveOperand(dest, destType, result.operands) ||
                 parser.addTypeToList(destType, result.types));
}

static LogicalResult verify(InsertOp op) {
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
        attr.getInt() >= destVectorType.getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "dest vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InsertSlicesOp
//===----------------------------------------------------------------------===//

static ParseResult parseInsertSlicesOp(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  ArrayAttr sizesAttr;
  StringRef sizesAttrName = InsertSlicesOp::getSizesAttrName();
  ArrayAttr stridesAttr;
  StringRef stridesAttrName = InsertSlicesOp::getStridesAttrName();
  TupleType tupleType;
  VectorType resultVectorType;
  return failure(
      parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(sizesAttr, sizesAttrName, result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(stridesAttr, stridesAttrName, result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tupleType) ||
      parser.parseKeywordType("into", resultVectorType) ||
      parser.resolveOperand(operandInfo, tupleType, result.operands) ||
      parser.addTypeToList(resultVectorType, result.types));
}

static void print(OpAsmPrinter &p, InsertSlicesOp op) {
  p << op.getOperationName() << ' ' << *op.vectors() << ", ";
  p << op.sizes() << ", " << op.strides();
  p.printOptionalAttrDict(
      op.getAttrs(),
      /*elidedAttrs=*/{InsertSlicesOp::getSizesAttrName(),
                       InsertSlicesOp::getStridesAttrName()});
  p << " : " << op.vectors()->getType();
  p << " into " << op.getResultVectorType();
}

static LogicalResult verify(InsertSlicesOp op) {
  SmallVector<int64_t, 4> sizes;
  op.getSizes(sizes);
  SmallVector<int64_t, 4> strides;
  op.getStrides(strides);
  return isValidExtractOrInsertSlicesType(
      op.getOperation(), op.getResultVectorType(), op.getSourceTupleType(),
      sizes, strides);
}

void InsertSlicesOp::getSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(sizes(), results);
}

void InsertSlicesOp::getStrides(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(strides(), results);
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
  if (!op.acc().empty())
    p << ", " << op.acc();
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
// ReshapeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ReshapeOp op) {
  p << op.getOperationName() << " " << *op.vector() << ", [" << op.input_shape()
    << "], [" << op.output_shape() << "], " << op.fixed_vector_sizes();
  SmallVector<StringRef, 2> elidedAttrs = {
      ReshapeOp::getOperandSegmentSizeAttr(),
      ReshapeOp::getFixedVectorSizesAttrName()};
  p.printOptionalAttrDict(op.getAttrs(), elidedAttrs);
  p << " : " << op.getInputVectorType() << " to " << op.getOutputVectorType();
}

// TODO(b/146516564) Consider passing number of inner vector dimensions that
// are fixed, instead of their values in 'fixesVectorSizes' array attr.
//
// operation ::= ssa-id `=` `vector.reshape` ssa-use, `[` ssa-use-list `]`,
//                          `[` ssa-use-list `]`, `[` array-attribute `]`
//                          `:` vector-type 'to' vector-type
//
static ParseResult parseReshapeOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputInfo;
  SmallVector<OpAsmParser::OperandType, 4> inputShapeInfo;
  SmallVector<OpAsmParser::OperandType, 4> outputShapeInfo;
  ArrayAttr fixedVectorSizesAttr;
  StringRef attrName = ReshapeOp::getFixedVectorSizesAttrName();
  auto indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(inputInfo) || parser.parseComma() ||
      parser.parseOperandList(inputShapeInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() ||
      parser.parseOperandList(outputShapeInfo,
                              OpAsmParser::Delimiter::Square) ||
      parser.parseComma()) {
    return failure();
  }

  auto builder = parser.getBuilder();
  result.addAttribute(
      ReshapeOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({1, static_cast<int32_t>(inputShapeInfo.size()),
                                static_cast<int32_t>(outputShapeInfo.size())}));
  Type inputType;
  Type outputType;
  return failure(
      parser.parseAttribute(fixedVectorSizesAttr, attrName,
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(inputType) ||
      parser.resolveOperand(inputInfo, inputType, result.operands) ||
      parser.resolveOperands(inputShapeInfo, indexType, result.operands) ||
      parser.resolveOperands(outputShapeInfo, indexType, result.operands) ||
      parser.parseKeywordType("to", outputType) ||
      parser.addTypeToList(outputType, result.types));
}

static LogicalResult verify(ReshapeOp op) {
  // Verify that rank(numInputs/outputs) + numFixedVec dim matches vec rank.
  auto inputVectorType = op.getInputVectorType();
  auto outputVectorType = op.getOutputVectorType();
  int64_t inputShapeRank = op.getNumInputShapeSizes();
  int64_t outputShapeRank = op.getNumOutputShapeSizes();
  SmallVector<int64_t, 4> fixedVectorSizes;
  op.getFixedVectorSizes(fixedVectorSizes);
  int64_t numFixedVectorSizes = fixedVectorSizes.size();

  if (inputVectorType.getRank() != inputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid input shape for vector type ")
           << inputVectorType;

  if (outputVectorType.getRank() != outputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid output shape for vector type ")
           << outputVectorType;

  // Verify that the 'fixedVectorSizes' match a input/output vector shape
  // suffix.
  unsigned inputVectorRank = inputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = inputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != inputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match input vector for dim ")
             << i;
  }

  unsigned outputVectorRank = outputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = outputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != outputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match output vector for dim ")
             << i;
  }

  // If all shape operands are produced by constant ops, verify that product
  // of dimensions for input/output shape match.
  auto isDefByConstant = [](Value *operand) {
    return isa_and_nonnull<ConstantIndexOp>(operand->getDefiningOp());
  };
  if (llvm::all_of(op.input_shape(), isDefByConstant) &&
      llvm::all_of(op.output_shape(), isDefByConstant)) {
    int64_t numInputElements = 1;
    for (auto *operand : op.input_shape())
      numInputElements *=
          cast<ConstantIndexOp>(operand->getDefiningOp()).getValue();
    int64_t numOutputElements = 1;
    for (auto *operand : op.output_shape())
      numOutputElements *=
          cast<ConstantIndexOp>(operand->getDefiningOp()).getValue();
    if (numInputElements != numOutputElements)
      return op.emitError("product of input and output shape sizes must match");
  }
  return success();
}

void ReshapeOp::getFixedVectorSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(fixed_vector_sizes(), results);
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

void StridedSliceOp::getOffsets(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(offsets(), results);
}

namespace {

// Pattern to rewrite a StridedSliceOp(ConstantMaskOp) -> ConstantMaskOp.
class StridedSliceConstantMaskFolder final
    : public OpRewritePattern<StridedSliceOp> {
public:
  using OpRewritePattern<StridedSliceOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(StridedSliceOp stridedSliceOp,
                                     PatternRewriter &rewriter) const override {
    // Return if 'stridedSliceOp' operand is not defined by a ConstantMaskOp.
    auto defOp = stridedSliceOp.vector()->getDefiningOp();
    auto constantMaskOp = dyn_cast_or_null<ConstantMaskOp>(defOp);
    if (!constantMaskOp)
      return matchFailure();
    // Return if 'stridedSliceOp' has non-unit strides.
    if (llvm::any_of(stridedSliceOp.strides(), [](Attribute attr) {
          return attr.cast<IntegerAttr>().getInt() != 1;
        }))
      return matchFailure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    populateFromInt64AttrArray(constantMaskOp.mask_dim_sizes(), maskDimSizes);
    // Gather strided slice offsets and sizes.
    SmallVector<int64_t, 4> sliceOffsets;
    populateFromInt64AttrArray(stridedSliceOp.offsets(), sliceOffsets);
    SmallVector<int64_t, 4> sliceSizes;
    populateFromInt64AttrArray(stridedSliceOp.sizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<int64_t, 4> sliceMaskDimSizes;
    assert(sliceOffsets.size() == maskDimSizes.size());
    for (const auto &it : llvm::zip(maskDimSizes, sliceOffsets, sliceSizes)) {
      int64_t maskDimSize = std::get<0>(it);
      int64_t sliceOffset = std::get<1>(it);
      int64_t sliceSize = std::get<2>(it);
      int64_t sliceMaskDimSize = std::max(
          static_cast<int64_t>(0),
          std::min(sliceOffset + sliceSize, maskDimSize) - sliceOffset);
      sliceMaskDimSizes.push_back(sliceMaskDimSize);
    }
    // If any of 'sliceMaskDimSizes' are zero, then set all to zero (masked
    // region is a conjunction of mask dim intervals).
    if (llvm::any_of(sliceMaskDimSizes, [](int64_t sz) { return sz == 0; }))
      sliceMaskDimSizes.assign(maskDimSizes.size(), 0);

    // Replace 'stridedSliceOp' with ConstantMaskOp with sliced mask region.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        stridedSliceOp, stridedSliceOp.getResult()->getType(),
        rewriter.getI64ArrayAttr(sliceMaskDimSizes));
    return matchSuccess();
  }
};

} // end anonymous namespace

void StridedSliceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  // Pattern to rewrite a StridedSliceOp(ConstantMaskOp) -> ConstantMaskOp.
  results.insert<StridedSliceConstantMaskFolder>(context);
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
  p << op.getOperationName() << " " << op.memref() << "[" << op.indices()
    << "], " << op.padding() << " ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getMemRefType() << ", " << op.getVectorType();
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
  p << op.getOperationName() << " " << *op.vector() << ", " << *op.memref()
    << "[" << op.indices() << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getVectorType() << ", " << op.getMemRefType();
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

static void print(OpAsmPrinter &p, TypeCastOp op) {
  auto type = op.getOperand()->getType().cast<MemRefType>();
  p << op.getOperationName() << ' ' << *op.memref() << " : " << type << " to "
    << inferVectorTypeCastResultType(type);
}

static LogicalResult verify(TypeCastOp op) {
  auto resultType = inferVectorTypeCastResultType(op.getMemRefType());
  if (op.getResultMemRefType() != resultType)
    return op.emitOpError("expects result type to be: ") << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

static ParseResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operandInfos;
  SmallVector<Type, 4> types;
  auto loc = parser.getCurrentLocation();
  auto *ctx = parser.getBuilder().getContext();
  return failure(
      parser.parseOperandList(operandInfos) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(operandInfos, types, loc, result.operands) ||
      parser.addTypeToList(TupleType::get(types, ctx), result.types));
}

static void print(OpAsmPrinter &p, TupleOp op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.getOperands());
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  interleaveComma(op.getOperation()->getOperandTypes(), p);
}

static LogicalResult verify(TupleOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TupleGetOp
//===----------------------------------------------------------------------===//

static ParseResult parseTupleGetOp(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  StringRef indexAttrName = TupleGetOp::getIndexAttrName();
  Type indexType = parser.getBuilder().getIndexType();
  TupleType tupleType;
  VectorType resultVectorType;
  if (parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(indexAttr, indexType, indexAttrName,
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tupleType) ||
      parser.resolveOperand(operandInfo, tupleType, result.operands))
    return failure();
  if (indexAttr.getInt() < 0 ||
      indexAttr.getInt() >= static_cast<int64_t>(tupleType.size()))
    return failure();
  parser.addTypeToList(tupleType.getType(indexAttr.getInt()), result.types);
  return success();
}

static void print(OpAsmPrinter &p, TupleGetOp op) {
  p << op.getOperationName() << ' ' << *op.getOperand() << ", " << op.index();
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{TupleGetOp::getIndexAttrName()});
  p << " : " << op.getOperand()->getType();
}

static LogicalResult verify(TupleGetOp op) {
  auto tupleType = op.getOperand()->getType().cast<TupleType>();
  if (op.getIndex() < 0 ||
      op.getIndex() >= static_cast<int64_t>(tupleType.size()))
    return op.emitOpError("tuple get index out of range");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantMaskOp
//===----------------------------------------------------------------------===//

ParseResult parseConstantMaskOp(OpAsmParser &parser, OperationState &result) {
  Type resultType;
  ArrayAttr maskDimSizesAttr;
  StringRef attrName = ConstantMaskOp::getMaskDimSizesAttrName();
  return failure(
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(maskDimSizesAttr, attrName, result.attributes) ||
      parser.parseColonType(resultType) ||
      parser.addTypeToList(resultType, result.types));
}

static void print(OpAsmPrinter &p, ConstantMaskOp op) {
  p << op.getOperationName() << ' ' << op.mask_dim_sizes() << " : "
    << op.getResult()->getType();
}

static LogicalResult verify(ConstantMaskOp &op) {
  // Verify that array attr size matches the rank of the vector result.
  auto resultType = op.getResult()->getType().cast<VectorType>();
  if (static_cast<int64_t>(op.mask_dim_sizes().size()) != resultType.getRank())
    return op.emitOpError(
        "must specify array attr of size equal vector result rank");
  // Verify that each array attr element is in bounds of corresponding vector
  // result dimension size.
  auto resultShape = resultType.getShape();
  SmallVector<int64_t, 4> maskDimSizes;
  for (auto it : llvm::enumerate(op.mask_dim_sizes())) {
    int64_t attrValue = it.value().cast<IntegerAttr>().getInt();
    if (attrValue < 0 || attrValue > resultShape[it.index()])
      return op.emitOpError(
          "array attr of size out of bounds of vector result dimension size");
    maskDimSizes.push_back(attrValue);
  }
  // Verify that if one mask dim size is zero, they all should be zero (because
  // the mask region is a conjunction of each mask dimension interval).
  bool any_zeros = llvm::is_contained(maskDimSizes, 0);
  bool all_zeros = llvm::all_of(maskDimSizes, [](int64_t s) { return s == 0; });
  if (any_zeros && !all_zeros)
    return op.emitOpError("expected all mask dim sizes to be zeros, "
                          "as a result of conjunction with zero mask dim");
  return success();
}

//===----------------------------------------------------------------------===//
// CreateMaskOp
//===----------------------------------------------------------------------===//

ParseResult parseCreateMaskOp(OpAsmParser &parser, OperationState &result) {
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

static void print(OpAsmPrinter &p, CreateMaskOp op) {
  p << op.getOperationName() << ' ' << op.operands() << " : " << op.getType();
}

static LogicalResult verify(CreateMaskOp op) {
  // Verify that an operand was specified for each result vector each dimension.
  if (op.getNumOperands() !=
      op.getResult()->getType().cast<VectorType>().getRank())
    return op.emitOpError(
        "must specify an operand for each result vector dimension");
  return success();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

ParseResult parsePrintOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType source;
  Type sourceType;
  return failure(parser.parseOperand(source) ||
                 parser.parseColonType(sourceType) ||
                 parser.resolveOperand(source, sourceType, result.operands));
}

static void print(OpAsmPrinter &p, PrintOp op) {
  p << op.getOperationName() << ' ' << *op.source() << " : "
    << op.getPrintType();
}

namespace {

// Pattern to rewrite a CreateMaskOp with a ConstantMaskOp.
class CreateMaskFolder final : public OpRewritePattern<CreateMaskOp> {
public:
  using OpRewritePattern<CreateMaskOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CreateMaskOp createMaskOp,
                                     PatternRewriter &rewriter) const override {
    // Return if any of 'createMaskOp' operands are not defined by a constant.
    auto is_not_def_by_constant = [](Value *operand) {
      return !isa_and_nonnull<ConstantIndexOp>(operand->getDefiningOp());
    };
    if (llvm::any_of(createMaskOp.operands(), is_not_def_by_constant))
      return matchFailure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    for (auto *operand : createMaskOp.operands()) {
      auto defOp = operand->getDefiningOp();
      maskDimSizes.push_back(cast<ConstantIndexOp>(defOp).getValue());
    }
    // Replace 'createMaskOp' with ConstantMaskOp.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        createMaskOp, createMaskOp.getResult()->getType(),
        rewriter.getI64ArrayAttr(maskDimSizes));
    return matchSuccess();
  }
};

} // end anonymous namespace

void CreateMaskOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CreateMaskFolder>(context);
}

void mlir::vector::populateVectorToVectorCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<CreateMaskFolder, StridedSliceConstantMaskFolder>(context);
}

namespace mlir {
namespace vector {

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.cpp.inc"

} // namespace vector
} // namespace mlir

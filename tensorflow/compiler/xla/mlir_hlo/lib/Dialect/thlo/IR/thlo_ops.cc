/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Destination-style ops tools
//===----------------------------------------------------------------------===//

LogicalResult verifyDestinationStyleOp(Operation *op) {
  auto dstStyleOp = cast<linalg::DestinationStyleOpInterface>(*op);
  if (dstStyleOp.hasBufferSemantics()) return success(op->getNumResults() == 0);

  if (!dstStyleOp.hasTensorSemantics())
    return op->emitOpError("expected either buffer or tensor semantics");

  return success();
}

template <typename DstOpTy>
void printDstStyleOp(
    DstOpTy op, OpAsmPrinter &p,
    function_ref<SmallVector<StringRef>(DstOpTy op, OpAsmPrinter &)>
        printAttrsFn = nullptr) {
  if (op.getNumInputs() != 0) {
    p << " ins(";
    llvm::interleaveComma(
        op.getOperands().take_front(op.getNumInputs()), p,
        [&](Value input) { p << input << " : " << input.getType(); });
    p << ")";
  }
  p << " outs(";
  llvm::interleaveComma(
      op.getOperands().take_back(op.getNumOutputs()), p,
      [&](Value output) { p << output << " : " << output.getType(); });
  p << ")";

  // Print attributes with custom printing logic.
  SmallVector<StringRef> elidedAttrs;
  if (printAttrsFn) elidedAttrs = printAttrsFn(op, p);

  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

ParseResult parseKeywordOperandListWithTypes(
    OpAsmParser &parser, OperationState &result, StringRef keyword,
    SmallVectorImpl<Type> *operandTypes) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    SMLoc operandsOperandsLoc = parser.getCurrentLocation();

    if (parser.parseCommaSeparatedList(
            AsmParser::Delimiter::Paren, [&]() -> ParseResult {
              if (parser.parseOperand(operands.emplace_back(),
                                      /*allowResultNumber=*/false) ||
                  parser.parseColon() ||
                  parser.parseType(operandTypes->emplace_back())) {
                return failure();
              }
              return success();
            }))
      return failure();

    if (parser.resolveOperands(operands, *operandTypes, operandsOperandsLoc,
                               result.operands))
      return failure();
  }
  return success();
}

ParseResult parseDstStyleOp(
    OpAsmParser &parser, OperationState &result,
    function_ref<ParseResult(OpAsmParser &, NamedAttrList &)> parseAttrsFn =
        nullptr) {
  // Parse `ins` and `outs`.
  SmallVector<Type, 4> inputTypes, outputTypes;
  if (parseKeywordOperandListWithTypes(parser, result, "ins", &inputTypes) ||
      parseKeywordOperandListWithTypes(parser, result, "outs", &outputTypes))
    return failure();

  // Add result types.
  for (Type outputType : outputTypes) {
    if (outputType.isa<RankedTensorType>()) result.addTypes(outputType);
  }

  // Parse required attributes.
  if (parseAttrsFn && failed(parseAttrsFn(parser, result.attributes)))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

ParseResult parseDenseI64ArrayAttr(OpAsmParser &parser,
                                   NamedAttrList &attributes,
                                   StringRef attributeName) {
  if (parser.parseKeyword(attributeName) || parser.parseEqual())
    return failure();

  attributes.set(attributeName, DenseI64ArrayAttr::parse(parser, Type{}));
  return success();
}

void printDenseI64ArrayAttr(OpAsmPrinter &p, StringRef attributeName,
                            ArrayRef<int64_t> attributeValue) {
  p << " " << attributeName << " = [" << attributeValue << "] ";
}

bool dimensionsMatch(int64_t d1, int64_t d2) {
  return ShapedType::isDynamic(d1) || ShapedType::isDynamic(d2) || d1 == d2;
}

SmallVector<utils::IteratorType> getParallelIteratorTypes(int64_t dimCount) {
  return SmallVector<utils::IteratorType>(dimCount,
                                          utils::IteratorType::parallel);
}

SmallVector<Range> getIterationDomainForTensor(OpBuilder &b, Location loc,
                                               Value tensor,
                                               int64_t dimCount = -1) {
  auto dimValues = tensor::createDimValues(b, loc, tensor);
  if (dimCount >= 0) dimValues.resize(dimCount);
  return llvm::to_vector(llvm::map_range(dimValues, [&](Value d) {
    return Range{b.getIndexAttr(0), d, b.getIndexAttr(1)};
  }));
}

Value getMaterializedTile(OpBuilder &b, Location loc,
                          TypedValue<TensorType> tensor,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> dynamicDims =
      tensor::createDynamicDimValues(b, loc, tensor);
  ArrayAttr staticDims = b.getI64ArrayAttr(tensor.getType().getShape());
  Value space = b.create<gml_st::SpaceOp>(loc, dynamicDims, staticDims);

  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  Value tile = b.create<gml_st::TileOp>(loc, space, offsets, sizes, strides);
  return b.create<gml_st::MaterializeOp>(loc, tensor, tile);
}

}  // namespace
}  // namespace mlir

// Generated dialect definitions.
#include "mlir-hlo/Dialect/thlo/IR/thlo_dialect.cc.inc"

namespace mlir {
namespace thlo {

void THLODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult checkYieldOutputs(YieldOp yieldOp,
                                TypeRange expectedElementTypes) {
  uint64_t numOutputs = expectedElementTypes.size();
  if (yieldOp.getValues().size() != numOutputs) {
    return yieldOp.emitOpError("expects number of tensor output args = ")
           << numOutputs << " to match the number of yield operands = "
           << yieldOp.getValues().size();
  }

  for (auto &item : llvm::enumerate(
           llvm::zip(expectedElementTypes, yieldOp.getOperandTypes()))) {
    Type outputElementType, resultType;
    unsigned index = item.index();
    std::tie(outputElementType, resultType) = item.value();
    if (outputElementType != resultType)
      return yieldOp.emitOpError("expects yield operand ")
             << index << " with type = " << resultType
             << " to match output arg element type = " << outputElementType;
  }

  return success();
}

LogicalResult YieldOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {

gml_st::TileOp createTileOp(OpBuilder &b, Location loc, Value tensor,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> sizes) {
  auto initTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<OpFoldResult> unitStrides(initTy.getRank(), b.getIndexAttr(1));
  SmallVector<Value> dynamicSpaceSizes =
      tensor::createDynamicDimValues(b, loc, tensor);
  ArrayAttr staticSpaceSizes = b.getI64ArrayAttr(initTy.getShape());
  auto space =
      b.create<gml_st::SpaceOp>(loc, dynamicSpaceSizes, staticSpaceSizes);
  return b.create<gml_st::TileOp>(loc, space, offsets, sizes, unitStrides);
}

}  // namespace

SmallVector<utils::IteratorType> ConcatenateOp::getLoopIteratorTypes() {
  return getParallelIteratorTypes(getInit().getType().getRank());
}

SmallVector<Value> ConcatenateOp::getDestinationOperands(OpBuilder &) {
  return {getInit()};
}

SmallVector<Range> ConcatenateOp::getIterationDomain(OpBuilder &b) {
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

namespace {

// TODO(frgossen): Fuse this as a switch statement if all the operands are unit
// size in the concatenation dimension.
Value fuseConcatenateOpThroughTile(ConcatenateOp op, OpBuilder &builder,
                                   Location loc, Value tile) {
  uint64_t concatDim = op.getDimension();
  RankedTensorType resultTy = op.getType(0).cast<RankedTensorType>();
  int64_t rank = resultTy.getRank();
  OperandRange allOperands = op.getInputs();
  Value anyOperand = allOperands.front();

  // Create the shared tile strides, which are the exact same for every operand
  // tile. Also create a basis for the space sizes, tile offsets, and tile
  // sizes. These hold the shared values in all non-concat dimensions and can be
  // amended in the concat dimension to create the individual operand tiles.
  SmallVector<Value> sharedTileStrides(rank);
  SmallVector<Value> baseSpaceSizes(rank);
  SmallVector<Value> baseTileOffsets(rank);
  SmallVector<Value> baseTileSizes(rank);
  for (int64_t i = 0; i < rank; ++i) {
    Value iCst = builder.create<arith::ConstantIndexOp>(loc, i);
    sharedTileStrides[i] = builder.create<gml_st::StrideOp>(loc, tile, iCst);

    // The space sizes, tile offsets, and tile sizes differ in the concat
    // dimension. Do not populate these.
    if (i == static_cast<int64_t>(concatDim)) {
      continue;
    }

    baseSpaceSizes[i] =
        builder.createOrFold<tensor::DimOp>(loc, anyOperand, iCst);
    baseTileOffsets[i] = builder.create<gml_st::OffsetOp>(loc, tile, iCst);
    baseTileSizes[i] = builder.create<gml_st::SizeOp>(loc, tile, iCst);
  }

  // Some shared values.
  ArrayAttr allDynamicStridesOrOffsetsAttr = builder.getI64ArrayAttr(
      SmallVector<int64_t>(rank, ShapedType::kDynamicStrideOrOffset));
  ArrayAttr allDynamicSizesAttr = builder.getI64ArrayAttr(
      SmallVector<int64_t>(rank, ShapedType::kDynamicSize));
  Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value concatDimCst = builder.create<arith::ConstantIndexOp>(loc, concatDim);
  Value maxTileSizeInConcatDim =
      builder.create<gml_st::SizeOp>(loc, tile, concatDimCst);

  // The remaining tile offset in the concat dimension is subtracted by each
  // operand's size in that dimension. We maintain the invariant
  // remainingTileOffsetInConcatDim >= 0.
  Value remainingTileOffsetInConcatDim =
      builder.create<gml_st::OffsetOp>(loc, tile, concatDimCst);

  // Create the relevant subsets per operand. These tiles can be empty at
  // runtime.
  SmallVector<Value> subOperands;
  subOperands.reserve(allOperands.size());
  for (Value operand : allOperands) {
    // Create operand space.
    Value operandSizeInConcatDim =
        builder.create<tensor::DimOp>(loc, operand, concatDimCst);
    baseSpaceSizes[concatDim] = operandSizeInConcatDim;
    Value operandSpace = builder.create<gml_st::SpaceOp>(loc, baseSpaceSizes,
                                                         allDynamicSizesAttr);

    // Find the current operand's tile offset in the concat dimension. This is
    // the remaining offset clamped into the bounds of the operand. Note that
    // the remaining offset is always >= 0.
    Value operandTileOffsetInConcatDim = builder.create<arith::MinUIOp>(
        loc, remainingTileOffsetInConcatDim, operandSizeInConcatDim);
    baseTileOffsets[concatDim] = operandTileOffsetInConcatDim;

    // Find the current operand's tile size in the concat dimension.
    Value remainingOperandSizeInConcatDim = builder.create<arith::SubIOp>(
        loc, operandSizeInConcatDim, operandTileOffsetInConcatDim);
    baseTileSizes[concatDim] = builder.create<arith::MinUIOp>(
        loc, remainingOperandSizeInConcatDim, maxTileSizeInConcatDim);

    // Create the operand tile and materialize the subset for this operand.
    Value tile = builder.create<gml_st::TileOp>(
        loc, operandSpace, baseTileOffsets, baseTileSizes, sharedTileStrides,
        allDynamicStridesOrOffsetsAttr, allDynamicSizesAttr,
        allDynamicStridesOrOffsetsAttr);
    subOperands.push_back(
        builder.create<gml_st::MaterializeOp>(loc, operand, tile));

    // Unless it is the last operand, update the remaining tile offset in the
    // concat dimension. The remaining offset is subtracted by the operand's
    // size but must remain >= 0.
    if (operand != allOperands.back()) {
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                                remainingTileOffsetInConcatDim,
                                                operandSizeInConcatDim);
      Value sub = builder.create<arith::SubIOp>(
          loc, remainingTileOffsetInConcatDim, operandSizeInConcatDim);
      remainingTileOffsetInConcatDim =
          builder.create<arith::SelectOp>(loc, cmp, zeroCst, sub);
    }
  }

  // Create the tiled concat op.
  auto tileType = tile.getType().cast<gml_st::TileType>();
  Value subInit =
      builder.create<gml_st::MaterializeOp>(loc, op.getInit(), tile);
  auto subResultType =
      RankedTensorType::get(tileType.getShape(), resultTy.getElementType());
  return builder
      .create<thlo::ConcatenateOp>(loc, subResultType, subOperands, subInit,
                                   concatDim)
      ->getResult(0);
}

Value fuseConcatenateOpThroughPointRecursively(
    OpBuilder &builder, Location loc, RankedTensorType rankedTy,
    uint64_t concatDim, SmallVector<Value> &remainingOffsets,
    ValueRange remainingOperands) {
  // Bail if called for no operands.
  if (remainingOperands.empty()) {
    return {};
  }
  Value leadingOperand = remainingOperands.front();

  // Terminal case of exactly one operand.
  if (remainingOperands.size() == 1) {
    // Create operand space.
    SmallVector<Value> dynamicDims =
        tensor::createDynamicDimValues(builder, loc, leadingOperand);
    ArrayAttr staticDims = builder.getI64ArrayAttr(rankedTy.getShape());
    Value operandSpace =
        builder.create<gml_st::SpaceOp>(loc, dynamicDims, staticDims);

    // Create operand point.
    SmallVector<int64_t> allDynamicOffsets(rankedTy.getRank(),
                                           ShapedType::kDynamicStrideOrOffset);

    auto sizeOrStride = builder.getI64ArrayAttr({1});
    Value operandPoint = builder.create<gml_st::TileOp>(
        loc, operandSpace, remainingOffsets, ValueRange{}, ValueRange{},
        builder.getI64ArrayAttr(allDynamicOffsets), sizeOrStride, sizeOrStride);

    return builder.create<gml_st::MaterializeOp>(loc, rankedTy.getElementType(),
                                                 leadingOperand, operandPoint);
  }

  // For more than 1 operand, distinguish between the leading operand and the
  // remainder.
  assert(remainingOperands.size() > 1 &&
         "expect more than 1 operand at this point");
  Value leadingOperandConcatDim =
      builder.create<tensor::DimOp>(loc, leadingOperand, concatDim);
  Value leadingOperandPredicate = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, remainingOffsets[concatDim],
      leadingOperandConcatDim);
  auto ifOp = builder.create<scf::IfOp>(
      loc, rankedTy.getElementType(), leadingOperandPredicate,
      [&](OpBuilder &builder, Location loc) {
        // For the leading operand, recur with the current offsets.
        Value fused = fuseConcatenateOpThroughPointRecursively(
            builder, loc, rankedTy, concatDim, remainingOffsets,
            leadingOperand);
        builder.create<scf::YieldOp>(loc, fused);
      },
      [&](OpBuilder &builder, Location loc) {
        // For the remaining operands, substract the leading operand's size from
        // the remaining offsets in the concatenation dimension.
        SmallVector<Value> thenRemainingOffsets(remainingOffsets.begin(),
                                                remainingOffsets.end());
        thenRemainingOffsets[concatDim] = builder.create<arith::SubIOp>(
            loc, remainingOffsets[concatDim], leadingOperandConcatDim);
        Value fused = fuseConcatenateOpThroughPointRecursively(
            builder, loc, rankedTy, concatDim, thenRemainingOffsets,
            remainingOperands.drop_front());
        builder.create<scf::YieldOp>(loc, fused);
      });
  return ifOp.getResults().front();
}

Value fuseConcatenateOpThroughPoint(ConcatenateOp op, OpBuilder &builder,
                                    Location loc, Value subset) {
  auto resultTy = op.getType(0).cast<RankedTensorType>();
  int64_t resultRank = resultTy.getRank();
  uint64_t concatDim = op.getDimension();

  // Materialize initial offsets.
  SmallVector<Value> initialOffsets;
  initialOffsets.reserve(resultRank);
  for (int64_t i = 0; i < resultRank; ++i) {
    initialOffsets.push_back(builder.create<gml_st::OffsetOp>(
        loc, subset, builder.create<arith::ConstantIndexOp>(loc, i)));
  }

  ValueRange initialOperands = op.getInputs();
  return fuseConcatenateOpThroughPointRecursively(
      builder, loc, resultTy, concatDim, initialOffsets, initialOperands);
}

}  // namespace

gml_st::TilingInterface ConcatenateOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // Create tile subset.
  auto loc = getLoc();
  gml_st::TileOp tile = createTileOp(b, loc, getInit(), offsets, sizes);

  auto tiled = fuseConcatenateOpThroughTile(*this, b, loc, tile);
  return llvm::cast<gml_st::TilingInterface>(tiled.getDefiningOp());
}

FailureOr<Value> ConcatenateOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  return getTiledImplementation(b, offsets, sizes)->getResults().front();
}

ParseResult ConcatenateOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void ConcatenateOp::print(OpAsmPrinter &p) { printDstStyleOp(*this, p); }

LogicalResult ConcatenateOp::verify() {
  Type outputElementType =
      getOutputs().front().getType().cast<ShapedType>().getElementType();

  for (Type inputArgType : TypeRange{getInputs()}) {
    Type inputArgElementType = inputArgType.cast<ShapedType>().getElementType();
    if (inputArgElementType != outputElementType) {
      return emitOpError() << "expected element type of input "
                           << inputArgElementType
                           << " to match output element type "
                           << outputElementType;
    }
  }

  return verifyDestinationStyleOp(getOperation());
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

ParseResult DynamicBroadcastInDimOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseDstStyleOp(parser, result,
                         [&](OpAsmParser &parser, NamedAttrList &attributes) {
                           return parseDenseI64ArrayAttr(
                               parser, attributes, "broadcast_dimensions");
                         });
}

void DynamicBroadcastInDimOp::print(OpAsmPrinter &p) {
  printDstStyleOp<DynamicBroadcastInDimOp>(
      *this, p,
      [](DynamicBroadcastInDimOp op,
         OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.getBroadcastDimensionsAttrName(),
                               op.getBroadcastDimensions());
        return {op.getBroadcastDimensionsAttrName()};
      });
}

LogicalResult DynamicBroadcastInDimOp::verify() {
  return verifyDestinationStyleOp(getOperation());
}

SmallVector<utils::IteratorType>
DynamicBroadcastInDimOp::getLoopIteratorTypes() {
  return getParallelIteratorTypes(getInit().getType().getRank());
}

SmallVector<Value> DynamicBroadcastInDimOp::getDestinationOperands(
    OpBuilder &) {
  return {getInit()};
}

SmallVector<Range> DynamicBroadcastInDimOp::getIterationDomain(OpBuilder &b) {
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

gml_st::TilingInterface DynamicBroadcastInDimOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // Create tile subset.
  auto loc = getLoc();
  auto tile = createTileOp(b, loc, getInit(), offsets, sizes);
  auto initRank = getInit().getType().cast<RankedTensorType>().getRank();

  // Create the needed constants only once.
  DenseMap<uint64_t, Value> localIndexConstants;
  auto getIndexConstant = [&](uint64_t c) -> Value {
    auto it = localIndexConstants.find(c);
    if (it != localIndexConstants.end()) return it->second;
    auto cst = b.create<arith::ConstantIndexOp>(loc, c);
    localIndexConstants[c] = cst;
    return cst;
  };

  DenseSet<int64_t> dimensionsThatStay(getBroadcastDimensions().begin(),
                                       getBroadcastDimensions().end());

  // Materialize operand space.
  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  auto operandSpaceTy = b.getType<gml_st::TileType>(operandTy.getShape());
  auto dynamicDims = tensor::createDynamicDimValues(b, loc, getOperand());
  auto staticDims = b.getI64ArrayAttr(operandTy.getShape());
  Value operandSpace =
      b.create<gml_st::SpaceOp>(loc, operandSpaceTy, dynamicDims, staticDims);

  // Materialize operand dimensions.
  SmallVector<Value> operandDims;
  int64_t dynamicDimsIdx = 0;
  operandDims.reserve(operandTy.getRank());
  for (const auto &it : llvm::enumerate(operandTy.getShape())) {
    int64_t d = it.value();
    Value dim = d == ShapedType::kDynamicSize ? dynamicDims[dynamicDimsIdx++]
                                              : getIndexConstant(d);
    operandDims.push_back(dim);
  }

  // Find the expanding dimensions. If corresponding operand and result
  // dimensions are different then the dimension is expanding.
  // TODO(frgossen): Use info from known expanding and known non-expanding
  // dimensions here.
  SmallVector<Value> operandExpandingDims;
  for (const auto &it : llvm::enumerate(getBroadcastDimensions())) {
    auto operandDim = operandDims[it.index()];
    auto resultDim =
        b.create<tensor::DimOp>(loc, getInit(), getIndexConstant(it.value()));
    operandExpandingDims.push_back(b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, operandDim, resultDim));
  }

  // Compute operand tile offsets.
  int64_t operandRank = operandTy.getRank();
  auto staticOffsets = b.getI64ArrayAttr(
      SmallVector<int64_t>(operandRank, ShapedType::kDynamicStrideOrOffset));
  SmallVector<Value> operandOffsets;
  Value zero = getIndexConstant(0);
  for (int initId = 0, operandId = 0; initId < initRank; ++initId) {
    if (!dimensionsThatStay.contains(initId)) continue;
    Value isExpanding = operandExpandingDims[operandId++];
    Value collapsedSubsetOffset =
        b.create<gml_st::OffsetOp>(loc, tile, getIndexConstant(initId));
    operandOffsets.push_back(b.create<arith::SelectOp>(loc, isExpanding, zero,
                                                       collapsedSubsetOffset));
  }

  // Compute operand tile sizes.
  auto staticTileSizes = b.getI64ArrayAttr(
      SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
  SmallVector<Value> tileSizes;
  Value one = getIndexConstant(1);
  for (int initId = 0, operandId = 0; initId < initRank; ++initId) {
    if (!dimensionsThatStay.contains(initId)) continue;
    Value isExpanding = operandExpandingDims[operandId++];
    Value tileSize =
        b.create<gml_st::SizeOp>(loc, tile, getIndexConstant(initId));
    tileSizes.push_back(
        b.create<arith::SelectOp>(loc, isExpanding, one, tileSize));
  }

  // Create operand tile.
  auto staticTileStrides =
      b.getI64ArrayAttr(SmallVector<int64_t>(operandRank, 1));
  SmallVector<Value> tileStrides = {};
  auto operandTileTy = b.getType<gml_st::TileType>(
      SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
  auto operandTile = b.create<gml_st::TileOp>(
      loc, operandTileTy, operandSpace, operandOffsets, tileSizes, tileStrides,
      staticOffsets, staticTileSizes, staticTileStrides);

  // Materialize operand tiles.
  Value tiledInit = b.create<gml_st::MaterializeOp>(loc, getInit(), tile);
  Value tiledOperand =
      b.create<gml_st::MaterializeOp>(loc, getOperand(), operandTile);

  // Finally, materialize tiled broadcast.
  auto tileTy = tile.getType();
  auto resultTy = getType(0).cast<RankedTensorType>();
  auto tiledResultTy =
      RankedTensorType::get(tileTy.getShape(), resultTy.getElementType());
  return b.create<DynamicBroadcastInDimOp>(
      loc, TypeRange{tiledResultTy}, tiledOperand, tiledInit,
      getBroadcastDimensionsAttr(), getKnownExpandingDimensionsAttr(),
      getKnownNonexpandingDimensionsAttr());
}

FailureOr<Value> DynamicBroadcastInDimOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  return getTiledImplementation(b, offsets, sizes)->getResults().front();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

ParseResult ScatterOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(parser, result)) return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  return success();
}

void ScatterOp::print(OpAsmPrinter &p) {
  printDstStyleOp<ScatterOp>(*this, p);

  p << "(";
  llvm::interleaveComma(getUpdateComputation().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getUpdateComputation(), /*printEntryBlockArgs=*/false);
}

LogicalResult ScatterOp::verify() {
  if (failed(verifyDestinationStyleOp(getOperation()))) return failure();

  auto indicesType = getIndices().getType().cast<ShapedType>();
  int64_t indicesRank = indicesType.getRank();

  if (indicesRank != 2)
    return emitOpError() << "expected `indices` to be a 2D tensor";

  auto updatesType = getUpdates().getType();
  int64_t updatesRank = updatesType.getRank();

  if (updatesType.getDimSize(0) != indicesType.getDimSize(0)) {
    return emitOpError() << "expected major dimension of `indices` to match "
                            "major dimension of `updates`";
  }

  int64_t indexVectorDim = indicesType.getDimSize(1);
  if (ShapedType::isDynamic(indexVectorDim))
    return emitOpError() << "expected index vector dimension size to be static";

  auto initType = getInit().getType();
  int64_t initRank = initType.getRank();

  if (indexVectorDim > initRank) {
    return emitOpError() << "expected index vector dimension size = "
                         << indexVectorDim
                         << " to be smaller or equal than `init` rank = "
                         << initRank;
  }

  if (updatesRank - 1 != initRank)
    return emitOpError() << "expected `updates` rank + 1 to match `init` rank";

  if (updatesType.getElementType() != initType.getElementType()) {
    return emitOpError()
           << "expected `updates` element type to match `init` element type";
  }

  // The update computation should yield exactly 1 result.
  auto updateTerminator = cast<YieldOp>(getBody()->getTerminator());
  Type outputElementType =
      getOutputs().front().getType().cast<ShapedType>().getElementType();
  if (!succeeded(checkYieldOutputs(updateTerminator, outputElementType)))
    return failure();

  return success();
}

SmallVector<utils::IteratorType> ScatterOp::getLoopIteratorTypes() {
  return {utils::IteratorType::reduction};
}

SmallVector<Value> ScatterOp::getDestinationOperands(OpBuilder &) {
  return {getInit()};
}

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &b) {
  Value indicesCount = b.create<tensor::DimOp>(getLoc(), getIndices(), 0);
  return {Range{b.getIndexAttr(0), indicesCount, b.getIndexAttr(1)}};
}

static Value getSlice(OpBuilder &b, Location loc, Value tensor,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  SmallVector<Value> dynSizes = tensor::createDynamicDimValues(b, loc, tensor);
  auto staticSizes = b.getI64ArrayAttr(tensorType.getShape());

  Value space = b.create<gml_st::SpaceOp>(loc, dynSizes, staticSizes);
  if (sizes.empty()) return b.create<gml_st::MaterializeOp>(loc, tensor, space);

  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  Value tile = b.create<gml_st::TileOp>(loc, space, offsets, sizes, strides);
  return b.create<gml_st::MaterializeOp>(loc, tensor, tile);
}

static Value getFullSpace(OpBuilder &b, Location loc, Value tensor) {
  return getSlice(b, loc, tensor, llvm::None, llvm::None);
}

mlir::gml_st::TilingInterface ScatterOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  IntegerAttr zeroAttr = b.getIndexAttr(0);

  OpFoldResult tileOffset = offsets.front();
  OpFoldResult tileSize = sizes.front();

  // Tile outer dimension of updates.
  Value update = this->getUpdates();
  auto updateType = update.getType().cast<RankedTensorType>();

  SmallVector<OpFoldResult> updateOffsets(updateType.getRank(), zeroAttr);
  updateOffsets.front() = tileOffset;
  SmallVector<OpFoldResult> updateSizes = tensor::getMixedSizes(b, loc, update);
  updateSizes.front() = tileSize;

  Value updateSlice = getSlice(b, loc, update, updateOffsets, updateSizes);

  // Tile outer dimension of indices.
  Value indices = this->getIndices();

  SmallVector<OpFoldResult> indicesOffsets{offsets.front(), zeroAttr};
  indicesOffsets.front() = tileOffset;
  SmallVector<OpFoldResult> indicesSizes =
      tensor::getMixedSizes(b, loc, indices);
  indicesSizes.front() = tileSize;

  Value indicesSlice = getSlice(b, loc, indices, indicesOffsets, indicesSizes);

  // Get full space of the `init` tensor.
  Value init = this->getInit();
  Value initSlice = getFullSpace(b, loc, init);

  auto dpsInterface =
      cast<linalg::DestinationStyleOpInterface>(this->getOperation());
  return dpsInterface.clone(b, loc, TypeRange{initSlice.getType()},
                            ValueRange{indicesSlice, updateSlice, initSlice});
}

FailureOr<Value> ScatterOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "variadic scatter is not implemented");
  return getTiledImplementation(b, offsets, sizes)->getResult(0);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

ParseResult GatherOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void GatherOp::print(OpAsmPrinter &p) { printDstStyleOp(*this, p); }

LogicalResult GatherOp::verify() {
  return verifyDestinationStyleOp(getOperation());
}

SmallVector<utils::IteratorType> GatherOp::getLoopIteratorTypes() {
  // Currently, `offset_dims` is empty, so the iteration domain is just the
  // entire output.
  return getParallelIteratorTypes(getInit().getType().getRank());
}

SmallVector<Value> GatherOp::getDestinationOperands(OpBuilder &) {
  return {getInit()};
}

SmallVector<Range> GatherOp::getIterationDomain(OpBuilder &b) {
  // Currently, `offset_dims` is empty, so the iteration domain is just the
  // entire output.
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

mlir::gml_st::TilingInterface GatherOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto offsetsWithVectorDim = offsets.vec();
  auto sizesWithVectorDim = sizes.vec();

  offsetsWithVectorDim.emplace_back(b.getIndexAttr(0));
  sizesWithVectorDim.emplace_back(
      b.getIndexAttr(getStartIndices().getType().getShape().back()));

  llvm::SmallVector<OpFoldResult> strides(offsets.size() + 1,
                                          b.getIndexAttr(1));

  auto subStartIndices = b.create<tensor::ExtractSliceOp>(
      getLoc(), getStartIndices(), offsetsWithVectorDim, sizesWithVectorDim,
      strides);
  Value initSlice = getMaterializedTile(b, getLoc(), getInit(), offsets, sizes);

  return b
      .create<GatherOp>(getLoc(), TypeRange{initSlice.getType()},
                        ValueRange{getOperand(), subStartIndices, initSlice})
      .getOperation();
}

FailureOr<Value> GatherOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "resultNumber > 0 not implemented");
  return getTiledImplementation(b, offsets, sizes)->getResult(0);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                   mlir::ArrayRef<mlir::NamedAttribute>)>
TransposeOp::getRegionBuilder() {
  return [](mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
            mlir::ArrayRef<mlir::NamedAttribute>) {
    b.create<mlir::thlo::YieldOp>(block.getArguments().back());
  };
}

void TransposeOp::createRegion(::mlir::OpBuilder &opBuilder,
                               ::mlir::OperationState &odsState) {
  Region *region = odsState.addRegion();

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (auto t : odsState.operands) {
    argTypes.push_back(getElementTypeOrSelf(t));
    argLocs.push_back(opBuilder.getUnknownLoc());
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(region, /*insertPt=*/{}, argTypes, argLocs);

  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  getRegionBuilder()(b, *body, odsState.attributes.getAttrs());
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, Type resultType,
                        Value input, Value init, DenseI64ArrayAttr permutation,
                        ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(input);
  odsState.addOperands(init);
  odsState.addAttribute(getPermutationAttrName(odsState.name), permutation);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultType);

  createRegion(odsBuilder, odsState);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, Type resultType,
                        Value input, Value init, ArrayRef<int64_t> permutation,
                        ArrayRef<NamedAttribute> attributes) {
  build(odsBuilder, odsState, resultType, input, init,
        odsBuilder.getDenseI64ArrayAttr(permutation), attributes);
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "permutation");
          })))
    return failure();

  OpBuilder opBuilder(parser.getContext());
  createRegion(opBuilder, result);
  return success();
}

void TransposeOp::print(OpAsmPrinter &p) {
  printDstStyleOp<TransposeOp>(
      *this, p, [](TransposeOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.getPermutationAttrName(),
                               op.getPermutation());
        return {op.getPermutationAttrName()};
      });
}

bool isValidPermutation(ArrayRef<int64_t> permutation) {
  SmallVector<bool> seen(permutation.size(), false);
  for (auto p : permutation) {
    // Verify that each element is in [0..n-1] range and is present only once.
    if (p < 0 || p >= static_cast<int64_t>(permutation.size()) || seen[p])
      return false;

    seen[p] = true;
  }
  return true;
}

LogicalResult TransposeOp::verify() {
  ArrayRef<int64_t> permutationRef = getPermutation();

  if (!isValidPermutation(permutationRef))
    return emitOpError("permutation is not valid");

  auto inputType = getInput().getType();
  auto initType = getInit().getType();

  int64_t rank = inputType.getRank();

  if (rank != initType.getRank())
    return emitOpError() << "input rank " << rank
                         << " does not match init rank " << initType.getRank();

  if (rank != static_cast<int64_t>(permutationRef.size()))
    return emitOpError() << "size of permutation " << permutationRef.size()
                         << " does not match the argument rank " << rank;

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  for (int64_t i = 0; i < rank; ++i) {
    int64_t inputDim = inputDims[permutationRef[i]];
    int64_t initDim = initDims[i];

    if (!dimensionsMatch(inputDim, initDim)) {
      return emitOpError() << "dim(result, " << i << ") = " << initDim
                           << " doesn't match dim(input, permutation[" << i
                           << "]) = " << inputDim;
    }
  }

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<StringRef> TransposeOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<StringRef>(rank, getParallelIteratorTypeName());
}

ArrayAttr TransposeOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  return builder.getAffineMapArrayAttr(
      {builder.getMultiDimIdentityMap(rank),
       AffineMap::getPermutationMap(
           llvm::to_vector_of<unsigned>(getPermutation()), getContext())});
}

bool TransposeOp::hasIndexSemantics() { return false; }

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

ParseResult ReductionOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            return parseDenseI64ArrayAttr(parser, attributes, "dimensions");
          }))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  return success();
}

void ReductionOp::print(OpAsmPrinter &p) {
  printDstStyleOp<ReductionOp>(
      *this, p, [](ReductionOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.getDimensionsAttrName(),
                               op.getDimensions());
        return {op.getDimensionsAttrName()};
      });

  p << "(";
  llvm::interleaveComma(getCombiner().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getCombiner(), /*printEntryBlockArgs=*/false);
}

LogicalResult ReductionOp::verify() {
  ArrayRef<int64_t> dimensionsRef = getDimensions();

  for (int64_t i = 1; i < getNumInputs(); ++i) {
    if (failed(mlir::verifyCompatibleShape(getInputs()[i].getType(),
                                           getInputs()[0].getType()))) {
      return emitOpError() << "expects all inputs to have compatible shapes. "
                              "Shape at input-index "
                           << i
                           << " is not compatible with shape at input-index 0.";
    }
  }
  for (int64_t i = 1; i < getNumOutputs(); ++i) {
    if (failed(mlir::verifyCompatibleShape(getInits()[i].getType(),
                                           getInits()[0].getType()))) {
      return emitOpError()
             << "expects all outputs to have compatible shapes. "
                "Shape at output-index "
             << i << " is not compatible with shape at output-index 0.";
    }
  }
  auto inputType = getInputs()[0].getType().cast<ShapedType>();
  auto initType = getInits()[0].getType().cast<ShapedType>();

  DenseSet<int64_t> dimensionsToReduce;
  int64_t lastDimension = -1;
  for (int64_t dimension : dimensionsRef) {
    if (dimension < 0 || dimension >= inputType.getRank()) {
      return emitOpError()
             << "dimensions for reduction should be in the range [0, "
             << inputType.getRank() - 1 << "].";
    }
    if (dimension <= lastDimension) {
      return emitOpError()
             << "reduction dimensions are not in increasing order: "
             << dimensionsRef;
    }

    lastDimension = dimension;
    dimensionsToReduce.insert(dimension);
  }

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  // Input dimensions that will be left after the reduction.
  SmallVector<int64_t> reducedInputDims;
  for (const auto &en : llvm::enumerate(inputDims)) {
    if (!dimensionsToReduce.count(en.index()))
      reducedInputDims.push_back(en.value());
  }

  if (reducedInputDims.size() != initType.getRank()) {
    return emitOpError() << "number of dimensions after reduction "
                         << reducedInputDims.size()
                         << " doesn't match the init rank "
                         << initType.getRank();
  }

  if (!all_of_zip(reducedInputDims, initDims, &dimensionsMatch))
    return emitOpError() << "init dimensions [" << initDims
                         << "] doesn't match input dimensions after reduction ["
                         << reducedInputDims << "]";

  Block *block = getBody();
  if (static_cast<int64_t>(block->getArguments().size()) !=
      getNumInputs() + getNumOutputs()) {
    return emitOpError()
           << "number of block arguments " << block->getArguments().size()
           << " doesn't match the number of inputs plus the number of outputs "
           << getNumInputs() + getNumOutputs();
  }

  // Check that the first block arguments match the element type of the inputs.
  auto inputElementTypes =
      llvm::to_vector<8>(llvm::map_range(getInputs().getTypes(), [](Type type) {
        return type.cast<ShapedType>().getElementType();
      }));
  auto blockArgumentInputTypes = llvm::to_vector<8>(
      llvm::map_range(block->getArguments().take_front(getNumInputs()),
                      [](BlockArgument arg) { return arg.getType(); }));
  if (blockArgumentInputTypes != inputElementTypes) {
    return emitOpError() << "input element types " << inputElementTypes
                         << " do not match block argument types "
                         << blockArgumentInputTypes;
  }

  // Check that the last block arguments match the element type of the outputs.
  auto outputElementTypes =
      llvm::to_vector<8>(llvm::map_range(getInits().getTypes(), [](Type type) {
        return type.cast<ShapedType>().getElementType();
      }));
  auto blockArgumentOutputTypes = llvm::to_vector<8>(
      llvm::map_range(block->getArguments().take_back(getNumOutputs()),
                      [](BlockArgument arg) { return arg.getType(); }));
  if (blockArgumentOutputTypes != outputElementTypes) {
    return emitOpError() << "output element types " << outputElementTypes
                         << " do not match block argument types "
                         << blockArgumentOutputTypes;
  }

  // The reducer should yield exactly getNumOutputs() outputs.
  YieldOp blockTerminator = cast<YieldOp>(block->getTerminator());
  if (!succeeded(checkYieldOutputs(blockTerminator, outputElementTypes)))
    return failure();

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<StringRef> ReductionOp::getIteratorTypesArray() {
  int64_t inputRank = getInputs()[0].getType().cast<ShapedType>().getRank();
  SmallVector<StringRef> iteratorTypes(inputRank,
                                       getParallelIteratorTypeName());
  for (int64_t reductionDim : getDimensions())
    iteratorTypes[reductionDim] = getReductionIteratorTypeName();
  return iteratorTypes;
}

ArrayAttr ReductionOp::getIndexingMaps() {
  SmallVector<AffineMap> affineMaps;
  int64_t inputRank = getInputs()[0].getType().cast<ShapedType>().getRank();
  for (int64_t i = 0, e = getNumInputs(); i < e; ++i) {
    affineMaps.push_back(
        AffineMap::getMultiDimIdentityMap(inputRank, getContext()));
  }
  SmallVector<AffineExpr, 4> exprs;
  ArrayRef<int64_t> dimensionsRef = getDimensions();
  for (int64_t i = 0, j = 0; i < inputRank; ++i) {
    bool isReductionDim = j < dimensionsRef.size() && dimensionsRef[j] == i;
    if (isReductionDim) {
      ++j;
    } else {
      exprs.push_back(getAffineDimExpr(i, getContext()));
    }
  }
  for (int64_t i = 0, e = getNumOutputs(); i < e; ++i) {
    affineMaps.push_back(
        AffineMap::get(inputRank, /*symbolCount=*/0, exprs, getContext()));
  }
  return Builder(getContext()).getAffineMapArrayAttr(affineMaps);
}

bool ReductionOp::hasIndexSemantics() { return false; }

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(parser, result)) return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  return success();
}

void MapOp::print(OpAsmPrinter &p) {
  printDstStyleOp<MapOp>(*this, p);

  p << "(";
  llvm::interleaveComma(getMapper().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getMapper(), /*printEntryBlockArgs=*/false);
}

LogicalResult MapOp::verify() {
  auto *bodyBlock = getBody();
  auto blockArgs = bodyBlock->getArguments();

  // Checks if the number of `inputs` match the arity of the `mapper` region.
  if (getInputs().size() != blockArgs.size())
    return emitOpError() << "expects number of operands to match the arity of "
                            "mapper, but got: "
                         << getInputs().size() << " and " << blockArgs.size();

  // The parameters of mapper should all match the element type // of inputs.
  for (const auto &[bbArgType, inputArg] :
       llvm::zip(bodyBlock->getArgumentTypes(), getInputs())) {
    auto inputElemType = inputArg.getType().cast<ShapedType>().getElementType();
    if (bbArgType != inputElemType) {
      return emitOpError() << "expected element type of input " << inputElemType
                           << " to match bbArg type " << bbArgType;
    }
  }

  // The shape of each input must match the shape of the output.
  auto outputShape =
      getOutputs().front().getType().cast<ShapedType>().getShape();
  for (Type inputArgType : TypeRange{getInputs()}) {
    auto inputElemShape = inputArgType.cast<ShapedType>().getShape();
    if (inputElemShape != outputShape) {
      return emitOpError() << "expected shape of input (" << inputElemShape
                           << ") to match shape of output (" << outputShape
                           << ")";
    }
  }

  // The mapper should yield exactly one output.
  YieldOp mapperTerminator = cast<YieldOp>(bodyBlock->getTerminator());
  Type outputElementType =
      getOutputs().front().getType().cast<ShapedType>().getElementType();
  if (!succeeded(checkYieldOutputs(mapperTerminator, outputElementType)))
    return failure();

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<StringRef> MapOp::getIteratorTypesArray() {
  int64_t rank = getInit().getType().getRank();
  return SmallVector<StringRef>(rank, getParallelIteratorTypeName());
}

ArrayAttr MapOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getInit().getType().getRank();
  int64_t numIndexingMaps = getOperands().size();
  return builder.getAffineMapArrayAttr(SmallVector<AffineMap>(
      numIndexingMaps, builder.getMultiDimIdentityMap(rank)));
}

bool MapOp::hasIndexSemantics() { return false; }

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

ParseResult SortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(parser, result)) return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true)) {
    return failure();
  }

  Region *comparator = result.addRegion();
  if (parser.parseRegion(*comparator, regionArgs)) return failure();

  return success();
}

void SortOp::print(OpAsmPrinter &p) {
  printDstStyleOp<SortOp>(*this, p);

  p << "(";
  llvm::interleaveComma(getComparator().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getComparator(), /*printEntryBlockArgs=*/false);
}

LogicalResult SortOp::verify() {
  auto *comparatorBlock = getBody();
  auto comparatorArgs = comparatorBlock->getArguments();

  // Checks that the arity of the comparator is equal to twice the number of
  // inputs.
  if (comparatorArgs.size() != getNumInputs() * 2)
    return emitOpError() << "expected the number of block arguments "
                         << comparatorArgs.size() << " to be twice the number "
                         << "of inputs (2*" << getNumInputs() << ")";

  // Checks that the comparator's arguments match the element type of the
  // inputs.
  TypeRange inputTypes = TypeRange{getInputs()};
  TypeRange comparatorArgElementTypes = comparatorBlock->getArgumentTypes();
  for (size_t i = 0; i < getInputs().size(); ++i) {
    Type inputArgElemType = inputTypes[i].cast<ShapedType>().getElementType(),
         comparatorArgElemType1 = comparatorArgElementTypes[2 * i],
         comparatorArgElemType2 = comparatorArgElementTypes[2 * i + 1];
    if (comparatorArgElemType1 != inputArgElemType ||
        comparatorArgElemType2 != inputArgElemType)
      return emitOpError() << "expected element type of input " << i
                           << " to match type of the corresponding "
                              "arguments to the comparison function but got "
                           << inputArgElemType << " and ("
                           << comparatorArgElemType1 << ", "
                           << comparatorArgElemType2 << ")";
  }

  // Checks that the comparator yields exactly one boolean output.
  YieldOp comparatorTerminator =
      cast<YieldOp>(comparatorBlock->getTerminator());
  if (!succeeded(
          checkYieldOutputs(comparatorTerminator,
                            TypeRange({IntegerType::get(getContext(), 1)}))))
    return failure();

  // Checks that the inputs all have the same shape.
  ArrayRef<int64_t> referenceShape =
      getInputs().front().getType().cast<ShapedType>().getShape();

  for (auto &item : llvm::enumerate(TypeRange{getInputs()})) {
    ArrayRef<int64_t> shape = item.value().cast<ShapedType>().getShape();
    if (shape != referenceShape) {
      return emitOpError() << "expected all inputs to have the same shape ("
                           << referenceShape << ") but input " << item.index()
                           << " has shape (" << shape << ")";
    }
  }

  // Checks that the outputs have the same shape as the inputs.
  for (auto &item : llvm::enumerate(TypeRange{getOutputs()})) {
    ArrayRef<int64_t> shape = item.value().cast<ShapedType>().getShape();
    if (shape != referenceShape) {
      return emitOpError() << "expected outputs to have shape ("
                           << referenceShape << ") but output " << item.index()
                           << " has shape (" << shape << ")";
    }
  }

  // Checks that the rank of the reference shape is larger than the absolute
  // value of the sorting dimension. This is enough to ensure that the dimension
  // is valid, since all inputs are known to have the same shape.
  int64_t referenceRank = referenceShape.size();
  if (getDimension() >= referenceRank || getDimension() < 0) {
    return emitOpError() << "sorting dimension must be in range [0, "
                         << referenceRank << ") but got " << getDimension();
  }

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<utils::IteratorType> SortOp::getLoopIteratorTypes() {
  return getParallelIteratorTypes(getType(0).cast<ShapedType>().getRank() - 1);
}

SmallVector<Value> SortOp::getDestinationOperands(OpBuilder &) {
  return {getInits()};
}

SmallVector<Range> SortOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  auto oneInit = getInits().front();
  auto operandsRank = oneInit.getType().cast<ShapedType>().getRank();

  SmallVector<Range> iterationDomain(operandsRank - 1);

  IntegerAttr zero = b.getIndexAttr(0);
  IntegerAttr one = b.getIndexAttr(1);
  int64_t sortDimension = getDimension();

  for (auto axis : llvm::seq<int64_t>(0, operandsRank - 1)) {
    int64_t operandAxis = (axis >= sortDimension) ? axis + 1 : axis;
    iterationDomain[axis].offset = zero;
    iterationDomain[axis].size =
        b.createOrFold<tensor::DimOp>(loc, oneInit, operandAxis);
    iterationDomain[axis].stride = one;
  }
  return iterationDomain;
}

mlir::gml_st::TilingInterface SortOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto loc = getLoc();
  SmallVector<OpFoldResult> tileOffsets = llvm::to_vector(offsets);
  SmallVector<OpFoldResult> tileSizes = llvm::to_vector(sizes);

  size_t numOutputs = getNumOutputs();
  int64_t sortDimension = getDimension();

  Value oneInput = getInputs().front();

  // Capture the entire sorting axis in each tile.
  tileOffsets.insert(tileOffsets.begin() + sortDimension, b.getIndexAttr(0));

  OpFoldResult sortDimensionSize =
      b.createOrFold<tensor::DimOp>(loc, oneInput, sortDimension);
  tileSizes.insert(tileSizes.begin() + sortDimension, sortDimensionSize);

  gml_st::TileOp tile = createTileOp(b, loc, oneInput, tileOffsets, tileSizes);

  // Materialize the tile for each input and init.
  SmallVector<Value> tiledInputsAndInits;
  SmallVector<Type> tiledResultTypes;
  tiledInputsAndInits.reserve(numOutputs * 2);
  tiledResultTypes.reserve(numOutputs);

  auto oneInputShape = oneInput.getType().cast<ShapedType>().getShape();

  for (const auto &input : getInputs()) {
    tiledInputsAndInits.push_back(
        b.create<gml_st::MaterializeOp>(loc, input, tile));
    tiledResultTypes.push_back(RankedTensorType::get(
        oneInputShape, input.getType().cast<ShapedType>().getElementType()));
  }

  for (const auto &init : getInits()) {
    tiledInputsAndInits.push_back(
        b.create<gml_st::MaterializeOp>(loc, init, tile));
  }

  auto dpsInterface =
      cast<linalg::DestinationStyleOpInterface>(this->getOperation());
  return dpsInterface.clone(b, loc, tiledResultTypes, tiledInputsAndInits);
}

FailureOr<Value> SortOp::generateResultTileValue(OpBuilder &b,
                                                 unsigned resultNumber,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes)->getResult(resultNumber);
}

}  // namespace thlo
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.cc.inc"

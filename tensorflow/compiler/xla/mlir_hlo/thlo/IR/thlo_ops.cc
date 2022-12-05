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

#include "thlo/IR/thlo_ops.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "gml_st/interfaces/tiling_interface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Destination-style ops tools
//===----------------------------------------------------------------------===//

LogicalResult verifyDestinationStyleOp(Operation *op) {
  auto dstStyleOp = cast<DestinationStyleOpInterface>(*op);
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
  p.increaseIndent();
  if (op.getNumDpsInputs() != 0) {
    p.printNewline();
    p << "ins(";
    llvm::interleaveComma(
        op.getOperands().take_front(op.getNumDpsInputs()), p,
        [&](Value input) { p << input << " : " << input.getType(); });
    p << ")";
  }
  p.printNewline();
  p << "outs(";
  llvm::interleaveComma(
      op.getOperands().take_back(op.getNumDpsInits()), p,
      [&](Value output) { p << output << " : " << output.getType(); });
  p << ")";

  // Print attributes with custom printing logic.
  SmallVector<StringRef> elidedAttrs;
  if (printAttrsFn) elidedAttrs = printAttrsFn(op, p);

  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  p.decreaseIndent();
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
  p << attributeName << " = [" << attributeValue << "] ";
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
  return llvm::to_vector(llvm::map_range(dimValues, [&](OpFoldResult d) {
    return Range{b.getIndexAttr(0), d, b.getIndexAttr(1)};
  }));
}

}  // namespace
}  // namespace mlir

// Generated dialect definitions.
#include "thlo/IR/thlo_dialect.cc.inc"

namespace mlir {
namespace thlo {

void THLODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "thlo/IR/thlo_ops.cc.inc"
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
Value fuseConcatenateOpThroughTile(ConcatenateOp op, OpBuilder &b, Location loc,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes,
                                   bool useExtractSlice) {
  int64_t concatDim = op.getDimension().getSExtValue();
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
  SmallVector<Value> tileOffsets = getAsValues(b, loc, offsets);
  SmallVector<Value> tileSizes = getAsValues(b, loc, sizes);
  SmallVector<Value> tileStrides(sizes.size(),
                                 b.create<arith::ConstantIndexOp>(loc, 1));
  for (int64_t i = 0; i < rank; ++i) {
    Value iCst = b.create<arith::ConstantIndexOp>(loc, i);
    sharedTileStrides[i] =
        getValueOrCreateConstantIndexOp(b, loc, tileStrides[i]);

    // The space sizes, tile offsets, and tile sizes differ in the concat
    // dimension. Do not populate these.
    if (i == static_cast<int64_t>(concatDim)) continue;

    baseSpaceSizes[i] = b.createOrFold<tensor::DimOp>(loc, anyOperand, iCst);
    baseTileOffsets[i] =
        getValueOrCreateConstantIndexOp(b, loc, tileOffsets[i]);
    baseTileSizes[i] = getValueOrCreateConstantIndexOp(b, loc, tileSizes[i]);
  }

  // Some shared values.
  SmallVector<int64_t> allDynamic(rank, ShapedType::kDynamic);
  Value zeroCst = b.create<arith::ConstantIndexOp>(loc, 0);
  Value concatDimCst = b.create<arith::ConstantIndexOp>(loc, concatDim);
  Value maxTileSizeInConcatDim = tileSizes[concatDim];

  // The remaining tile offset in the concat dimension is subtracted by each
  // operand's size in that dimension. We maintain the invariant
  // remainingTileOffsetInConcatDim >= 0.
  Value remainingTileOffsetInConcatDim = tileOffsets[concatDim];

  // Create the relevant subsets per operand. These tiles can be empty at
  // runtime.
  SmallVector<Value> subOperands;
  subOperands.reserve(allOperands.size());
  for (Value operand : allOperands) {
    // Create operand space.
    Value operandSizeInConcatDim =
        b.create<tensor::DimOp>(loc, operand, concatDimCst);
    baseSpaceSizes[concatDim] = operandSizeInConcatDim;

    // Find the current operand's tile offset in the concat dimension. This is
    // the remaining offset clamped into the bounds of the operand. Note that
    // the remaining offset is always >= 0.
    Value operandTileOffsetInConcatDim = b.create<arith::MinUIOp>(
        loc, remainingTileOffsetInConcatDim, operandSizeInConcatDim);
    baseTileOffsets[concatDim] = operandTileOffsetInConcatDim;

    // Find the current operand's tile size in the concat dimension.
    Value remainingOperandSizeInConcatDim = b.create<arith::SubIOp>(
        loc, operandSizeInConcatDim, operandTileOffsetInConcatDim);
    baseTileSizes[concatDim] = b.create<arith::MinUIOp>(
        loc, remainingOperandSizeInConcatDim, maxTileSizeInConcatDim);

    // Create the operand tile and materialize the subset for this operand.
    subOperands.push_back(gml_st::materializeSlice(
        b, loc, operand, getMixedValues(allDynamic, baseTileOffsets, b),
        getMixedValues(allDynamic, baseTileSizes, b),
        getMixedValues(allDynamic, sharedTileStrides, b), false));

    // Unless it is the last operand, update the remaining tile offset in the
    // concat dimension. The remaining offset is subtracted by the operand's
    // size but must remain >= 0.
    if (operand != allOperands.back()) {
      Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                          remainingTileOffsetInConcatDim,
                                          operandSizeInConcatDim);
      Value sub = b.create<arith::SubIOp>(loc, remainingTileOffsetInConcatDim,
                                          operandSizeInConcatDim);
      remainingTileOffsetInConcatDim =
          b.create<arith::SelectOp>(loc, cmp, zeroCst, sub);
    }
  }

  // Create the tiled concat op.
  Value subInit = gml_st::materializeSlice(b, loc, op.getInit(), offsets, sizes,
                                           useExtractSlice);
  auto subResultType =
      RankedTensorType::get(subInit.getType().cast<ShapedType>().getShape(),
                            resultTy.getElementType());
  return b
      .create<thlo::ConcatenateOp>(loc, subResultType, subOperands, subInit,
                                   b.getIndexAttr(concatDim))
      ->getResult(0);
}

Value fuseConcatenateOpThroughPointRecursively(
    OpBuilder &b, Location loc, RankedTensorType rankedTy, uint64_t concatDim,
    SmallVector<Value> &remainingOffsets, ValueRange remainingOperands,
    bool useExtractSlice) {
  // Bail if called for no operands.
  if (remainingOperands.empty()) {
    return {};
  }
  Value leadingOperand = remainingOperands.front();

  // Terminal case of exactly one operand.
  if (remainingOperands.size() == 1) {
    // Create operand point.
    SmallVector<int64_t> allDynamicOffsets(rankedTy.getRank(),
                                           ShapedType::kDynamic);

    SmallVector<int64_t> sizeOrStride({1});

    auto slice = gml_st::materializeSlice(
        b, loc, leadingOperand,
        getMixedValues(allDynamicOffsets, remainingOffsets, b),
        getMixedValues(sizeOrStride, ValueRange{}, b),
        getMixedValues(sizeOrStride, ValueRange{}, b), useExtractSlice);

    return b.create<tensor::ExtractOp>(
        loc, slice, ValueRange{b.create<arith::ConstantIndexOp>(loc, 0)});
  }

  // For more than 1 operand, distinguish between the leading operand and the
  // remainder.
  assert(remainingOperands.size() > 1 &&
         "expect more than 1 operand at this point");
  Value leadingOperandConcatDim =
      b.create<tensor::DimOp>(loc, leadingOperand, concatDim);
  Value leadingOperandPredicate = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, remainingOffsets[concatDim],
      leadingOperandConcatDim);
  auto ifOp = b.create<scf::IfOp>(
      loc, rankedTy.getElementType(), leadingOperandPredicate,
      [&](OpBuilder &b, Location loc) {
        // For the leading operand, recur with the current offsets.
        Value fused = fuseConcatenateOpThroughPointRecursively(
            b, loc, rankedTy, concatDim, remainingOffsets, leadingOperand,
            useExtractSlice);
        b.create<scf::YieldOp>(loc, fused);
      },
      [&](OpBuilder &b, Location loc) {
        // For the remaining operands, substract the leading operand's size from
        // the remaining offsets in the concatenation dimension.
        SmallVector<Value> thenRemainingOffsets(remainingOffsets.begin(),
                                                remainingOffsets.end());
        thenRemainingOffsets[concatDim] = b.create<arith::SubIOp>(
            loc, remainingOffsets[concatDim], leadingOperandConcatDim);
        Value fused = fuseConcatenateOpThroughPointRecursively(
            b, loc, rankedTy, concatDim, thenRemainingOffsets,
            remainingOperands.drop_front(), useExtractSlice);
        b.create<scf::YieldOp>(loc, fused);
      });
  return ifOp.getResults().front();
}

}  // namespace

gml_st::TilingInterface ConcatenateOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    bool useExtractSlice) {
  auto tiled = fuseConcatenateOpThroughTile(*this, b, getLoc(), offsets, sizes,
                                            useExtractSlice);
  return llvm::cast<gml_st::TilingInterface>(tiled.getDefiningOp());
}

FailureOr<Value> ConcatenateOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  return getTiledImplementation(b, offsets, sizes, /*useExtractSlice=*/false)
      ->getResults()
      .front();
}

ParseResult ConcatenateOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(
      parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
        int64_t dimension = 0;
        if (parser.parseKeyword("dimension") || parser.parseEqual() ||
            parser.parseInteger(dimension))
          return failure();

        attributes.set("dimension",
                       parser.getBuilder().getIndexAttr(dimension));
        return success();
      });
}

void ConcatenateOp::print(OpAsmPrinter &p) {
  printDstStyleOp<ConcatenateOp>(
      *this, p,
      [](ConcatenateOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        p.printNewline();
        p << op.getDimensionAttrName().str() << " = " << op.getDimension();

        return {op.getDimensionAttrName()};
      });
}

LogicalResult ConcatenateOp::verify() {
  int64_t concatDim = getDimension().getSExtValue();

  ShapedType inputType =
      getDpsInputOperand(0)->get().getType().cast<ShapedType>();
  int64_t rank = inputType.getRank();
  auto inputShape = inputType.getShape();

  Type outputElementType =
      getDpsInitOperand(0)->get().getType().cast<ShapedType>().getElementType();

  for (const auto &en : llvm::enumerate(getInputs())) {
    ShapedType inputArgShapedType = en.value().getType().cast<ShapedType>();
    auto inputArgShape = inputArgShapedType.getShape();

    if (inputArgShapedType.getElementType() != outputElementType)
      return emitOpError() << "expected element type of input "
                           << inputArgShapedType.getElementType()
                           << " to match output element type "
                           << outputElementType;

    if (inputArgShapedType.getRank() != rank)
      return emitOpError() << "expected all args to be rank " << rank
                           << ", got " << inputArgShapedType.getRank()
                           << " in arg " << en.index();

    // Make sure that all dimensions, expect for concatenation dim, in the input
    // arg are equal.
    // TODO(shyshkov): Also check output dims once tiling is fixed for
    // ConcatenateOp.
    for (int64_t i = 0; i < rank; ++i) {
      if (i == concatDim) continue;

      if (inputShape[i] != inputArgShape[i])
        return emitOpError()
               << "shape of input arg " << en.index() << ": "
               << inputArgShapedType << " doesn't match expected shape "
               << inputType << " (all dims except concat dim(" << concatDim
               << ") should match exactly)";
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
        p.printNewline();
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
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    bool useExtractSlice) {
  // Create tile subset.
  auto loc = getLoc();
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
  auto dynamicDims = tensor::createDynamicDimValues(b, loc, getOperand());

  // Materialize operand dimensions.
  SmallVector<Value> operandDims;
  int64_t dynamicDimsIdx = 0;
  operandDims.reserve(operandTy.getRank());
  for (const auto &it : llvm::enumerate(operandTy.getShape())) {
    int64_t d = it.value();
    Value dim = d == ShapedType::kDynamic ? dynamicDims[dynamicDimsIdx++]
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
  auto tileOpOffsets = getValueOrCreateConstantIndexOp(b, loc, offsets);
  int64_t operandRank = operandTy.getRank();
  auto staticOffsets = SmallVector<int64_t>(operandRank, ShapedType::kDynamic);
  SmallVector<Value> operandOffsets;
  Value zero = getIndexConstant(0);
  for (int initId = 0, operandId = 0; initId < initRank; ++initId) {
    if (!dimensionsThatStay.contains(initId)) continue;
    Value isExpanding = operandExpandingDims[operandId++];
    Value collapsedSubsetOffset = tileOpOffsets[initId];
    operandOffsets.push_back(b.create<arith::SelectOp>(loc, isExpanding, zero,
                                                       collapsedSubsetOffset));
  }

  // Compute operand tile sizes.
  auto staticTileSizes =
      SmallVector<int64_t>(operandRank, ShapedType::kDynamic);
  SmallVector<Value> tileSizes;
  Value one = getIndexConstant(1);
  auto tileOpSizes = getValueOrCreateConstantIndexOp(b, loc, sizes);
  for (int initId = 0, operandId = 0; initId < initRank; ++initId) {
    if (!dimensionsThatStay.contains(initId)) continue;
    Value isExpanding = operandExpandingDims[operandId++];
    Value tileSize = tileOpSizes[initId];
    tileSizes.push_back(
        b.create<arith::SelectOp>(loc, isExpanding, one, tileSize));
  }

  // Create operand tile.
  auto staticTileStrides = SmallVector<int64_t>(operandRank, 1);
  SmallVector<Value> tileStrides = {};

  // Materialize operand tiles.
  Value tiledInit = gml_st::materializeSlice(b, loc, getInit(), offsets, sizes,
                                             useExtractSlice);
  Value tiledOperand = gml_st::materializeSlice(
      b, loc, getOperand(), getMixedValues(staticOffsets, operandOffsets, b),
      getMixedValues(staticTileSizes, tileSizes, b),
      getMixedValues(staticTileStrides, tileStrides, b), useExtractSlice);

  // Finally, materialize tiled broadcast.
  auto resultTy = getType(0).cast<RankedTensorType>();
  auto tiledResultTy =
      RankedTensorType::get(tiledInit.getType().cast<ShapedType>().getShape(),
                            resultTy.getElementType());
  return b.create<DynamicBroadcastInDimOp>(
      loc, TypeRange{tiledResultTy}, tiledOperand, tiledInit,
      getBroadcastDimensionsAttr(), getKnownExpandingDimensionsAttr(),
      getKnownNonexpandingDimensionsAttr());
}

FailureOr<Value> DynamicBroadcastInDimOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  return getTiledImplementation(b, offsets, sizes, /*useExtractSlice=*/false)
      ->getResults()
      .front();
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

  p.increaseIndent();
  p.printNewline();
  p << "(";
  llvm::interleaveComma(getUpdateComputation().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getUpdateComputation(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
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
      getDpsInitOperand(0)->get().getType().cast<ShapedType>().getElementType();
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

mlir::gml_st::TilingInterface ScatterOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    bool useExtractSlice) {
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

  Value updateSlice = gml_st::materializeSlice(b, loc, update, updateOffsets,
                                               updateSizes, useExtractSlice);

  // Tile outer dimension of indices.
  Value indices = this->getIndices();

  SmallVector<OpFoldResult> indicesOffsets{offsets.front(), zeroAttr};
  indicesOffsets.front() = tileOffset;
  SmallVector<OpFoldResult> indicesSizes =
      tensor::getMixedSizes(b, loc, indices);
  indicesSizes.front() = tileSize;

  Value indicesSlice = gml_st::materializeSlice(b, loc, indices, indicesOffsets,
                                                indicesSizes, useExtractSlice);

  // Get full space of the `init` tensor.
  Value init = this->getInit();
  Value initSlice =
      gml_st::materializeIdentitySlice(b, loc, init, useExtractSlice);

  return mlir::clone(b, this->getOperation(), TypeRange{initSlice.getType()},
                     ValueRange{indicesSlice, updateSlice, initSlice});
}

FailureOr<Value> ScatterOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "variadic scatter is not implemented");
  return getTiledImplementation(b, offsets, sizes, /*useExtractSlice=*/false)
      ->getResult(0);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

ParseResult GatherOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void GatherOp::print(OpAsmPrinter &p) { printDstStyleOp(*this, p); }

LogicalResult GatherOp::verify() {
  auto indicesType = getStartIndices().getType();
  int64_t indicesRank = indicesType.getRank();

  if (indicesRank != 2)
    return emitOpError() << "expected `indices` to be a 2D tensor";

  auto initType = getInit().getType();
  if (indicesType.getDimSize(0) != getInit().getType().getDimSize(0)) {
    return emitOpError()
           << "expected major dimension of `startIndices` to match "
              "major dimension of `init`";
  }

  if (initType.getNumDynamicDims() > 1 ||
      (initType.getNumDynamicDims() == 1 && !initType.isDynamicDim(0))) {
    return emitOpError() << "only the major dimenion of `init` may be dynamic";
  }

  if (indicesType.isDynamic(1)) {
    return emitOpError()
           << "the minor dimensions of `startIndices` must be static";
  }

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<utils::IteratorType> GatherOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel};
}

SmallVector<Value> GatherOp::getDestinationOperands(OpBuilder &) {
  return {getInit()};
}

SmallVector<Range> GatherOp::getIterationDomain(OpBuilder &b) {
  Value indicesCount = b.create<tensor::DimOp>(getLoc(), getStartIndices(), 0);
  return {Range{b.getIndexAttr(0), indicesCount, b.getIndexAttr(1)}};
}

mlir::gml_st::TilingInterface GatherOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    bool useExtractSlice) {
  SmallVector<OpFoldResult> startIndexOffsets{offsets.front(),
                                              b.getIndexAttr(0)};
  SmallVector<OpFoldResult> startIndexSizes{
      sizes.front(),
      b.getIndexAttr(getStartIndices().getType().getShape().back())};
  auto subStartIndices = gml_st::materializeSlice(
      b, getLoc(), getStartIndices(), startIndexOffsets, startIndexSizes,
      useExtractSlice);

  int64_t initRank = getInit().getType().getRank();
  SmallVector<OpFoldResult> initOffsets(initRank, b.getIndexAttr(0));
  initOffsets[0] = offsets.front();
  auto initSizes = tensor::getMixedSizes(b, getLoc(), getInit());
  initSizes[0] = sizes.front();
  Value initSlice = gml_st::materializeSlice(
      b, getLoc(), getInit(), initOffsets, initSizes, useExtractSlice);

  return b
      .create<GatherOp>(getLoc(), TypeRange{initSlice.getType()},
                        ValueRange{getOperand(), subStartIndices, initSlice})
      .getOperation();
}

FailureOr<Value> GatherOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "resultNumber > 0 not implemented");
  return getTiledImplementation(b, offsets, sizes, /*useExtractSlice=*/false)
      ->getResult(0);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  ResultRange results = getResults();
  for (int i = 0; i < results.size(); i++) {
    setNameFn(results[i], "sorted" + std::to_string(i));
  }
}

void SortOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  for (int i = 0, e = region.getNumArguments(); i < e; i += 2) {
    setNameFn(region.getArgument(i), "lhs" + std::to_string(i / 2));
    setNameFn(region.getArgument(i + 1), "rhs" + std::to_string(i / 2));
  }
}

ParseResult SortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            int64_t dimension = 0;
            int64_t isStable = 0;
            if (parser.parseKeyword("dimension") || parser.parseEqual() ||
                parser.parseInteger(dimension) ||
                parser.parseKeyword("is_stable") || parser.parseEqual() ||
                parser.parseInteger(isStable))
              return failure();

            auto b = parser.getBuilder();
            attributes.set("dimension", b.getIndexAttr(dimension));
            attributes.set("is_stable", b.getBoolAttr(isStable != 0));
            return success();
          }))
    return failure();

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
  printDstStyleOp<SortOp>(
      *this, p, [](SortOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        p.printNewline();
        p << op.getDimensionAttrName().str() << " = " << op.getDimension();

        p.printNewline();
        p << op.getIsStableAttrName().str() << " = " << op.getIsStable();
        return {op.getDimensionAttrName(), op.getIsStableAttrName()};
      });

  p.increaseIndent();
  p.printNewline();
  p << "(";
  llvm::interleaveComma(getComparator().getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getComparator(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
}

LogicalResult SortOp::verify() {
  auto *comparatorBlock = getBody();
  auto comparatorArgs = comparatorBlock->getArguments();

  // Checks that the arity of the comparator is equal to twice the number of
  // inputs.
  int64_t numInputs = getNumDpsInputs();
  int64_t numOutputs = getNumDpsInits();
  if (getNumDpsInits() != numInputs) {
    return emitOpError() << "expected the number of inputs " << numInputs
                         << " to match the number of outputs " << numOutputs;
  }
  if (static_cast<int64_t>(comparatorArgs.size()) != numInputs * 2) {
    return emitOpError() << "expected the number of block arguments "
                         << comparatorArgs.size() << " to be twice the number "
                         << "of inputs (2*" << numInputs << ")";
  }
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
  for (auto &item : llvm::enumerate(getInits())) {
    ArrayRef<int64_t> shape =
        item.value().getType().cast<ShapedType>().getShape();
    if (shape != referenceShape) {
      return emitOpError() << "expected outputs to have shape ("
                           << referenceShape << ") but output " << item.index()
                           << " has shape (" << shape << ")";
    }
  }

  // Checks that the rank of the reference shape is larger than the absolute
  // value of the sorting dimension. This is enough to ensure that the dimension
  // is valid, since all inputs are known to have the same shape. `getDimension`
  // returns an unsigned int, so no need to check for negative values.
  size_t referenceRank = referenceShape.size();
  if (getDimension().getSExtValue() >= referenceRank) {
    return emitOpError() << "sorting dimension must be in range [0, "
                         << referenceRank << ") but got "
                         << getDimension().getSExtValue();
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
  int64_t sortDimension = getDimension().getSExtValue();

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
    OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    bool useExtractSlice) {
  auto loc = getLoc();
  SmallVector<OpFoldResult> tileOffsets = llvm::to_vector(offsets);
  SmallVector<OpFoldResult> tileSizes = llvm::to_vector(sizes);

  size_t numOutputs = getNumDpsInits();
  int64_t sortDimension = getDimension().getSExtValue();

  Value oneInput = getInputs().front();

  // Capture the entire sorting axis in each tile.
  tileOffsets.insert(tileOffsets.begin() + sortDimension, b.getIndexAttr(0));

  OpFoldResult sortDimensionSize =
      b.createOrFold<tensor::DimOp>(loc, oneInput, sortDimension);
  tileSizes.insert(tileSizes.begin() + sortDimension, sortDimensionSize);

  // Materialize the tile for each input and init.
  SmallVector<Value> tiledInputsAndInits;
  SmallVector<Type> tiledResultTypes;
  tiledInputsAndInits.reserve(numOutputs * 2);
  tiledResultTypes.reserve(numOutputs);

  for (const auto &input : getInputs()) {
    tiledInputsAndInits.push_back(gml_st::materializeSlice(
        b, loc, input, tileOffsets, tileSizes, useExtractSlice));
    auto tileShape =
        tiledInputsAndInits.back().getType().cast<ShapedType>().getShape();
    tiledResultTypes.push_back(RankedTensorType::get(
        tileShape, input.getType().cast<ShapedType>().getElementType()));
  }

  for (const auto &init : getInits()) {
    tiledInputsAndInits.push_back(gml_st::materializeSlice(
        b, loc, init, tileOffsets, tileSizes, useExtractSlice));
  }

  return mlir::clone(b, this->getOperation(), tiledResultTypes,
                     tiledInputsAndInits);
}

FailureOr<Value> SortOp::generateResultTileValue(OpBuilder &b,
                                                 unsigned resultNumber,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes, /*useExtractSlice=*/false)
      ->getResult(resultNumber);
}

}  // namespace thlo
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "thlo/IR/thlo_ops.cc.inc"

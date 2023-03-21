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
#include <cassert>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

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
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace {

Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       ArrayRef<OpFoldResult> strides) {
  return b.create<tensor::ExtractSliceOp>(loc, valueToTile, offsets, sizes,
                                          strides);
}

Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  return materializeSlice(b, loc, valueToTile, offsets, sizes, strides);
}

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
  if (op.getNumDpsInputs() != 0) {
    p << " ins(";
    llvm::interleaveComma(
        op.getOperands().take_front(op.getNumDpsInputs()), p,
        [&](Value input) { p << input << " : " << input.getType(); });
    p << ")";
  }
  p << " outs(";
  llvm::interleaveComma(
      op.getOperands().take_back(op.getNumDpsInits()), p,
      [&](Value output) { p << output << " : " << output.getType(); });
  p << ")";

  // Print attributes with custom printing logic.
  SmallVector<StringRef> elidedAttrs;
  if (printAttrsFn) {
    p << ' ';
    elidedAttrs = printAttrsFn(op, p);
  }

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
                                      /*allowResultNumber=*/true) ||
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

static void getDstStyleOpEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, const OpOperandVector &inputOperands,
    const OpOperandVector &outputOperands) {
  for (auto *operand : inputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
  for (auto *operand : outputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
}

}  // namespace
}  // namespace mlir

//===----------------------------------------------------------------------===//
// THLO Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace mlir {
namespace {

struct THLOInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Operations in THLO dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {}
};

}  // namespace
}  // namespace mlir

//===----------------------------------------------------------------------===//
// THLODialect
//===----------------------------------------------------------------------===//

// Generated dialect definitions.
#include "thlo/IR/thlo_dialect.cc.inc"

namespace mlir {
namespace thlo {

void THLODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "thlo/IR/thlo_ops.cc.inc"
      >();

  addInterfaces<THLOInlinerInterface>();
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

  for (const auto &item : llvm::enumerate(
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

SmallVector<Range> ConcatenateOp::getIterationDomain(OpBuilder &b) {
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

namespace {

Value getSingleOperandTiledImplementationForConcatRecursively(
    OpBuilder &b, Location loc, int64_t concatDim, ValueRange remainingOperands,
    SmallVector<OpFoldResult> &remainingOffsets, ArrayRef<OpFoldResult> sizes) {
  assert(!remainingOperands.empty() && "expect at least one remaining operand");
  assert(sizes[concatDim].get<Attribute>().cast<IntegerAttr>().getInt() == 1 &&
         "expect unit size in concat dim");

  // Terminal case of exactly one operand.
  Value leadingOperand = remainingOperands.front();
  if (remainingOperands.size() == 1) {
    return materializeSlice(b, loc, leadingOperand, remainingOffsets, sizes);
  }

  // For more than one operand, distinguish between the leading operand and the
  // remainder.
  assert(remainingOperands.size() > 1 &&
         "expect more than one operand at this point");
  Value leadingOperandSizeInConcatDim =
      b.create<tensor::DimOp>(loc, leadingOperand, concatDim);
  Value remainingOffsetInConcatDim =
      getValueOrCreateConstantIndexOp(b, loc, remainingOffsets[concatDim]);
  Value leadingOperandPredicate = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, remainingOffsetInConcatDim,
      leadingOperandSizeInConcatDim);
  auto ifOp = b.create<scf::IfOp>(
      loc, leadingOperandPredicate,
      [&](OpBuilder &b, Location loc) {
        Value tiledConcat =
            getSingleOperandTiledImplementationForConcatRecursively(
                b, loc, concatDim, {leadingOperand}, remainingOffsets, sizes);
        b.create<scf::YieldOp>(loc, tiledConcat);
      },
      [&](OpBuilder &b, Location loc) {
        remainingOffsets[concatDim] =
            b.create<arith::SubIOp>(loc, remainingOffsetInConcatDim,
                                    leadingOperandSizeInConcatDim)
                .getResult();
        Value tiledConcat =
            getSingleOperandTiledImplementationForConcatRecursively(
                b, loc, concatDim, remainingOperands.drop_front(),
                remainingOffsets, sizes);
        b.create<scf::YieldOp>(loc, tiledConcat);
      });
  return ifOp.getResults().front();
}

Value getSingleOperandTiledImplementationForConcat(
    ConcatenateOp op, OpBuilder &b, Location loc,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  int64_t concatDim = op.getDimension().getSExtValue();
  SmallVector<OpFoldResult> remainingOffsets(offsets);
  return getSingleOperandTiledImplementationForConcatRecursively(
      b, loc, concatDim, op.getInputs(), remainingOffsets, sizes);
}

Value getGenericTiledImplementationForConcat(ConcatenateOp op, OpBuilder &b,
                                             Location loc,
                                             ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes) {
  // Create a basis for the tile offsets and sizes. These hold the shared values
  // in all non-concat dimensions and are amended in the concat dimension to
  // create the individual operand tiles. Also, create the shared tile strides,
  // which are the exact same for every operand tile.
  SmallVector<OpFoldResult> operandTileOffsetsBase(offsets);
  SmallVector<OpFoldResult> operandTileSizesBase(sizes);
  SmallVector<OpFoldResult> operandTileStrides(sizes.size(), b.getIndexAttr(1));

  // Some shared values.
  Value zeroCst = b.create<arith::ConstantIndexOp>(loc, 0);
  int64_t concatDim = op.getDimension().getSExtValue();
  Value concatDimCst = b.create<arith::ConstantIndexOp>(loc, concatDim);
  Value maxTileSizeInConcatDim =
      getValueOrCreateConstantIndexOp(b, loc, sizes[concatDim]);

  // The remaining tile offset in the concat dimension is subtracted by each
  // operand's size in that dimension. We maintain the invariant
  // remainingTileOffsetInConcatDim >= 0.
  Value remainingTileOffsetInConcatDim =
      getValueOrCreateConstantIndexOp(b, loc, offsets[concatDim]);

  // Create the relevant subsets per operand. These tiles can be empty at
  // runtime.
  SmallVector<Value> tiledOperands;
  tiledOperands.reserve(op.getNumDpsInputs());
  for (Value operand : op.getInputs()) {
    // Find the current operand's tile offset in the concat dimension. This is
    // the remaining offset clamped into the bounds of the operand. Note that
    // the remaining offset is always >= 0.
    Value operandSizeInConcatDim =
        b.create<tensor::DimOp>(loc, operand, concatDimCst);
    Value operandTileOffsetInConcatDim = b.create<arith::MinUIOp>(
        loc, remainingTileOffsetInConcatDim, operandSizeInConcatDim);
    operandTileOffsetsBase[concatDim] = operandTileOffsetInConcatDim;

    // Find the current operand's tile size in the concat dimension.
    Value remainingOperandSizeInConcatDim = b.create<arith::SubIOp>(
        loc, operandSizeInConcatDim, operandTileOffsetInConcatDim);
    operandTileSizesBase[concatDim] = b.createOrFold<arith::MinUIOp>(
        loc, remainingOperandSizeInConcatDim, maxTileSizeInConcatDim);

    // Create the operand tile and materialize the subset for this operand.
    tiledOperands.push_back(
        materializeSlice(b, loc, operand, operandTileOffsetsBase,
                         operandTileSizesBase, operandTileStrides));

    // Unless it is the last operand, update the remaining tile offset in the
    // concat dimension. The remaining offset is subtracted by the operand's
    // size but must remain >= 0.
    if (operand != op.getInputs().back()) {
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
  Value tiledInit = materializeSlice(b, loc, op.getInit(), offsets, sizes);
  auto tiledConcat =
      b.create<thlo::ConcatenateOp>(loc, tiledInit.getType(), tiledOperands,
                                    tiledInit, b.getIndexAttr(concatDim));
  return tiledConcat.getResults().front();
}

Value getTiledImplementationForConcat(ConcatenateOp op, OpBuilder &b,
                                      Location loc,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes) {
  // If the tile is of unit size in the concatenation dimension, we can generate
  // the tiled implementation based on a single operand.
  int64_t concatDim = op.getDimension().getSExtValue();
  OpFoldResult tileSizeInConcatDim = sizes[concatDim];
  if (tileSizeInConcatDim.is<Attribute>() &&
      tileSizeInConcatDim.get<Attribute>().cast<IntegerAttr>().getInt() == 1) {
    return getSingleOperandTiledImplementationForConcat(op, b, loc, offsets,
                                                        sizes);
  }

  // Otherwise, rely on the generic implementation.
  return getGenericTiledImplementationForConcat(op, b, loc, offsets, sizes);
}

}  // namespace

FailureOr<TilingResult> ConcatenateOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto tiled =
      getTiledImplementationForConcat(*this, b, getLoc(), offsets, sizes);
  return TilingResult{{tiled.getDefiningOp()}, {tiled}};
}

LogicalResult ConcatenateOp::getResultTilePosition(
    OpBuilder & /*b*/, unsigned /*resultNumber*/,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> ConcatenateOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult.value();
}

LogicalResult ConcatenateOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Location loc = getLoc();
  Value init = getInit();

  // Assume unique result.
  if (getNumResults() != 1) return failure();
  SmallVector<OpFoldResult> &shape = reifiedReturnShapes.emplace_back();

  // Derive shape from init operand.
  int64_t rank = init.getType().cast<RankedTensorType>().getRank();
  shape.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    shape.push_back(b.create<tensor::DimOp>(loc, init, i).getResult());
  }

  return success();
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

void ConcatenateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
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

SmallVector<Range> DynamicBroadcastInDimOp::getIterationDomain(OpBuilder &b) {
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

FailureOr<TilingResult> DynamicBroadcastInDimOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // Create tile subset.
  auto loc = getLoc();
  auto initRank = getInit().getType().cast<RankedTensorType>().getRank();

  DenseMap<uint64_t, Value> localIndexConstants;

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
    Value dim = d == ShapedType::kDynamic
                    ? dynamicDims[dynamicDimsIdx++]
                    : b.create<arith::ConstantIndexOp>(loc, d);
    operandDims.push_back(dim);
  }

  // Find the expanding dimensions. If corresponding operand and result
  // dimensions are different then the dimension is expanding.
  // TODO(frgossen): Use info from known expanding and known non-expanding
  // dimensions here.
  SmallVector<Value> operandExpandingDims;
  for (const auto &it : llvm::enumerate(getBroadcastDimensions())) {
    auto operandDim = operandDims[it.index()];
    auto resultDim = b.create<tensor::DimOp>(
        loc, getInit(), b.create<arith::ConstantIndexOp>(loc, it.value()));
    operandExpandingDims.push_back(b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, operandDim, resultDim));
  }

  // Compute operand tile offsets.
  auto tileOpOffsets = getValueOrCreateConstantIndexOp(b, loc, offsets);
  int64_t operandRank = operandTy.getRank();
  auto staticOffsets = SmallVector<int64_t>(operandRank, ShapedType::kDynamic);
  SmallVector<Value> operandOffsets;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
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
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
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
  Value tiledInit = materializeSlice(b, loc, getInit(), offsets, sizes);
  Value tiledOperand = materializeSlice(
      b, loc, getOperand(), getMixedValues(staticOffsets, operandOffsets, b),
      getMixedValues(staticTileSizes, tileSizes, b),
      getMixedValues(staticTileStrides, tileStrides, b));

  // Finally, materialize tiled broadcast.
  auto resultTy = getType(0).cast<RankedTensorType>();
  auto tiledResultTy =
      RankedTensorType::get(tiledInit.getType().cast<ShapedType>().getShape(),
                            resultTy.getElementType());
  auto tiledOp = b.create<DynamicBroadcastInDimOp>(
      loc, TypeRange{tiledResultTy}, tiledOperand, tiledInit,
      getBroadcastDimensionsAttr(), getKnownExpandingDimensionsAttr(),
      getKnownNonexpandingDimensionsAttr());
  return TilingResult{{tiledOp}, {tiledOp.getResult()}};
}

LogicalResult DynamicBroadcastInDimOp::getResultTilePosition(
    OpBuilder & /*b*/, unsigned /*resultNumber*/,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DynamicBroadcastInDimOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "expect unique result idx");
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult.value();
}

void DynamicBroadcastInDimOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
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

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &b) {
  Value indicesCount = b.create<tensor::DimOp>(getLoc(), getIndices(), 0);
  return {Range{b.getIndexAttr(0), indicesCount, b.getIndexAttr(1)}};
}

FailureOr<TilingResult> ScatterOp::getTiledImplementation(
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

  Value updateSlice =
      materializeSlice(b, loc, update, updateOffsets, updateSizes);

  // Tile outer dimension of indices.
  Value indices = this->getIndices();

  SmallVector<OpFoldResult> indicesOffsets{offsets.front(), zeroAttr};
  indicesOffsets.front() = tileOffset;
  SmallVector<OpFoldResult> indicesSizes =
      tensor::getMixedSizes(b, loc, indices);
  indicesSizes.front() = tileSize;

  Value indicesSlice =
      materializeSlice(b, loc, indices, indicesOffsets, indicesSizes);

  // Get full space of the `init` tensor. We use an extract_slice op because
  // otherwise, tileUsingSCFForOp won't replace the arg with the bbarg.
  int64_t initRank = getInit().getType().getRank();
  Value init = materializeSlice(b, loc, this->getInit(),
                                SmallVector<OpFoldResult>(initRank, zeroAttr),
                                tensor::getMixedSizes(b, loc, this->getInit()));

  Operation *tiledOp =
      mlir::clone(b, this->getOperation(), TypeRange{init.getType()},
                  ValueRange{indicesSlice, updateSlice, init});
  return TilingResult{{tiledOp}, {tiledOp->getResult(0)}};
}

LogicalResult ScatterOp::getResultTilePosition(
    OpBuilder &b, unsigned /*resultNumber*/, ArrayRef<OpFoldResult> /*offsets*/,
    ArrayRef<OpFoldResult> /*sizes*/, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  ScatterOp scatterOp = cast<ScatterOp>(this->getOperation());
  auto init = scatterOp.getInit();
  resultOffsets =
      SmallVector<OpFoldResult>(init.getType().getRank(), b.getIndexAttr(0));
  resultSizes = tensor::createDimValues(b, scatterOp.getLoc(), init);
  return success();
}

FailureOr<TilingResult> ScatterOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "variadic scatter is not implemented");
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult;
}

void ScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
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

SmallVector<Range> GatherOp::getIterationDomain(OpBuilder &b) {
  Value indicesCount = b.create<tensor::DimOp>(getLoc(), getStartIndices(), 0);
  return {Range{b.getIndexAttr(0), indicesCount, b.getIndexAttr(1)}};
}

FailureOr<TilingResult> GatherOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> startIndexOffsets{offsets.front(),
                                              b.getIndexAttr(0)};
  SmallVector<OpFoldResult> startIndexSizes{
      sizes.front(),
      b.getIndexAttr(getStartIndices().getType().getShape().back())};
  auto subStartIndices = materializeSlice(b, getLoc(), getStartIndices(),
                                          startIndexOffsets, startIndexSizes);

  int64_t initRank = getInit().getType().getRank();
  SmallVector<OpFoldResult> initOffsets(initRank, b.getIndexAttr(0));
  initOffsets[0] = offsets.front();
  auto initSizes = tensor::getMixedSizes(b, getLoc(), getInit());
  initSizes[0] = sizes.front();
  Value initSlice =
      materializeSlice(b, getLoc(), getInit(), initOffsets, initSizes);

  auto gatherOp =
      b.create<GatherOp>(getLoc(), TypeRange{initSlice.getType()},
                         ValueRange{getOperand(), subStartIndices, initSlice});
  return TilingResult{{gatherOp}, {gatherOp.getResult()}};
}

LogicalResult GatherOp::getResultTilePosition(
    OpBuilder &b, unsigned /*resultNumber*/, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  GatherOp gatherOp = cast<GatherOp>(this->getOperation());
  auto init = gatherOp.getInit();
  resultOffsets =
      SmallVector<OpFoldResult>(init.getType().getRank(), b.getIndexAttr(0));
  resultOffsets.front() = offsets.front();
  resultSizes = tensor::createDimValues(b, gatherOp.getLoc(), init);
  resultSizes.front() = sizes.front();
  return success();
}

FailureOr<TilingResult> GatherOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  assert(resultNumber == 0 && "resultNumber > 0 not implemented");
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult.value();
}

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  ResultRange results = getResults();
  for (size_t i = 0; i < results.size(); i++) {
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
        p << op.getDimensionAttrName().str() << " = " << op.getDimension()
          << ' ' << op.getIsStableAttrName().str() << " = " << op.getIsStable();
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

  for (const auto &item : llvm::enumerate(TypeRange{getInputs()})) {
    ArrayRef<int64_t> shape = item.value().cast<ShapedType>().getShape();
    if (shape != referenceShape) {
      return emitOpError() << "expected all inputs to have the same shape ("
                           << referenceShape << ") but input " << item.index()
                           << " has shape (" << shape << ")";
    }
  }

  // Checks that the outputs have the same shape as the inputs.
  for (const auto &item : llvm::enumerate(getInits())) {
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
  if (getDimension().getSExtValue() >= (int64_t)referenceRank) {
    return emitOpError() << "sorting dimension must be in range [0, "
                         << referenceRank << ") but got "
                         << getDimension().getSExtValue();
  }

  return verifyDestinationStyleOp(getOperation());
}

SmallVector<utils::IteratorType> SortOp::getLoopIteratorTypes() {
  return getParallelIteratorTypes(getType(0).cast<ShapedType>().getRank() - 1);
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

FailureOr<TilingResult> SortOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
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
    tiledInputsAndInits.push_back(
        materializeSlice(b, loc, input, tileOffsets, tileSizes));
    auto tileShape =
        tiledInputsAndInits.back().getType().cast<ShapedType>().getShape();
    tiledResultTypes.push_back(RankedTensorType::get(
        tileShape, input.getType().cast<ShapedType>().getElementType()));
  }

  for (const auto &init : getInits()) {
    tiledInputsAndInits.push_back(
        materializeSlice(b, loc, init, tileOffsets, tileSizes));
  }

  Operation *tiledOp = mlir::clone(b, this->getOperation(), tiledResultTypes,
                                   tiledInputsAndInits);
  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult SortOp::getResultTilePosition(
    OpBuilder &b, unsigned /*resultNumber*/, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  SortOp sortOp = cast<SortOp>(this->getOperation());
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);

  int64_t sortDimIndex = sortOp.getDimension().getSExtValue();
  Value sortDimValue = b.create<tensor::DimOp>(
      sortOp.getLoc(), sortOp.getInputs().front(), sortDimIndex);
  resultOffsets.insert(resultOffsets.begin() + sortDimIndex, b.getIndexAttr(0));
  resultSizes.insert(resultSizes.begin() + sortDimIndex, sortDimValue);
  return success();
}

FailureOr<TilingResult> SortOp::generateResultTileValue(
    OpBuilder &b, unsigned /*resultNumber*/, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult.value();
}

void SortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

ParseResult ReverseOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(
      parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
        return parseDenseI64ArrayAttr(parser, attributes, "reverse_dimensions");
      });
}

void ReverseOp::print(OpAsmPrinter &p) {
  printDstStyleOp<ReverseOp>(
      *this, p, [](ReverseOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.getReverseDimensionsAttrName(),
                               op.getReverseDimensions());
        return {op.getReverseDimensionsAttrName()};
      });
}

LogicalResult ReverseOp::verify() {
  return verifyDestinationStyleOp(getOperation());
}

void ReverseOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "reversed");
}

SmallVector<utils::IteratorType> ReverseOp::getLoopIteratorTypes() {
  int64_t rank = getType().cast<ShapedType>().getRank();
  return getParallelIteratorTypes(rank);
}

SmallVector<Range> ReverseOp::getIterationDomain(OpBuilder &b) {
  return getIterationDomainForTensor(b, getLoc(), getInit());
}

namespace {
SmallVector<OpFoldResult> getInputTileOffsetsForReverse(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<int64_t> reverseDimensions,
    TypedValue<ShapedType> &input) {
  auto tileOpOffsets = getValueOrCreateConstantIndexOp(b, loc, offsets);
  auto sizes = getValueOrCreateConstantIndexOp(b, loc, tileSizes);
  SmallVector<OpFoldResult> inputTileOffsets;
  for (size_t i = 0; i < tileOpOffsets.size(); ++i) {
    if (llvm::is_contained(reverseDimensions, i)) {
      inputTileOffsets.push_back(OpFoldResult{b.createOrFold<arith::SubIOp>(
          loc,
          b.createOrFold<arith::SubIOp>(
              loc, b.createOrFold<tensor::DimOp>(loc, input, i),
              Value(tileOpOffsets[i])),
          sizes[i])});
    } else {
      inputTileOffsets.push_back(tileOpOffsets[i]);
    }
  }

  return inputTileOffsets;
}
}  // namespace

FailureOr<TilingResult> ReverseOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto loc = getLoc();
  auto input = getInput();
  SmallVector<OpFoldResult> inputTileOffsets = getInputTileOffsetsForReverse(
      b, loc, offsets, sizes, getReverseDimensions(), input);

  // Materialize the tile for input and init.
  SmallVector<Value, 2> tiledInputsAndInits;

  tiledInputsAndInits.push_back(
      materializeSlice(b, loc, input, inputTileOffsets, sizes));
  tiledInputsAndInits.push_back(
      materializeSlice(b, loc, getInit(), offsets, sizes));
  auto tileShape =
      tiledInputsAndInits.back().getType().cast<ShapedType>().getShape();
  auto tiledResultType = RankedTensorType::get(
      tileShape, input.getType().cast<ShapedType>().getElementType());

  Operation *tiledOp = mlir::clone(b, this->getOperation(), tiledResultType,
                                   tiledInputsAndInits);
  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult ReverseOp::getResultTilePosition(
    OpBuilder & /*b*/, unsigned /*resultNumber*/,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> ReverseOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult)) return failure();
  return tilingResult.value();
}

OpFoldResult ReverseOp::fold(
    ReverseOpGenericAdaptor<ArrayRef<Attribute>>) /*operands*/ {
  auto inputType = getInput().getType();
  for (unsigned i = 0; i < getReverseDimensions().size(); ++i) {
    if (inputType.getDimSize(getReverseDimensions()[i]) != 1) return nullptr;
  }
  return getInput();
}

void ReverseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getDstStyleOpEffectsImpl(effects, getOperation()->getResults(),
                           getDpsInputOperands(), getDpsInitOperands());
}

}  // namespace thlo
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "thlo/IR/thlo_ops.cc.inc"

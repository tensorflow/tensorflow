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
#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Destination-style ops tools
//===----------------------------------------------------------------------===//

bool hasTensorSemantics(OperandRange operands, unsigned numOutputArgs) {
  for (auto operand : operands.drop_back(numOutputArgs)) {
    if (!operand.getType().isa<ShapedType>()) continue;
    if (!operand.getType().isa<RankedTensorType>()) return false;
  }
  return llvm::all_of(operands.take_back(numOutputArgs), [](Value operand) {
    return operand.getType().isa<RankedTensorType>();
  });
}

bool hasBufferSemantics(OperandRange operands) {
  return llvm::all_of(operands, [](Value operand) {
    return operand.getType().isa<MemRefType>();
  });
}

LogicalResult verifyDestinationStyleOp(Operation *op,
                                       unsigned numOutputArgs = 1) {
  if (hasBufferSemantics(op->getOperands()))
    return success(op->getNumResults() == 0);

  if (!hasTensorSemantics(op->getOperands(), numOutputArgs))
    return op->emitOpError("expected either buffer or tensor semantics");

  if (op->getNumResults() != numOutputArgs) {
    return op->emitOpError(
        "expected the number of output args to match the number of results");
  }
  for (auto &en : llvm::enumerate(llvm::zip(
           op->getResultTypes(), op->getOperands().take_back(numOutputArgs)))) {
    size_t index = en.index();
    Type resultType = std::get<0>(en.value());
    Type outputOperandType = std::get<1>(en.value()).getType();
    if (resultType != outputOperandType)
      op->emitOpError() << "type " << resultType << " of result " << index
                        << " does not match output operand type "
                        << outputOperandType;
  }
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
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {

Value fuseConcatenateOpThroughTile(ConcatenateOp op, OpBuilder &builder,
                                   Location loc, Value tile) {
  uint64_t concatDim = op.dimension();
  auto resultTy = op.getResult().getType().cast<RankedTensorType>();
  int64_t rank = resultTy.getRank();
  OperandRange allOperands = op.operands();
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
    if (i == concatDim) {
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
      remainingTileOffsetInConcatDim = builder.create<arith::SelectOp>(
          loc,
          builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                        remainingTileOffsetInConcatDim,
                                        operandSizeInConcatDim),
          zeroCst,
          builder.create<arith::SubIOp>(loc, remainingTileOffsetInConcatDim,
                                        operandSizeInConcatDim));
    }
  }

  // Create the tiled concat op.
  auto tileType = tile.getType().cast<gml_st::TileType>();
  Value subInit = builder.create<gml_st::MaterializeOp>(loc, op.init(), tile);
  auto subResultType =
      RankedTensorType::get(tileType.getShape(), resultTy.getElementType());
  return builder.create<thlo::ConcatenateOp>(loc, subResultType, subOperands,
                                             subInit, concatDim);
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
    Value operandPoint = builder.create<gml_st::PointOp>(
        loc, operandSpace, remainingOffsets,
        builder.getI64ArrayAttr(allDynamicOffsets));

    return builder.create<gml_st::MaterializeOp>(loc, leadingOperand,
                                                 operandPoint);
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
  auto resultTy = op.getType().cast<RankedTensorType>();
  int64_t resultRank = resultTy.getRank();
  uint64_t concatDim = op.dimension();

  // Materialize initial offsets.
  SmallVector<Value> initialOffsets;
  initialOffsets.reserve(resultRank);
  for (int64_t i = 0; i < resultRank; ++i) {
    initialOffsets.push_back(builder.create<gml_st::OffsetOp>(
        loc, subset, builder.create<arith::ConstantIndexOp>(loc, i)));
  }

  ValueRange initialOperands = op.operands();
  return fuseConcatenateOpThroughPointRecursively(
      builder, loc, resultTy, concatDim, initialOffsets, initialOperands);
}

}  // namespace

Value ConcatenateOp::fuse(Location loc, Value subset, OpBuilder &builder) {
  Type subsetTy = subset.getType();
  if (subsetTy.isa<gml_st::TileType>()) {
    return fuseConcatenateOpThroughTile(*this, builder, loc, subset);
  }
  if (subsetTy.isa<gml_st::PointType>()) {
    return fuseConcatenateOpThroughPoint(*this, builder, loc, subset);
  }
  return {};
}

ParseResult ConcatenateOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void ConcatenateOp::print(OpAsmPrinter &p) {
  printDstStyleOp(cast<ConcatenateOp>(getOperation()), p);
}

LogicalResult ConcatenateOp::verify() {
  return verifyDestinationStyleOp(getOperation(), getNumOutputs());
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
      cast<DynamicBroadcastInDimOp>(getOperation()), p,
      [](DynamicBroadcastInDimOp op,
         OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.broadcast_dimensionsAttrName(),
                               op.broadcast_dimensions());
        return {op.broadcast_dimensionsAttrName()};
      });
}

LogicalResult DynamicBroadcastInDimOp::verify() {
  return verifyDestinationStyleOp(getOperation(), getNumOutputs());
}

Value DynamicBroadcastInDimOp::fuse(Location loc, Value subset,
                                    OpBuilder &builder) {
  Type subsetTy = subset.getType();
  auto operandTy = operand().getType().cast<RankedTensorType>();
  auto resultTy = getType(0).cast<RankedTensorType>();
  int64_t operandRank = operandTy.getRank();

  // Create the needed constants only once.
  DenseMap<uint64_t, Value> localIndexConstants;
  auto getIndexConstant = [&](uint64_t c) -> Value {
    auto it = localIndexConstants.find(c);
    if (it != localIndexConstants.end()) return it->second;
    auto cst = builder.create<arith::ConstantIndexOp>(loc, c);
    localIndexConstants[c] = cst;
    return cst;
  };

  // Materialize operand space.
  auto operandSpaceTy = builder.getType<gml_st::TileType>(operandTy.getShape());
  auto dynamicDims = tensor::createDynamicDimValues(builder, loc, operand());
  auto staticDims = builder.getI64ArrayAttr(operandTy.getShape());
  Value operandSpace = builder.create<gml_st::SpaceOp>(loc, operandSpaceTy,
                                                       dynamicDims, staticDims);

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

  // Collapse the subset to operate only on corresponding dimensions.
  // TODO(frgossen): Only generate this when needed.
  auto collapsedSubset = builder.create<gml_st::DropDimsOp>(
      loc, subset, broadcast_dimensionsAttr());

  // Find the expanding dimensions. If corresponding operand and result
  // dimensions are different then the dimension is expanding.
  // TODO(frgossen): Use info from known expanding and known non-expanding
  // dimensions here.
  SmallVector<Value> operandExpandingDims;
  for (const auto &it : llvm::enumerate(broadcast_dimensions())) {
    auto operandDim = operandDims[it.index()];
    auto resultDim = builder.create<tensor::DimOp>(
        loc, init(), getIndexConstant(it.value()));
    operandExpandingDims.push_back(builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, operandDim, resultDim));
  }

  // Compute operand offsets, which are needed for tile and point subsets.
  auto staticOffsets = builder.getI64ArrayAttr(
      SmallVector<int64_t>(operandRank, ShapedType::kDynamicStrideOrOffset));
  SmallVector<Value> offsets;
  Value zero = getIndexConstant(0);
  for (int i = 0; i < operandRank; ++i) {
    Value isExpanding = operandExpandingDims[i];
    Value collapsedSubsetOffset = builder.create<gml_st::OffsetOp>(
        loc, collapsedSubset, getIndexConstant(i));
    offsets.push_back(builder.create<arith::SelectOp>(loc, isExpanding, zero,
                                                      collapsedSubsetOffset));
  }

  // If the regarded subset is of point type, we can already construct the
  // operand point and materialize it.
  if (auto pointTy = subsetTy.dyn_cast<gml_st::PointType>()) {
    auto operandPoint = builder.create<gml_st::PointOp>(
        loc, pointTy, operandSpace, offsets, staticOffsets);
    return builder.create<gml_st::MaterializeOp>(
        loc, operandTy.getElementType(), operand(), operandPoint);
  }

  // If the regarded subset is of tile type, we still need the operand tile
  // sizes to materialize a fused broadcast.
  if (auto tileTy = subsetTy.dyn_cast<gml_st::TileType>()) {
    // Compute operand tile sizes.
    auto staticTileSizes = builder.getI64ArrayAttr(
        SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
    SmallVector<Value> tileSizes;
    Value one = getIndexConstant(1);
    for (int i = 0; i < operandRank; ++i) {
      Value isExpanding = operandExpandingDims[i];
      Value tileSize = builder.create<gml_st::SizeOp>(loc, collapsedSubset,
                                                      getIndexConstant(i));
      tileSizes.push_back(
          builder.create<arith::SelectOp>(loc, isExpanding, one, tileSize));
    }

    // Create operand tile.
    auto staticTileStrides =
        builder.getI64ArrayAttr(SmallVector<int64_t>(operandRank, 1));
    SmallVector<Value> tileStrides = {};
    auto operandTileTy = builder.getType<gml_st::TileType>(
        SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
    auto operandTile = builder.create<gml_st::TileOp>(
        loc, operandTileTy, operandSpace, offsets, tileSizes, tileStrides,
        staticOffsets, staticTileSizes, staticTileStrides);

    // Materialize operand subsets.
    Value tiledInit =
        builder.create<gml_st::MaterializeOp>(loc, init(), subset);
    Value tiledOperand =
        builder.create<gml_st::MaterializeOp>(loc, operand(), operandTile);

    // Finally, materialize tiled broadcast.
    auto tiledResultTy =
        RankedTensorType::get(tileTy.getShape(), resultTy.getElementType());
    return builder
        .create<DynamicBroadcastInDimOp>(
            loc, TypeRange{tiledResultTy}, tiledOperand, tiledInit,
            broadcast_dimensionsAttr(), known_expanding_dimensionsAttr(),
            known_nonexpanding_dimensionsAttr())
        .getResult(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

ParseResult ScatterOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void ScatterOp::print(OpAsmPrinter &p) {
  printDstStyleOp(cast<ScatterOp>(getOperation()), p);
}

LogicalResult ScatterOp::verify() {
  return verifyDestinationStyleOp(getOperation(), getNumOutputs());
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

ParseResult GatherOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(parser, result);
}

void GatherOp::print(OpAsmPrinter &p) {
  printDstStyleOp(cast<GatherOp>(getOperation()), p);
}

LogicalResult GatherOp::verify() {
  return verifyDestinationStyleOp(getOperation(), getNumOutputs());
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp(
      parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
        return parseDenseI64ArrayAttr(parser, attributes, "permutation");
      });
}

void TransposeOp::print(OpAsmPrinter &p) {
  printDstStyleOp<TransposeOp>(
      cast<TransposeOp>(getOperation()), p,
      [](TransposeOp op, OpAsmPrinter &p) -> SmallVector<StringRef> {
        printDenseI64ArrayAttr(p, op.permutationAttrName(), op.permutation());
        return {op.permutationAttrName()};
      });
}

bool isValidPermutation(ArrayRef<int64_t> permutation) {
  SmallVector<bool> seen(permutation.size(), false);
  for (auto p : permutation) {
    // Verify that each element is in [0..n-1] range and is present only once.
    if (p < 0 || p >= permutation.size() || seen[p]) return false;

    seen[p] = true;
  }
  return true;
}

LogicalResult TransposeOp::verify() {
  ArrayRef<int64_t> permutationRef = permutation();

  if (!isValidPermutation(permutationRef))
    return emitOpError("permutation is not valid");

  auto inputType = input().getType().cast<ShapedType>();
  auto initType = init().getType().cast<ShapedType>();

  int64_t rank = inputType.getRank();

  if (rank != initType.getRank())
    return emitOpError() << "input rank " << rank
                         << " does not match init rank " << initType.getRank();

  if (rank != permutationRef.size())
    return emitOpError() << "size of permutation " << permutationRef.size()
                         << " does not match the argument rank " << rank;

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  for (size_t i = 0; i < rank; ++i) {
    int64_t inputDim = inputDims[permutationRef[i]];
    int64_t initDim = initDims[i];

    if (inputDim != ShapedType::kDynamicSize &&
        initDim != ShapedType::kDynamicSize && inputDim != initDim) {
      return emitOpError() << "dim(result, " << i << ") = " << initDim
                           << " doesn't match dim(input, permutation[" << i
                           << "]) = " << inputDim;
    }
  }

  return verifyDestinationStyleOp(getOperation(), getNumOutputs());
}

}  // namespace thlo
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.cc.inc"

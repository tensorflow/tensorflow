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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace {

void printShapeTypeDimensionsList(AsmPrinter &printer,
                                  ArrayRef<int64_t> integers) {
  llvm::interleave(
      integers, printer,
      [&](int64_t val) {
        if (val == ShapedType::kDynamicSize)
          printer << '?';
        else
          printer << val;
      },
      "x");
}

ParseResult parseShapeTypeDimensionsList(
    AsmParser &parser, FailureOr<SmallVector<int64_t>> &dims) {
  SmallVector<int64_t> vals;
  if (failed(parser.parseDimensionList(vals, /*allowDynamic=*/true,
                                       /*withTrailingX=*/false))) {
    return failure();
  }
  dims = vals;
  return success();
}

ParseResult parseAssignmentListWithTypes(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &lhs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &rhs,
    SmallVectorImpl<Type> &types) {
  auto parseElt = [&]() -> ParseResult {
    if (parser.parseOperand(lhs.emplace_back(), /*allowResultNumber=*/false) ||
        parser.parseEqual() || parser.parseOperand(rhs.emplace_back()) ||
        parser.parseColon() || parser.parseType(types.emplace_back())) {
      return failure();
    }
    return success();
  };
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt);
}

}  // namespace
}  // namespace mlir

// Generated dialect definitions.
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_dialect.cc.inc"

// Generated type classes.
#define GET_TYPEDEF_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"

// Generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_attrs.cc.inc"

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// GmlStDialect
//===----------------------------------------------------------------------===//

void GmlStDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_attrs.cc.inc"
      >();
}

// Helper function to ensure index types for some attrbutes when folding.
static OpFoldResult ensureIndexTypeForAttribute(OpFoldResult foldResult) {
  if (foldResult.is<Attribute>()) {
    auto attr = foldResult.get<Attribute>().dyn_cast<IntegerAttr>();
    if (!attr.getType().isa<IndexType>()) {
      Builder b(attr.getContext());
      return b.getIndexAttr(attr.getInt());
    }
  }
  return foldResult;
}

Operation *GmlStDialect::materializeConstant(OpBuilder &builder, Attribute attr,
                                             Type type, Location loc) {
  if (type.isa<IndexType>()) {
    int64_t intValue = attr.cast<IntegerAttr>().getInt();
    return builder.create<arith::ConstantIndexOp>(loc, intValue);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

static FailureOr<Type> inferReturnType(ShapedType sourceType, Type setType) {
  if (auto tileType = setType.dyn_cast<TileType>()) {
    return sourceType.clone(tileType.getShape(), sourceType.getElementType());
  }
  return failure();
}

void MaterializeOp::build(OpBuilder &builder, OperationState &result,
                          Value source, Value set) {
  auto sourceType = source.getType().cast<ShapedType>();
  auto resultTypeOr = inferReturnType(sourceType, set.getType());
  assert(succeeded(resultTypeOr) && "could not infer result type");
  build(builder, result, *resultTypeOr, source, set);
}

LogicalResult verifyCompatibleExtractedSubset(Operation *op,
                                              ShapedType shapedType,
                                              Type extractedType,
                                              Type setType) {
  auto sourceRank = shapedType.getRank();
  auto elementType = shapedType.getElementType();

  // If the result is a scalar, check that the tile had a single element.
  if (!extractedType.isa<ShapedType>()) {
    auto tileType = setType.cast<TileType>();
    if (extractedType != elementType) {
      return op->emitOpError("expected the result type ")
             << extractedType << " to match source element type "
             << elementType;
    }
    if (tileType.hasStaticShape() && tileType.getNumElements() == 1)
      return success();

    return op->emitOpError("expected tile type ")
           << tileType << " to have a single element shape";
  }

  // If the result is a shaped type, compare with the inferred type.
  auto extractedShapedType = extractedType.cast<ShapedType>();
  auto tileType = setType.cast<TileType>();
  int64_t tileRank = tileType.getRank();
  if (tileRank != sourceRank) {
    return op->emitOpError("expected source rank = ")
           << sourceRank << " to match tile rank = " << tileRank;
  }

  auto inferredType =
      shapedType.clone(tileType.getShape(), shapedType.getElementType());
  if (extractedShapedType != inferredType) {
    return op->emitOpError("expected result type = ")
           << extractedShapedType
           << " to match the inferred type = " << inferredType;
  }

  return success();
}

LogicalResult MaterializeOp::verify() {
  return verifyCompatibleExtractedSubset(getOperation(), getSource().getType(),
                                         getType(), getSet().getType());
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange lowerBounds, ValueRange upperBounds,
                   ValueRange steps, ValueRange inputs, ValueRange outputs,
                   ArrayAttr iteratorTypes,
                   function_ref<void(OpBuilder &, Location, ValueRange,
                                     ValueRange, ValueRange)>
                       bodyBuilderFn) {
  build(builder, result, lowerBounds, upperBounds, steps, inputs, outputs,
        iteratorTypes, llvm::None, bodyBuilderFn);
}

void LoopOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange lowerBounds, ValueRange upperBounds,
                   ValueRange steps, ValueRange inputs, ValueRange outputs,
                   ArrayAttr iteratorTypes,
                   Optional<ArrayAttr> distributionTypes,
                   function_ref<void(OpBuilder &, Location, ValueRange,
                                     ValueRange, ValueRange)>
                       bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(inputs);
  result.addOperands(outputs);
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBounds.size()),
                                    static_cast<int32_t>(upperBounds.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(inputs.size()),
                                    static_cast<int32_t>(outputs.size())}));
  result.addAttribute(getIteratorTypesAttrStrName(), iteratorTypes);

  if (distributionTypes.has_value())
    result.addAttribute(getDistributionTypesAttrStrName(),
                        distributionTypes.value());

  // Add output types for `RankedTensorType` output arguments.
  for (Value output : outputs) {
    Type outputType = output.getType();
    if (outputType.isa<RankedTensorType>()) result.addTypes(outputType);
  }

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = steps.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIVs, result.location);
  for (Value input : inputs) {
    argTypes.push_back(input.getType());
    argLocs.push_back(input.getLoc());
  }
  for (Value output : outputs) {
    argTypes.push_back(output.getType());
    argLocs.push_back(output.getLoc());
  }
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().slice(numIVs, inputs.size()),
                  bodyBlock->getArguments().take_back(outputs.size()));
    LoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void LoopOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << getLowerBound() << ") to ("
    << getUpperBound() << ") step (" << getStep() << ")";

  if (!getInputs().empty()) {
    p << " ins (";
    llvm::interleaveComma(llvm::zip(getRegionInputArgs(), getInputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }
  if (!getOutputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(llvm::zip(getRegionOutputArgs(), getOutputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }

  if (llvm::any_of(getIteratorTypes(), [](Attribute attr) {
        return attr.cast<IteratorTypeAttr>().getValue() !=
               utils::IteratorType::parallel;
      }))
    p << " iterators" << getIteratorTypes();

  if (getDistributionTypes().has_value())
    p << " distribution" << getDistributionTypes().value();

  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{LoopOp::getOperandSegmentSizeAttr(),
                       LoopOp::getIteratorTypesAttrName(),
                       LoopOp::getDistributionTypesAttrName()});
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::UnresolvedOperand, 4> ivs;
  if (parser.parseOperandList(ivs, OpAsmParser::Delimiter::Paren,
                              /*allowResultNumber=*/false))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Parse input tensors.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs, inputRegionArgs;
  SmallVector<Type, 4> inputTypes;
  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    SMLoc inputsOperandsLoc = parser.getCurrentLocation();

    if (parseAssignmentListWithTypes(parser, inputRegionArgs, inputs,
                                     inputTypes))
      return failure();

    if (parser.resolveOperands(inputs, inputTypes, inputsOperandsLoc,
                               result.operands))
      return failure();
  }

  // Parse output tensors.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs, outputRegionArgs;
  SmallVector<Type, 4> outputTypes;
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    SMLoc outputsOperandsLoc = parser.getCurrentLocation();

    if (parseAssignmentListWithTypes(parser, outputRegionArgs, outputs,
                                     outputTypes))
      return failure();

    if (parser.resolveOperands(outputs, outputTypes, outputsOperandsLoc,
                               result.operands))
      return failure();
    for (Type outputType : outputTypes)
      if (outputType.isa<RankedTensorType>()) result.addTypes(outputType);
  }

  Attribute iterTypes;
  if (succeeded(parser.parseOptionalKeyword("iterators"))) {
    if (parser.parseAttribute(iterTypes)) return failure();
  } else {
    // Set all loop iterator types to "parallel" if they are not printed in IR.
    auto parallelIter =
        builder.getAttr<IteratorTypeAttr>(utils::IteratorType::parallel);
    iterTypes = builder.getArrayAttr(
        SmallVector<Attribute, 4>(ivs.size(), parallelIter));
  }

  result.addAttribute(LoopOp::getIteratorTypesAttrStrName(), iterTypes);

  if (succeeded(parser.parseOptionalKeyword("distribution"))) {
    Attribute distributionTypes;
    if (failed(parser.parseAttribute(distributionTypes))) return failure();
    result.addAttribute(LoopOp::getDistributionTypesAttrStrName(),
                        distributionTypes);
  }

  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lower.size()),
                                    static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(inputs.size()),
                                    static_cast<int32_t>(outputs.size())}));

  // Parse the body.
  Region *body = result.addRegion();

  SmallVector<Type, 4> regionTypes(ivs.size(), builder.getIndexType());
  regionTypes.append(inputTypes);
  regionTypes.append(outputTypes);

  SmallVector<OpAsmParser::UnresolvedOperand, 4> regionOperands(ivs);
  regionOperands.append(inputRegionArgs);
  regionOperands.append(outputRegionArgs);

  SmallVector<OpAsmParser::Argument, 4> regionArgs;

  for (auto argAndType : llvm::zip(regionOperands, regionTypes)) {
    auto &arg = regionArgs.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }

  if (parser.parseRegion(*body, regionArgs)) return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  return success();
}

Region &LoopOp::getLoopBody() { return getRegion(); }

LogicalResult LoopOp::verify() {
  // Check if iterator types are provided for every loop dimension.
  if (getIteratorTypes().size() != getNumLoops())
    return emitOpError("expected iterator types array attribute size = ")
           << getIteratorTypes().size()
           << " to match the number of loops = " << getNumLoops();

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(getInputs(), getRegionInputArgs()))) {
    Value input, inputRegionArg;
    unsigned index = item.index();
    std::tie(input, inputRegionArg) = item.value();
    if (input.getType() != inputRegionArg.getType())
      return emitOpError("expected input arg ")
             << index << " with type = " << input.getType()
             << " to match region arg " << index + getNumLoops()
             << " type = " << inputRegionArg.getType();
  }

  // Check if types of output arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(getOutputs(), getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType())
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg "
             << index + getNumLoops() + getInputs().size()
             << " type = " << outputRegionArg.getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoopLikeOp
//===----------------------------------------------------------------------===//

namespace {

ParseResult parseForOpOutputArgs(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &regionOperands,
    SmallVectorImpl<Type> &regionTypes, int32_t *outputCount) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs, outputRegionArgs;
  SmallVector<Type, 4> outputTypes;

  auto parseElt = [&]() -> ParseResult {
    if (parser.parseOperand(outputRegionArgs.emplace_back(),
                            /*allowResultNumber=*/false) ||
        parser.parseEqual()) {
      return failure();
    }
    if (parser.parseOperand(outputs.emplace_back()) || parser.parseColon() ||
        parser.parseType(outputTypes.emplace_back())) {
      return failure();
    }
    *outputCount = outputs.size();
    return success();
  };
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    SMLoc loc = parser.getCurrentLocation();

    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt))
      return failure();
    if (parser.resolveOperands(outputs, outputTypes, loc, result.operands))
      return failure();
  }
  regionOperands.append(outputRegionArgs);
  regionTypes.append(outputTypes);
  return success();
}

}  // namespace

template <typename LoopTy>
ParseResult parseLoopLikeOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::UnresolvedOperand, 4> ivs;
  if (parser.parseOperandList(ivs, OpAsmParser::Delimiter::Paren,
                              /*allowResultNumber=*/false))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<int32_t> segmentSizes{static_cast<int32_t>(lower.size()),
                                    static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(steps.size())};

  // Parse distribution type (only for ParallelOp)
  if (std::is_same<LoopTy, ParallelOp>::value) {
    if (succeeded(parser.parseOptionalKeyword("distribution"))) {
      StringAttr distributionType;
      if (parser.parseLParen() || parser.parseAttribute(distributionType) ||
          parser.parseRParen())
        return failure();
      result.addAttribute(ParallelOp::getDistributionTypeAttrName(result.name),
                          distributionType);
    }
  }

  // Parse the output tensors (only for ForOp) and the body.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> regionOperands(ivs);
  SmallVector<Type, 4> regionTypes(ivs.size(), builder.getIndexType());

  if (std::is_same<LoopTy, ForOp>::value) {
    int32_t outputCount = 0;
    if (parseForOpOutputArgs(parser, result, regionOperands, regionTypes,
                             &outputCount))
      return failure();
    segmentSizes.push_back(outputCount);
  }

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  for (auto argAndType : llvm::zip(regionOperands, regionTypes)) {
    auto &arg = regionArgs.emplace_back();
    std::tie(arg.ssaName, arg.type) = argAndType;
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // Parser result types.
  if (parser.parseOptionalColonTypeList(result.types)) return failure();

  // Add segment sizes.
  result.addAttribute(LoopTy::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));

  return success();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

Region &ParallelOp::getLoopBody() { return getRegion(); }

SetYieldOp ParallelOp::getTerminator() {
  return cast<SetYieldOp>(getBody()->getTerminator());
}

LogicalResult ParallelOp::verify() { return success(); }

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    Optional<StringAttr> distributionType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addTypes(resultTypes);
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBounds.size()),
                                    static_cast<int32_t>(upperBounds.size()),
                                    static_cast<int32_t>(steps.size())}));

  if (distributionType.has_value())
    result.addAttribute(getDistributionTypeAttrName(result.name),
                        distributionType.value());

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIvs = steps.size();
  SmallVector<Type, 8> argTypes(numIvs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIvs, result.location);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIvs));
    ParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void ParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << getLowerBound() << ") to ("
    << getUpperBound() << ") step (" << getStep() << ") ";

  if (getDistributionType().has_value())
    p << "distribution (" << getDistributionTypeAttr() << ") ";

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{ParallelOp::getOperandSegmentSizeAttr(),
                       getDistributionTypeAttrName()});

  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleave(getResultTypes(), p, ", ");
  }
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLoopLikeOp<ParallelOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

Region &ForOp::getLoopBody() { return getRegion(); }

SetYieldOp ForOp::getTerminator() {
  return cast<SetYieldOp>(getBody()->getTerminator());
}

LogicalResult ForOp::verify() {
  // Check if types of output arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(getOutputs(), getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType()) {
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg " << index + getNumLoops()
             << " type = " << outputRegionArg.getType();
    }
    if (getTerminator().getDstOperand(index)->get() != outputRegionArg) {
      return getTerminator().emitOpError("expected output block argument ")
             << index << " to match set_yield destination";
    }
  }
  return success();
}

void ForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    ValueRange outputs,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(outputs);
  result.addTypes(resultTypes);
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBounds.size()),
                                    static_cast<int32_t>(upperBounds.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(outputs.size())}));

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIvs = steps.size();
  SmallVector<Type, 8> argTypes(numIvs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIvs, result.location);
  for (Value output : outputs) {
    argTypes.push_back(output.getType());
    argLocs.push_back(output.getLoc());
  }
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIvs),
                  bodyBlock->getArguments().take_back(outputs.size()));
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void ForOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << getLowerBound() << ") to ("
    << getUpperBound() << ") step (" << getStep() << ")";

  if (!getOutputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(
        llvm::zip(getRegionOutputArgs(), getOutputs()), p, [&](auto it) {
          Value outputRegionArg, output;
          std::tie(outputRegionArg, output) = it;
          p << outputRegionArg << " = " << output << ": " << output.getType();
        });
    p << ")";
  }

  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{ForOp::getOperandSegmentSizeAttr()});

  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleave(getResultTypes(), p, ", ");
  }
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLoopLikeOp<ForOp>(parser, result);
}

namespace {

static constexpr int64_t kNoMatch = -1;

// Folds away LoopOp inputs if they have no uses within the body.
//
// Example:
//
// %0 = gml_st.loop ...  ins (%in_ = %in: tensor<...>,
//                                  %in_buf_ = %in_buf: memref<...>) {...}
// Becomes
//
// gml_st.loop ...  ins (%in_buf_ = %in_buf: memref<...>) {...}
struct LoopInputsFolder : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newInputs, regionInputTensorArgs;
    // Store ids of the corresponding old and new input operands.
    SmallVector<int64_t, 2> oldInputIdToNew(loop.getInputs().size(), kNoMatch);
    for (const auto &en : llvm::enumerate(
             llvm::zip(loop.getInputs(), loop.getRegionInputArgs()))) {
      Value in, bbArg;
      size_t index = en.index();
      std::tie(in, bbArg) = en.value();
      if (!bbArg.use_empty()) {
        oldInputIdToNew[index] = newInputs.size();
        newInputs.push_back(in);
      }
    }
    if (newInputs.size() == loop.getInputs().size()) return failure();
    Location loc = loop.getLoc();
    auto newLoop = rewriter.create<LoopOp>(
        loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
        newInputs, loop.getOutputs(), loop.getIteratorTypes(),
        loop.getDistributionTypes());

    // Clone the region.
    BlockAndValueMapping bvm;
    bvm.map(loop.getInductionVars(), newLoop.getInductionVars());
    bvm.map(loop.getRegionOutputArgs(), newLoop.getRegionOutputArgs());
    for (const auto &en : llvm::enumerate(oldInputIdToNew))
      if (en.value() != kNoMatch)
        bvm.map(loop.getRegionInputArgs()[en.index()],
                newLoop.getRegionInputArgs()[en.value()]);
    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(newLoop.getBody(), rewriter.getListener());
    for (auto &op : *loop.getBody()) innerBuilder.clone(op, bvm);
    rewriter.replaceOp(loop, newLoop.getResults());

    return success();
  }
};

}  // namespace

/// A simple, conservative analysis to determine if the loop is shape
/// conserving. I.e., the type of the arg-th yielded value is the same as the
/// type of the corresponding basic block argument of the loop.
/// Note: This function handles only simple cases. Expand as needed.
static bool isShapePreserving(LoopOp loopOp, int64_t arg) {
  auto yieldOp = cast<YieldOp>(loopOp.getLoopBody().front().getTerminator());
  if (yieldOp.getValues().empty())
    // Loop either has no outputs or is a "memref-based version". In either
    // case, the loop is shape conserving.
    return true;
  assert(arg < static_cast<int64_t>(yieldOp.getValues().size()) &&
         "arg is out of bounds");
  Value value = yieldOp.getValues()[arg];
  while (value) {
    if (value == loopOp.getRegionOutputArgs()[arg]) return true;
    OpResult opResult = value.dyn_cast<OpResult>();
    if (!opResult) return false;

    using tensor::InsertSliceOp;
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
                .template Case<InsertSliceOp>(
                    [&](InsertSliceOp op) { return op.getDest(); })
                .template Case<LoopOp>([&](LoopOp loopOp) {
                  return isShapePreserving(loopOp, opResult.getResultNumber())
                             ? loopOp.getOutputs()[opResult.getResultNumber()]
                             : Value();
                })
                .Default([&](auto /*op*/) { return Value(); });
  }
  return false;
}

namespace {

/// Fold dim(x) where `x` is an input/output argument of a LoopOp block
/// to dim(y) where `y` is the initial input/output value of the argument.
///
/// E.g.:
/// %y = ... : tensor<...>
/// gml_st.loop ... ins(%x = %y : tensor<...>) {
///   tensor.dim %x, %c0 : tensor<...>
/// }
///
/// is folded to:
/// %y = ... : tensor<...>
/// gml_st.loop ... ins(%x = %y : tensor<...>) {
///   tensor.dim %y, %c0 : tensor<...>
/// }
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the yielded value (in case of outputs) does not change with loop iterations.
template <typename OpTy>
struct DimOfLoopInsOutsFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const final {
    auto src = dimOp.getSource().template dyn_cast<BlockArgument>();
    if (!src) return failure();
    auto loopOp = dyn_cast<LoopOp>(src.getOwner()->getParent()->getParentOp());
    if (!loopOp) return failure();
    unsigned numLoops = loopOp.getNumLoops();
    unsigned numInputArgs = loopOp.getRegionInputArgs().size();
    if (src.getArgNumber() >= numInputArgs + numLoops &&
        !isShapePreserving(loopOp,
                           src.getArgNumber() - numInputArgs - numLoops))
      return failure();

    auto inputArgs = loopOp.getRegionInputArgs();
    auto it1 = llvm::find(inputArgs, src);
    if (it1 != inputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.getSourceMutable().assign(
            loopOp.getInputs()[it1 - inputArgs.begin()]);
      });
      return success();
    }

    auto outputArgs = loopOp.getRegionOutputArgs();
    auto it2 = llvm::find(outputArgs, src);
    if (it2 != outputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.getSourceMutable().assign(
            loopOp.getOutputs()[it2 - outputArgs.begin()]);
      });
      return success();
    }

    return failure();
  }
};

/// Fold dim(r) where `r` is the result of a LoopOp to dim(y) where `y`
/// is the initial output value of the loop.
///
/// E.g.:
/// %y = ... : tensor<...>
/// %r = gml_st.loop ... outs(%i = %y : tensor<...>) {
///   ...
/// }
/// %0 = tensor.dim %r, %c0 : tensor<...>
///
/// is folded to:
/// %y = ... : tensor<...>
/// gml_st.loop ... outs(%i = %y : tensor<...>) {
///   ...
/// }
/// %0 = tensor.dim %y, %c0 : tensor<...>
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the yielded value (in case of outputs) does not change with loop iterations.
template <typename OpTy>
struct DimOfLoopResultFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const final {
    auto loopOp = dimOp.getSource().template getDefiningOp<LoopOp>();
    if (!loopOp) return failure();
    auto opResult = dimOp.getSource().template cast<OpResult>();
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(loopOp, resultNumber)) return failure();
    rewriter.updateRootInPlace(dimOp, [&]() {
      dimOp.getSourceMutable().assign(loopOp.getOutputs()[resultNumber]);
    });
    return success();
  }
};

// Folds away LoopOp output tensors when the following conditions are met:
// * result of `gml_st.loop` has no uses
// * output tensor is the argument of `gml_st.yield`
//
// Example:
//
// %0 = gml_st.loop ...  outs (%o_ = %out: tensor<...>,
//                                   %obuf_ = %out_buf: memref<...>) {
//   ...
//   gml_st.yield %o_ : tensor ...
// }
//
// Becomes
//
// gml_st.loop ...  outs (%obuf_ = %out_buf: memref<...>) {
//   ...
//   gml_st.yield
// }
struct LoopResultsFolder : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &rewriter) const final {
    if (loop.getNumResults() == 0) return failure();

    Block *block = loop.getBody();
    auto yieldOp = cast<YieldOp>(block->getTerminator());

    // Match the pattern and collect output buffers that will replace the output
    // tensors and also the ops that will be ignored when cloning the body.
    SmallVector<Value, 2> newOutputOperands, newYieldArgs;
    int resultId = 0;
    // Store ids of the corresponding old and new output operands.
    SmallVector<int64_t, 2> oldOutputIdToNew(loop.getOutputs().size(),
                                             kNoMatch);
    // Store ids of the corresponding old and new results.
    SmallVector<int64_t, 2> oldResultIdToNew(loop.getNumResults(), kNoMatch);
    SmallVector<Value, 2> resultReplacement(loop.getNumResults());
    for (const auto &en : llvm::enumerate(
             llvm::zip(loop.getOutputs(), loop.getRegionOutputArgs()))) {
      size_t index = en.index();
      Value out = std::get<0>(en.value());
      Value outRegionArg = std::get<1>(en.value());

      if (!out.getType().isa<RankedTensorType>()) {
        oldOutputIdToNew[index] = newOutputOperands.size();
        newOutputOperands.push_back(out);
        continue;
      }
      Value result = loop.getResult(resultId);
      Value yieldArg = yieldOp.getOperand(resultId);
      if (yieldArg != outRegionArg || !result.use_empty()) {
        oldOutputIdToNew[index] = newOutputOperands.size();
        oldResultIdToNew[resultId] = newYieldArgs.size();
        resultReplacement[resultId] = out;
        newOutputOperands.push_back(out);
        newYieldArgs.push_back(yieldArg);
      }
      ++resultId;
    }
    if (newOutputOperands.size() == loop.getOutputs().size()) return failure();

    Location loc = loop.getLoc();
    auto newLoop = rewriter.create<LoopOp>(
        loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
        loop.getInputs(), newOutputOperands, loop.getIteratorTypes(),
        loop.getDistributionTypes());

    // Clone the region.
    BlockAndValueMapping bvm;
    bvm.map(loop.getInductionVars(), newLoop.getInductionVars());
    bvm.map(loop.getRegionInputArgs(), newLoop.getRegionInputArgs());
    for (const auto &en : llvm::enumerate(oldOutputIdToNew)) {
      if (en.value() != kNoMatch)
        bvm.map(loop.getRegionOutputArgs()[en.index()],
                newLoop.getRegionOutputArgs()[en.value()]);
      else
        bvm.map(loop.getRegionOutputArgs()[en.index()],
                loop.getOutputs()[en.index()]);
    }
    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(newLoop.getBody(), rewriter.getListener());
    for (auto &op : loop.getBody()->without_terminator())
      innerBuilder.clone(op, bvm);
    innerBuilder.create<YieldOp>(
        loc, llvm::to_vector<2>(llvm::map_range(
                 newYieldArgs, [&](Value arg) { return bvm.lookup(arg); })));

    for (const auto &en : llvm::enumerate(oldResultIdToNew))
      if (en.value() != kNoMatch)
        resultReplacement[en.index()] = newLoop.getResult(en.value());
    rewriter.replaceOp(loop, resultReplacement);

    return success();
  }
};

/// Pull `gml_st.loop` input/output arguments that are produced by
/// `tensor.cast` ops inside `gml_st.loop`:
///
/// ```
///   %in = tensor.cast %t0 : tensor<32x1024xf32> to tensor<?x?xf32>
///   %out = tensor.cast %t1 : tensor<32x1024xf32> to tensor<?x?xf32>
///   %result = gml_st.loop %i = %c0 to %c1024 step %c32
///       ins (%in_ = %in: tensor<?x?xf32>)
///       outs (%out_ = %out: tensor<?x?xf32>) {
///     %0 = call @do(%in_, %out_)
///       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
///     scf.yield %0 : tensor<?x?xf32>
///   }
///   %result_cast = tensor.cast %result
///     : tensor<?x?xf32> to tensor<32x1024xf32>
///   use_of(%result_cast)
/// ```
///
/// folds into:
//
/// ```
///   %result = gml_st.loop %i = %c0 to %c1024 step %c32
///       ins (%in_ = %t0: tensor<32x1024xf32>)
///       outs (%out_ = %t1: tensor<32x1024xf32>) {
///     %in_cast = tensor.cast %in_ : tensor<32x1024xf32> to tensor<?x?xf32>
///     %out_cast = tensor.cast %out_ : tensor<32x1024xf32> to tensor<?x?xf32>
///     %0 = call @do(%in_, %out_)
///       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
///     %0_cast = tensor.cast %0 : tensor<?x?xf32> to tensor<32x1024xf32>
///     scf.yield %0 : tensor<32x1024xf32>
///   }
///   use_of(%result)
/// ```
struct TensorCastOfLoopInsOutsFolder : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &rewriter) const override {
    CastOpsOfArgs inputCasts = findTensorCastOps(loop.getInputs());
    CastOpsOfArgs outputCasts = findTensorCastOps(loop.getOutputs());
    if (!inputCasts.castFound && !outputCasts.castFound) return failure();

    auto newLoop = rewriter.create<LoopOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), inputCasts.updatedArgs, outputCasts.updatedArgs,
        loop.getIteratorTypes(), loop.getDistributionTypes());

    rewriter.replaceOp(loop, insertCastsAndCloneBody(inputCasts, outputCasts,
                                                     loop, newLoop, rewriter));
    return success();
  }

 private:
  struct CastOpsOfArgs {
    SmallVector<tensor::CastOp, 4> ops;
    // Contains either old arguments or arguments of `tensor.cast`.
    SmallVector<Value, 4> updatedArgs;
    bool castFound = false;
  };

  // Scans through args to find what args are produced by `tensor.cast` ops.
  CastOpsOfArgs findTensorCastOps(ValueRange args) const {
    CastOpsOfArgs result;
    for (auto arg : args) {
      if (auto cast = arg.getDefiningOp<tensor::CastOp>()) {
        result.ops.push_back(cast);
        result.updatedArgs.push_back(cast.getSource());
        result.castFound = true;
        continue;
      }
      result.ops.push_back(nullptr);
      result.updatedArgs.push_back(arg);
    }
    return result;
  }

  SmallVector<Value, 4> insertCastsAndCloneBody(
      const CastOpsOfArgs &inputCasts, const CastOpsOfArgs &outputCasts,
      LoopOp loop, LoopOp newLoop, PatternRewriter &rewriter) const {
    auto loc = newLoop.getLoc();
    BlockAndValueMapping bvm;
    bvm.map(loop.getInductionVars(), newLoop.getInductionVars());

    auto innerBuilder =
        OpBuilder::atBlockEnd(newLoop.getBody(), rewriter.getListener());

    Value oldArg, newArg, yieldArg, result;
    tensor::CastOp argCast;

    // Map inputs, insert `tensor.cast` if necessary.
    for (auto item : llvm::zip(loop.getRegionInputArgs(),
                               newLoop.getRegionInputArgs(), inputCasts.ops)) {
      std::tie(oldArg, newArg, argCast) = item;
      if (!argCast) {
        bvm.map(oldArg, newArg);
        continue;
      }
      Value newCast =
          innerBuilder.create<tensor::CastOp>(loc, argCast.getType(), newArg);
      bvm.map(oldArg, newCast);
    }

    // Map outputs, insert `tensor.cast` and cast the loop results if necessary.
    SmallVector<Value, 4> newResults;
    rewriter.setInsertionPointAfter(newLoop);
    for (auto item :
         llvm::zip(loop.getRegionOutputArgs(), newLoop.getRegionOutputArgs(),
                   outputCasts.ops, newLoop.getResults())) {
      std::tie(oldArg, newArg, argCast, result) = item;
      if (!argCast) {
        bvm.map(oldArg, newArg);
        newResults.push_back(result);
        continue;
      }
      Value newCast =
          innerBuilder.create<tensor::CastOp>(loc, argCast.getType(), newArg);
      bvm.map(oldArg, newCast);

      newResults.push_back(
          rewriter.create<tensor::CastOp>(loc, argCast.getType(), result));
    }

    // Clone loop body.
    for (auto &op : loop.getBody()->without_terminator())
      innerBuilder.clone(op, bvm);

    // Cast yield arguments to the new type.
    SmallVector<Value, 4> yieldArgs =
        loop.getBody()->getTerminator()->getOperands();
    SmallVector<Value, 4> newYieldArgs;
    for (auto item : llvm::zip(yieldArgs, outputCasts.ops)) {
      std::tie(yieldArg, argCast) = item;
      if (!argCast) {
        newYieldArgs.push_back(bvm.lookup(yieldArg));
        continue;
      }
      newYieldArgs.push_back(innerBuilder.create<tensor::CastOp>(
          loc, argCast.getSource().getType(), bvm.lookup(yieldArg)));
    }
    innerBuilder.create<YieldOp>(loc, newYieldArgs);
    return newResults;
  }
};

/// Removes loops in which at least one lower/upper bound pair consists
/// of the same values - such loops have an empty iteration domain.
struct FoldEmptyLoops : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    for (auto dim : llvm::zip(op.getLowerBound(), op.getUpperBound())) {
      if (std::get<0>(dim) != std::get<1>(dim)) continue;
      SmallVector<Value> tensorOutputs;
      for (Value out : op.getOutputs()) {
        if (out.getType().isa<RankedTensorType>()) tensorOutputs.push_back(out);
      }
      rewriter.replaceOp(op, tensorOutputs);
      return success();
    }
    return failure();
  }
};

}  // namespace

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results
      .add<FoldEmptyLoops, LoopInputsFolder, LoopResultsFolder,
           DimOfLoopInsOutsFolder<tensor::DimOp>,
           DimOfLoopInsOutsFolder<memref::DimOp>,
           DimOfLoopResultFolder<tensor::DimOp>,
           DimOfLoopResultFolder<memref::DimOp>, TensorCastOfLoopInsOutsFolder>(
          context);
}

/// This is used for patterns of the form
/// ```
///    gml_st.loop(memrefcast(%src)) -> gml_st.loop(%src)
/// ```
/// It folds the source of the memref.cast into the root operation directly.
LogicalResult LoopOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  LoopOp op = *this;
  bool folded = false;
  Location loc = op->getLoc();

  Block *body = op.getBody();
  OpBuilder b = OpBuilder::atBlockBegin(body);

  // Update `input` and `output` operands and block arguments if necessary.
  // Operands list: [lbs, ubs, steps, inputs, outputs].
  // Block args list: [ivs, inputs, outputs].
  for (size_t operandIndex = op.getNumControlOperands(),
              bbArgIndex = op.getNumLoops(), e = op.getNumOperands();
       operandIndex < e; ++operandIndex, ++bbArgIndex) {
    OpOperand &operand = op->getOpOperand(operandIndex);

    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      BlockArgument newBbArg = body->insertArgument(
          bbArgIndex, castOp.getOperand().getType(), op.getLoc());
      BlockArgument oldBbArg = body->getArgument(newBbArg.getArgNumber() + 1);

      // Insert memref.cast back to the original type.
      oldBbArg.replaceAllUsesWith(
          b.create<memref::CastOp>(loc, oldBbArg.getType(), newBbArg));
      body->eraseArgument(oldBbArg.getArgNumber());

      folded = true;
    }
  }
  return success(folded);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  auto *parentOp = getOperation()->getParentOp();

  if (auto setYield = dyn_cast<SetYieldOp>(parentOp)) {
    if (getValues().size() != 1)
      return emitOpError(
          "expected a single argument for the terminator of accumulator "
          "region");
    return success();
  }
  auto loopOp = cast<LoopOp>(parentOp);
  // Check if output args with tensor types match results types.
  SmallVector<Value, 2> tensorOuts;
  llvm::copy_if(
      loopOp.getOutputs(), std::back_inserter(tensorOuts),
      [&](Value out) { return out.getType().isa<RankedTensorType>(); });
  if (tensorOuts.size() != getValues().size())
    return emitOpError("expected number of tensor output args = ")
           << tensorOuts.size()
           << " to match the number of yield operands = " << getValues().size();

  TypeRange tensorTypes{ValueRange{tensorOuts}};
  for (auto &item :
       llvm::enumerate(llvm::zip(tensorTypes, getOperandTypes()))) {
    Type outType, resultType;
    unsigned index = item.index();
    std::tie(outType, resultType) = item.value();
    if (outType != resultType)
      return emitOpError("expected yield operand ")
             << index << " with type = " << resultType
             << " to match output arg type = " << outType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SpaceOp
//===----------------------------------------------------------------------===//

void SpaceOp::build(OpBuilder &builder, OperationState &result,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<NamedAttribute> attrs) {
  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticSizes;
  for (OpFoldResult size : sizes)
    dispatchIndexOpFoldResult(size, dynamicSizes, staticSizes,
                              ShapedType::kDynamicSize);
  build(builder, result, TileType::get(builder.getContext(), staticSizes),
        dynamicSizes, builder.getI64ArrayAttr(staticSizes));
  result.addAttributes(attrs);
}

LogicalResult SpaceOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SpaceOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(
      llvm::map_range(adaptor.getStaticSizes(), [&](const Attribute &val) {
        return val.cast<IntegerAttr>().getValue().getSExtValue();
      }));
  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult SpaceOp::verify() {
  auto resultTy = getType().cast<TileType>();
  return mlir::verifyListOfOperandsOrIntegers(
      getOperation(), "size", resultTy.getShape().size(), getStaticSizes(),
      getDynamicSizes(), ShapedType::isDynamic);
}

unsigned SpaceOp::getNumDynamicEntriesUpToIdx(unsigned idx) {
  return std::count_if(getStaticSizes().begin(), getStaticSizes().begin() + idx,
                       [&](const mlir::Attribute size) {
                         return mlir::ShapedType::isDynamic(
                             size.cast<mlir::IntegerAttr>().getInt());
                       });
}

mlir::Value SpaceOp::getDynamicSize(unsigned idx) {
  auto numDynamic = getNumDynamicEntriesUpToIdx(idx);
  return getDynamicSizes()[numDynamic];
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

void TileOp::build(OpBuilder &b, OperationState &result, Value superset,
                   ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                   ArrayRef<OpFoldResult> strides,
                   ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  auto tileType = TileType::get(b.getContext(), staticSizes);
  build(b, result, tileType, superset, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

LogicalResult TileOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Derive result shape.
  TileOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(
      llvm::map_range(adaptor.getStaticSizes(), [&](const auto &size) {
        return size.template dyn_cast<mlir::IntegerAttr>()
            .getValue()
            .getSExtValue();
      }));

  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult TileOp::verify() {
  auto supersetTy = getSuperset().getType().cast<TileType>();
  auto rank = supersetTy.getShape().size();
  if (failed(mlir::verifyListOfOperandsOrIntegers(getOperation(), "size", rank,
                                                  getStaticSizes(), getSizes(),
                                                  ShapedType::isDynamic))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "offset", rank, getStaticOffsets(), getOffsets(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "stride", rank, getStaticStrides(), getStrides(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  for (auto it : llvm::zip(supersetTy.getShape(), getStaticOffsets(),
                           getStaticSizes(), getStaticStrides())) {
    auto offset =
        std::get<1>(it).dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    if (offset < 0 && offset != ShapedType::kDynamicStrideOrOffset) {
      return emitOpError("expected offset = ")
             << offset << " to be non-negative";
    }
    auto size =
        std::get<2>(it).dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    if (size < 0 && size != ShapedType::kDynamicSize) {
      return emitOpError("expected size = ") << size << " to be non-negative";
    }
    auto stride =
        std::get<3>(it).dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    if (stride < 0 && stride != ShapedType::kDynamicStrideOrOffset) {
      return emitOpError("expected stride = ")
             << stride << " to be non-negative";
    }
    auto argSize = std::get<0>(it);
    // If the argument tile has a dynamic dimension, no additional verification
    // is possible.
    if (argSize == ShapedType::kDynamicSize) continue;
    if (offset >= 0) {
      if (stride >= 0 && size > 0) {
        int64_t largestIndex = offset + stride * (size - 1);
        if (largestIndex >= argSize) {
          return emitOpError("offset = ")
                 << offset << " size = " << size << " stride = " << stride
                 << " causes access out of bounds at " << largestIndex
                 << " for argument dimension size = " << argSize;
        }
      } else if (offset >= argSize) {
        return emitOpError("offset = ")
               << offset
               << " is out of bounds for argument dimension size = " << argSize;
      }
    } else if (stride > 0 && size > 0 && stride * (size - 1) >= argSize) {
      return emitOpError("size = ")
             << size << " stride = " << stride
             << " causes access out of bounds for argument dimension size = "
             << argSize;
    }
  }
  return success();
}

namespace {

OpFoldResult multiplyOperandsOrIntegers(OpBuilder &builder, Location loc,
                                        OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return builder.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() *
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return lhs;
    if (lhsInt == 1) return rhs;

    lhs = builder.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Multiply.
  return builder.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

OpFoldResult addOperandsOrIntegers(OpBuilder &builder, Location loc,
                                   OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return builder.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() +
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return rhs;

    lhs = builder.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Add.
  return builder.create<arith::AddIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

// Compose offsets with newOffset = supersetOffset + supersetStride * offset.
SmallVector<OpFoldResult> composeOffsets(
    const llvm::SmallVectorImpl<OpFoldResult> &supersetOffsets,
    const llvm::SmallVectorImpl<OpFoldResult> &supersetStrides,
    const llvm::SmallVectorImpl<OpFoldResult> &offsets, Location loc,
    OpBuilder &builder) {
  SmallVector<OpFoldResult> composedOffsets;
  for (auto it : llvm::zip(supersetOffsets, supersetStrides, offsets)) {
    composedOffsets.push_back(addOperandsOrIntegers(
        builder, loc, std::get<0>(it),
        multiplyOperandsOrIntegers(builder, loc, std::get<1>(it),
                                   std::get<2>(it))));
  }
  return composedOffsets;
}

// Compose strides with newStride = supersetStride * stride.
SmallVector<OpFoldResult> composeStrides(
    OpBuilder &builder, Location loc,
    const llvm::SmallVectorImpl<OpFoldResult> &supersetStrides,
    const llvm::SmallVectorImpl<OpFoldResult> &strides) {
  SmallVector<OpFoldResult> composedStrides;
  for (auto it : llvm::zip(supersetStrides, strides)) {
    composedStrides.push_back(multiplyOperandsOrIntegers(
        builder, loc, std::get<0>(it), std::get<1>(it)));
  }
  return composedStrides;
}

}  // namespace

Value TileOp::compose(OpBuilder &builder) {
  auto supersetOp =
      llvm::dyn_cast_or_null<TileOp>(getSuperset().getDefiningOp());
  if (!supersetOp) return {};

  // Compose offsets with newOffset = supersetOffset + supersetStride *
  // offset.
  auto loc = getLoc();
  auto composedOffsets =
      composeOffsets(supersetOp.getMixedOffsets(), supersetOp.getMixedStrides(),
                     getMixedOffsets(), loc, builder);

  // Compose strides with newStride = supersetStride * stride.
  auto composedStrides = composeStrides(
      builder, loc, supersetOp.getMixedStrides(), getMixedStrides());

  // Build the composed tile op.
  return builder.create<TileOp>(loc, supersetOp.getSuperset(), composedOffsets,
                                getMixedSizes(), composedStrides);
}

//===----------------------------------------------------------------------===//
// SetYieldOp
//===----------------------------------------------------------------------===//

using AccumulatorRegionBuilderFn =
    function_ref<void(OpBuilder &, Location, Value, Value)>;

void SetYieldOp::build(OpBuilder &builder, OperationState &result) {
  build(builder, result, llvm::None, llvm::None, llvm::None);
}

void SetYieldOp::build(OpBuilder &builder, OperationState &result,
                       ValueRange srcs, ValueRange dsts, ValueRange sets) {
  SmallVector<bool, 2> accumulatorFlags(srcs.size(), false);
  build(builder, result, srcs, dsts, sets,
        builder.getBoolArrayAttr(accumulatorFlags), llvm::None);
}

void SetYieldOp::build(
    OpBuilder &builder, OperationState &result, ValueRange srcs,
    ValueRange dsts, ValueRange sets, ArrayAttr accumulatorFlags,
    ArrayRef<AccumulatorRegionBuilderFn> accumulatorBuilderFns) {
  assert(dsts.size() == srcs.size() &&
         "`dsts` and `srcs` should have the same size");
  assert(sets.size() == srcs.size() &&
         "`sets` and `srcs` should have the same size");
  assert(accumulatorFlags.size() == srcs.size() &&
         "`accumulatorFlags` and `srcs` should have the same size");

  auto accumulatorCount = llvm::count_if(accumulatorFlags, [](Attribute attr) {
    return attr.cast<BoolAttr>().getValue();
  });
  (void)accumulatorCount;
  assert(accumulatorCount ==
             static_cast<int64_t>(accumulatorBuilderFns.size()) &&
         "the number of flags set in `accumulatorFlags` attribute should be "
         "equal to the number of `accumulatorBuilderFns`");

  result.addOperands(srcs);
  result.addOperands(dsts);
  result.addOperands(sets);
  result.addAttribute(SetYieldOp::getAccumulatorFlagsAttrName(result.name),
                      accumulatorFlags);

  const auto *builderFnIt = accumulatorBuilderFns.begin();
  for (auto item : llvm::zip(srcs, accumulatorFlags)) {
    Value src = std::get<0>(item);
    auto accumulatorFlag = std::get<1>(item).cast<BoolAttr>();

    if (!accumulatorFlag.getValue()) continue;
    Region *region = result.addRegion();
    OpBuilder::InsertionGuard g(builder);
    SmallVector<Type, 2> argTypes(2, src.getType());
    builder.createBlock(region);
    Block &bodyBlock = region->front();
    bodyBlock.addArguments(argTypes, {result.location, result.location});

    builder.setInsertionPointToStart(&bodyBlock);
    (*builderFnIt)(builder, result.location, bodyBlock.getArgument(0),
                   bodyBlock.getArgument(1));
    ++builderFnIt;
  }
}

LogicalResult SetYieldOp::verify() {
  for (const auto [dst, src, set] :
       llvm::zip(getDsts(), getSrcs(), getSets())) {
    if (failed(verifyCompatibleExtractedSubset(getOperation(),
                                               dst.getType().cast<ShapedType>(),
                                               src.getType(), set.getType())))
      return failure();
  }
  auto accumulatorCount = llvm::count_if(
      getAccumulatorFlags(),
      [](Attribute attr) { return attr.cast<BoolAttr>().getValue(); });
  if (accumulatorCount != static_cast<int64_t>(getAccumulators().size()))
    return emitOpError("expected the number of accumulator regions ")
           << getAccumulators().size()
           << " to match the number of set accumulator flags "
           << accumulatorCount;

  auto *regionIt = getAccumulators().begin();
  for (auto item : llvm::zip(getSrcs(), getAccumulatorFlags())) {
    Type srcType = std::get<0>(item).getType();
    BoolAttr accumulatorFlag = std::get<1>(item).cast<BoolAttr>();
    if (!accumulatorFlag.getValue()) continue;

    Block &block = regionIt->front();
    if (block.getArgumentTypes() != SmallVector<Type>{srcType, srcType})
      return emitOpError()
             << "expected accumulator region to have 2 arguments of type "
             << srcType;
    ++regionIt;
  }
  return success();
}

void SetYieldOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs = */
                          {getAccumulatorFlagsAttrName().str()});

  auto *regionIt = getOperation()->getRegions().begin();
  for (auto &en : llvm::enumerate(
           llvm::zip(getSrcs(), getDsts(), getSets(), getAccumulatorFlags()))) {
    if (en.index() > 0) {
      p << ',';
      p.printNewline();
    }
    Value src = std::get<0>(en.value());
    Value dst = std::get<1>(en.value());
    Value set = std::get<2>(en.value());
    auto accumulatorFlag = std::get<3>(en.value()).cast<BoolAttr>();

    p << ' ' << src << " into " << dst << '[' << set << ']';

    if (accumulatorFlag.getValue()) {
      auto &block = regionIt->getBlocks().front();
      Value newValue = block.getArgument(0);
      Value oldValue = block.getArgument(1);
      p << " acc (" << newValue << ", " << oldValue << ": "
        << oldValue.getType() << ") ";

      p.printRegion(*regionIt, false);
      ++regionIt;
    }

    p << " : " << src.getType() << " into " << dst.getType() << '['
      << set.getType() << ']';
  }
}

ParseResult SetYieldOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  SmallVector<bool, 2> accumulatorFlags;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs, dsts, sets;
  SmallVector<Type, 4> srcTypes, dstTypes, setTypes;

  auto parseElt = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand src;
    auto parseResult = parser.parseOptionalOperand(src, false);

    if (!parseResult.has_value()) return success();
    srcs.push_back(src);

    if (parser.parseKeyword("into") ||
        parser.parseOperand(dsts.emplace_back()) || parser.parseLSquare() ||
        parser.parseOperand(sets.emplace_back()) || parser.parseRSquare())
      return failure();

    OpBuilder b(parser.getBuilder().getContext());
    bool hasAccumulatorRegion = succeeded(parser.parseOptionalKeyword("acc"));
    accumulatorFlags.push_back(hasAccumulatorRegion);
    if (hasAccumulatorRegion) {
      auto region = std::make_unique<Region>();
      OpAsmParser::UnresolvedOperand newValue, oldValue;
      Type argType;
      if (parser.parseLParen() || parser.parseOperand(newValue) ||
          parser.parseComma() || parser.parseOperand(oldValue) ||
          parser.parseColonType(argType) || parser.parseRParen())
        return failure();

      SmallVector<OpAsmParser::Argument, 4> regionArgs;
      for (auto value : {newValue, oldValue}) {
        auto &arg = regionArgs.emplace_back();
        arg.ssaName = value;
        arg.type = argType;
      }

      if (parser.parseRegion(*region, regionArgs)) return failure();
      result.addRegion(std::move(region));
    }
    if (parser.parseColon() || parser.parseType(srcTypes.emplace_back()) ||
        parser.parseKeyword("into") ||
        parser.parseType(dstTypes.emplace_back()) || parser.parseLSquare() ||
        parser.parseType(setTypes.emplace_back()) || parser.parseRSquare())
      return failure();

    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseElt))
    return failure();

  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(dsts, dstTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(sets, setTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  result.addAttribute(SetYieldOp::getAccumulatorFlagsAttrName(result.name),
                      parser.getBuilder().getBoolArrayAttr(accumulatorFlags));
  return success();
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

OpFoldResult OffsetOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};
  int64_t idx = idxAttr.getInt();

  // Case: offset(tile(space))
  Operation *subsetDef = getSubset().getDefiningOp();
  if (auto tileOp = llvm::dyn_cast_or_null<TileOp>(subsetDef)) {
    Operation *supersetDef = tileOp.getSuperset().getDefiningOp();

    // Can only fold locally if the superset is the root space. Otherwise, rely
    // on subset composition.
    if (!llvm::isa_and_nonnull<SpaceOp>(supersetDef)) return {};

    return ensureIndexTypeForAttribute(mlir::getMixedStridesOrOffsets(
        tileOp.getStaticOffsets(), tileOp.getOffsets())[idx]);
  }

  // Case: offset(space)
  if (llvm::isa_and_nonnull<SpaceOp>(subsetDef)) {
    Builder b(getContext());
    return b.getIndexAttr(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

OpFoldResult SizeOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};
  int64_t idx = idxAttr.getInt();

  // Case: size(tile(...))
  // Note that sizes can also be folded in the presence of nested tiling. There
  // is no need to check for an immediate root space here.
  Operation *tileDef = getTile().getDefiningOp();
  if (auto tileOp = llvm::dyn_cast_or_null<TileOp>(tileDef)) {
    return ensureIndexTypeForAttribute(tileOp.getMixedSizes()[idx]);
  }

  // Case: size(space)
  if (auto spaceOp = llvm::dyn_cast_or_null<SpaceOp>(tileDef)) {
    return ensureIndexTypeForAttribute(mlir::getMixedSizes(
        spaceOp.getStaticSizes(), spaceOp.getDynamicSizes())[idx]);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// StrideOp
//===----------------------------------------------------------------------===//

OpFoldResult StrideOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};
  int64_t idx = idxAttr.getInt();

  // Case: offset(tile(space))
  Operation *subsetDef = getTile().getDefiningOp();
  if (auto tileOp = llvm::dyn_cast_or_null<TileOp>(subsetDef)) {
    Operation *supersetDef = tileOp.getSuperset().getDefiningOp();

    // Can only fold locally if the superset is the root space. Otherwise, rely
    // on subset composition.
    if (!llvm::isa_and_nonnull<SpaceOp>(supersetDef)) return {};

    return ensureIndexTypeForAttribute(mlir::getMixedStridesOrOffsets(
        tileOp.getStaticStrides(), tileOp.getStrides())[idx]);
  }

  // Case: offset(space)
  if (llvm::isa_and_nonnull<SpaceOp>(subsetDef)) {
    Builder b(getContext());
    return b.getIndexAttr(1);
  }

  return {};
}

}  // namespace gml_st
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"

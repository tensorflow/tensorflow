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

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
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

// TODO(frgossen): Move this to MHLO or even to MLIR.
ParseResult parseI64ElementsAttr(OpAsmParser &parser,
                                 DenseIntElementsAttr &attr) {
  SmallVector<int64_t> values;

  // Parse opening bracket.
  if (failed(parser.parseLSquare())) return failure();

  auto tryParseInt = [&]() {
    int64_t val;
    auto parsingRes = parser.parseOptionalInteger(val);
    if (parsingRes.hasValue() && succeeded(*parsingRes)) {
      values.push_back(val);
      return true;
    }
    return false;
  };

  // Parse comma-separated ints.
  if (tryParseInt()) {
    while (succeeded(parser.parseOptionalComma())) {
      int64_t val;
      if (failed(parser.parseInteger(val))) return failure();
      values.push_back(val);
    }
  }

  // Parse closing bracket.
  if (failed(parser.parseRSquare())) return failure();

  // Build attribute.
  OpBuilder b(parser.getContext());
  attr = b.getI64TensorAttr(values);
  return success();
}

// TODO(frgossen): Move this to MHLO or even to MLIR.
template <class OpTy>
void printI64ElementsAttr(OpAsmPrinter &printer, OpTy op,
                          DenseIntElementsAttr attr) {
  printer << "[";
  llvm::interleave(
      attr.getValues<int64_t>(), printer, [&](int64_t val) { printer << val; },
      ", ");
  printer << "]";
}

}  // namespace
}  // namespace mlir

// Generated dialect definitions.
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_dialect.cc.inc"

// Generated type classes.
#define GET_TYPEDEF_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"

namespace mlir {
namespace gml_st {

void GmlStDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult MaterializeOp::inferReturnTypes(
    MLIRContext *, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  MaterializeOp::Adaptor adaptor(operands, attributes);

  ShapedType sourceType = adaptor.source().getType().cast<ShapedType>();
  Type subsetType = adaptor.subset().getType();

  if (auto tileType = subsetType.dyn_cast<TileType>()) {
    if (auto memrefType = sourceType.dyn_cast<MemRefType>()) {
      inferredReturnTypes.push_back(
          MemRefType::get(tileType.getShape(), sourceType.getElementType()));
    } else if (auto tensorType = sourceType.dyn_cast<RankedTensorType>()) {
      inferredReturnTypes.push_back(RankedTensorType::get(
          tileType.getShape(), sourceType.getElementType()));
    } else {
      return failure();
    }
  } else if (subsetType.isa<PointType>()) {
    inferredReturnTypes.push_back(sourceType.getElementType());
  } else {
    return failure();
  }
  return success();
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
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
                                static_cast<int32_t>(upperBounds.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(inputs.size()),
                                static_cast<int32_t>(outputs.size())}));
  result.addAttribute(getIteratorTypesAttrName(), iteratorTypes);

  if (distributionTypes.hasValue())
    result.addAttribute(getDistributionTypesAttrName(),
                        distributionTypes.getValue());

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
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ")";

  if (!inputs().empty()) {
    p << " ins (";
    llvm::interleaveComma(llvm::zip(getRegionInputArgs(), inputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }
  if (!outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(llvm::zip(getRegionOutputArgs(), outputs()), p,
                          [&](auto it) {
                            p << std::get<0>(it) << " = " << std::get<1>(it)
                              << ": " << std::get<1>(it).getType();
                          });
    p << ")";
  }

  if (llvm::any_of(iterator_types(), [](Attribute attr) {
        return attr.cast<StringAttr>().getValue() !=
               LoopOp::getParallelIteratorTypeName();
      }))
    p << " iterators" << iterator_types();

  if (distribution_types().hasValue())
    p << " distribution" << distribution_types().getValue();

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{LoopOp::getOperandSegmentSizeAttr(),
                       LoopOp::getIteratorTypesAttrName(),
                       LoopOp::getDistributionTypesAttrName()});
}

namespace {
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

  // Parse attributes.
  SmallVector<Attribute, 4> iterTypes, distributionTypes;
  auto parseAttr = [&](StringRef keyword, SmallVector<Attribute, 4> *attrs) {
    if (succeeded(parser.parseOptionalKeyword(keyword))) {
      StringAttr attr;

      if (parser.parseLSquare() || parser.parseAttribute(attr))
        return failure();
      attrs->push_back(attr);
      for (int i = 1, e = ivs.size(); i < e; ++i) {
        if (parser.parseComma() || parser.parseAttribute(attr))
          return failure();
        attrs->push_back(attr);
      }
      if (parser.parseRSquare()) return failure();
    }
    return success();
  };
  if (failed(parseAttr("iterators", &iterTypes)) ||
      failed(parseAttr("distribution", &distributionTypes)))
    return failure();

  // Set all loop iterator types to "parallel" if they are not printed in IR.
  if (iterTypes.empty()) {
    auto parallelIter =
        builder.getStringAttr(LoopOp::getParallelIteratorTypeName());
    iterTypes = SmallVector<Attribute, 4>(ivs.size(), parallelIter);
  }
  result.addAttribute(LoopOp::getIteratorTypesAttrName(),
                      builder.getArrayAttr(iterTypes));
  if (!distributionTypes.empty())
    result.addAttribute(LoopOp::getDistributionTypesAttrName(),
                        builder.getArrayAttr(distributionTypes));
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
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

Region &LoopOp::getLoopBody() { return region(); }

LogicalResult LoopOp::verify() {
  // Check if iterator types are provided for every loop dimension.
  if (iterator_types().size() != getNumLoops())
    return emitOpError("expected iterator types array attribute size = ")
           << iterator_types().size()
           << " to match the number of loops = " << getNumLoops();

  // Check if types of input arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(inputs(), getRegionInputArgs()))) {
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
       llvm::enumerate(llvm::zip(outputs(), getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType())
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg "
             << index + getNumLoops() + inputs().size()
             << " type = " << outputRegionArg.getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoopLikeOp
//===----------------------------------------------------------------------===//

template <typename LoopTy>
void buildLoopLikeOp(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    ValueRange outputs, ValueRange subsets,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(outputs);
  result.addOperands(subsets);
  result.addTypes(resultTypes);
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
                                static_cast<int32_t>(upperBounds.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(outputs.size()),
                                static_cast<int32_t>(subsets.size())}));

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
    LoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

namespace {
template <typename LoopTy>
ParseResult parseOutputArgs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &outputRegionArgs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &outputs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &subsets,
    SmallVectorImpl<Type> &outputTypes, SmallVectorImpl<Type> &subsetTypes) {
  auto parseElt = [&]() -> ParseResult {
    if (std::is_same<LoopTy, ForOp>::value) {
      if (parser.parseOperand(outputRegionArgs.emplace_back(),
                              /*allowResultNumber=*/false) ||
          parser.parseEqual()) {
        return failure();
      }
    }
    if (parser.parseOperand(outputs.emplace_back()) ||
        parser.parseKeyword("at") ||
        parser.parseOperand(subsets.emplace_back()) || parser.parseColon() ||
        parser.parseType(outputTypes.emplace_back()) ||
        parser.parseKeyword("at") ||
        parser.parseType(subsetTypes.emplace_back())) {
      return failure();
    }
    return success();
  };
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt);
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

  // Parse output tensors.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs, outputRegionArgs,
      subsets;
  SmallVector<Type, 4> outputTypes, subsetTypes;
  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    SMLoc loc = parser.getCurrentLocation();

    if (parseOutputArgs<LoopTy>(parser, outputRegionArgs, outputs, subsets,
                                outputTypes, subsetTypes))
      return failure();

    if (parser.resolveOperands(outputs, outputTypes, loc, result.operands) ||
        parser.resolveOperands(subsets, subsetTypes, loc, result.operands))
      return failure();
  }

  // Parse the body.
  SmallVector<Type, 4> regionTypes(ivs.size(), builder.getIndexType());
  SmallVector<OpAsmParser::UnresolvedOperand, 4> regionOperands(ivs);

  if (!outputRegionArgs.empty()) {
    regionOperands.append(outputRegionArgs);
    regionTypes.append(outputTypes);
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
  result.addAttribute(
      LoopTy::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
                                static_cast<int32_t>(upper.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(outputs.size()),
                                static_cast<int32_t>(subsets.size())}));

  // Parser result types.
  if (parser.parseColon() || parser.parseTypeList(result.types))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

Region &ParallelOp::getLoopBody() { return region(); }

LogicalResult ParallelOp::verify() { return success(); }

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    ValueRange outputs, ValueRange subsets,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  buildLoopLikeOp<ParallelOp>(builder, result, resultTypes, lowerBounds,
                              upperBounds, steps, outputs, subsets,
                              bodyBuilderFn);
}

void ParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ")";

  if (!outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(llvm::zip(outputs(), subsets()), p, [&](auto it) {
      Value output, subset;
      std::tie(output, subset) = it;
      p << output << " at " << subset << ": " << output.getType() << " at "
        << subset.getType();
    });
    p << ")";
  }

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{ParallelOp::getOperandSegmentSizeAttr()});
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseLoopLikeOp<ParallelOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

Region &ForOp::getLoopBody() { return region(); }

LogicalResult ForOp::verify() {
  // Check if types of output arguments match region args types.
  for (auto &item :
       llvm::enumerate(llvm::zip(outputs(), getRegionOutputArgs()))) {
    Value output, outputRegionArg;
    unsigned index = item.index();
    std::tie(output, outputRegionArg) = item.value();
    if (output.getType() != outputRegionArg.getType()) {
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg " << index + getNumLoops()
             << " type = " << outputRegionArg.getType();
    }
  }
  return success();
}

void ForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    ValueRange outputs, ValueRange subsets,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  buildLoopLikeOp<ForOp>(builder, result, resultTypes, lowerBounds, upperBounds,
                         steps, outputs, subsets, bodyBuilderFn);
}

void ForOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ")";

  if (!outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(
        llvm::zip(getRegionOutputArgs(), outputs(), subsets()), p,
        [&](auto it) {
          Value outputRegionArg, output, subset;
          std::tie(outputRegionArg, output, subset) = it;
          p << outputRegionArg << " = " << output << " at " << subset << ": "
            << output.getType() << " at " << subset.getType();
        });
    p << ")";
  }

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{ForOp::getOperandSegmentSizeAttr()});
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
    SmallVector<int64_t, 2> oldInputIdToNew(loop.inputs().size(), kNoMatch);
    for (const auto &en :
         llvm::enumerate(llvm::zip(loop.inputs(), loop.getRegionInputArgs()))) {
      Value in, bbArg;
      size_t index = en.index();
      std::tie(in, bbArg) = en.value();
      if (!bbArg.use_empty()) {
        oldInputIdToNew[index] = newInputs.size();
        newInputs.push_back(in);
      }
    }
    if (newInputs.size() == loop.inputs().size()) return failure();
    Location loc = loop.getLoc();
    auto newLoop = rewriter.create<LoopOp>(
        loc, loop.lowerBound(), loop.upperBound(), loop.step(), newInputs,
        loop.outputs(), loop.iterator_types(), loop.distribution_types());

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
  if (yieldOp.values().empty())
    // Loop either has no outputs or is a "memref-based version". In either
    // case, the loop is shape conserving.
    return true;
  assert(arg < static_cast<int64_t>(yieldOp.values().size()) &&
         "arg is out of bounds");
  Value value = yieldOp.values()[arg];
  while (value) {
    if (value == loopOp.getRegionOutputArgs()[arg]) return true;
    OpResult opResult = value.dyn_cast<OpResult>();
    if (!opResult) return false;

    using tensor::InsertSliceOp;
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
                .template Case<InsertSliceOp>(
                    [&](InsertSliceOp op) { return op.dest(); })
                .template Case<LoopOp>([&](LoopOp loopOp) {
                  return isShapePreserving(loopOp, opResult.getResultNumber())
                             ? loopOp.outputs()[opResult.getResultNumber()]
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
    auto src = dimOp.source().template dyn_cast<BlockArgument>();
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
        dimOp.sourceMutable().assign(loopOp.inputs()[it1 - inputArgs.begin()]);
      });
      return success();
    }

    auto outputArgs = loopOp.getRegionOutputArgs();
    auto it2 = llvm::find(outputArgs, src);
    if (it2 != outputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.sourceMutable().assign(
            loopOp.outputs()[it2 - outputArgs.begin()]);
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
    auto loopOp = dimOp.source().template getDefiningOp<LoopOp>();
    if (!loopOp) return failure();
    auto opResult = dimOp.source().template cast<OpResult>();
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(loopOp, resultNumber)) return failure();
    rewriter.updateRootInPlace(dimOp, [&]() {
      dimOp.sourceMutable().assign(loopOp.outputs()[resultNumber]);
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
    SmallVector<int64_t, 2> oldOutputIdToNew(loop.outputs().size(), kNoMatch);
    // Store ids of the corresponding old and new results.
    SmallVector<int64_t, 2> oldResultIdToNew(loop.getNumResults(), kNoMatch);
    SmallVector<Value, 2> resultReplacement(loop.getNumResults());
    for (const auto &en : llvm::enumerate(
             llvm::zip(loop.outputs(), loop.getRegionOutputArgs()))) {
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
    if (newOutputOperands.size() == loop.outputs().size()) return failure();

    Location loc = loop.getLoc();
    auto newLoop = rewriter.create<LoopOp>(
        loc, loop.lowerBound(), loop.upperBound(), loop.step(), loop.inputs(),
        newOutputOperands, loop.iterator_types(), loop.distribution_types());

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
                loop.outputs()[en.index()]);
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
    CastOpsOfArgs inputCasts = findTensorCastOps(loop.inputs());
    CastOpsOfArgs outputCasts = findTensorCastOps(loop.outputs());
    if (!inputCasts.castFound && !outputCasts.castFound) return failure();

    auto newLoop = rewriter.create<LoopOp>(
        loop.getLoc(), loop.lowerBound(), loop.upperBound(), loop.step(),
        inputCasts.updatedArgs, outputCasts.updatedArgs, loop.iterator_types(),
        loop.distribution_types());

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
        result.updatedArgs.push_back(cast.source());
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
          loc, argCast.source().getType(), bvm.lookup(yieldArg)));
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
    for (auto dim : llvm::zip(op.lowerBound(), op.upperBound())) {
      if (std::get<0>(dim) != std::get<1>(dim)) continue;
      SmallVector<Value> tensorOutputs;
      for (Value out : op.outputs()) {
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
  auto loopOp = dyn_cast<LoopOp>(parentOp);
  // Check if output args with tensor types match results types.
  SmallVector<Value, 2> tensorOuts;
  llvm::copy_if(
      loopOp.outputs(), std::back_inserter(tensorOuts),
      [&](Value out) { return out.getType().isa<RankedTensorType>(); });
  if (tensorOuts.size() != values().size())
    return emitOpError("expected number of tensor output args = ")
           << tensorOuts.size()
           << " to match the number of yield operands = " << values().size();

  TypeRange tensorTypes(llvm::makeArrayRef(tensorOuts));
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

LogicalResult SpaceOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SpaceOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(
      llvm::map_range(adaptor.static_shapes(), [&](const Attribute &val) {
        return val.cast<IntegerAttr>().getValue().getSExtValue();
      }));
  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult SpaceOp::verify() {
  auto resultTy = getType().cast<TileType>();
  return mlir::verifyListOfOperandsOrIntegers(
      getOperation(), "shapes", resultTy.getShape().size(), static_shapes(),
      shapes(), ShapedType::isDynamic);
}

//===----------------------------------------------------------------------===//
// PointOp
//===----------------------------------------------------------------------===//

LogicalResult PointOp::verify() {
  auto subsetTy = subset().getType();
  if (subsetTy.isa<PointType>()) {
    if (!static_indices().empty() || !indices().empty()) {
      return emitOpError(
          "expected empty indices and static_indices for a subset of type "
          "PointType");
    }
  } else {
    auto tileTy = subsetTy.cast<TileType>();
    auto tileShape = tileTy.getShape();
    if (failed(mlir::verifyListOfOperandsOrIntegers(
            getOperation(), "indices", tileShape.size(), static_indices(),
            indices(), ShapedType::isDynamicStrideOrOffset))) {
      return failure();
    }
    // Check whether the known indices are in-bounds of known dimension sizes.
    for (auto dimAndIndex : llvm::zip(tileShape, static_indices())) {
      auto dimSize = std::get<0>(dimAndIndex);
      auto index = std::get<1>(dimAndIndex)
                       .dyn_cast<mlir::IntegerAttr>()
                       .getValue()
                       .getSExtValue();
      if (index == ShapedType::kDynamicStrideOrOffset) continue;
      if (index < 0) {
        return emitOpError("expected index = ")
               << index << " to be non-negative";
      }
      if (dimSize != ShapedType::kDynamicSize && index >= dimSize) {
        return emitOpError("expected index = ")
               << index << " to be between 0 and " << (dimSize - 1);
      }
    }
  }
  return success();
}
//
//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Derive result shape.
  TileOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(
      llvm::map_range(adaptor.static_sizes(), [&](const auto &size) {
        return size.template dyn_cast<mlir::IntegerAttr>()
            .getValue()
            .getSExtValue();
      }));

  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult TileOp::verify() {
  auto subsetTy = subset().getType().cast<TileType>();
  auto rank = subsetTy.getShape().size();
  if (failed(mlir::verifyListOfOperandsOrIntegers(getOperation(), "sizes", rank,
                                                  static_sizes(), sizes(),
                                                  ShapedType::isDynamic))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "offsets", rank, static_offsets(), offsets(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "strides", rank, static_strides(), strides(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  for (auto it : llvm::zip(subsetTy.getShape(), static_offsets(),
                           static_sizes(), static_strides())) {
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

//===----------------------------------------------------------------------===//
// CollapseTileOp
//===----------------------------------------------------------------------===//

LogicalResult CollapseTileOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get argument tile type.
  Value argTile = operands.front();
  auto argTy = argTile.getType().dyn_cast<TileType>();
  if (!argTy) return failure();
  auto argShape = argTy.getShape();

  // Derive result shape.
  CollapseTileOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(llvm::map_range(
      adaptor.remaining_dims(),
      [&](const auto &d) { return argShape[d.getLimitedValue()]; }));

  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// SubsetYieldOp
//===----------------------------------------------------------------------===//

LogicalResult SubsetYieldOp::verify() { return success(); }

}  // namespace gml_st
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"

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
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
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

template <typename DstOpTy>
void printDstStyleOp(DstOpTy op, OpAsmPrinter &p) {
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

  p.printOptionalAttrDict(op->getAttrs());
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

ParseResult parseDstStyleOp(OpAsmParser &parser, OperationState &result) {
  // Parse `ins` and `outs`.
  SmallVector<Type, 4> inputTypes, outputTypes;
  if (parseKeywordOperandListWithTypes(parser, result, "ins", &inputTypes) ||
      parseKeywordOperandListWithTypes(parser, result, "outs", &outputTypes))
    return failure();

  // Add result types.
  for (Type outputType : outputTypes) {
    if (outputType.isa<RankedTensorType>()) result.addTypes(outputType);
  }

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
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
  Type setType = adaptor.set().getType();

  if (auto tileType = setType.dyn_cast<TileType>()) {
    if (auto memrefType = sourceType.dyn_cast<MemRefType>()) {
      inferredReturnTypes.push_back(
          MemRefType::get(tileType.getShape(), sourceType.getElementType()));
    } else if (auto tensorType = sourceType.dyn_cast<RankedTensorType>()) {
      inferredReturnTypes.push_back(RankedTensorType::get(
          tileType.getShape(), sourceType.getElementType()));
    } else {
      return failure();
    }
  } else if (setType.isa<PointType>()) {
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

  if (distributionTypes.has_value())
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

  if (distribution_types().has_value())
    p << " distribution" << distribution_types().getValue();

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
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
                      builder.getI32VectorAttr(segmentSizes));

  return success();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

Region &ParallelOp::getLoopBody() { return region(); }

SetYieldOp ParallelOp::getTerminator() {
  return cast<SetYieldOp>(getBody()->getTerminator());
}

LogicalResult ParallelOp::verify() { return success(); }

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addTypes(resultTypes);
  result.addAttribute(
      LoopOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
                                static_cast<int32_t>(upperBounds.size()),
                                static_cast<int32_t>(steps.size())}));

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
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ") ";

  p.printRegion(region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{ParallelOp::getOperandSegmentSizeAttr()});

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

Region &ForOp::getLoopBody() { return region(); }

SetYieldOp ForOp::getTerminator() {
  return cast<SetYieldOp>(getBody()->getTerminator());
}

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
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
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
  p << " (" << getInductionVars() << ") = (" << lowerBound() << ") to ("
    << upperBound() << ") step (" << step() << ")";

  if (!outputs().empty()) {
    p << " outs (";
    llvm::interleaveComma(
        llvm::zip(getRegionOutputArgs(), outputs()), p, [&](auto it) {
          Value outputRegionArg, output;
          std::tie(outputRegionArg, output) = it;
          p << outputRegionArg << " = " << output << ": " << output.getType();
        });
    p << ")";
  }

  p << ' ';
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
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
                    [&](InsertSliceOp op) { return op.getDest(); })
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
            loopOp.inputs()[it1 - inputArgs.begin()]);
      });
      return success();
    }

    auto outputArgs = loopOp.getRegionOutputArgs();
    auto it2 = llvm::find(outputArgs, src);
    if (it2 != outputArgs.end()) {
      rewriter.updateRootInPlace(dimOp, [&] {
        dimOp.getSourceMutable().assign(
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
    auto loopOp = dimOp.getSource().template getDefiningOp<LoopOp>();
    if (!loopOp) return failure();
    auto opResult = dimOp.getSource().template cast<OpResult>();
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(loopOp, resultNumber)) return failure();
    rewriter.updateRootInPlace(dimOp, [&]() {
      dimOp.getSourceMutable().assign(loopOp.outputs()[resultNumber]);
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

  if (auto setYield = dyn_cast<SetYieldOp>(parentOp)) {
    if (values().size() != 1)
      return emitOpError(
          "expected a single argument for the terminator of accumulator "
          "region");
    return success();
  }
  auto loopOp = cast<LoopOp>(parentOp);
  // Check if output args with tensor types match results types.
  SmallVector<Value, 2> tensorOuts;
  llvm::copy_if(
      loopOp.outputs(), std::back_inserter(tensorOuts),
      [&](Value out) { return out.getType().isa<RankedTensorType>(); });
  if (tensorOuts.size() != values().size())
    return emitOpError("expected number of tensor output args = ")
           << tensorOuts.size()
           << " to match the number of yield operands = " << values().size();

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

LogicalResult SpaceOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SpaceOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(
      llvm::map_range(adaptor.static_sizes(), [&](const Attribute &val) {
        return val.cast<IntegerAttr>().getValue().getSExtValue();
      }));
  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult SpaceOp::verify() {
  auto resultTy = getType().cast<TileType>();
  return mlir::verifyListOfOperandsOrIntegers(
      getOperation(), "size", resultTy.getShape().size(), static_sizes(),
      dynamic_sizes(), ShapedType::isDynamic);
}

unsigned SpaceOp::getNumDynamicEntriesUpToIdx(unsigned idx) {
  return std::count_if(static_sizes().begin(), static_sizes().begin() + idx,
                       [&](const mlir::Attribute size) {
                         return mlir::ShapedType::isDynamic(
                             size.cast<mlir::IntegerAttr>().getInt());
                       });
}

mlir::Value SpaceOp::getDynamicSize(unsigned idx) {
  auto numDynamic = getNumDynamicEntriesUpToIdx(idx);
  return dynamic_sizes()[numDynamic];
}

//===----------------------------------------------------------------------===//
// PointOp
//===----------------------------------------------------------------------===//

LogicalResult PointOp::verify() {
  auto tileShape = superset().getType().cast<TileType>().getShape();
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "index", tileShape.size(), static_indices(),
          dynamic_indices(), ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  // Check whether the known indices are in-bounds of known dimension sizes.
  for (auto dimAndIndex : llvm::zip(tileShape, static_indices())) {
    auto dimSize = std::get<0>(dimAndIndex);
    auto index =
        std::get<1>(dimAndIndex).dyn_cast<mlir::IntegerAttr>().getInt();
    if (index == ShapedType::kDynamicStrideOrOffset) continue;
    if (index < 0) {
      return emitOpError("expected index = ") << index << " to be non-negative";
    }
    if (dimSize != ShapedType::kDynamicSize && index >= dimSize) {
      return emitOpError("expected index = ")
             << index << " to be between 0 and " << (dimSize - 1);
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
  auto supersetTy = superset().getType().cast<TileType>();
  auto rank = supersetTy.getShape().size();
  if (failed(mlir::verifyListOfOperandsOrIntegers(getOperation(), "size", rank,
                                                  static_sizes(), sizes(),
                                                  ShapedType::isDynamic))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "offset", rank, static_offsets(), offsets(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "stride", rank, static_strides(), strides(),
          ShapedType::isDynamicStrideOrOffset))) {
    return failure();
  }
  for (auto it : llvm::zip(supersetTy.getShape(), static_offsets(),
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
  auto supersetOp = llvm::dyn_cast_or_null<TileOp>(superset().getDefiningOp());
  if (!supersetOp) return {};

  // Compose offsets with newOffset = supersetOffset + supersetStride *
  // offset.
  auto loc = getLoc();
  auto composedOffsets = decomposeMixedStridesOrOffsets(
      builder,
      composeOffsets(supersetOp.getMixedOffsets(), supersetOp.getMixedStrides(),
                     getMixedOffsets(), loc, builder));

  // Compose strides with newStride = supersetStride * stride.
  auto composedStrides = decomposeMixedStridesOrOffsets(
      builder, composeStrides(builder, loc, supersetOp.getMixedStrides(),
                              getMixedStrides()));

  // Build the composed tile op.
  return builder.create<TileOp>(loc, supersetOp.superset(),
                                composedOffsets.second, sizes(),
                                composedStrides.second, composedOffsets.first,
                                static_sizes(), composedStrides.first);
}

//===----------------------------------------------------------------------===//
// PointOp
//===----------------------------------------------------------------------===//

namespace {

// TODO(frgossen): Move this upstream to the ViewLikeInterface
SmallVector<OpFoldResult> getMixedImpl(ArrayAttr staticValues,
                                       ValueRange dynamicValues,
                                       const int64_t dynamicValuePlaceholder) {
  int64_t idxDynamic = 0;
  SmallVector<OpFoldResult> result;
  for (const auto &staticAttr : staticValues) {
    int64_t staticInt = staticAttr.cast<IntegerAttr>().getInt();
    if (staticInt == dynamicValuePlaceholder) {
      result.push_back(dynamicValues[idxDynamic++]);
    } else {
      result.push_back(staticAttr);
    }
  }
  return result;
}

// TODO(frgossen): Move this upstream to the ViewLikeInterface
SmallVector<OpFoldResult> getMixedStridesOrOffsets(ArrayAttr staticValues,
                                                   ValueRange dynamicValues) {
  return getMixedImpl(staticValues, dynamicValues,
                      ShapedType::kDynamicStrideOrOffset);
}

}  // namespace

Value PointOp::compose(OpBuilder &builder) {
  auto supersetOp = llvm::dyn_cast_or_null<TileOp>(superset().getDefiningOp());
  if (!supersetOp) return {};

  // Compose offsets with newOffset = supersetOffset + supersetStride *
  // offset.
  auto loc = getLoc();
  auto composedOffsets = decomposeMixedStridesOrOffsets(
      builder,
      composeOffsets(
          supersetOp.getMixedOffsets(), supersetOp.getMixedStrides(),
          getMixedStridesOrOffsets(static_indices(), dynamic_indices()), loc,
          builder));

  // Build the composed point op.
  return builder.create<PointOp>(loc, supersetOp.superset(),
                                 composedOffsets.second, composedOffsets.first);
}

//===----------------------------------------------------------------------===//
// DropDimsOp
//===----------------------------------------------------------------------===//

LogicalResult DropDimsOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DropDimsOp::Adaptor adaptor(operands, attributes, regions);
  Type argTy = adaptor.superset().getType();

  // If the argument is of point type, so is the result.
  if (auto pointTy = argTy.dyn_cast<PointType>()) {
    inferredReturnTypes.push_back(argTy);
    return success();
  }

  // If the argument is of tile type, we can skip the dropped dimensions to
  // derive the result type.
  if (auto tileTy = argTy.dyn_cast<TileType>()) {
    auto argShape = tileTy.getShape();
    SmallVector<int64_t> resultShape = llvm::to_vector(llvm::map_range(
        adaptor.remaining_dims(), [&](const auto &d) { return argShape[d]; }));
    auto resultTy = TileType::get(ctx, resultShape);
    inferredReturnTypes.push_back(resultTy);
    return success();
  }

  return failure();
}

namespace {
// Composition with a superset by selecting a subset of dimensions from the
// superset. Both the dimensions to select, and the order in which they should
// be selected, are specified by 'dims'.
Value selectDimsFromSuperset(OpBuilder &builder, Location loc, Type type,
                             Value superset, ArrayRef<int64_t> dims) {
  Operation *definingOp = superset.getDefiningOp();
  auto spaceOp = llvm::dyn_cast_or_null<SpaceOp>(definingOp);
  auto tileOp = llvm::dyn_cast_or_null<TileOp>(definingOp);
  if (tileOp) {
    spaceOp =
        llvm::dyn_cast_or_null<SpaceOp>(tileOp.superset().getDefiningOp());
  }
  if (!spaceOp) return {};

  // Create a new space op consisting of the subset of dimensions defined by
  // 'dims'.
  SmallVector<Value> dynamicDims;
  SmallVector<Attribute> staticDims;
  SmallVector<int64_t> shape;
  auto originalShape = spaceOp.getType().getShape();
  const size_t rank = dims.size();
  staticDims.reserve(rank);
  shape.reserve(rank);
  for (const int64_t dim : dims) {
    shape.push_back(originalShape[dim]);
    staticDims.push_back(spaceOp.static_sizes()[dim]);
    if (ShapedType::isDynamic(staticDims.back().cast<IntegerAttr>().getInt())) {
      dynamicDims.push_back(spaceOp.getDynamicSize(dim));
    }
  }
  auto spaceTy = builder.getType<TileType>(shape);
  Value newSpace = builder.create<SpaceOp>(loc, spaceTy, dynamicDims,
                                           builder.getArrayAttr(staticDims));
  if (!tileOp) return newSpace;

  // Otherwise we need to extract 'dims' dimensions from the 'tileOp' operand.
  SmallVector<Value> inputTileOffsets, inputTileSizes, inputTileStrides;
  SmallVector<int64_t> inputStaticOffsets, inputStaticSizes, inputStaticStrides;
  inputStaticOffsets.reserve(rank);
  inputStaticSizes.reserve(rank);
  inputStaticStrides.reserve(rank);
  inputTileOffsets.reserve(tileOp.offsets().size());
  inputTileSizes.reserve(tileOp.sizes().size());
  inputTileStrides.reserve(tileOp.strides().size());
  for (const int64_t dim : dims) {
    if (tileOp.isDynamicOffset(dim)) {
      inputTileOffsets.push_back(tileOp.getDynamicOffset(dim));
      inputStaticOffsets.push_back(ShapedType::kDynamicStrideOrOffset);
    } else {
      inputStaticOffsets.push_back(tileOp.getStaticOffset(dim));
    }
    if (tileOp.isDynamicSize(dim)) {
      inputTileSizes.push_back(tileOp.getDynamicSize(dim));
      inputStaticSizes.push_back(ShapedType::kDynamicSize);
    } else {
      inputStaticSizes.push_back(tileOp.getStaticSize(dim));
    }
    if (tileOp.isDynamicStride(dim)) {
      inputTileStrides.push_back(tileOp.getDynamicStride(dim));
      inputStaticStrides.push_back(ShapedType::kDynamicStrideOrOffset);
    } else {
      inputStaticStrides.push_back(tileOp.getStaticStride(dim));
    }
  }

  return builder.create<TileOp>(loc, type, newSpace, inputTileOffsets,
                                inputTileSizes, inputTileStrides,
                                builder.getI64ArrayAttr(inputStaticOffsets),
                                builder.getI64ArrayAttr(inputStaticSizes),
                                builder.getI64ArrayAttr(inputStaticStrides));
}
}  // namespace

Value DropDimsOp::compose(OpBuilder &builder) {
  // We can compose with a TileOp operand which has a SpaceOp operand, or
  // compose with a SpaceOp operand.
  return selectDimsFromSuperset(builder, getLoc(), getType(), superset(),
                                remaining_dims());
}

//===----------------------------------------------------------------------===//
// TransposeDimsOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeDimsOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TransposeDimsOp::Adaptor adaptor(operands, attributes, regions);
  const Type argTy = adaptor.superset().getType();

  // If the argument is of point type, so is the result.
  if (auto pointTy = argTy.dyn_cast<PointType>()) {
    inferredReturnTypes.push_back(argTy);
    return success();
  }

  // If the argument is of tile type, we can transpose the type's dimensions.
  if (auto tileTy = argTy.dyn_cast<TileType>()) {
    auto argShape = tileTy.getShape();
    const SmallVector<int64_t> resultShape = llvm::to_vector(llvm::map_range(
        adaptor.permutation(), [&](const auto &d) { return argShape[d]; }));
    auto resultTy = TileType::get(ctx, resultShape);
    inferredReturnTypes.push_back(resultTy);
    return success();
  }

  return failure();
}

Value TransposeDimsOp::compose(OpBuilder &builder) {
  // We can compose with a TileOp operand which has a SpaceOp operand, or
  // compose with a SpaceOp operand. transpose_tile(tile(space, offsets, sizes,
  // strides)) is replaced by tile(transpose(space), transpose(offsets),
  // transpose(sizes), transpose(strides)). transpose_tile(space) is replaced by
  // transpose(space).

  return selectDimsFromSuperset(builder, getLoc(), getType(), superset(),
                                permutation());
}

LogicalResult TransposeDimsOp::verify() {
  // Verify that `permutation` is in fact a permutation.
  size_t rank = permutation().size();
  SmallVector<int64_t> position(rank, -1);
  for (const auto &it : llvm::enumerate(permutation())) {
    int64_t dim = it.value();
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
      return emitOpError("permutation[")
             << it.index() << "] = " << dim << " is outside of range [0, "
             << rank - 1 << "]";
    }
    if (position[dim] >= 0) {
      return emitOpError(
                 "expected permutation attribute to contain no duplicate "
                 "values, but got ")
             << dim << " at positions " << position[dim] << " and "
             << it.index();
    }
    position[dim] = it.index();
  }

  // Verify tile-specific relationship between types and permutation. The
  // constraints between argument and result type are verified through the
  // implementation of `inferReturnTypes`.
  if (auto tileTy = getType().dyn_cast<TileType>()) {
    size_t tileRank = tileTy.getShape().size();
    if (tileRank != rank) {
      return emitOpError("expected result rank ")
             << tileRank << " to match the permutation size of " << rank << ".";
    }
  }

  return success();
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
  result.addAttribute(SetYieldOp::accumulatorFlagsAttrName(result.name),
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
  auto accumulatorCount = llvm::count_if(
      accumulatorFlags(),
      [](Attribute attr) { return attr.cast<BoolAttr>().getValue(); });
  if (accumulatorCount != static_cast<int64_t>(accumulators().size()))
    return emitOpError("expected the number of accumulator regions ")
           << accumulators().size()
           << " to match the number of set accumulator flags "
           << accumulatorCount;

  auto *regionIt = accumulators().begin();
  for (auto item : llvm::zip(srcs(), accumulatorFlags())) {
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
                          {accumulatorFlagsAttrName().str()});

  auto *regionIt = getOperation()->getRegions().begin();
  for (auto &en :
       llvm::enumerate(llvm::zip(srcs(), dsts(), sets(), accumulatorFlags()))) {
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

    if (!parseResult.hasValue()) return success();
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

  result.addAttribute(SetYieldOp::accumulatorFlagsAttrName(result.name),
                      parser.getBuilder().getBoolArrayAttr(accumulatorFlags));
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

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
  return parseDstStyleOp(parser, result);
}

void DynamicBroadcastInDimOp::print(OpAsmPrinter &p) {
  printDstStyleOp(cast<DynamicBroadcastInDimOp>(getOperation()), p);
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
  auto operandSpaceTy = builder.getType<TileType>(operandTy.getShape());
  auto dynamicDims = tensor::createDynamicDimValues(builder, loc, operand());
  auto staticDims = builder.getI64ArrayAttr(operandTy.getShape());
  Value operandSpace =
      builder.create<SpaceOp>(loc, operandSpaceTy, dynamicDims, staticDims);

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
  auto collapsedSubset =
      builder.create<DropDimsOp>(loc, subset, broadcast_dimensionsAttr());

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
    Value collapsedSubsetOffset =
        builder.create<OffsetOp>(loc, collapsedSubset, getIndexConstant(i));
    offsets.push_back(builder.create<arith::SelectOp>(loc, isExpanding, zero,
                                                      collapsedSubsetOffset));
  }

  // If the regarded subset is of point type, we can already construct the
  // operand point and materialize it.
  if (auto pointTy = subsetTy.dyn_cast<PointType>()) {
    auto operandPoint = builder.create<PointOp>(loc, pointTy, operandSpace,
                                                offsets, staticOffsets);
    return builder.create<MaterializeOp>(loc, operandTy.getElementType(),
                                         operand(), operandPoint);
  }

  // If the regarded subset is of tile type, we still need the operand tile
  // sizes to materialize a fused broadcast.
  if (auto tileTy = subsetTy.dyn_cast<TileType>()) {
    // Compute operand tile sizes.
    auto staticTileSizes = builder.getI64ArrayAttr(
        SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
    SmallVector<Value> tileSizes;
    Value one = getIndexConstant(1);
    for (int i = 0; i < operandRank; ++i) {
      Value isExpanding = operandExpandingDims[i];
      Value tileSize =
          builder.create<SizeOp>(loc, collapsedSubset, getIndexConstant(i));
      tileSizes.push_back(
          builder.create<arith::SelectOp>(loc, isExpanding, one, tileSize));
    }

    // Create operand tile.
    auto staticTileStrides =
        builder.getI64ArrayAttr(SmallVector<int64_t>(operandRank, 1));
    SmallVector<Value> tileStrides = {};
    auto operandTileTy = builder.getType<TileType>(
        SmallVector<int64_t>(operandRank, ShapedType::kDynamicSize));
    auto operandTile = builder.create<TileOp>(
        loc, operandTileTy, operandSpace, offsets, tileSizes, tileStrides,
        staticOffsets, staticTileSizes, staticTileStrides);

    // Materialize operand subsets.
    Value tiledInit = builder.create<MaterializeOp>(loc, init(), subset);
    Value tiledOperand =
        builder.create<MaterializeOp>(loc, operand(), operandTile);

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
// OffsetOp
//===----------------------------------------------------------------------===//

OpFoldResult OffsetOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};

  if (auto tileOp = tile().getDefiningOp<TileOp>()) {
    auto idx = idxAttr.getInt();
    if (tileOp.isDynamicOffset(idx)) return tileOp.getDynamicOffset(idx);

    Builder b(idxAttr.getContext());
    return b.getIndexAttr(tileOp.getStaticOffset(idx));
  }
  // TODO(unknown): Handle space op, as well.
  return {};
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

OpFoldResult SizeOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};

  if (auto tileOp = tile().getDefiningOp<TileOp>()) {
    auto idx = idxAttr.getInt();
    if (tileOp.isDynamicSize(idx)) return tileOp.getDynamicSize(idx);

    Builder b(idxAttr.getContext());
    return b.getIndexAttr(tileOp.getStaticSize(idx));
  }
  // TODO(unknown): Handle space op, as well.
  return {};
}

//===----------------------------------------------------------------------===//
// StrideOp
//===----------------------------------------------------------------------===//

OpFoldResult StrideOp::fold(ArrayRef<Attribute> operands) {
  auto idxAttr = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!idxAttr) return {};

  if (auto tileOp = tile().getDefiningOp<TileOp>()) {
    auto idx = idxAttr.getInt();
    if (tileOp.isDynamicStride(idx)) return tileOp.getDynamicStride(idx);

    Builder b(idxAttr.getContext());
    return b.getIndexAttr(tileOp.getStaticStride(idx));
  }
  // TODO(unknown): Handle space op, as well.
  return {};
}

}  // namespace gml_st
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"

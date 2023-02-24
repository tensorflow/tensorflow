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

#include "gml_st/IR/gml_st_ops.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace {

void printShapeTypeDimensionsList(AsmPrinter &printer,
                                  ArrayRef<int64_t> integers) {
  llvm::interleave(
      integers, printer,
      [&](int64_t val) {
        if (val == ShapedType::kDynamic) {
          printer << '?';
        } else {
          printer << val;
        }
      },
      "x");
}

ParseResult parseShapeTypeDimensionsList(AsmParser &parser,
                                         SmallVectorImpl<int64_t> &dims) {
  SmallVector<int64_t> vals;
  if (failed(parser.parseDimensionList(vals, /*allowDynamic=*/true,
                                       /*withTrailingX=*/false))) {
    return failure();
  }
  dims = vals;
  return success();
}

LogicalResult verifyCompatibleExtractedSubset(Operation *op,
                                              ShapedType shapedType,
                                              Type extractedType,
                                              ArrayRef<int64_t> tileShape) {
  auto sourceRank = shapedType.getRank();
  auto elementType = shapedType.getElementType();

  // If the result is a scalar, check that the tile had a single element.
  if (!extractedType.isa<ShapedType>()) {
    if (extractedType != elementType) {
      return op->emitOpError("expected the result type ")
             << extractedType << " to match source element type "
             << elementType;
    }
    if (!ShapedType::isDynamicShape(tileShape) &&
        ShapedType::getNumElements(tileShape) == 1)
      return success();

    return op->emitOpError("expected tile type ")
           << tileShape << " to have a single element shape";
  }

  // If the result is a shaped type, compare with the inferred type.
  auto extractedShapedType = extractedType.cast<ShapedType>();
  unsigned tileRank = tileShape.size();
  if (tileRank != sourceRank) {
    return op->emitOpError("expected source rank = ")
           << sourceRank << " to match tile rank = " << tileRank;
  }

  auto inferredType = shapedType.clone(tileShape, shapedType.getElementType());
  if (extractedShapedType != inferredType) {
    return op->emitOpError("expected result type = ")
           << extractedShapedType
           << " to match the inferred type = " << inferredType;
  }

  return success();
}

}  // namespace
}  // namespace mlir

// Generated dialect definitions.
#include "gml_st/IR/gml_st_dialect.cc.inc"

// Generated type classes.
#define GET_TYPEDEF_CLASSES
#include "gml_st/IR/gml_st_types.cc.inc"

// Generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "gml_st/IR/gml_st_attrs.cc.inc"

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// GmlSt Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct GmlStInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Operations in GmlSt dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    auto yieldOp = dyn_cast<gml_st::YieldOp>(op);
    if (!yieldOp) return;

    for (auto [valueToRepl, operand] :
         llvm::zip(valuesToRepl, yieldOp.getOperands())) {
      valueToRepl.replaceAllUsesWith(operand);
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// GmlStDialect
//===----------------------------------------------------------------------===//

void GmlStDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gml_st/IR/gml_st_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "gml_st/IR/gml_st_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "gml_st/IR/gml_st_attrs.cc.inc"
      >();

  addInterfaces<GmlStInlinerInterface>();
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
// LoopLikeOp
//===----------------------------------------------------------------------===//

namespace {

ParseResult parseLoopLikeOpOutputArgs(
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
    *outputCount = static_cast<int32_t>(outputs.size());
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

  // Parse the output tensors and the body.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> regionOperands(ivs);
  SmallVector<Type, 4> regionTypes(ivs.size(), builder.getIndexType());

  int32_t outputCount = 0;
  if (failed(parseLoopLikeOpOutputArgs(parser, result, regionOperands,
                                       regionTypes, &outputCount)))
    return failure();

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
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(lower.size()),
                           static_cast<int32_t>(upper.size()),
                           static_cast<int32_t>(steps.size()), outputCount}));

  return success();
}

template <typename LoopTy>
void buildLoopLikeOp(
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
      LoopTy::getOperandSegmentSizeAttr(),
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
    LoopTy::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

struct CollapseSingleIterationLoops : public OpRewritePattern<ParallelOp> {
  explicit CollapseSingleIterationLoops(
      MLIRContext *context,
      llvm::function_ref<bool(ParallelOp)> filterFn = nullptr,
      PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelOp>(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (filterFn && !filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");

    IRMapping mapping;
    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<Value> newLowerBounds, newUpperBounds, newSteps;
    newLowerBounds.reserve(op.getLowerBound().size());
    newUpperBounds.reserve(op.getUpperBound().size());
    newSteps.reserve(op.getStep().size());
    auto getConstant = [](Value v) -> Optional<int64_t> {
      auto constant =
          dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
      if (constant) return constant.value();
      return std::nullopt;
    };
    for (auto [lowerBound, upperBound, step, iv] :
         llvm::zip(op.getLowerBound(), op.getUpperBound(), op.getStep(),
                   op.getInductionVars())) {
      // Collect the statically known loop bounds.
      auto lowerBoundConstant = getConstant(lowerBound);
      auto upperBoundConstant = getConstant(upperBound);
      auto stepConstant = getConstant(step);
      // Remove the loop if it performs zero iterations.
      if (lowerBoundConstant && upperBoundConstant &&
          *lowerBoundConstant == *upperBoundConstant) {
        rewriter.replaceOp(op, op.getOutputs());
        return success();
      }
      // Replace the loop induction variable by the lower bound if the loop
      // performs a single iteration. Otherwise, copy the loop bounds.
      if (lowerBoundConstant && upperBoundConstant && stepConstant &&
          (*upperBoundConstant - *lowerBoundConstant) > 0 &&
          (*upperBoundConstant - *lowerBoundConstant) <= *stepConstant) {
        mapping.map(iv, lowerBound);
      } else {
        newLowerBounds.push_back(lowerBound);
        newUpperBounds.push_back(upperBound);
        newSteps.push_back(step);
      }
    }
    // Exit if none of the loop dimensions perform a single iteration.
    if (newLowerBounds.size() == op.getLowerBound().size()) return failure();

    // All of the loop dimensions perform a single iteration. Inline loop body.
    if (newLowerBounds.empty()) {
      mapping.map(op.getRegionOutputArgs(), op.getOutputs());
      for (auto &bodyOp : op.getBody()->without_terminator()) {
        rewriter.clone(bodyOp, mapping);
      }
      SmallVector<Value> results;
      results.reserve(op.getResults().size());
      SetYieldOp terminator = op.getTerminator();
      for (const auto &[dst, src, set] :
           llvm::zip(terminator.getDsts(), terminator.getSrcs(),
                     terminator.getSets())) {
        auto tileOp = set.template getDefiningOp<TileOp>();

        if (!tileOp) {
          return terminator.emitOpError(
              "expected the SetYieldOp terminator of gml_st loop to have a "
              "TileOp set");
        }
        auto getMappedValues = [&](ValueRange values) {
          return llvm::to_vector(llvm::map_range(values, [&](Value value) {
            return mapping.lookupOrDefault(value);
          }));
        };

        if (dst.getType().template isa<TensorType>()) {
          Value srcVal = mapping.lookupOrDefault(src);
          if (srcVal.getType().isa<TensorType>()) {
            results.push_back(rewriter.create<tensor::InsertSliceOp>(
                op.getLoc(), dst.getType(), srcVal,
                mapping.lookupOrDefault(dst),
                getMappedValues(tileOp.getOffsets()),
                getMappedValues(tileOp.getSizes()),
                getMappedValues(tileOp.getStrides()), tileOp.getStaticOffsets(),
                tileOp.getStaticSizes(), tileOp.getStaticStrides()));
          } else {
            SmallVector<Value> mappedOffsets =
                getMappedValues(tileOp.getOffsets());
            SmallVector<OpFoldResult> ofrs;
            int idx = 0;
            for (int64_t offset : tileOp.getStaticOffsets()) {
              if (ShapedType::isDynamic(offset)) {
                ofrs.push_back(mappedOffsets[idx++]);
              } else {
                ofrs.push_back(rewriter.getIndexAttr(offset));
              }
            }
            results.push_back(rewriter.create<tensor::InsertOp>(
                op.getLoc(), srcVal, mapping.lookupOrDefault(dst),
                getAsValues(rewriter, op.getLoc(), ofrs)));
          }
        } else if (dst.getType().template isa<VectorType>()) {
          results.push_back(rewriter.create<vector::InsertStridedSliceOp>(
              op.getLoc(), dst.getType(), mapping.lookupOrDefault(src),
              mapping.lookupOrDefault(dst),
              rewriter.getI64ArrayAttr(tileOp.getStaticSizes()),
              rewriter.getI64ArrayAttr(tileOp.getStaticStrides())));
        } else {
          return op.emitOpError(
              "expected output of gml_st loop to be either a tensor or a "
              "vector");
        }
      }
      rewriter.replaceOp(op, results);
      return success();
    }

    // Replace the loop by a lower-dimensional loop.
    ParallelOp newOp;
    newOp = rewriter.create<ParallelOp>(op.getLoc(), op.getResultTypes(),
                                        newLowerBounds, newUpperBounds,
                                        newSteps, op.getOutputs());
    // The new loop needs to keep all attributes from the old one, except for
    // "operand_segment_sizes" which captures the outdated information of the
    // old iteration domain.
    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == ParallelOp::getOperandSegmentSizeAttr())
        continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

 private:
  llvm::function_ref<bool(ParallelOp)> filterFn;
};

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

Region &ParallelOp::getLoopBody() { return getRegion(); }

SetYieldOp ParallelOp::getTerminator() {
  return cast<SetYieldOp>(getBody()->getTerminator());
}

LogicalResult ParallelOp::verify() {
  if (getNumResults() != getNumOutputs()) {
    return emitOpError() << "expected the number of output arguments to match "
                            "the number of results";
  }

  // Check if types of output arguments match region args types.
  for (auto &item : llvm::enumerate(
           llvm::zip(getOutputs(), getRegionOutputArgs(), getResultTypes()))) {
    Value output, outputRegionArg;
    Type resultType;
    unsigned index = item.index();
    std::tie(output, outputRegionArg, resultType) = item.value();
    if (output.getType() != outputRegionArg.getType()) {
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match region arg " << index + getNumLoops()
             << " type = " << outputRegionArg.getType();
    }
    if (output.getType() != resultType) {
      return emitOpError("expected output arg ")
             << index << " with type = " << output.getType()
             << " to match resultType " << index << " type = " << resultType;
    }
    auto terminator = getTerminator();
    auto numDstOperands = terminator.getNumDstOperands();
    if (index >= numDstOperands) {
      const auto *s = index ? "s" : "";
      return terminator.emitOpError("expected to have at least ")
             << index + 1 << " destination operand" << s << " (currently "
             << numDstOperands << ")";
    }

    if (terminator.getDstOperand(index)->get() != outputRegionArg) {
      return terminator.emitOpError("expected output block argument ")
             << index << " to match set_yield destination";
    }
  }
  return success();
}

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps,
    ValueRange outputs, std::optional<StringAttr> distributionType,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  if (distributionType.has_value())
    result.addAttribute(getDistributionTypeAttrName(result.name),
                        distributionType.value());

  buildLoopLikeOp<ParallelOp>(builder, result, resultTypes, lowerBounds,
                              upperBounds, steps, outputs, bodyBuilderFn);
}

void ParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getInductionVars() << ") = (" << getLowerBound() << ") to ("
    << getUpperBound() << ") step (" << getStep() << ") ";

  if (!getOutputs().empty()) {
    p << "outs (";
    llvm::interleaveComma(
        llvm::zip(getRegionOutputArgs(), getOutputs()), p, [&](auto it) {
          Value outputRegionArg, output;
          std::tie(outputRegionArg, output) = it;
          p << outputRegionArg << " = " << output << ": " << output.getType();
        });
    p << ") ";
  }

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

namespace {

/// Fold tensor.dim(gml_st.parallel outs(... = %t)) to tensor.dim(%t).
struct DimOfParallelOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const final {
    auto parallelOp = dimOp.getSource().getDefiningOp<ParallelOp>();
    if (!parallelOp) return failure();

    OpOperand &out =
        parallelOp.getOpOperandForResult(dimOp.getSource().cast<OpResult>());
    rewriter.updateRootInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(out.get()); });
    return success();
  }
};

/// Fold tensor.casts into the output arguments of gml_st.parallel.
struct FoldTensorCastIntoParallelOp
    : public OpRewritePattern<gml_st::ParallelOp> {
  using OpRewritePattern<gml_st::ParallelOp>::OpRewritePattern;

  struct TypeCast {
    Type srcType;
    Type dstType;
  };

  LogicalResult matchAndRewrite(gml_st::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const final {
    llvm::SmallMapVector<unsigned, TypeCast, 2> tensorCastProducers;
    llvm::SmallVector<Value> newOutputTensors = parallelOp.getOutputs();
    for (auto &en : llvm::enumerate(newOutputTensors)) {
      if (auto castOp = en.value().getDefiningOp<tensor::CastOp>()) {
        tensorCastProducers[en.index()] =
            TypeCast{castOp.getSource().getType(), castOp.getType()};
        en.value() = castOp.getSource();
      }
    }

    if (tensorCastProducers.empty()) return failure();

    // Create new loop.
    Location loc = parallelOp.getLoc();
    std::optional<StringAttr> distTypeAttr;
    if (auto distType = parallelOp.getDistributionType())
      distTypeAttr = rewriter.getStringAttr(*distType);
    auto newParallelOp = rewriter.create<ParallelOp>(
        loc, TypeRange{ValueRange{newOutputTensors}},
        parallelOp.getLowerBound(), parallelOp.getUpperBound(),
        parallelOp.getStep(), newOutputTensors, distTypeAttr, nullptr);

    Block *loopBody = newParallelOp.getBody();

    // Cast bbArgs back to the original types.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> castBBArgs =
        ValueRange{newParallelOp.getRegionOutputArgs()};
    for (auto &item : tensorCastProducers) {
      Value &oldTypeBBArg = castBBArgs[item.first];
      oldTypeBBArg = rewriter.create<tensor::CastOp>(loc, item.second.dstType,
                                                     oldTypeBBArg);
    }

    // Move old body into new parallel loop.
    SmallVector<Value> blockArgs = newParallelOp.getInductionVars();
    blockArgs.append(castBBArgs);
    rewriter.mergeBlocks(parallelOp.getBody(), loopBody, blockArgs);

    // Cast `set_yield` destination operands to the new types.
    SetYieldOp terminator = newParallelOp.getTerminator();
    rewriter.setInsertionPoint(terminator);
    SmallVector<Value> castDsts = terminator.getDsts();
    for (auto &item : tensorCastProducers) {
      Value &newTypeDsts = castDsts[item.first];
      newTypeDsts = rewriter.create<tensor::CastOp>(loc, item.second.srcType,
                                                    newTypeDsts);
    }
    terminator.getDstsMutable().assign(castDsts);

    // Cast results back to the original types.
    rewriter.setInsertionPointAfter(newParallelOp);
    SmallVector<Value> castResults = newParallelOp.getResults();
    for (auto &item : tensorCastProducers) {
      Value &oldTypeResult = castResults[item.first];
      oldTypeResult = rewriter.create<tensor::CastOp>(loc, item.second.dstType,
                                                      oldTypeResult);
    }
    rewriter.replaceOp(parallelOp, castResults);

    return success();
  }
};

}  // namespace

void ParallelOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<CollapseSingleIterationLoops>(context, [&](ParallelOp op) {
    return !op.getDistributionType().has_value();
  });
  results.add<DimOfParallelOp, FoldTensorCastIntoParallelOp>(context);
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

namespace {
/// Fold gml_st.tile [%c0] ... into gml_st.tile [0] ...
/// Adapted from OpWithOffsetSizesAndStridesConstantArgumentFolder, which makes
/// slightly incompatible assumptions about the op.
struct FoldConstantsIntoTileType : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(op.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(rewriter, mixedOffsets)) &&
        failed(foldDynamicIndexList(rewriter, mixedSizes)) &&
        failed(foldDynamicIndexList(rewriter, mixedStrides)))
      return failure();

    // Create the new tile in canonical form.
    TileOp newOp = rewriter.create<TileOp>(op.getLoc(), mixedOffsets,
                                           mixedSizes, mixedStrides);
    // Cast the result back to the original type. This will be folded further
    // materialize ops.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{op.getType()}, ValueRange{newOp});

    return success();
  }
};
}  // namespace

void TileOp::build(OpBuilder &b, OperationState &result,
                   ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                   ArrayRef<OpFoldResult> strides,
                   ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto tileType = TileType::get(b.getContext(), staticSizes);
  build(b, result, tileType, dynamicOffsets, dynamicSizes, dynamicStrides,
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

void TileOp::build(OpBuilder &b, OperationState &result,
                   ArrayRef<OpFoldResult> offsets,
                   ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> unitSizesAndStrides(offsets.size(),
                                                b.getIndexAttr(1));
  return build(b, result, offsets, unitSizesAndStrides, unitSizesAndStrides,
               attrs);
}

LogicalResult TileOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> /*loc*/, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Derive result shape.
  TileOp::Adaptor adaptor(operands, attributes, regions);
  SmallVector<int64_t> shape = llvm::to_vector(adaptor.getStaticSizes());

  auto resultTy = TileType::get(ctx, shape);
  inferredReturnTypes.push_back(resultTy);
  return success();
}

LogicalResult TileOp::verify() {
  auto resultType = getType();
  auto rank = resultType.getRank();
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "size", rank, getStaticSizes(), getSizes()))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "offset", rank, getStaticOffsets(), getOffsets()))) {
    return failure();
  }
  if (failed(mlir::verifyListOfOperandsOrIntegers(
          getOperation(), "stride", rank, getStaticStrides(), getStrides()))) {
    return failure();
  }
  for (auto [tileSize, offset, size, stride] :
       llvm::zip(resultType.getShape(), getStaticOffsets(), getStaticSizes(),
                 getStaticStrides())) {
    if (offset < 0 && offset != ShapedType::kDynamic) {
      return emitOpError("expected offset = ")
             << offset << " to be non-negative";
    }
    if (size < 0 && size != ShapedType::kDynamic) {
      return emitOpError("expected size = ") << size << " to be non-negative";
    }
    if (stride < 0 && stride != ShapedType::kDynamic) {
      return emitOpError("expected stride = ")
             << stride << " to be non-negative";
    }
    if (tileSize != size) {
      return emitOpError("size arg = ")
             << size << " does not match tile size = " << tileSize;
    }
  }
  return success();
}

void TileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldConstantsIntoTileType>(context);
}

//===----------------------------------------------------------------------===//
// SetYieldOp
//===----------------------------------------------------------------------===//

using AccumulatorRegionBuilderFn =
    function_ref<void(OpBuilder &, Location, Value, Value)>;

void SetYieldOp::build(OpBuilder &builder, OperationState &result) {
  build(builder, result, std::nullopt, std::nullopt, std::nullopt);
}

void SetYieldOp::build(OpBuilder &builder, OperationState &result,
                       ValueRange srcs, ValueRange dsts, ValueRange sets) {
  SmallVector<bool, 2> accumulatorFlags(srcs.size(), false);
  build(builder, result, srcs, dsts, sets,
        builder.getBoolArrayAttr(accumulatorFlags), std::nullopt);
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
    if (failed(verifyCompatibleExtractedSubset(
            getOperation(), dst.getType().cast<ShapedType>(), src.getType(),
            set.getType().cast<TileType>().getShape())))
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
    auto parseResult = parser.parseOptionalOperand(src);

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

namespace {
/// Folds UnrealizedConversionCast of TileType into SetYieldOp.
struct FoldTileCastIntoSetYield : public OpRewritePattern<SetYieldOp> {
  using OpRewritePattern<SetYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SetYieldOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::any_of(op.getSets(), [](auto set) {
          return set.template getDefiningOp<UnrealizedConversionCastOp>() !=
                 nullptr;
        }))
      return failure();
    SmallVector<Value> newSrcs{op.getSrcs()};
    SmallVector<Value> newSets{op.getSets()};
    for (auto &&[src, set] : llvm::zip(newSrcs, newSets)) {
      auto cast = set.getDefiningOp<UnrealizedConversionCastOp>();
      if (!cast) continue;
      set = cast.getOperand(0);
      Type castResultType = src.getType();
      if (auto shapedType = dyn_cast<ShapedType>(castResultType)) {
        castResultType =
            shapedType.clone(set.getType().cast<TileType>().getShape(),
                             shapedType.getElementType());
        src = rewriter.create<tensor::CastOp>(op.getLoc(), castResultType, src);
      }
    }
    rewriter.replaceOpWithNewOp<SetYieldOp>(op, newSrcs, op.getDsts(), newSets);
    return success();
  }
};
}  // namespace

void SetYieldOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<FoldTileCastIntoSetYield>(context);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// FusionOp
//===----------------------------------------------------------------------===//

YieldOp FusionOp::getTerminator() {
  return cast<YieldOp>(getBody()->getTerminator());
}

void FusionOp::print(OpAsmPrinter &p) {
  p << " (";
  llvm::interleaveComma(
      llvm::zip(getBody()->getArguments(), getInputs()), p, [&](auto it) {
        Value inputRegionArg, input;
        std::tie(inputRegionArg, input) = it;
        p << inputRegionArg << " = " << input << ": " << input.getType();
      });
  p << ") ";

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(getOperation()->getAttrs());

  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleave(getResultTypes(), p, ", ");
  }
}

ParseResult FusionOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands, regionOperands;
  SmallVector<Type, 4> operandTypes;

  auto parseElt = [&]() -> ParseResult {
    if (parser.parseOperand(regionOperands.emplace_back(),
                            /*allowResultNumber=*/false) ||
        parser.parseEqual()) {
      return failure();
    }
    if (parser.parseOperand(operands.emplace_back()) || parser.parseColon() ||
        parser.parseType(operandTypes.emplace_back())) {
      return failure();
    }
    return success();
  };

  // Parse argument list.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt))
    return failure();

  SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(operands, operandTypes, loc, result.operands))
    return failure();

  // Parse region.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  for (auto argAndType : llvm::zip(regionOperands, operandTypes)) {
    auto &arg = regionArgs.emplace_back();
    std::tie(arg.ssaName, arg.type) = argAndType;
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // Parser result types.
  if (parser.parseOptionalColonTypeList(result.types)) return failure();

  return success();
}

LogicalResult FusionOp::verify() {
  llvm::SetVector<Value> valuesDefinedAbove;
  getUsedValuesDefinedAbove(getRegion(), getRegion(), valuesDefinedAbove);

  for (Value v : valuesDefinedAbove) {
    auto *definingOp = v.getDefiningOp();

    if (!isa_and_nonnull<arith::ConstantOp>(definingOp))
      return emitOpError() << "using value defined outside the region that is "
                              "not 'arith.constant'.";
  }

  return success();
}

}  // namespace gml_st
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "gml_st/IR/gml_st_ops.cc.inc"

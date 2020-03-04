/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include "absl/memory/memory.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/AffineExpr.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace {

ArrayAttr GetNParallelLoopsAttrs(unsigned nParallelLoops, Builder* b) {
  auto parallelLoopTypeAttr = b->getStringAttr("parallel");
  SmallVector<Attribute, 3> iteratorTypes;
  for (int i = 0; i < nParallelLoops; ++i) {
    iteratorTypes.push_back(parallelLoopTypeAttr);
  }
  return b->getArrayAttr(iteratorTypes);
}

template <bool isLHLO = true>
ShapedType getXLAOpResultType(Operation* op) {
  if (isLHLO) {
    return op->getOperand(op->getNumOperands() - 1)
        .getType()
        .cast<ShapedType>();
  }
  return op->getResult(0).getType().cast<ShapedType>();
}

template <bool isLHLO = true>
bool verifyXLAOpBufferOrTensorSemantics(Operation* op) {
  auto verifyType = [&](Value val) -> bool {
    return (isLHLO && val.getType().isa<MemRefType>()) ||
           (!isLHLO && val.getType().isa<RankedTensorType>());
  };
  if (!llvm::all_of(op->getOperands(), verifyType)) return false;
  return isLHLO ? op->getResults().empty()
                : llvm::all_of(op->getResults(), verifyType);
}

template <typename OpTy, bool isLHLO = true>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto argType =
        op.getOperation()->getOperand(0).getType().template cast<ShapedType>();
    if (!argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects ranked args");
      return ConversionPattern::matchFailure();
    }
    if (!argType.getElementType().isSignlessIntOrFloat()) {
      return ConversionPattern::matchFailure();
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Attribute, 2> indexingMaps;
    SmallVector<Type, 4> bodyArgTypes, bodyResultTypes, opResultTypes;

    // This doesnt account for implicit broadcast, but the working assumption
    // here is that are broadcasts have been made explicit.
    unsigned nloops = argType.getRank();
    if (!nloops) {
      return ConversionPattern::matchFailure();
    }
    int operandCount = (isLHLO ? args.size() - 1 : args.size());
    auto verifyArgOrResultType = [&](Value val) -> ShapedType {
      auto shapedType = val.getType().dyn_cast<ShapedType>();
      if (!shapedType ||
          (!shapedType.isa<MemRefType>() &&
           !shapedType.isa<RankedTensorType>()) ||
          shapedType.getRank() != nloops)
        return nullptr;
      indexingMaps.emplace_back(
          AffineMapAttr::get(rewriter.getMultiDimIdentityMap(nloops)));
      return shapedType;
    };
    for (const auto& arg : llvm::enumerate(args)) {
      auto shapedType = verifyArgOrResultType(arg.value());
      if (!shapedType) return ConversionPattern::matchFailure();
      auto& result_or_body_arg =
          arg.index() < operandCount ? bodyArgTypes : bodyResultTypes;
      result_or_body_arg.emplace_back(shapedType.getElementType());
    }
    if (!isLHLO) {
      // HLO operations have return as tensor types.
      assert(bodyResultTypes.empty() &&
             "When lowering HLO ops result can't be part of arguments");
      Value result = op.getOperation()->getResult(0);
      auto shapedType = verifyArgOrResultType(result);
      if (!shapedType) return ConversionPattern::matchFailure();
      bodyResultTypes.push_back(shapedType.getElementType());
      opResultTypes.push_back(shapedType);
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, opResultTypes, args,
        rewriter.getI64IntegerAttr(bodyArgTypes.size()),     // args_in
        rewriter.getI64IntegerAttr(bodyResultTypes.size()),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, &rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(bodyArgTypes);
    if (isLHLO) block->addArguments(bodyResultTypes);

    SmallVector<Value, 4> bodyArgs;
    for (int i = 0, e = bodyArgTypes.size(); i < e; ++i) {
      bodyArgs.push_back(block->getArgument(i));
    }

    rewriter.setInsertionPointToEnd(block);
    // TODO(ravishankarm) : For now use the method in xla_lhlo namespace. That
    // method needs to be moved out of there.
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<OpTy>(
        llvm::cast<OpTy>(op), bodyResultTypes, bodyArgs, &rewriter);
    if (!opResult) {
      return ConversionPattern::matchFailure();
    }
    rewriter.create<linalg::YieldOp>(loc, opResult);
    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return ConversionPattern::matchSuccess();
  }
};

template <typename LhloOp>
class ScalarPointwiseToStandardConverter : public OpConversionPattern<LhloOp> {
 public:
  using OpConversionPattern<LhloOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      LhloOp lhlo_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = lhlo_op.getLoc();
    auto argType =
        lhlo_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.getElementType().isSignlessIntOrFloat() ||
        (argType.getRank() != 0)) {
      return ConversionPattern::matchFailure();
    }

    // Create two loads from the input.
    auto lhs = rewriter.create<LoadOp>(loc, lhlo_op.lhs());
    auto rhs = rewriter.create<LoadOp>(loc, lhlo_op.rhs());
    // TODO(ravishankarm) : Move this method out of xla_lhlo namespace.
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<LhloOp>(
        llvm::cast<LhloOp>(lhlo_op), argType.getElementType(),
        llvm::ArrayRef<Value>{lhs, rhs}, &rewriter);
    rewriter.create<StoreOp>(loc, opResult, lhlo_op.out());
    rewriter.eraseOp(lhlo_op);
    return ConversionPattern::matchSuccess();
  }
};

/// Base class for lowering xla operations that have one operand and one result,
/// and are semantically equivalent to a copy of the input to the output (like
/// transpose, some reshape, etc.). The derived classes need to provide a method
/// `getIndexingMapsAttr` that returns an ArrayAttr containing AffineMapAttr for
/// the index maps of the input and the output.
template <typename Derived, typename OpTy, bool isLHLO = true>
class DataMovementOpConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!verifyXLAOpBufferOrTensorSemantics<isLHLO>(op))
      return ConversionPattern::matchFailure();
    auto operandType = op.operand().getType().template cast<ShapedType>();
    auto resultType = getXLAOpResultType<isLHLO>(op);
    if (!verifyXLAOpBufferOrTensorSemantics<isLHLO>(op))
      return ConversionPattern::matchFailure();
    // TODO(b/150203558) Enable once tiling/fusion works in this case.
    if (isLHLO && (operandType.getRank() == 0))
      return ConversionPattern::matchFailure();
    ArrayAttr indexingMapsAttr =
        static_cast<const Derived&>(*this).getIndexingMapsAttr(op, &rewriter);
    if (!indexingMapsAttr) return ConversionPattern::matchFailure();

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    auto nloops = resultType.getRank();
    auto loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, isLHLO ? ArrayRef<Type>{} : resultType, args,
        rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1),
        indexingMapsAttr, GetNParallelLoopsAttrs(nloops, &rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(operandType.getElementType());
    if (isLHLO) block->addArgument(resultType.getElementType());

    rewriter.setInsertionPointToEnd(block);
    rewriter.create<linalg::YieldOp>(loc, block->getArgument(0));

    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return ConversionPattern::matchSuccess();
  }
};

template <typename OpTy, bool isLHLO = true>
class BroadcastInDimConverter
    : public DataMovementOpConverter<BroadcastInDimConverter<OpTy, isLHLO>,
                                     OpTy, isLHLO> {
 public:
  using DataMovementOpConverter<BroadcastInDimConverter<OpTy, isLHLO>, OpTy,
                                isLHLO>::DataMovementOpConverter;

  ArrayAttr getIndexingMapsAttr(OpTy broadcastOp, Builder* b) const {
    auto resultType = getXLAOpResultType<isLHLO>(broadcastOp);
    auto operandType =
        broadcastOp.operand().getType().template cast<ShapedType>();
    unsigned nloops = resultType.getRank();

    auto operandShape = operandType.getShape();
    SmallVector<AffineExpr, 4> dimExprs;
    {
      dimExprs.reserve(nloops);

      if (broadcastOp.broadcast_dimensions()) {
        for (const auto& broadcastDim :
             enumerate(broadcastOp.broadcast_dimensions()
                           .getValue()
                           .getIntValues())) {
          int size = broadcastDim.value().getSExtValue();
          // TODO(pifon): Add support for args with dynamic shapes for the case
          // when a dimension of size 1 is broadcasted into dim of size N.
          AffineExpr affineExpr = operandShape[broadcastDim.index()] == 1
                                      ? b->getAffineConstantExpr(0)
                                      : b->getAffineDimExpr(size);
          dimExprs.push_back(affineExpr);
        }
      }
      if (dimExprs.empty()) {
        // The input is a scalar, i.e. this is a scalar broadcast op.
        dimExprs.push_back(b->getAffineConstantExpr(0));
      }
    }
    return b->getAffineMapArrayAttr(
        {AffineMap::get(nloops, /*symbolCount=*/0, dimExprs),
         b->getMultiDimIdentityMap(nloops)});
  }
};

// Special case for scalar broadcast in lhlo.
// TODO(b/150203558) Remove once the bug is fixed.
class ScalarBroadcastInDimConverter
    : public OpConversionPattern<xla_lhlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern<xla_lhlo::BroadcastInDimOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::BroadcastInDimOp broadcastOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto operandMemrefType =
        broadcastOp.operand().getType().dyn_cast<MemRefType>();
    // Only support scalar operands.
    if (operandMemrefType.getRank() != 0) return matchFailure();
    auto resultMemrefType =
        broadcastOp.output().getType().dyn_cast<MemRefType>();
    if (!operandMemrefType || !resultMemrefType) return matchFailure();
    auto broadcastDims = broadcastOp.broadcast_dimensions();
    if (!broadcastDims.hasValue()) return matchFailure();

    unsigned nloops = resultMemrefType.getRank();
    SmallVector<Attribute, 1> indexingMaps{
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(nloops))};
    auto loc = broadcastOp.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>{}, broadcastOp.output(),
        rewriter.getI64IntegerAttr(0),  // args_in
        rewriter.getI64IntegerAttr(1),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, &rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(resultMemrefType.getElementType());

    rewriter.setInsertionPointToEnd(block);
    auto scalar =
        rewriter.create<LoadOp>(loc, broadcastOp.operand(), llvm::None);
    rewriter.create<linalg::YieldOp>(loc, scalar.getResult());
    rewriter.eraseOp(broadcastOp);
    return matchSuccess();
  }
};

template <typename OpTy, bool isLHLO = true>
class TransposeConverter
    : public DataMovementOpConverter<TransposeConverter<OpTy, isLHLO>, OpTy,
                                     isLHLO> {
 public:
  using DataMovementOpConverter<TransposeConverter<OpTy, isLHLO>, OpTy,
                                isLHLO>::DataMovementOpConverter;
  ArrayAttr getIndexingMapsAttr(OpTy op, Builder* b) const {
    auto resultType =
        getXLAOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultType.getRank());
    for (auto permutation : llvm::enumerate(op.permutation())) {
      inputExprs[permutation.value().getZExtValue()] =
          b->getAffineDimExpr(permutation.index());
    }
    return b->getAffineMapArrayAttr(
        {AffineMap::get(nloops, /*symbolCount=*/0, inputExprs),
         b->getMultiDimIdentityMap(nloops)});
  }
};

/// Pattern for the special case where reshape is adding or removing a dimension
/// of size 1. These can be lowered to a linalg.generic op.
///
/// For example a
///   "xla_hlo.reshape"(..) : (tensor<12x1x42xi32) -> tensor<12x42xi32>
/// can have indexing maps
/// [affine_map<(d0, d1) -> (d0, 0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
///
/// Similarly a
///   "xla_hlo.reshape"(..) : (tensor<12x42xi32>) -> tensor<12x1x42xi32>
/// can have indexing maps
/// [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1,
/// d2)>]
template <typename OpTy, bool isLHLO = true>
class ReshapeAddRemoveDimConverter
    : public DataMovementOpConverter<ReshapeAddRemoveDimConverter<OpTy, isLHLO>,
                                     OpTy, isLHLO> {
 public:
  using DataMovementOpConverter<ReshapeAddRemoveDimConverter<OpTy, isLHLO>,
                                OpTy, isLHLO>::DataMovementOpConverter;

  ArrayAttr getIndexingMapsAttr(OpTy op, Builder* b) const {
    auto resultType =
        getXLAOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto operandType =
        op.getOperation()->getOperand(0).getType().template cast<ShapedType>();
    if (!resultType.hasStaticShape() || !operandType.hasStaticShape())
      return nullptr;

    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    unsigned resultIndex = 0, operandIndex = 0;
    auto resultShape = resultType.getShape();
    auto operandShape = operandType.getShape();

    while (resultIndex < resultShape.size() &&
           operandIndex < operandShape.size()) {
      if (resultShape[resultIndex] == operandShape[operandIndex]) {
        // Copy over the affine expr when the size of the result and operand
        // match at a dim
        inputExprs.push_back(b->getAffineDimExpr(resultIndex));
        resultIndex++;
        operandIndex++;
      } else if (resultShape[resultIndex] == 1) {
        // If size at result is 1, then ignore this dimension for the input, it
        // is an extra dim added.
        resultIndex++;
      } else if (operandShape[operandIndex] == 1) {
        // If the operandShape is 1, then add a (0) for the operand map since
        // this dimension is dropped.
        inputExprs.push_back(b->getAffineConstantExpr(0));
        operandIndex++;
      } else {
        return nullptr;
      }
    }
    // Make sure all remaining dimensions of the operand and result are ones.
    auto checkRemainingDims = [](int64_t dim) { return dim != 1; };
    if ((resultIndex < resultShape.size() &&
         llvm::any_of(resultShape.drop_front(resultIndex),
                      checkRemainingDims)) ||
        (operandIndex < operandShape.size() &&
         llvm::any_of(operandShape.drop_front(operandIndex),
                      checkRemainingDims)))
      return nullptr;
    inputExprs.resize(operandShape.size(), b->getAffineConstantExpr(0));
    return b->getAffineMapArrayAttr(
        {AffineMap::get(nloops, /*symbolCount=*/0, inputExprs),
         b->getMultiDimIdentityMap(nloops)});
  }
};

class IotaConverter : public OpConversionPattern<xla_lhlo::IotaOp> {
 public:
  using OpConversionPattern<xla_lhlo::IotaOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::IotaOp iotaOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto resultMemrefType =
        iotaOp.getOperand().getType().dyn_cast<MemRefType>();
    if (!resultMemrefType) return matchFailure();

    auto resultElementType = resultMemrefType.getElementType();
    if (!resultElementType.isSignlessIntOrFloat()) return matchFailure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = resultMemrefType.getRank();
    SmallVector<Attribute, 2> indexingMaps;
    indexingMaps.emplace_back(
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(nloops)));

    auto loc = iotaOp.getLoc();
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, ArrayRef<Type>{}, args,
        rewriter.getI64IntegerAttr(0),  // args_in
        rewriter.getI64IntegerAttr(1),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, &rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    for (unsigned i = 0; i < nloops; ++i) {
      block->addArgument(rewriter.getIndexType());
    }
    block->addArguments(llvm::makeArrayRef(resultElementType));

    rewriter.setInsertionPointToEnd(block);
    Operation* castOp = rewriter.create<IndexCastOp>(
        loc, block->getArgument(iotaOp.iota_dimension().getZExtValue()),
        rewriter.getIntegerType(resultElementType.getIntOrFloatBitWidth()));
    if (resultElementType.isa<FloatType>()) {
      castOp = rewriter.create<SIToFPOp>(loc, castOp->getResult(0),
                                         resultElementType);
    }
    rewriter.create<linalg::YieldOp>(loc, castOp->getResult(0));
    rewriter.eraseOp(iotaOp);
    return matchSuccess();
  }
};

class ConstConverter : public OpConversionPattern<xla_lhlo::ConstOp> {
 public:
  using OpConversionPattern<xla_lhlo::ConstOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::ConstOp constOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = constOp.getLoc();
    auto valueAttr = constOp.value().cast<DenseElementsAttr>();
    if (valueAttr.getType().getRank() != 0) return matchFailure();
    auto stdConstOp =
        rewriter.create<mlir::ConstantOp>(loc, valueAttr.getValue({}));
    rewriter.create<mlir::StoreOp>(loc, stdConstOp, constOp.getOperand());
    rewriter.eraseOp(constOp);
    return matchSuccess();
  }
};

class SliceConverter : public OpConversionPattern<xla_lhlo::SliceOp> {
 public:
  using OpConversionPattern<xla_lhlo::SliceOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::SliceOp sliceOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = sliceOp.getLoc();
    auto argType =
        sliceOp.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects known-rank args");
      return ConversionPattern::matchFailure();
    }

    SmallVector<Value, 3> ranges;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      Value start_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.start_indices().getValue<int64_t>(i));
      Value limit_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.limit_indices().getValue<int64_t>(i));
      Value stride = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.strides().getValue<int64_t>(i));
      ranges.push_back(rewriter.create<linalg::RangeOp>(loc, start_index,
                                                        limit_index, stride));
    }
    auto linalg_slice =
        rewriter.create<linalg::SliceOp>(loc, sliceOp.getOperand(0), ranges);
    rewriter.create<linalg::CopyOp>(loc, linalg_slice, sliceOp.getOperand(1));
    rewriter.eraseOp(sliceOp);
    return matchSuccess();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<BroadcastInDimConverter<xla_lhlo::BroadcastInDimOp>,
                   ConstConverter,
                   IotaConverter,
                   PointwiseToLinalgConverter<xla_lhlo::AbsOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AddOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AndOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CeilOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CompareOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ConvertOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CopyOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CosOp>,
                   PointwiseToLinalgConverter<xla_lhlo::DivOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ExpOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MaxOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MinOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MulOp>,
                   PointwiseToLinalgConverter<xla_lhlo::NegOp>,
                   PointwiseToLinalgConverter<xla_lhlo::RemOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SelectOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SignOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SubOp>,
                   PointwiseToLinalgConverter<xla_lhlo::TanhOp>,
                   ReshapeAddRemoveDimConverter<xla_lhlo::ReshapeOp>,
                   ScalarBroadcastInDimConverter,
                   ScalarPointwiseToStandardConverter<xla_lhlo::AddOp>,
                   SliceConverter
                  >(context);
  // clang-format on
}

// Converts LHLO ops to Linalg generic.
// Sample result for xla_lhlo::AddOp.
//
// "xla_lhlo.add"(%arg1, %arg2, %out) :
//      (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//
// will be converted to
//
// #map0 = (d0, d1) -> (d0, d1)
// "linalg.generic"(%arg1, %arg2, %out) ( {
//   ^bb0(%arg4: f32, %arg5: f32):
//     %0 = addf %arg4, %arg5 : f32
//     "linalg.yield"(%0) : (f32) -> ()
//   }) {
//     args_in = 2,
//     args_out = 1,
//     indexing_maps = [#map0, #map0, #map0],
//     iterator_types = ["parallel", "parallel"],
//   } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
// }
struct LhloLegalizeToLinalg : public FunctionPass<LhloLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

struct HloLegalizeToLinalg : public FunctionPass<HloLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    xla_hlo::populateHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace xla_lhlo {
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeLhloToLinalgPass() {
  return absl::make_unique<LhloLegalizeToLinalg>();
}

static PassRegistration<LhloLegalizeToLinalg> legalize_lhlo_pass(
    "lhlo-legalize-to-linalg", "Legalize from LHLO dialect to Linalg dialect");
}  // namespace xla_lhlo

namespace xla_hlo {

void populateHLOToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns) {
  patterns->insert<BroadcastInDimConverter<xla_hlo::BroadcastInDimOp, false>,
                   ReshapeAddRemoveDimConverter<xla_hlo::ReshapeOp, false>,
                   TransposeConverter<xla_hlo::TransposeOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AbsOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AddOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AndOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CeilOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CompareOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CopyOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::DivOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ExpOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MaxOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MinOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MulOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::NegOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::RemOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SelectOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SubOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::TanhOp, false>>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeHloToLinalgPass() {
  return absl::make_unique<HloLegalizeToLinalg>();
}

static PassRegistration<HloLegalizeToLinalg> legalize_hlo_pass(
    "hlo-legalize-to-linalg", "Legalize from HLO dialect to Linalg dialect");
}  // namespace xla_hlo
}  // namespace mlir

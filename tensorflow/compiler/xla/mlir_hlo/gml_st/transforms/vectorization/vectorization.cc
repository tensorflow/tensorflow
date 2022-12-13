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

#include <limits>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/vector_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

using mlir::linalg::BroadcastOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::MapOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::Mmt4DOp;
using mlir::linalg::ReduceOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

#define GEN_PASS_DEF_VECTORIZEGMLSTLOOPSPASS
#define GEN_PASS_DEF_VECTORIZEPERFECTLYTILEDLOOPSPASS
#include "gml_st/transforms/passes.h.inc"

// The upper limit for vectorization of untiled `linalg.fill`. If a tensor has a
// static shape with more elements, then `linalg.fill` won't be vectorized. It
// is expected that such operations are tiled to get to small static shapes.
constexpr int64_t kNumElementsThreshold = 1024;

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransferReadOp vectorRead,
      mlir::PatternRewriter &rewriter) const override {
    auto expand = vectorRead.getSource().getDefiningOp<ExpandShapeOp>();
    if (!expand) return failure();

    auto expandSrc = expand.getSrc();
    auto expandSrcType = expand.getSrcType();
    auto expandDstType = expand.getResultType();
    if (expandSrcType.getRank() != 1 || expandDstType.getRank() != 2)
      return failure();

    auto resultType = vectorRead.getType().dyn_cast<mlir::ShapedType>();
    if (!resultType || resultType.getShape() != expandDstType.getShape())
      return failure();

    auto zero = rewriter.create<arith::ConstantIndexOp>(vectorRead.getLoc(), 0);
    auto map = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                    vectorRead.getContext());
    // TODO(pifon): Also support canonicalization in case the map is not an
    // identity.
    if (!map.isIdentity()) return failure();

    auto newRead = rewriter.create<TransferReadOp>(
        vectorRead.getLoc(),
        mlir::VectorType::get(expandSrcType.getShape(),
                              expandSrcType.getElementType()),
        expandSrc, mlir::ValueRange{zero}, mlir::AffineMapAttr::get(map),
        vectorRead.getPadding(),
        /*mask=*/mlir::Value(), rewriter.getBoolArrayAttr({true}));
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        vectorRead, vectorRead.getType(), newRead);
    return success();
  }
};

// Rewrite materialize of scalar from 1-element vector into a vector.extract /
// vector.extractelement.
struct MaterializeFromSingleElementToExtractPattern
    : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    auto sourceType = source.getType().dyn_cast<VectorType>();
    if (!sourceType || sourceType.getNumDynamicDims() > 0 ||
        sourceType.getNumElements() > 1) {
      return rewriter.notifyMatchFailure(
          op, "source should be a single element vector");
    }
    if (op.getResult().getType().isa<ShapedType>())
      return rewriter.notifyMatchFailure(op, "result should be a scalar");

    int64_t rank = sourceType.getRank();
    if (rank == 0) {
      // vector.extract doesn't support 0D tensors at the moment,
      // use vector.extractelement.
      rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(op, source);
      return success();
    }
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        op, source, SmallVector<int64_t>(rank, 0));
    return success();
  }
};

// Prepend a set_yield of scalar into 1-element vector with a vector.insert.
struct SetYieldOfScalarToVectorPattern : public OpRewritePattern<SetYieldOp> {
  using OpRewritePattern<SetYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SetYieldOp op,
                                PatternRewriter &rewriter) const override {
    auto tryRewrite = [&](Value dst, Value set, OpOperand &src) {
      if (!dst.getType().isa<VectorType>()) return failure();
      if (src.get().getType().isa<VectorType>()) return failure();
      auto tileOp = set.getDefiningOp<TileOp>();
      if (!tileOp || !tileOp.getOffsets().empty()) return failure();

      src.set(rewriter.create<vector::InsertOp>(op.getLoc(), src.get(), dst,
                                                tileOp.getStaticOffsets()));
      return success();
    };

    if (llvm::none_of(
            llvm::zip_first(op.getDsts(), op.getSets(), op->getOpOperands()),
            [&](auto &&tuple) {
              return succeeded(std::apply(tryRewrite, tuple));
            })) {
      return rewriter.notifyMatchFailure(
          op, "expected scalar srcs and static offsets");
    }

    return success();
  }
};

/// Update tensor operand of vector.transfer_write that uses MaterializeOp.
struct MaterializeUpdateTransferWriteTensorOperand
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    // Sanity checks of TransferWriteOp.
    if (op.hasOutOfBoundsDim()) return failure();
    if (op.getVectorType().getRank() != op.getShapedType().getRank())
      return failure();
    if (op.getMask()) return failure();
    // Fold only if the TransferWriteOp completely overwrites the `source`
    // with a vector, i.e. the result of the TransferWriteOp is a new tensor
    // whose content is the data of the vector.
    if (!llvm::equal(op.getVectorType().getShape(),
                     op.getShapedType().getShape()))
      return failure();
    if (!op.getPermutationMap().isIdentity()) return failure();

    auto src = op.getSource().getDefiningOp<MaterializeOp>();
    if (!src) return failure();

    auto tileOp = src.getSet().getDefiningOp<TileOp>();
    if (!tileOp) return failure();

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, op.getLoc(), tileOp.getMixedOffsets());
    SmallVector<bool> inBounds(op.getTransferRank(), true);
    rewriter.setInsertionPointAfter(op);
    auto newOp = rewriter.create<vector::TransferWriteOp>(
        op.getLoc(), op.getVector(), src.getSource(), indices,
        ArrayRef<bool>{inBounds});
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, op.getResult().getType().cast<RankedTensorType>(),
        newOp.getResult(), tileOp.getOffsets(), tileOp.getSizes(),
        tileOp.getStrides(), tileOp.getStaticOffsets(), tileOp.getStaticSizes(),
        tileOp.getStaticStrides());

    return success();
  }
};

/// Update tensor operand of vector.transfer_write used by SetYieldOp.
struct SetYieldUpdateTransferWriteTensorOperand
    : public OpRewritePattern<SetYieldOp> {
  using OpRewritePattern<SetYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SetYieldOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (const auto &[src, dst, set] :
         llvm::zip(op.getSrcs(), op.getDsts(), op.getSets())) {
      auto xferOp = src.getDefiningOp<vector::TransferWriteOp>();

      // Sanity checks of TransferWriteOp.
      if (!xferOp) continue;
      if (xferOp.getSource() == dst) continue;
      if (xferOp.hasOutOfBoundsDim()) continue;
      if (xferOp.getVectorType().getRank() != xferOp.getShapedType().getRank())
        continue;
      if (xferOp.getMask()) continue;
      // Fold only if the TransferWriteOp completely overwrites the `source`
      // with a vector, i.e. the result of the TransferWriteOp is a new tensor
      // whose content is the data of the vector.
      if (!llvm::equal(xferOp.getVectorType().getShape(),
                       xferOp.getShapedType().getShape()))
        continue;
      if (!xferOp.getPermutationMap().isIdentity()) continue;

      auto tileOp = set.getDefiningOp<TileOp>();

      if (!tileOp) continue;

      SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
          rewriter, op.getLoc(), tileOp.getMixedOffsets());
      SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
      auto newOp = rewriter.create<vector::TransferWriteOp>(
          xferOp.getLoc(), xferOp.getVector(), dst, indices,
          ArrayRef<bool>{inBounds});
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          xferOp, xferOp.getResult().getType().cast<RankedTensorType>(),
          newOp.getResult(), tileOp.getOffsets(), tileOp.getSizes(),
          tileOp.getStrides(), tileOp.getStaticOffsets(),
          tileOp.getStaticSizes(), tileOp.getStaticStrides());
      changed = true;
    }
    return success(changed);
  }
};

template <typename OpTy>
struct VectorizationPattern : public mlir::OpRewritePattern<OpTy> {
  VectorizationPattern(MLIRContext *context,
                       llvm::function_ref<bool(OpTy)> matchFn,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<OpTy>(context, benefit), filterFn(matchFn) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");
    return mlir::linalg::vectorize(rewriter, op);
  }

 private:
  llvm::function_ref<bool(OpTy)> filterFn;
};

// Generates an offset of all 0s suitable as the index paramter for the builder
// of vector.transfer_read or vector.transfer_write with input or output
// `value`, respectively.
SmallVector<Value, 4> generateDefaultOffsetFor(Value value,
                                               OpBuilder &builder) {
  auto shapedType = value.getType().dyn_cast<ShapedType>();
  if (!shapedType) return {};
  Value offset = builder.create<arith::ConstantIndexOp>(value.getLoc(), 0);
  return SmallVector<Value, 4>(shapedType.getRank(), offset);
}

// Converts the ranked-tensor-typed `bvm`-mapped operands of `op` into vectors
// via vector.transfer_read. Updates `bvm`'s mapping of `op`'s operands to the
// newly created vector values.
void convertTensorOperandsToVector(Operation *op, BlockAndValueMapping &bvm,
                                   OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  for (Value operand : op->getOperands()) {
    Value mappedOperand = bvm.lookupOrDefault(operand);
    auto tensorType = mappedOperand.getType().dyn_cast<RankedTensorType>();
    if (!tensorType || tensorType.getNumDynamicDims() > 0) continue;
    builder.setInsertionPointAfterValue(mappedOperand);
    Value vectorOperand = builder.createOrFold<TransferReadOp>(
        mappedOperand.getLoc(),
        VectorType::get(tensorType.getShape(), tensorType.getElementType()),
        mappedOperand, generateDefaultOffsetFor(mappedOperand, builder));
    bvm.map(operand, vectorOperand);
  }
}

// Converts the `bvm`-mapped `results` from vectors to tensors using
// vector.transfer_write, passing in corresponding `destinations` as the
// destination parameter of vector.transfer_write. Updates `bvm`'s mapping of
// `op`'s results to the newly generated tensors. Expects that the operation's
// results are vectors, and the destinations tensors.
void convertVectorResultsToTensor(ValueRange results, ValueRange destinations,
                                  BlockAndValueMapping &bvm,
                                  OpBuilder &builder) {
  for (auto [result, dest] : llvm::zip(results, destinations)) {
    Value mappedResult = bvm.lookupOrDefault(result);
    // Skip over scalars and leave them as is.
    if (!mappedResult.getType().isa<ShapedType>()) continue;
    assert(mappedResult.getType().isa<VectorType>() &&
           "op's result should be a vector");
    assert(dest.getType().isa<RankedTensorType>() &&
           "destination should be a tensor");
    auto writeOp = builder.create<TransferWriteOp>(
        mappedResult.getLoc(), mappedResult, dest,
        generateDefaultOffsetFor(dest, builder));
    bvm.map(result, writeOp.getResult());
  }
}

// Rewrite tensor.extract on single-element tensors into a vector.extract.
struct TensorToElementVectorizationPattern
    : public mlir::OpRewritePattern<tensor::ExtractOp> {
  TensorToElementVectorizationPattern(
      MLIRContext *context, llvm::function_ref<bool(tensor::ExtractOp)> matchFn,
      mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<tensor::ExtractOp>(context, benefit),
        filterFn(matchFn) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");
    TensorType tensorType = op.getTensor().getType();
    if (tensorType.getNumDynamicDims() > 0 || tensorType.getNumElements() > 1)
      return rewriter.notifyMatchFailure(op, "should have a single element");

    BlockAndValueMapping bvm;
    convertTensorOperandsToVector(op, bvm, rewriter);
    if (tensorType.getRank() == 0) {
      // ExtractOp only supports ranks > 0, for rank = 0 use ExtractElementOp
      rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(
          op, bvm.lookupOrDefault(op.getTensor()));
    } else {
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(
          op, bvm.lookupOrDefault(op.getTensor()),
          SmallVector<int64_t, 1>(tensorType.getRank(), 0));
    }
    return success();
  }

 private:
  llvm::function_ref<bool(tensor::ExtractOp)> filterFn;
};

// Rewrite vector.transfer_read(tensor.empty) into a constant vector of the
// right size. This is our temporary way of expressing the nonexistent
// vector.undef, which creates a vector to be used in destination-passing-style
// ops.
// TODO(b/255779480): Figure out how to properly solve this issue.
struct TensorEmptyToVectorBroadcastPattern
    : public OpRewritePattern<TransferReadOp> {
  TensorEmptyToVectorBroadcastPattern(
      MLIRContext *context, llvm::function_ref<bool(TransferReadOp)> filterFn,
      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(matchSimpleTransferOp(op, rewriter))) return failure();
    auto tensorEmpty = op.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!tensorEmpty)
      return rewriter.notifyMatchFailure(op, "source should be tensor.empty");
    VectorType vectorType = op.getResult().getType().dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "result should be a vector");
    Type elementType = vectorType.getElementType();
    TypedAttr nanAttr;
    if (elementType.isa<IntegerType>()) {
      nanAttr = rewriter.getIntegerAttr(elementType, 0l);
    } else if (elementType.isa<FloatType>()) {
      nanAttr = rewriter.getFloatAttr(elementType,
                                      std::numeric_limits<double>::quiet_NaN());
    } else {
      return rewriter.notifyMatchFailure(
          op, "should operate on integer or floating point vectors");
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, DenseElementsAttr::get(vectorType, nanAttr));
    return success();
  }

 private:
  llvm::function_ref<bool(TransferReadOp)> filterFn;
};

struct MaterializeOpVectorizationPattern
    : public OpRewritePattern<MaterializeOp> {
  MaterializeOpVectorizationPattern(
      MLIRContext *context, llvm::function_ref<bool(MaterializeOp)> filterFn,
      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");
    TypedValue<ShapedType> source = op.getSource();
    ShapedType sourceType = source.getType();
    // TODO(b/244314345): Support imperfect tiling, which results in dynamic
    // shapes.
    if (!sourceType.isa<RankedTensorType>() ||
        sourceType.getNumDynamicDims() > 0 ||
        !op.getSet().getType().cast<TileType>().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "input is not statically shaped");

    Location loc = op.getLoc();
    BlockAndValueMapping bvm;
    convertTensorOperandsToVector(op, bvm, rewriter);
    Type newResult = op.getResult().getType();
    if (auto tensorResult = newResult.dyn_cast<RankedTensorType>()) {
      newResult = VectorType::get(tensorResult.getShape(),
                                  tensorResult.getElementType());
    }
    Value vectorMaterialize = rewriter.create<MaterializeOp>(
        loc, newResult, bvm.lookupOrDefault(source), op.getSet());
    bvm.map(op, vectorMaterialize);
    if (auto vectorType = newResult.dyn_cast<VectorType>()) {
      // The result is not a scalar, generate a TransferWrite back to tensor.
      // transfer_write uses destination passing style, so we need to "invent" a
      // destination tensor. The entinre tensor_write op, together with the
      // invented tensor will be folded when vectorizing the final
      // gml_st.set_yield op.
      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, vectorType.getShape(), vectorType.getElementType());
      convertVectorResultsToTensor(op->getResults(), {emptyTensor}, bvm,
                                   rewriter);
    }
    rewriter.replaceOp(op, bvm.lookupOrDefault(op));
    return success();
  }

 private:
  llvm::function_ref<bool(MaterializeOp)> filterFn;
};

struct IdentityMaterializeOpFoldingPattern
    : public OpRewritePattern<MaterializeOp> {
  explicit IdentityMaterializeOpFoldingPattern(MLIRContext *context,
                                               PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto set = op.getSet().getDefiningOp<TileOp>();
    // Only fold identity materialize of ForOp's block argument.
    // Set has to be an identity tile op and source and result are static and
    // have the same shapes.
    if (!op->getParentOfType<ForOp>() || !src.isa<BlockArgument>() || !set ||
        !isIdentityTileOp(set) || !haveSameStaticShape(src, op.getResult()))
      return rewriter.notifyMatchFailure(op, "did not match filter");

    op.replaceAllUsesWith(src);
    return success();
  }
};

// Converts static tensors among `types` to their equivalent vectors.
SmallVector<Type, 1> convertToVectorTypes(TypeRange types) {
  return llvm::to_vector<1>(llvm::map_range(types, [&](Type type) -> Type {
    if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
      return VectorType::get(tensorType.getShape(),
                             tensorType.getElementType());
    }
    return type;
  }));
}

// Copies the body of a loop `op` that is being vectorized, vectorizing the
// terminator, and stores the mapping to new values into `bvm`.
void copyLoopBodyAndVectorizeTerminator(LoopLikeOpInterface op,
                                        OpBuilder &builder,
                                        BlockAndValueMapping &bvm) {
  auto &blocks = op.getLoopBody().getBlocks();
  assert(blocks.size() == 1 && "loop body should contain a single block");
  Block &block = blocks.front();
  for (Operation &bodyMember : block.without_terminator()) {
    builder.clone(bodyMember, bvm);
  }
  convertTensorOperandsToVector(block.getTerminator(), bvm, builder);
  builder.clone(*block.getTerminator(), bvm);
}

// Vectorizes a gml_st.parallel `op`, and stores the mapping from old to new
// values into `bvm`.
ParallelOp vectorizeLoopLikeOp(ParallelOp op, BlockAndValueMapping &bvm,
                               PatternRewriter &rewriter) {
  Optional<StringAttr> distTypeAttr;
  if (auto distType = op.getDistributionType())
    distTypeAttr = rewriter.getStringAttr(*distType);
  return rewriter.create<ParallelOp>(
      op.getLoc(), convertToVectorTypes(op->getResultTypes()),
      op.getLowerBound(), op.getUpperBound(), op.getStep(), distTypeAttr,
      [&](OpBuilder &builder, Location, ValueRange inductionVars) {
        bvm.map(op.getInductionVars(), inductionVars);
        copyLoopBodyAndVectorizeTerminator(op, builder, bvm);
      });
}

// Vectorizes a gml_st.for `op`, and stores the mapping from old to new
// values into `bvm`.
ForOp vectorizeLoopLikeOp(ForOp op, BlockAndValueMapping &bvm,
                          PatternRewriter &rewriter) {
  convertTensorOperandsToVector(op, bvm, rewriter);
  auto outputs = llvm::to_vector(llvm::map_range(
      op.getOutputs(), [&](Value v) { return bvm.lookupOrDefault(v); }));
  return rewriter.create<ForOp>(
      op.getLoc(), convertToVectorTypes(op->getResultTypes()),
      op.getLowerBound(), op.getUpperBound(), op.getStep(), outputs,
      [&](OpBuilder &builder, Location, ValueRange inductionVars,
          ValueRange outputs) {
        bvm.map(op.getInductionVars(), inductionVars);
        bvm.map(op.getRegionOutputArgs(), outputs);
        convertVectorResultsToTensor(op.getRegionOutputArgs(), op.getOutputs(),
                                     bvm, builder);
        copyLoopBodyAndVectorizeTerminator(op, builder, bvm);
      });
}

template <typename LoopLikeOp>
struct LoopLikeOpVectorizationPattern : public OpRewritePattern<LoopLikeOp> {
  LoopLikeOpVectorizationPattern(MLIRContext *context,
                                 llvm::function_ref<bool(LoopLikeOp)> filterFn,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern<LoopLikeOp>(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(LoopLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");
    SetYieldOp setYield = op.getTerminator();
    // Make sure that all the arguments are either tiles or ranked tensors, and
    // that we have at least one tensor (so that the rewrite is not a no-op).
    bool hasTensor = false;
    for (auto [srcType, dstType] : llvm::zip(setYield.getSrcs().getTypes(),
                                             setYield.getDsts().getTypes())) {
      // gcc is failing without `template dyn_cast` here.
      auto dstTensor = dstType.template dyn_cast<RankedTensorType>();
      // TODO(b/244314345): Support imperfect tiling, which results in dynamic
      // shapes.
      if (!dstTensor || dstTensor.getNumDynamicDims() > 0)
        return rewriter.notifyMatchFailure(
            op, "destination tensors should be statically shaped");
      hasTensor = true;
      if (!srcType.template isa<ShapedType>()) continue;
      auto srcTensor = srcType.template dyn_cast<RankedTensorType>();
      if (!srcTensor || srcTensor.getNumDynamicDims() > 0)
        return rewriter.notifyMatchFailure(
            op, "source tensors should be statically shaped");
    }
    if (!hasTensor) {
      return rewriter.notifyMatchFailure(
          op, "should yield at least one tensor to be vectorized");
    }
    // We currently only support set_yield without an accumulator, since this
    // pattern is only needed for GPU, where accumulators are not used.
    if (!setYield.getAccumulators().empty()) {
      return rewriter.notifyMatchFailure(
          op, "shoud not use set_yield accumulators");
    }

    BlockAndValueMapping bvm;

    auto vectorLoopLikeOp = vectorizeLoopLikeOp(op, bvm, rewriter);
    bvm.map(op.getResults(), vectorLoopLikeOp.getResults());

    convertVectorResultsToTensor(op->getResults(), op.getLoopLikeOpInits(), bvm,
                                 rewriter);
    SmallVector<Value, 1> mappedResults = llvm::to_vector<1>(llvm::map_range(
        op.getResults(), [&](Value v) { return bvm.lookupOrDefault(v); }));

    rewriter.replaceOp(op, mappedResults);
    return success();
  }

 private:
  llvm::function_ref<bool(LoopLikeOp)> filterFn;
};

RewritePatternSet getDefaultVectorizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  mlir::vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<mlir::linalg::LinalgCopyVTRForwardingPattern,
               mlir::linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                             /*benefit=*/2);
  TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  return patterns;
}

bool isInsideGmlStLoop(Operation *op) {
  Operation *parent = op->getParentOp();
  return isa<LoopOp>(parent) || isa<ParallelOp>(parent) || isa<ForOp>(parent);
}

bool isFillTiledOrSmall(FillOp fill) {
  if (isInsideGmlStLoop(fill)) return true;

  // Allow vectorization for static shapes with low number of elements.
  auto outputType = fill.output().getType().cast<mlir::RankedTensorType>();
  return outputType.hasStaticShape() &&
         outputType.getNumElements() < kNumElementsThreshold;
}

bool isLinalgOpTiledOrOneDimReduction(LinalgOp op) {
  if (isInsideGmlStLoop(op)) return true;

  // Allow vectorization of 1D reductions.
  return op.getNumLoops() == 1 && op.getNumReductionLoops() == 1;
}

bool isGenericOpTiledOrOneDimReduction(GenericOp generic) {
  if (isInsideGmlStLoop(generic)) return true;

  // Allow vectorization of 1D reductions.
  return generic.getNumLoops() == 1 && generic.getNumReductionLoops() == 1;
}

struct VectorizeGmlStLoopsPass
    : public impl::VectorizeGmlStLoopsPassBase<VectorizeGmlStLoopsPass> {
  VectorizeGmlStLoopsPass(bool vectorizeGmlStOpsParam,
                          ArrayRef<StringRef> distributionLabelsParam) {
    vectorizeGmlStOps = vectorizeGmlStOpsParam;
    for (StringRef distribution : distributionLabelsParam)
      distributionLabels.push_back(distribution.str());
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    auto isValidDistribution = [&](Operation *op) {
      if (distributionLabels.empty()) return true;
      ParallelOp parent = op->getParentOfType<ParallelOp>();
      if (!parent || !parent.getDistributionType().has_value()) return false;
      return llvm::find(distributionLabels,
                        parent.getDistributionType().value()) !=
             distributionLabels.end();
    };
    // These lambdas have to be assigned to local variables, so that they
    // survive beyond patterns.add() and applyPatternsAndFoldGreedily() calls.
    auto fillOpFilter = [&](FillOp op) {
      bool filter = isValidDistribution(op) && isFillTiledOrSmall(op);
      return filter;
    };
    auto linalgOpFilter = [&](LinalgOp op) {
      return isValidDistribution(op) && isLinalgOpTiledOrOneDimReduction(op);
    };
    auto genericOpFilter = [&](GenericOp op) {
      return isValidDistribution(op) && isGenericOpTiledOrOneDimReduction(op);
    };
    auto matmulOpFilter = [&](MatmulOp op) {
      if (isInsideGmlStLoop(op)) return true;
      // Allow vectorization for static shapes.
      auto outputType =
          op.getResult(0).getType().cast<mlir::RankedTensorType>();
      return outputType.hasStaticShape();
    };
    auto materializeOpFilter = [&](MaterializeOp op) {
      // Materialize op should only be vectorized if the producer of its
      // source is within the vectorized region, otherwise we vectorize one
      // level too much. (E.g., for GPU, if we are vectorizing up to warp level,
      // we should not vectorize materializes of warp-level tiles from
      // block-level tiles, since it means we are inserting a
      // vector.transfer_read on the source, i.e., a block-level tile).
      Operation *sourceOp = op.getSource().getDefiningOp();
      // Only vectorize MaterializeOp inside a loop, since we are only enabling
      // this pattern when vectorizing ForOp and ParallelOp anyway.
      Operation *parent = op->getParentOp();
      bool opInsideLoop = isa<ParallelOp>(parent) || isa<ForOp>(parent);
      return sourceOp != nullptr && opInsideLoop &&
             isValidDistribution(sourceOp);
    };
    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      patterns.add<TransferReadOfOneDimExpandShape,
                   MaterializeFromSingleElementToExtractPattern,
                   SetYieldOfScalarToVectorPattern>(ctx);
      patterns.add<VectorizationPattern<FillOp>>(ctx, fillOpFilter);
      patterns.add<VectorizationPattern<GenericOp>>(ctx, genericOpFilter);
      patterns.add<VectorizationPattern<BroadcastOp>,
                   VectorizationPattern<MapOp>, VectorizationPattern<ReduceOp>>(
          ctx, linalgOpFilter);
      patterns.add<VectorizationPattern<MatmulOp>>(ctx, matmulOpFilter);
      patterns.add<TensorToElementVectorizationPattern,
                   TensorEmptyToVectorBroadcastPattern>(ctx,
                                                        isValidDistribution);
      if (vectorizeGmlStOps) {
        patterns.add<MaterializeOpVectorizationPattern>(ctx,
                                                        materializeOpFilter);
        patterns.add<LoopLikeOpVectorizationPattern<ParallelOp>,
                     LoopLikeOpVectorizationPattern<ForOp>>(
            ctx, isValidDistribution);
      }
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      patterns.add<MaterializeUpdateTransferWriteTensorOperand,
                   SetYieldUpdateTransferWriteTensorOperand>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
  }
};

struct VectorizePerfectlyTiledLoopsPass
    : public impl::VectorizePerfectlyTiledLoopsPassBase<
          VectorizePerfectlyTiledLoopsPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    auto hasSmallStaticOutputs = [&](Operation *op) {
      return llvm::all_of(op->getResultTypes(), [](Type type) {
        auto outputType = type.dyn_cast<mlir::RankedTensorType>();
        return outputType && outputType.hasStaticShape() &&
               outputType.getNumElements() < kNumElementsThreshold;
      });
    };
    auto isPerfectlyTiledLoop = [&](Operation *op) {
      return (isa<ForOp>(op) || isa<ParallelOp>(op)) &&
             hasLabel(op, kPerfectlyTiledLoopLabel);
    };
    auto isInsidePerfectlyTiledLoop = [&](Operation *op) {
      return isPerfectlyTiledLoop(op->getParentOp());
    };
    auto isInsidePerfectlyTiledLoopOrSmall = [&](Operation *op) {
      return isInsidePerfectlyTiledLoop(op) || hasSmallStaticOutputs(op);
    };
    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      // clang-format off
      patterns.add<
        VectorizationPattern<BroadcastOp>,
        VectorizationPattern<GenericOp>,
        VectorizationPattern<MapOp>,
        VectorizationPattern<MatmulOp>,
        VectorizationPattern<Mmt4DOp>,
        VectorizationPattern<ReduceOp>
      >(ctx, isInsidePerfectlyTiledLoopOrSmall);
      // clang-format on
      patterns.add<VectorizationPattern<FillOp>>(ctx, isFillTiledOrSmall);
      patterns.add<TransferReadOfOneDimExpandShape>(ctx);
      patterns.add<TensorToElementVectorizationPattern>(
          ctx, isInsidePerfectlyTiledLoop);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      patterns.add<MaterializeUpdateTransferWriteTensorOperand,
                   SetYieldUpdateTransferWriteTensorOperand>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    // Hoisting transfer_read/transfer_write.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<IdentityMaterializeOpFoldingPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

      hoistRedundantVectorTransfersOnTensor(func);
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass(
    bool vectorizeGmlStOps, ArrayRef<StringRef> distributionLabels) {
  return std::make_unique<VectorizeGmlStLoopsPass>(vectorizeGmlStOps,
                                                   distributionLabels);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createVectorizePerfectlyTiledLoopsPass() {
  return std::make_unique<VectorizePerfectlyTiledLoopsPass>();
}

}  // namespace gml_st
}  // namespace mlir

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

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_VECTORIZEGMLSTLOOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

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

template <typename OpTy>
struct VectorizationPattern : public mlir::OpRewritePattern<OpTy> {
  VectorizationPattern(MLIRContext *context,
                       llvm::function_ref<bool(OpTy)> matchFn,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<OpTy>(context, benefit), matchFn(matchFn) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!matchFn(op)) return failure();
    return mlir::linalg::vectorize(rewriter, op);
  }

 private:
  llvm::function_ref<bool(OpTy)> matchFn;
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

// Converts the `bvm`-mapped results of `op` from vectors to tensors using
// vector.transfer_write, passing in corresponding `destinations` as the
// destination parameter of vector.transfer_write. Updates `bvm`'s mapping of
// `op`'s results to the newly generated tensors. Expects that the operation's
// results are vectors, and the destinations tensors.
void convertVectorResultsToTensor(Operation *op, ValueRange destinations,
                                  BlockAndValueMapping &bvm,
                                  OpBuilder &builder) {
  for (auto [result, dest] : llvm::zip(op->getResults(), destinations)) {
    Value mappedResult = bvm.lookupOrDefault(result);
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

struct MaterializeOpVectorizationPattern
    : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    TypedValue<ShapedType> source = op.getSource();
    ShapedType sourceType = source.getType();
    // TODO(b/244314345): Support imperfect tiling, which results in dynamic
    // shapes.
    if (!sourceType.isa<RankedTensorType>() ||
        sourceType.getNumDynamicDims() > 0)
      return failure();

    Location loc = op.getLoc();
    BlockAndValueMapping bvm;
    convertTensorOperandsToVector(op, bvm, rewriter);
    Value vectorMaterialize = rewriter.create<MaterializeOp>(
        loc, bvm.lookupOrDefault(source), op.getSet());
    bvm.map(op, vectorMaterialize);
    if (auto vectorType = vectorMaterialize.getType().dyn_cast<VectorType>()) {
      // The result is not a scalar, generate a TransferWrite back to tensor.
      // transfer_write uses destination passing style, so we need to "invent" a
      // destination tensor. The entinre tensor_write op, together with the
      // invented tensor will be folded when vectorizing the final
      // gml_st.set_yield op.
      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, vectorType.getShape(), vectorType.getElementType());
      convertVectorResultsToTensor(op, {emptyTensor}, bvm, rewriter);
    }
    rewriter.replaceOp(op, bvm.lookupOrDefault(op));
    return success();
  }
};

struct ParallelOpVectorizationPattern : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult match(ParallelOp op) const override {
    SetYieldOp setYield = op.getTerminator();
    // Make sure that all the arguments are either tiles or ranked tensors, and
    // that we have at least one tensor (so that the rewrite is not a no-op).
    bool hasTensor = false;
    for (auto [srcType, dstType] : llvm::zip(setYield.getSrcs().getTypes(),
                                             setYield.getDsts().getTypes())) {
      auto tensorType = srcType.dyn_cast<RankedTensorType>();
      // TODO(b/244314345): Support imperfect tiling, which results in dynamic
      // shapes.
      if (!tensorType || tensorType.getNumDynamicDims() > 0 ||
          dstType.cast<RankedTensorType>().getNumDynamicDims() > 0)
        return failure();

      hasTensor = true;
    }
    // We currently only support set_yield without an accumulator, since this
    // pattern is only needed for GPU, where accumulators are not used.
    return success(hasTensor && setYield.getAccumulators().empty());
  }

  void rewrite(ParallelOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    BlockAndValueMapping bvm;

    // Convert result types of the parallel op from tensor to vector.
    SmallVector<Type, 1> resultTypes = llvm::to_vector<1>(
        llvm::map_range(op.getResultTypes(), [&](Type resultType) -> Type {
          if (auto tensorType = resultType.dyn_cast<RankedTensorType>()) {
            return VectorType::get(tensorType.getShape(),
                                   tensorType.getElementType());
          }
          return resultType;
        }));

    // Convert gml_st.parallel op to its vector variant
    auto bodyBuilder = [&](OpBuilder &builder, Location,
                           ValueRange inductionVars) {
      bvm.map(op.getInductionVars(), inductionVars);
      for (Operation &bodyMember : op.getLoopBody().getOps()) {
        if (isa<SetYieldOp>(&bodyMember))
          convertTensorOperandsToVector(&bodyMember, bvm, rewriter);
        builder.clone(bodyMember, bvm);
      }
    };
    auto vectorParallel = rewriter.create<ParallelOp>(
        loc, resultTypes, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        llvm::None, bodyBuilder);
    bvm.map(op.getResults(), vectorParallel.getResults());

    convertVectorResultsToTensor(op, op.getTerminator().getDsts(), bvm,
                                 rewriter);
    SmallVector<Value, 1> mappedResults = llvm::to_vector<1>(llvm::map_range(
        op.getResults(), [&](Value v) { return bvm.lookupOrDefault(v); }));

    rewriter.replaceOp(op, mappedResults);
  }
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

bool isGenericOpTiledOrOneDimReduction(GenericOp generic) {
  if (isInsideGmlStLoop(generic)) return true;

  // Allow vectorization of 1D reductions.
  return generic.getNumLoops() == 1 && generic.getNumReductionLoops() == 1;
}

struct VectorizeGmlStLoopsPass
    : public impl::VectorizeGmlStLoopsPassBase<VectorizeGmlStLoopsPass> {
  explicit VectorizeGmlStLoopsPass(bool vectorizeGmlStOpsParam) {
    vectorizeGmlStOps = vectorizeGmlStOpsParam;
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
    patterns.add<TransferReadOfOneDimExpandShape>(func.getContext());
    patterns.add<VectorizationPattern<FillOp>>(ctx, isFillTiledOrSmall);
    patterns.add<VectorizationPattern<GenericOp>>(
        ctx, isGenericOpTiledOrOneDimReduction);
    if (vectorizeGmlStOps) {
      patterns.add<MaterializeOpVectorizationPattern,
                   ParallelOpVectorizationPattern>(ctx);
    }
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass(
    bool vectorizeGmlStOps) {
  return std::make_unique<VectorizeGmlStLoopsPass>(vectorizeGmlStOps);
}

}  // namespace gml_st
}  // namespace mlir

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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_LEGALIZECOLLECTIVEOPSPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class LegalizeCollectiveOpsPass
    : public impl::LegalizeCollectiveOpsPassBase<LegalizeCollectiveOpsPass> {
  void runOnOperation() override;
};

std::optional<xla_cpu::ReductionKind> MatchReductionComputation(
    Region& region) {
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }

  auto ret = dyn_cast<mhlo::ReturnOp>(region.front().getTerminator());
  if (!ret || ret->getNumOperands() != 1) {
    return std::nullopt;
  }

  auto computation = ret.getOperand(0).getDefiningOp();
  if (computation->getNumOperands() != 2 ||
      computation->getOperand(0) != region.front().getArgument(0) ||
      computation->getOperand(1) != region.front().getArgument(1)) {
    return std::nullopt;
  }

  if (isa<mhlo::AddOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_SUM;
  }
  if (isa<mhlo::MulOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_PRODUCT;
  }
  if (isa<mhlo::MinOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MIN;
  }
  if (isa<mhlo::MaxOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MAX;
  }

  auto type = computation->getOperandTypes().front().dyn_cast<ShapedType>();
  if (!type || !type.getElementType().isInteger(1)) {
    return std::nullopt;
  }

  if (isa<mhlo::AndOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MIN;
  }
  if (isa<mhlo::OrOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MAX;
  }

  return std::nullopt;
}

// Returns a `tensor.empty` with the same shape as `tensor`.
Value CreateEmptyLike(OpBuilder& b, Location loc, Value tensor) {
  auto ty = tensor.getType().cast<ShapedType>();
  auto sizes = tensor::getMixedSizes(b, loc, tensor);
  return b.create<tensor::EmptyOp>(loc, sizes, ty.getElementType());
}

class AllReduceLowering : public OpRewritePattern<mhlo::AllReduceOp> {
  using OpRewritePattern<mhlo::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto reduction_kind = MatchReductionComputation(op.getRegion());
    if (!reduction_kind) {
      return failure();
    }

    SmallVector<Value> dsts;
    for (auto operand : op->getOperands()) {
      // The operands and results have the same shapes.
      dsts.push_back(CreateEmptyLike(rewriter, op.getLoc(), operand));
    }

    rewriter.replaceOpWithNewOp<xla_cpu::AllReduceOp>(
        op, op->getResultTypes(), op->getOperands(), dsts,
        op.getReplicaGroupsAttr(),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}),
        rewriter.getI32IntegerAttr(op.getUseGlobalDeviceIdsAttr() ? 1 : 0),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(*reduction_kind)));

    return success();
  };
};

template <typename IdOp, typename XlaIdOp>
class IdLowering : public OpRewritePattern<IdOp> {
  using OpRewritePattern<IdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    Value id = rewriter.create<XlaIdOp>(op.getLoc());
    // Wrap the scalar in a tensor.
    Value id_tensor = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(), RankedTensorType::get({}, rewriter.getI32Type()), id);
    // And convert it to unsigned. This becomes a noop later.
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        RankedTensorType::get({}, IntegerType::get(rewriter.getContext(), 32,
                                                   IntegerType::Unsigned)),
        id_tensor);
    return success();
  };
};

class CollectivePermuteLowering
    : public OpRewritePattern<mhlo::CollectivePermuteOp> {
  using OpRewritePattern<mhlo::CollectivePermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CollectivePermuteOp op,
                                PatternRewriter& rewriter) const override {
    // The result of collective_permute has the same shape as the operand.
    Value dst = CreateEmptyLike(rewriter, op.getLoc(), op.getOperand());
    rewriter.replaceOpWithNewOp<xla_cpu::CollectivePermuteOp>(
        op, op->getResultTypes(), op->getOperand(0), dst,
        op.getSourceTargetPairsAttr(),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}));
    return success();
  };
};

class AllToAllLowering : public OpRewritePattern<mhlo::AllToAllOp> {
  using OpRewritePattern<mhlo::AllToAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllToAllOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SmallVector<Value> dsts;

    if (!op.getConcatDimensionAttr()) {
      for (auto operand : op->getOperands()) {
        // The operands and results of TupleAllToAll the same shapes.
        dsts.push_back(CreateEmptyLike(rewriter, op.getLoc(), operand));
      }
    } else {
      auto sizes = getValueOrCreateConstantIndexOp(
          b, b.getLoc(),
          tensor::getMixedSizes(b, op.getLoc(), op->getOperand(0)));
      uint64_t split_dimension = *op.getSplitDimension();
      Value split_count = b.create<arith::ConstantIndexOp>(*op.getSplitCount());
      sizes[split_dimension] = b.createOrFold<arith::DivUIOp>(
          b.getIndexType(), sizes[split_dimension], split_count);
      uint64_t concat_dimension = *op.getConcatDimension();
      sizes[concat_dimension] =
          b.createOrFold<arith::MulIOp>(sizes[concat_dimension], split_count);

      dsts.push_back(rewriter.create<tensor::EmptyOp>(
          op.getLoc(), getAsOpFoldResult(sizes),
          op->getResultTypes()[0].cast<ShapedType>().getElementType()));
    }

    rewriter.replaceOpWithNewOp<xla_cpu::AllToAllOp>(
        op, op->getResultTypes(), op->getOperands(), dsts,
        op.getReplicaGroupsAttr(), op.getSplitDimensionAttr(),
        op.getConcatDimensionAttr(), op.getSplitCountAttr());
    return success();
  };
};

class FftLowering : public OpRewritePattern<mhlo::FftOp> {
  using OpRewritePattern<mhlo::FftOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::FftOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // TODO(jreiffers): Support dynamic sizes.
    auto dst = b.create<tensor::EmptyOp>(op.getLoc(), op.getType().getShape(),
                                         op.getType().getElementType());

    auto lengths =
        llvm::to_vector<3>(op.getFftLengthAttr().getValues<int64_t>());
    rewriter.replaceOpWithNewOp<xla_cpu::FftOp>(
        op, op->getResultTypes(), op->getOperand(0), dst,
        static_cast<int32_t>(op.getFftType()),
        rewriter.getI64ArrayAttr(lengths));
    return success();
  };
};

class OutfeedLowering : public OpRewritePattern<mhlo::OutfeedOp> {
  using OpRewritePattern<mhlo::OutfeedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::OutfeedOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Attribute> result_types;
    for (auto operand : op.getInputs()) {
      result_types.push_back(
          TypeAttr::get(operand.getType().cast<ShapedType>().getElementType()));
    }
    rewriter.create<xla_cpu::OutfeedOp>(
        op.getLoc(), std::nullopt, op.getInputs(), op.getOutfeedConfigAttr(),
        ArrayAttr::get(op->getContext(), result_types));

    // Replacing the op with the token.
    rewriter.replaceOp(op, op.getToken());
    return success();
  };
};

class RngBitGeneratorLowering
    : public OpRewritePattern<mhlo::RngBitGeneratorOp> {
  using OpRewritePattern<mhlo::RngBitGeneratorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngBitGeneratorOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto state_init = CreateEmptyLike(b, op.getLoc(), op.getOperand());
    auto output_init =
        b.create<tensor::EmptyOp>(op.getLoc(), op.getType(1), ValueRange{});

    rewriter.replaceOpWithNewOp<xla_cpu::RngBitGeneratorOp>(
        op, op->getResultTypes(), op->getOperand(0), state_init, output_init,
        op.getRngAlgorithmAttr());
    return success();
  };
};

class AddDependencyLowering : public OpRewritePattern<mhlo::AddDependencyOp> {
  using OpRewritePattern<mhlo::AddDependencyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddDependencyOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<xla_cpu::AddDependencyOp>(
        op, op->getResultTypes(), op->getOperands());
    return success();
  };
};

void LegalizeCollectiveOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  // Convert mhlo collective operations to XLA cpu ops.
  RewritePatternSet patterns(ctx);
  patterns.insert<AddDependencyLowering, AllReduceLowering, AllToAllLowering,
                  CollectivePermuteLowering, FftLowering,
                  IdLowering<mhlo::PartitionIdOp, xla_cpu::PartitionIdOp>,
                  IdLowering<mhlo::ReplicaIdOp, xla_cpu::ReplicaIdOp>,
                  OutfeedLowering, RngBitGeneratorLowering>(ctx);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeCollectiveOpsPass() {
  return std::make_unique<LegalizeCollectiveOpsPass>();
}

}  // namespace cpu
}  // namespace xla

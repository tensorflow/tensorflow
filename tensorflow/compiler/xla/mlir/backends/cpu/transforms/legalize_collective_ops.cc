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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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

Optional<xla_cpu::ReductionKind> MatchReductionComputation(Region& region) {
  if (!region.hasOneBlock()) {
    return None;
  }

  auto ret = dyn_cast<mhlo::ReturnOp>(region.front().getTerminator());
  if (!ret || ret->getNumOperands() != 1) {
    return None;
  }

  auto computation = ret.getOperand(0).getDefiningOp();
  if (computation->getNumOperands() != 2 ||
      computation->getOperand(0) != region.front().getArgument(0) ||
      computation->getOperand(1) != region.front().getArgument(1)) {
    return None;
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

  if (!computation->getOperandTypes().front().isInteger(1)) {
    return None;
  }

  if (isa<mhlo::AndOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MIN;
  }
  if (isa<mhlo::OrOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MAX;
  }

  return None;
}

class AllReduceLowering : public OpRewritePattern<mhlo::AllReduceOp> {
  using OpRewritePattern<mhlo::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto reduction_kind = MatchReductionComputation(op.getRegion());
    if (!reduction_kind) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<xla_cpu::AllReduceOp>(
        op, op->getResultTypes(), op->getOperands(), op.getReplicaGroupsAttr(),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}),
        op.getUseGlobalDeviceIdsAttr(),
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

void LegalizeCollectiveOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  // Convert mhlo collective operations to XLA cpu ops.
  RewritePatternSet patterns(ctx);
  patterns.insert<AllReduceLowering,
                  IdLowering<mhlo::PartitionIdOp, xla_cpu::PartitionIdOp>,
                  IdLowering<mhlo::ReplicaIdOp, xla_cpu::ReplicaIdOp>>(ctx);

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

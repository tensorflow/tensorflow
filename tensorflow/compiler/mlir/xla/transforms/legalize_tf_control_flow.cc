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

// This file implements logic for lowering TensorFlow dialect's control flow to
// the XLA dialect.

#include <iterator>
#include <utility>

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

using mlir::PassRegistration;

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_LEGALIZETFCONTROLFLOW
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class LegalizeTFControlFlow
    : public impl::LegalizeTFControlFlowBase<LegalizeTFControlFlow> {
 public:
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLegalizeTFControlFlowPass() {
  return std::make_unique<LegalizeTFControlFlow>();
}

namespace {

class LowerYieldOp : public OpConversionPattern<TF::YieldOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::YieldOp op, TF::YieldOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

template <typename SrcOpT, typename DstOpT>
class LowerControlFlowOp : public OpConversionPattern<SrcOpT> {
 public:
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpT op, typename SrcOpT::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    DstOpT mhlo_op;
    Location loc = op.getLoc();
    if constexpr (std::is_same<DstOpT, mhlo::CaseOp>::value) {
      // Explicitly handle the Case op because it has variadic regions and takes
      // the number of regions as an input along with the operands.
      mhlo_op = rewriter.create<DstOpT>(loc, op.getResultTypes(),
                                        adaptor.getBranchIndex(),
                                        op.getBranches().size());
    } else {
      mhlo_op = rewriter.create<DstOpT>(loc, op.getResultTypes(),
                                        adaptor.getOperands());
    }

    // Replace all uses of `op` results with the newly created op.
    rewriter.replaceOp(op, mhlo_op.getResults());

    int64_t num_regions = op.getNumRegions();
    for (int64_t idx = 0; idx < num_regions; ++idx) {
      rewriter.inlineRegionBefore(op.getBodyRegion(idx),
                                  mhlo_op.getBodyRegion(idx),
                                  mhlo_op.getBodyRegion(idx).end());
    }
    return success();
  }
};
}  // namespace

void LegalizeTFControlFlow::runOnOperation() {
  Operation* op = getOperation();
  MLIRContext* context = op->getContext();

  ConversionTarget target(*context);
  target.addLegalOp<CaseOp, IfOp, WhileOp, ReturnOp>();

  RewritePatternSet patterns(context);
  patterns
      .add<LowerControlFlowOp<TF::CaseRegionOp, mhlo::CaseOp>,
           LowerControlFlowOp<TF::IfRegionOp, mhlo::IfOp>,
           LowerControlFlowOp<TF::WhileRegionOp, mhlo::WhileOp>, LowerYieldOp>(
          context);

  if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace mhlo
}  // namespace mlir

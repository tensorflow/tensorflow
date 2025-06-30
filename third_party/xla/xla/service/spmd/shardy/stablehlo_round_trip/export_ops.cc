/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_ops.h"

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace sdy {

namespace {

namespace stablehlo = ::mlir::stablehlo;
namespace mhlo = ::mlir::mhlo;

using ::mlir::ConversionPatternRewriter;
using ::mlir::LogicalResult;
using ::mlir::OpConversionPattern;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::StringRef;
using ::mlir::success;

using ::mlir::sdy::AllGatherOp;
using ::mlir::sdy::AllReduceOp;
using ::mlir::sdy::AllSliceOp;
using ::mlir::sdy::AllToAllOp;
using ::mlir::sdy::CollectivePermuteOp;
using ::mlir::sdy::ConstantOp;
using ::mlir::sdy::PropagationBarrierOp;
using ::mlir::sdy::ReduceScatterOp;
using ::mlir::sdy::ReshardOp;
using ::mlir::sdy::ShardingConstraintOp;
using ::mlir::sdy::TensorShardingAttr;

// Converts `sdy::ConstantOp` to `stablehlo::ConstantOp`.
class ConstantPattern : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We use the generic op builder so that unregistered attributes will be
    // added to the new op.
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

// We erase an `AllReduceOp` instead of converting it to a copy op, since it
// does not reshard the tensor.
class AllReducePattern : public OpConversionPattern<AllReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      AllReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensor());
    return success();
  }
};

class PropagationBarrierPattern
    : public OpConversionPattern<PropagationBarrierOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      PropagationBarrierOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

void rewriteCollectiveOp(mlir::Operation* op, mlir::Value input,
                         TensorShardingAttr sharding,
                         ConversionPatternRewriter& rewriter) {
  auto copyOp = rewriter.replaceOpWithNewOp<mhlo::CopyOp>(op, input);
  mlir::sdy::setShardings(copyOp, sharding);
}

template <class OpTy>
class ShardingPattern : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriteCollectiveOp(op, adaptor.getInput(), adaptor.getSharding(),
                        rewriter);
    return success();
  }
};

template <class OpTy>
class CollectivePattern : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriteCollectiveOp(op, adaptor.getTensor(), adaptor.getOutSharding(),
                        rewriter);
    return success();
  }
};

class ExportOpsPass
    : public mlir::PassWrapper<ExportOpsPass, OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportOpsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    // We do not expect to see ShardingConstraintOp in the input module.
    // ShardingConstraintOp should be replaced by ReshardOp before this pass.
    // Hence, we add ShardingConstraintOp as an illegal op.
    target.addIllegalOp<ConstantOp, ReshardOp, AllGatherOp, AllReduceOp,
                        AllSliceOp, AllToAllOp, CollectivePermuteOp,
                        ReduceScatterOp, ShardingConstraintOp,
                        PropagationBarrierOp>();
    target.addLegalOp<stablehlo::ConstantOp, mhlo::CopyOp>();
    mlir::RewritePatternSet patterns(&context);
    // After converting `sdy.constant` into `stablehlo.constant`, the constants
    // should not be deduped via folding. Fortunately, folding only happens in
    // greedy pattern rewriters. ExportHloShardingsPass does a simple walk,
    // which keeps the constants as is.
    patterns.add<ConstantPattern, AllReducePattern, ShardingPattern<ReshardOp>,
                 ShardingPattern<ShardingConstraintOp>,
                 PropagationBarrierPattern, CollectivePattern<AllGatherOp>,
                 CollectivePattern<AllSliceOp>, CollectivePattern<AllToAllOp>,
                 CollectivePattern<CollectivePermuteOp>,
                 CollectivePattern<ReduceScatterOp>>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "xla-sdy-export-ops"; }

  StringRef getDescription() const override {
    return "Exports Shardy ops to StableHLO ops. Processes sdy::ReshardOp and "
           "sdy::ConstantOp.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect, mlir::mhlo::MhloDialect>();
  }
};

}  // namespace

std::unique_ptr<Pass> createExportOpsPass() {
  return std::make_unique<ExportOpsPass>();
}

void registerExportOpsPass() { mlir::registerPass(createExportOpsPass); }

}  // namespace sdy
}  // namespace xla

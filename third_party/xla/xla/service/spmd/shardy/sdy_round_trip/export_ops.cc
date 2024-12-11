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

#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"

#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace stablehlo = ::mlir::stablehlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ConversionPatternRewriter;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpConversionPattern;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::StringRef;
using ::mlir::success;

using ::mlir::sdy::ConstantOp;
using ::mlir::sdy::ShardingConstraintOp;
using ::mlir::sdy::ShardingGroupOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

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

class ShardingConstraintPattern
    : public OpConversionPattern<ShardingConstraintOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      ShardingConstraintOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr sharding = op.getSharding();

    auto customCallOp = rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, op.getType(), adaptor.getInput());

    customCallOp.setCallTargetName(kShardingCustomCallTargetName);
    // Copy over any existing attrs other than the sharding.
    for (mlir::NamedAttribute attr : op->getDiscardableAttrs()) {
      customCallOp->setAttr(attr.getName(), attr.getValue());
    }
    customCallOp->setAttr(
        mlir::sdy::kShardingAttr,
        TensorShardingPerValueAttr::get(customCallOp.getContext(), sharding));

    return success();
  }
};

class ShardingGroupPattern : public OpConversionPattern<ShardingGroupOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      ShardingGroupOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto customCallOp = rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, op->getResultTypes(), adaptor.getInput());

    customCallOp.setCallTargetName(kShardingGroupCustomCallTargetName);
    setFrontendAttribute(customCallOp, kShardingGroupIdAttr,
                         op.getGroupIdAttr());
    customCallOp.setHasSideEffectAttr(rewriter.getBoolAttr(true));
    return success();
  }
};

class SdyRoundTripExportOpsPass
    : public PassWrapper<SdyRoundTripExportOpsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripExportOpsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addIllegalOp<ConstantOp, ShardingConstraintOp>();
    target.addLegalOp<stablehlo::ConstantOp, stablehlo::CustomCallOp>();
    mlir::RewritePatternSet patterns(&context);
    patterns
        .add<ConstantPattern, ShardingConstraintPattern, ShardingGroupPattern>(
            &context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-export-ops";
  }

  StringRef getDescription() const override {
    return "Exports Shardonnay ops to MHLO ops.";
  }
};

}  // namespace

std::unique_ptr<Pass> createSdyRoundTripExportOpsPass() {
  return std::make_unique<SdyRoundTripExportOpsPass>();
}

void registerSdyRoundTripExportOpsPass() {
  mlir::registerPass(createSdyRoundTripExportOpsPass);
}

}  // namespace sdy
}  // namespace xla

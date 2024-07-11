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

#include "xla/service/spmd/shardy/mhlo_round_trip/export_ops.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "shardy/dialect/sdy/ir/constants.h"  // from @shardy
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/sharding_op_util.h"

namespace xla {
namespace sdy {

namespace {

namespace mhlo = ::mlir::mhlo;

using ::mlir::ConversionPatternRewriter;
using ::mlir::LogicalResult;
using ::mlir::OpConversionPattern;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::success;

using ::mlir::sdy::ConstantOp;
using ::mlir::sdy::IdentityOp;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ReshardOp;
using ::mlir::sdy::ShardingConstraintOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Converts `sdy::ConstantOp` to `mhlo::ConstantOp`.
class ConstantPattern : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We use the generic op builder so that unregistered attributes will be
    // added to the new op.
    rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

// Removes `sdy::IdentityOp`.
class IdentityPattern : public OpConversionPattern<IdentityOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IdentityOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class ReshardPattern : public OpConversionPattern<ReshardOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      ReshardOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto copyOp =
        rewriter.replaceOpWithNewOp<mhlo::CopyOp>(op, adaptor.getInput());

    TensorShardingAttr sdySharding = adaptor.getShardingAttr();
    copyOp->setAttr(kShardingAttr, TensorShardingPerValueAttr::get(
                                       op.getContext(), sdySharding));

    SmallVector<int64_t> unspecifiedDims;
    for (auto [dim, dimSharding] :
         llvm::enumerate(sdySharding.getDimShardings())) {
      // Unspecified dims are those that are marked open but is not partitioned
      // on any axes.
      if (!dimSharding.getIsClosed() && dimSharding.emptyAxes()) {
        unspecifiedDims.push_back(dim);
      }
    }
    if (!unspecifiedDims.empty()) {
      copyOp->setAttr(kXlaBackendConfigAttr,
                      StringAttr::get(op.getContext(),
                                      xla::sharding_op_util::EncodeAttributes(
                                          unspecifiedDims)));
    }

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
    target.addIllegalOp<ConstantOp, IdentityOp, ReshardOp,
                        ShardingConstraintOp>();
    target.addLegalOp<mhlo::ConstantOp, mhlo::CopyOp>();
    mlir::RewritePatternSet patterns(&context);
    // After converting `sdy.constant` into `mhlo.constant`, the constants
    // should not be deduped via folding. Fortunately, folding only happens in
    // greedy pattern rewriters. ExportHloShardingsPass does a simple walk,
    // which keeps the constants as is.
    patterns.add<ConstantPattern, IdentityPattern, ReshardPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "xla-sdy-export-ops"; }

  StringRef getDescription() const override {
    return "Exports Shardonnay ops to MHLO ops. Processes sdy::IdentityOp, "
           "sdy::ReshardOp, and sdy::ConstantOp.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<Pass> createExportOpsPass() {
  return std::make_unique<ExportOpsPass>();
}

void registerExportOpsPass() { mlir::registerPass(createExportOpsPass); }

}  // namespace sdy
}  // namespace xla

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

#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_export.h"

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "src/shardy/dialect/sdy/ir/constants.h"  // from @shardy
#include "src/shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "src/shardy/dialect/sdy/ir/utils.h"  // from @shardy
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ConversionPatternRewriter;
using ::mlir::LogicalResult;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::OpConversionPattern;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::Value;
using ::mlir::mhlo::CopyOp;
using ::mlir::mhlo::CustomCallOp;

namespace sdy = ::mlir::sdy;
using sdy::kShardingAttr;
using sdy::ManualComputationOp;
using sdy::SdyDialect;
using sdy::TensorShardingPerValueAttr;

class ManualComputationPattern
    : public OpConversionPattern<ManualComputationOp> {
 public:
  explicit ManualComputationPattern(MLIRContext* context)
      : OpConversionPattern<ManualComputationOp>(context) {
    // We call this function so that MLIR applies the pattern to any
    // ManualComputationOp that uses another ManualComputationOp.
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(
      ManualComputationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MLIRContext* context = rewriter.getContext();

    mlir::StringAttr manualSharding =
        rewriter.getStringAttr(HloSharding::Manual().ToString());

    auto createAttributes =
        [&](StringRef callTargetName) -> SmallVector<NamedAttribute, 2> {
      return {rewriter.getNamedAttr("call_target_name",
                                    rewriter.getStringAttr(callTargetName)),
              rewriter.getNamedAttr(kXlaShardingAttr, manualSharding)};
    };
    SmallVector<NamedAttribute, 2> fullToShardAttributes =
        createAttributes(kSPMDFullToShardShapeCallTargetName);
    SmallVector<NamedAttribute, 2> shardToFullAttributes =
        createAttributes(kSPMDShardToFullShapeCallTargetName);

    // Set manual shardings in the body of ManualComputationOp.
    bool hasSdySharding = false;
    op.getBody().front().walk([&](Operation* op) {
      if (op->hasAttr(kShardingAttr)) {
        // TODO(b/333910165). Handle the cases with existing sdy.sharding.
        op->emitError(
            "Operation in ManualComputationOp has a sdy.sharding attribute.");
        hasSdySharding = true;
      }
      op->setAttr(kXlaShardingAttr, manualSharding);
    });
    if (hasSdySharding) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();

    // Add copy and custom_call @SPMDFullToShardShape for each operand. The
    // copy corresponds to custom_call @Sharding before sharding propagation.
    SmallVector<Value> fullToShardResults;
    for (auto [globalOperand, localArgumentType, inSharding] :
         llvm::zip_equal(adaptor.getOperands(), op.getBody().getArgumentTypes(),
                         adaptor.getInShardings().getShardings())) {
      auto copy = rewriter.create<CopyOp>(loc, globalOperand);
      copy->setAttr(kShardingAttr,
                    TensorShardingPerValueAttr::get(context, inSharding));
      auto fullToShard = rewriter.create<CustomCallOp>(
          loc, localArgumentType, copy.getResult(), fullToShardAttributes);
      fullToShardResults.push_back(fullToShard.getResult(0));
    }

    Operation* terminator = getBodyTerminator(op);
    rewriter.inlineBlockBefore(&op.getBody().front(), op, fullToShardResults);

    // Add custom_call @SPMDShardToFullShape and copy for each operand of
    // terminator.
    for (auto [terminatorOperand, opResult, outSharding] :
         llvm::zip_equal(terminator->getOperands(), op.getResults(),
                         adaptor.getOutShardings().getShardings())) {
      auto copy = rewriter.create<CopyOp>(loc, terminatorOperand);
      copy->setAttr(kXlaShardingAttr, manualSharding);

      shardToFullAttributes.back() = rewriter.getNamedAttr(
          kShardingAttr, TensorShardingPerValueAttr::get(context, outSharding));
      auto shardToFull = rewriter.create<CustomCallOp>(
          loc, opResult.getType(), copy.getResult(), shardToFullAttributes);
      rewriter.replaceAllUsesWith(opResult, shardToFull.getResult(0));
    }

    rewriter.eraseOp(terminator);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ShardMapExportPass
    : public mlir::PassWrapper<ShardMapExportPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardMapExportPass)

 private:
  void runOnOperation() final {
    MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addIllegalOp<ManualComputationOp>();
    // We need to explicitly mark FuncDialect as legal because when inlining
    // the ManualComputationOp we will replace its block arguments with
    // the func arguments or the func return operands with the inlined return
    // values. Similarly for the MhloDialect since a ManualComputationOp might
    // be nested within an MHLO op, e.g., a while loop.
    target.addLegalDialect<mlir::func::FuncDialect, mlir::mhlo::MhloDialect>();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ManualComputationPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "xla-sdy-shard-map-export"; }

  StringRef getDescription() const override {
    return "Replaces sdy::ManualComputationOp with the pattern that XLA "
           "recognizes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createShardMapExportPass() {
  return std::make_unique<ShardMapExportPass>();
}

void registerShardMapExportPass() {
  mlir::registerPass(createShardMapExportPass);
}

}  // namespace sdy
}  // namespace xla

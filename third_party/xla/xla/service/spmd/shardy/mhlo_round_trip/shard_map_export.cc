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

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
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
#include "shardy/dialect/sdy/ir/constants.h"  // from @shardy
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "shardy/dialect/sdy/ir/utils.h"  // from @shardy
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"
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
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::Value;
using ::mlir::mhlo::CopyOp;
using ::mlir::mhlo::CustomCallOp;

namespace sdy = ::mlir::sdy;
using sdy::AxisRefAttr;
using sdy::kShardingAttr;
using sdy::ManualComputationOp;
using sdy::MeshAttr;
using sdy::SdyDialect;
using sdy::TensorShardingAttr;
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
    auto inOutShardings = llvm::concat<const TensorShardingAttr>(
        adaptor.getInShardings().getShardings(),
        adaptor.getOutShardings().getShardings());
    if (inOutShardings.begin() == inOutShardings.end()) {
      // If there are no in/out shardings, op.getManualAxes() must be empty. We
      // directly inline the body and erase the op.
      rewriter.eraseOp(getBodyTerminator(op));
      rewriter.inlineBlockBefore(&op.getBody().front(), op);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // For a ManualComputationOp, all in/out shardings, shardings in the body,
    // and manual axes must refer to the same mesh.
    StringRef meshName = inOutShardings.begin()->getMeshName();
    MeshAttr mesh = mlir::sdy::getMeshAttr(op, meshName);
    CHECK(mesh);
    // If `fullyManual` is true, all axes are manual. Otherwise, partial axes
    // are manual and other axes are free (sharded or replicated) in the body of
    // the manual computation.
    bool fullyManual = mesh.getAxes().size() == op.getManualAxes().size();

    MLIRContext* context = rewriter.getContext();
    SmallVector<AxisRefAttr> manualAxes;
    llvm::transform(op.getManualAxes(), std::back_inserter(manualAxes),
                    [&](StringAttr manualAxis) {
                      return AxisRefAttr::get(context, manualAxis);
                    });

    std::function<StringAttr(const HloSharding&)> getStringAttr =
        [&](const HloSharding& hloSharding) {
          return rewriter.getStringAttr(hloSharding.ToString());
        };
    auto getMeshAttr = [&](TensorShardingAttr) { return mesh; };

    StringAttr fullyManualSharding = getStringAttr(HloSharding::Manual());
    auto partialManualSharding = [&](mlir::Type type) {
      int64_t rank = 0;
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        rank = tensorType.getRank();
      }

      TensorShardingAttr fullyOpen =
          TensorShardingAttr::getFullyOpen(context, rank, meshName);
      HloSharding hloSharding =
          convertToHloSharding(fullyOpen, getMeshAttr, manualAxes);
      return getStringAttr(hloSharding);
    };

    auto createAttributes =
        [&](StringRef callTargetName) -> SmallVector<NamedAttribute, 2> {
      return {rewriter.getNamedAttr("call_target_name",
                                    rewriter.getStringAttr(callTargetName)),
              rewriter.getNamedAttr(kXlaShardingAttr, fullyManualSharding)};
    };
    SmallVector<NamedAttribute, 2> fullToShardAttributes =
        createAttributes(kSPMDFullToShardShapeCallTargetName);
    SmallVector<NamedAttribute, 2> shardToFullAttributes =
        createAttributes(kSPMDShardToFullShapeCallTargetName);

    // We export the shardings in the body.
    if (fullyManual) {
      // All operations in the body have fully manual sharding.
      op.getBody().front().walk([&](Operation* opInBody) {
        opInBody->setAttr(kXlaShardingAttr, fullyManualSharding);
        // Remove the possible fully replicated sdy.sharding attribute.
        opInBody->removeAttr(kShardingAttr);
      });
    } else {
      // All operations in the body must be sharded or replicated along free
      // axes. If an operation does not have sharding annotation, it is fully
      // replicated along free axes.
      op.getBody().front().walk([&](Operation* opInBody) {
        TensorShardingPerValueAttr shardingPerValue =
            opInBody->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr);
        if (!shardingPerValue) {
          shardingPerValue = TensorShardingPerValueAttr::getFullyOpen(
              context, opInBody->getResultTypes(), meshName);
        }
        opInBody->setAttr(
            kXlaShardingAttr,
            convertToHloShardingAttr(opInBody, shardingPerValue.getShardings(),
                                     getMeshAttr, getStringAttr, manualAxes));
        opInBody->removeAttr(kShardingAttr);
      });
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

      if (!fullyManual) {
        fullToShardAttributes.back() = rewriter.getNamedAttr(
            kXlaShardingAttr, partialManualSharding(localArgumentType));
      }
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
      copy->setAttr(kXlaShardingAttr,
                    fullyManual
                        ? fullyManualSharding
                        : partialManualSharding(copy.getResult().getType()));

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

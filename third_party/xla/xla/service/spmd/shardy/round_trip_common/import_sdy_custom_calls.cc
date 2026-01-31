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

#include "xla/service/spmd/shardy/round_trip_common/import_sdy_custom_calls.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/sharding_op_util.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::IntegerAttr;
using ::mlir::StringRef;
using ::mlir::sdy::PropagationBarrierOp;
using ::mlir::sdy::PropagationDirectionAttr;
using ::mlir::sdy::ShardingConstraintOp;
using ::mlir::sdy::ShardingGroupOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::stablehlo::CustomCallOp;
using ::mlir::stablehlo::CustomCallOpAdaptor;

mlir::LogicalResult rewriteShardingCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) {
  CHECK_EQ(op.getNumOperands(), 1);

  std::vector<int64_t> unspecDims;
  if (std::optional<mlir::Attribute> backendConfig = op.getBackendConfig()) {
    StringRef configStr =
        mlir::dyn_cast<mlir::StringAttr>(*backendConfig).getValue();
    CHECK_OK(xla::sharding_op_util::ParseAttributes(
        absl::string_view(configStr.data(), configStr.size()), &unspecDims));
  }

  if (op->getNumResults() != 1) {
    op.emitError() << "expected CustomCallOp with exactly one result";
    return mlir::failure();
  }
  TensorShardingAttr sharding = mlir::sdy::getSharding(op->getResult(0));
  if (!sharding) {
    op.emitError() << "expected CustomCallOp with a sharding attribute";
    return mlir::failure();
  }

  if (!unspecDims.empty()) {
    sharding = sharding.openShardingDims(unspecDims);
  }

  rewriter.replaceOpWithNewOp<ShardingConstraintOp>(
      op, adaptor.getInputs().front(), sharding);

  return mlir::success();
}

mlir::LogicalResult rewritePropagationBarrierCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) {
  CHECK_EQ(op.getNumOperands(), 1);
  CHECK_EQ(op.getNumResults(), 1);
  std::optional<PropagationDirectionAttr> allowedDirection =
      tryGetFrontendAttr<PropagationDirectionAttr>(op, kAllowedDirectionAttr);
  if (!allowedDirection.has_value()) {
    op.emitError() << "expected PropagationBarrier CustomCall Op with a "
                      "propagation direction.";
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<PropagationBarrierOp>(
      op, adaptor.getInputs().front(), allowedDirection->getValue());

  return mlir::success();
}
mlir::LogicalResult rewriteShardingGroupCustomCall(
    CustomCallOp op, CustomCallOpAdaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) {
  CHECK_EQ(op.getNumOperands(), 1);
  CHECK_LE(op.getNumResults(), 1);

  std::optional<IntegerAttr> shardingGroupId =
      tryGetFrontendAttr<IntegerAttr>(op, kShardingGroupIdAttr);
  if (!shardingGroupId.has_value()) {
    return op.emitError() << "expected CustomCallOp with a sharding group id.";
  }
  if (!op.use_empty()) {
    return op.emitError()
           << "xla.sdy.ShardingGroup CustomCallOp should have no uses.";
  }

  rewriter.replaceOpWithNewOp<ShardingGroupOp>(op, adaptor.getInputs().front(),
                                               shardingGroupId->getInt());

  return mlir::success();
}

class SdyCustomCallPattern : public mlir::OpConversionPattern<CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getCallTargetName() == kShardingCustomCallTargetName) {
      return rewriteShardingCustomCall(op, adaptor, rewriter);
    }

    if (op.getCallTargetName() == kShardingGroupCustomCallTargetName) {
      return rewriteShardingGroupCustomCall(op, adaptor, rewriter);
    }
    if (op.getCallTargetName() == kPropagationBarrierCustomCallTargetName) {
      return rewritePropagationBarrierCustomCall(op, adaptor, rewriter);
    }

    return rewriter.notifyMatchFailure(
        op, "expected CustomCallOp with xla.sdy target name.");
  }
};

// Convert custom calls into sdy APIs.
// * xla.sdy.Sharding -> ShardingConstraintOp
// * xla.sdy.ShardingGroup -> ShardingGroupOp
class ImportSdyCustomCallsPass
    : public mlir::PassWrapper<ImportSdyCustomCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportSdyCustomCallsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::sdy::SdyDialect>();
    target.addDynamicallyLegalOp<CustomCallOp>([](CustomCallOp op) {
      return op.getCallTargetName() != kShardingCustomCallTargetName &&
             op.getCallTargetName() != kShardingGroupCustomCallTargetName &&
             op.getCallTargetName() != kPropagationBarrierCustomCallTargetName;
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.add<SdyCustomCallPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-import-sdy-custom-calls";
  }

  StringRef getDescription() const override {
    return "Converts a CustomCall with target name Sharding into a "
           "ShardingConstraintOp and with target name ShardingGroup into a "
           "ShardingGroupOp.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportSdyCustomCallsPass() {
  return std::make_unique<ImportSdyCustomCallsPass>();
}

void registerImportSdyCustomCallsPass() {
  mlir::registerPass(createImportSdyCustomCallsPass);
}

}  // namespace sdy
}  // namespace xla

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

#include "xla/service/spmd/shardy/round_trip_common/convert_sharding_group_custom_calls.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::StringRef;
using ::mlir::mhlo::CustomCallOp;
using ::mlir::sdy::ShardingGroupOp;

class ShardingGroupCustomCallPattern
    : public mlir::OpConversionPattern<CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != kShardingGroupCustomCallTargetName) {
      return rewriter.notifyMatchFailure(
          op, "expected CustomCallOp with target name " +
                  kShardingCustomCallTargetName.str());
    }

    CHECK_EQ(op.getNumOperands(), 1);
    if (op->getNumResults() != 0) {
      op.emitError() << "expected dangling CustomCallOp with no results.";
      return mlir::failure();
    }

    std::optional<int64_t> shardingGroupId = mlir::sdy::getShardingGroupId(op);
    if (!shardingGroupId.has_value()) {
      op.emitError() << "expected CustomCallOp with a sharding group id.";
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ShardingGroupOp>(
        op, adaptor.getInputs().front(), shardingGroupId.value());

    return mlir::success();
  }
};

// Converts a CustomCall with target name ShardingGroup into a ShardingGroupOp.
class ConvertShardingGroupCustomCallsPass
    : public mlir::PassWrapper<ConvertShardingGroupCustomCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertShardingGroupCustomCallsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::sdy::SdyDialect>();
    target.addDynamicallyLegalOp<CustomCallOp>([](CustomCallOp op) {
      return op.getCallTargetName() != kShardingGroupCustomCallTargetName;
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ShardingGroupCustomCallPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-convert-sharding-group-custom-calls";
  }

  StringRef getDescription() const override {
    return "Converts a CustomCall with target name ShardingGroup into a "
           "ShardingGroupOp.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createConvertShardingGroupCustomCallsPass() {
  return std::make_unique<ConvertShardingGroupCustomCallsPass>();
}

void registerConvertShardingGroupCustomCallsPass() {
  mlir::registerPass(createConvertShardingGroupCustomCallsPass);
}

}  // namespace sdy
}  // namespace xla

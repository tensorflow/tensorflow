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

#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpConversionPattern;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;

// Converts `CustomCallOp`s called `@local_xla.sdy.ManualComputation` with in/out
// shardings and manual axes as frontend attrs to `ManualComputationOp`s.
class ManualComputationPattern : public OpConversionPattern<CustomCallOp> {
 public:
  explicit ManualComputationPattern(MLIRContext* context,
                                    const SymbolTable& symbolTable)
      : OpConversionPattern<CustomCallOp>(context), symbolTable(symbolTable) {}

  mlir::LogicalResult matchAndRewrite(
      CustomCallOp customCallOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (customCallOp.getCallTargetName() !=
        kManualComputationCustomCallTargetName) {
      return mlir::failure();
    }

    CHECK_EQ(customCallOp.getCalledComputations().size(), 1);
    auto shmapBodyFunc =
        symbolTable.lookup<FuncOp>((*customCallOp.getCalledComputations()
                                         .getAsRange<mlir::FlatSymbolRefAttr>()
                                         .begin())
                                       .getValue());
    if (shmapBodyFunc.empty()) {
      return customCallOp->emitOpError(
          "expected a unique FuncOp per "
          "@local_xla.sdy.ManualComputation custom call. Were "
          "functions maybe somehow shared/de-duped between "
          "two ManualComputations?");
    }

    mlir::DictionaryAttr frontendAttrs = getFrontendAttrs(customCallOp);
    CHECK(frontendAttrs);
    auto manualComputationOp =
        rewriter.replaceOpWithNewOp<sdy::ManualComputationOp>(
            customCallOp, customCallOp->getResultTypes(),
            customCallOp->getOperands(),
            parseStringAttr<sdy::TensorShardingPerValueAttr>(frontendAttrs,
                                                             kInShardings),
            parseStringAttr<sdy::TensorShardingPerValueAttr>(frontendAttrs,
                                                             kOutShardings),
            parseStringAttr<sdy::ManualAxesAttr>(frontendAttrs, kManualAxes));
    sdy::inlineRegionAndConvertTerminatorOp<sdy::ReturnOp>(
        shmapBodyFunc.getBody(), manualComputationOp.getRegion(), rewriter);
    rewriter.eraseOp(shmapBodyFunc);
    return mlir::success();
  }

 private:
  const SymbolTable& symbolTable;
};

class SdyRoundTripShardMapImportPass
    : public mlir::PassWrapper<SdyRoundTripShardMapImportPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapImportPass)

 private:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(module);
    MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addDynamicallyLegalOp<CustomCallOp>([](CustomCallOp op) {
      return op.getCallTargetName() != kManualComputationCustomCallTargetName;
    });
    target.addLegalOp<sdy::ManualComputationOp, sdy::ReturnOp>();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ManualComputationPattern>(&context, symbolTable);
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-import";
  }

  StringRef getDescription() const override {
    return "converts CustomCalls called @local_xla.sdy.manual_computation_body "
           "with in/out shardings and manual axes as frontend attrs to a "
           "`ManualComputationOp`";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect>();
  }
};

}  // namespace

void registerSdyRoundTripShardMapImportPass() {
  mlir::registerPass(createSdyRoundTripShardMapImportPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapImportPass() {
  return std::make_unique<SdyRoundTripShardMapImportPass>();
}

}  // namespace sdy
}  // namespace xla

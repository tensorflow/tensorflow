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

#include <cassert>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;

// Converts a CallOp calling a @local_xla.sdy.manual_computation_body func with in/out
// shardings and manual axes as frontend attrs, wrapped with custom calls that
// change the shape of the arguments/results to a `ManualComputationOp`. See
// `SdyRoundTripShardMapExportPass` for its counterpart.
class ManualComputationPattern : public OpConversionPattern<CallOp> {
 public:
  explicit ManualComputationPattern(MLIRContext* context,
                                    const SymbolTable& symbolTable)
      : OpConversionPattern<CallOp>(context), symbolTable(symbolTable) {}

  mlir::LogicalResult matchAndRewrite(
      CallOp callOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!absl::StrContains(callOp.getCallee(),
                           kManualComputationBodyFuncName)) {
      return mlir::failure();
    }

    // NOTE: if the original `ManualComputationOp` had no operands (results),
    // then a @FullToShard (@ShardToFull) custom call won't be present. So
    // we have to take the operands/results of the newly created
    // `ManualComputationOp` differently depending on whether the original had
    // operands/results.
    CustomCallOp fullToShard;
    mlir::ValueRange operands = callOp->getOperands();
    if (!operands.empty()) {
      fullToShard = callOp->getOperand(0).getDefiningOp<CustomCallOp>();
      CHECK(fullToShard);
      CHECK(fullToShard.getCallTargetName() ==
            kGlobalToLocalShapeCallTargetName);
      operands = fullToShard->getOperands();
    }
    mlir::TypeRange resultTypes = callOp->getResultTypes();
    CustomCallOp shardToFull;
    if (!resultTypes.empty()) {
      CHECK(callOp->getResult(0).hasOneUse())
          << "all CallOp results should be used by a single ShardToFull";
      shardToFull =
          mlir::cast<CustomCallOp>(*callOp->getResult(0).getUsers().begin());
      CHECK(shardToFull.getCallTargetName() ==
            kLocalToGlobalShapeCallTargetName);
      resultTypes = shardToFull->getResultTypes();
    }

    auto shmapBodyFunc = symbolTable.lookup<FuncOp>(callOp.getCallee());
    if (shmapBodyFunc.empty()) {
      return callOp->emitOpError(
          "expected a unique FuncOp per "
          "@local_xla.sdy.manual_computation_body call. Were "
          "functions maybe somehow shared/de-duped between "
          "two ManualComputations?");
    }

    mlir::DictionaryAttr frontendAttrs = getFrontendAttrs(callOp);
    CHECK(frontendAttrs)
        << "Expected in/out shardings and manual axes as frontend attrs on the "
           "CallOp during round tripping.";
    auto manualComputationOp =
        rewriter.replaceOpWithNewOp<sdy::ManualComputationOp>(
            callOp, resultTypes, operands,
            parseStringAttr<sdy::TensorShardingPerValueAttr>(frontendAttrs,
                                                             kInShardings),
            parseStringAttr<sdy::TensorShardingPerValueAttr>(frontendAttrs,
                                                             kOutShardings),
            parseStringAttr<sdy::ManualAxesAttr>(frontendAttrs, kManualAxes));
    sdy::inlineRegionAndConvertTerminatorOp<sdy::ReturnOp>(
        shmapBodyFunc.getBody(), manualComputationOp.getRegion(), rewriter);
    rewriter.eraseOp(shmapBodyFunc);
    if (fullToShard) {
      rewriter.eraseOp(fullToShard);
    }
    if (shardToFull) {
      rewriter.replaceOp(shardToFull, manualComputationOp->getResults());
    }
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
    target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
      return !absl::StrContains(op.getCallee(), kManualComputationBodyFuncName);
    });
    target.addLegalOp<sdy::ManualComputationOp, sdy::ReturnOp, CustomCallOp>();
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
    return "converts a CallOp calling a @local_xla.sdy.manual_computation_body func "
           "with in/out shardings and manual axes as frontend attrs, wrapped "
           "with a pair of `CustomCallOps` that change the shape of the "
           "arguments/results, to a ManualComputationOp";
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

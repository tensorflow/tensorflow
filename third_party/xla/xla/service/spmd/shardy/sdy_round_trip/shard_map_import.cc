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
#include "llvm/ADT/STLExtras.h"
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
    if (!callOp.getCallee().contains(kManualComputationBodyFuncName)) {
      return mlir::failure();
    }

    // NOTE: if the original `ManualComputationOp` had no operands (results),
    // then a @GlobalToLocalShape (@LocalToGlobalShape) custom call won't be
    // present. So we have to take the operands/results of the newly created
    // `ManualComputationOp` differently depending on whether the original had
    // operands/results.
    CustomCallOp globalToLocalShape;
    mlir::ValueRange operands = adaptor.getOperands();
    if (!operands.empty()) {
      // An input to `sdy.manual_computation` can have a dimension of size 0
      // (i.e. 0 num-elements), in which case, the corresponding result of
      // `GlobalToLocalShape` custom call would be replaced with a constant of
      // the same shape. Therefore, we skip such operands until we find the
      // first one that is produced by the custom call.
      auto customCallResIt = llvm::find_if(operands, [](mlir::Value operand) {
        return operand.getDefiningOp<CustomCallOp>();
      });
      if (customCallResIt == operands.end()) {
        return callOp->emitOpError("expected at least one operand of ")
               << callOp.getCalleeAttr() << " to be produced by a "
               << kGlobalToLocalShapeCallTargetName << " CustomCallOp";
      }
      globalToLocalShape = (*customCallResIt).getDefiningOp<CustomCallOp>();
      CHECK(globalToLocalShape.getCallTargetName() ==
            kGlobalToLocalShapeCallTargetName);
      operands = globalToLocalShape->getOperands();
    }
    mlir::TypeRange resultTypes = callOp->getResultTypes();
    CustomCallOp localToGlobalShape;
    if (!resultTypes.empty()) {
      // Same as above, a result of `sdy.manual_computation` can have a
      // dimension of size 0, in which case, the corresponding result of
      // `@local_xla.sdy.manual_computation_body` call would be replaced with a
      // constant. Therefore, we check the first use rather than first result.
      if (!callOp->use_empty()) {
        localToGlobalShape =
            mlir::dyn_cast<CustomCallOp>(*callOp->user_begin());
      }
      if (!localToGlobalShape) {
        return callOp->emitOpError("expected the first use of ")
               << callOp.getCalleeAttr() << " to be by a "
               << kLocalToGlobalShapeCallTargetName << " CustomCallOp";
      }
      CHECK(localToGlobalShape.getCallTargetName() ==
            kLocalToGlobalShapeCallTargetName);
      resultTypes = localToGlobalShape->getResultTypes();
    }

    auto shmapBodyFunc = symbolTable.lookup<FuncOp>(callOp.getCallee());
    if (shmapBodyFunc.empty()) {
      return callOp->emitOpError(
          "expected a unique FuncOp per "
          "@local_xla.sdy.manual_computation_body call. Were "
          "functions maybe somehow shared/de-duped between "
          "two ManualComputations?");
    }

    MLIRContext* context = rewriter.getContext();
    sdy::TensorShardingPerValueAttr inShardings =
        sdy::TensorShardingPerValueAttr::get(context, {});
    sdy::TensorShardingPerValueAttr outShardings =
        sdy::TensorShardingPerValueAttr::get(context, {});
    sdy::ManualAxesAttr manualAxes = sdy::ManualAxesAttr::get(context, {});
    bool newCodePath = false;

    auto setShardingAttrs = [&newCodePath, &manualAxes](
                                CustomCallOp customCallOp,
                                sdy::TensorShardingPerValueAttr& shardings,
                                llvm::StringRef shardingAttrName) {
      if (!customCallOp) {
        return;
      }
      if (mlir::DictionaryAttr frontendAttrs = getFrontendAttrs(customCallOp)) {
        newCodePath = true;
        shardings = parseStringAttr<sdy::TensorShardingPerValueAttr>(
            frontendAttrs, shardingAttrName);
        if (manualAxes.empty()) {
          manualAxes =
              parseStringAttr<sdy::ManualAxesAttr>(frontendAttrs, kManualAxes);
        }
      }
    };

    setShardingAttrs(globalToLocalShape, inShardings, kInShardings);
    setShardingAttrs(localToGlobalShape, outShardings, kOutShardings);
    // TODO(b/410499196): Code to handle loading an old checkpoint. Remove after
    // 6 months of cl/745735176 being submitted.
    mlir::DictionaryAttr callOpFrontendAttrs = getFrontendAttrs(callOp);
    if (!newCodePath && callOpFrontendAttrs) {
      inShardings = parseStringAttr<sdy::TensorShardingPerValueAttr>(
          callOpFrontendAttrs, kInShardings);
      outShardings = parseStringAttr<sdy::TensorShardingPerValueAttr>(
          callOpFrontendAttrs, kOutShardings);
      manualAxes = parseStringAttr<sdy::ManualAxesAttr>(callOpFrontendAttrs,
                                                        kManualAxes);
    }
    auto manualComputationOp =
        rewriter.replaceOpWithNewOp<sdy::ManualComputationOp>(
            callOp, resultTypes, operands, inShardings, outShardings,
            manualAxes);
    sdy::inlineRegionAndConvertTerminatorOp<sdy::ReturnOp>(
        shmapBodyFunc.getBody(), manualComputationOp.getRegion(), rewriter);
    rewriter.eraseOp(shmapBodyFunc);
    if (globalToLocalShape) {
      rewriter.eraseOp(globalToLocalShape);
    }
    if (localToGlobalShape) {
      rewriter.replaceOp(localToGlobalShape, manualComputationOp->getResults());
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
      return !op.getCallee().contains(kManualComputationBodyFuncName);
    });
    target.addLegalOp<sdy::ManualComputationOp, sdy::ReturnOp, CustomCallOp>();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ManualComputationPattern>(&context, symbolTable);
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    // At this point, there may be stray `xla.sdy.GlobalToLocalShape` and
    // `xla.sdy.LocalToGlobalShape`, if the `@local_xla.sdy.manual_computation_body`
    // call was eliminated through DCE and the custom call uses were replaced
    // with constants as they had 0 elements, then it's safe to erase.
    module->walk([](CustomCallOp op) {
      if (op.getCallTargetName() == kGlobalToLocalShapeCallTargetName ||
          op.getCallTargetName() == kLocalToGlobalShapeCallTargetName) {
        CHECK(op.use_empty());
        op.erase();
      }
    });
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

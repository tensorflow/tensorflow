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

#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
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
using ::mlir::StringRef;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

namespace stablehlo = ::mlir::stablehlo;
namespace sdy = ::mlir::sdy;

class SdyRoundTripShardMapExportPass
    : public mlir::PassWrapper<SdyRoundTripShardMapExportPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapExportPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext* context = moduleOp.getContext();
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);
    auto rewriter = mlir::IRRewriter(context);
    moduleOp->walk([&](sdy::ManualComputationOp manualComputation) {
      rewriter.setInsertionPointToEnd(&moduleOp.getRegion().front());
      mlir::Location loc = manualComputation.getLoc();
      mlir::Region& manualCompBody = manualComputation.getBody();
      mlir::TypeRange manualCompBodyArgTypes =
          manualCompBody.getArgumentTypes();
      mlir::TypeRange localResultTypes =
          sdy::getBodyTerminatorOpOperandTypes(manualComputation);
      auto funcOp = rewriter.create<FuncOp>(
          loc, kManualComputationBodyFuncName,
          rewriter.getFunctionType(manualCompBodyArgTypes, localResultTypes));
      mlir::StringAttr funcName = symbolTable.insert(funcOp);

      rewriter.setInsertionPoint(manualComputation);
      stablehlo::CustomCallOp fullToShard;
      mlir::ValueRange operands = manualComputation->getOperands();
      if (!operands.empty()) {
        fullToShard = rewriter.create<stablehlo::CustomCallOp>(
            loc, manualCompBodyArgTypes, operands);
        fullToShard.setCallTargetName(kGlobalToLocalShapeCallTargetName);
        operands = fullToShard->getResults();
      }

      auto callOp =
          rewriter.create<CallOp>(loc, localResultTypes, funcName, operands);
      setFrontendAttribute(callOp, kInShardings,
                           manualComputation.getInShardings());
      setFrontendAttribute(callOp, kOutShardings,
                           manualComputation.getOutShardings());
      setFrontendAttribute(callOp, kManualAxes,
                           manualComputation.getManualAxesAttr());

      mlir::ResultRange results = manualComputation->getResults();
      if (!results.empty()) {
        auto shardToFull = rewriter.create<stablehlo::CustomCallOp>(
            loc, manualComputation.getResultTypes(), callOp->getResults());
        shardToFull.setCallTargetName(kLocalToGlobalShapeCallTargetName);
        results = shardToFull->getResults();
      }
      sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
          manualCompBody, funcOp.getBody());
      rewriter.replaceOp(manualComputation, results);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-export";
  }

  StringRef getDescription() const override {
    return "Converts the body of a ManualComputationOp to a separate function "
           "with a CallOp and a pair of CustomCallOps that change the shape of "
           "the arguments/results. The CallOp saves the in/out shardings and "
           "manual axes as frontend attrs.";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<stablehlo::StablehloDialect>();
  }
};

}  // namespace

void registerSdyRoundTripShardMapExportPass() {
  mlir::registerPass(createSdyRoundTripShardMapExportPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapExportPass() {
  return std::make_unique<SdyRoundTripShardMapExportPass>();
}

}  // namespace sdy
}  // namespace xla

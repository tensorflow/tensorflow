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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/CallGraph.h"
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
#include "mlir/Support/WalkResult.h"
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
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;

mlir::LogicalResult rewriteManualComputation(
    CallOp callOp, mlir::IRRewriter& rewriter,
    const mlir::SymbolTable& symbolTable) {
  auto shmapBodyFunc = symbolTable.lookup<FuncOp>(callOp.getCallee());

  // If the callOp has no uses, but has at least one result, then it means
  // all its results have a dimension of size 0 (i.e. 0 num-elements), and
  // therefore they were replaced with constants of the same shape. In which
  // case, we can safely erase the callOp and the manual computation body
  // function.
  if (callOp.use_empty() && !callOp->getResults().empty()) {
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  // NOTE: if the original `ManualComputationOp` had no operands (results),
  // then a @GlobalToLocalShape (@LocalToGlobalShape) custom call won't be
  // present. So we have to take the operands/results of the newly created
  // `ManualComputationOp` differently depending on whether the original had
  // operands/results.
  CustomCallOp globalToLocalShape;
  mlir::ValueRange operands = callOp.getOperands();
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
    CHECK_EQ(globalToLocalShape.getCallTargetName(),
             kGlobalToLocalShapeCallTargetName);
    operands = globalToLocalShape->getOperands();
  }

  mlir::TypeRange resultTypes = callOp->getResultTypes();
  CustomCallOp localToGlobalShape;
  if (!resultTypes.empty()) {
    // Same as above, a result of `sdy.manual_computation` can have a
    // dimension of size 0, in which case, the corresponding result of
    // `@xla.sdy.manual_computation_body` call would be replaced with a
    // constant. Therefore, we check the first use rather than first result.
    CHECK(!callOp->use_empty());
    localToGlobalShape = mlir::dyn_cast<CustomCallOp>(*callOp->user_begin());
    if (!localToGlobalShape) {
      return callOp->emitOpError("expected the first use of ")
             << callOp.getCalleeAttr() << " to be a "
             << kLocalToGlobalShapeCallTargetName << " CustomCallOp";
    }
    CHECK_EQ(localToGlobalShape.getCallTargetName(),
             kLocalToGlobalShapeCallTargetName);
    resultTypes = localToGlobalShape->getResultTypes();
  }

  MLIRContext* context = rewriter.getContext();
  sdy::TensorShardingPerValueAttr inShardings =
      sdy::TensorShardingPerValueAttr::get(context, {});
  sdy::TensorShardingPerValueAttr outShardings =
      sdy::TensorShardingPerValueAttr::get(context, {});
  sdy::ManualAxesAttr manualAxes = sdy::ManualAxesAttr::get(context, {});

  auto setShardingAttrs = [&manualAxes](
                              CustomCallOp customCallOp,
                              sdy::TensorShardingPerValueAttr& shardings,
                              llvm::StringRef shardingAttrName) {
    if (!customCallOp) {
      return;
    }
    if (mlir::DictionaryAttr frontendAttrs = getFrontendAttrs(customCallOp)) {
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
  auto manualComputationOp =
      rewriter.replaceOpWithNewOp<sdy::ManualComputationOp>(
          callOp, resultTypes, operands, inShardings, outShardings, manualAxes);
  sdy::inlineRegionAndConvertTerminatorOp<sdy::ReturnOp>(
      shmapBodyFunc.getBody(), manualComputationOp.getRegion(), rewriter);
  if (localToGlobalShape) {
    rewriter.replaceAllUsesWith(localToGlobalShape.getResults(),
                                manualComputationOp->getResults());
  }
  return mlir::success();
}

std::pair<FuncOp, bool> cloneManualComputationsRecursively(
    CallOp callOp, SymbolTable& symbolTable,
    llvm::SmallDenseSet<StringRef>& manualComputationNames) {
  FuncOp funcOp = symbolTable.lookup<FuncOp>(callOp.getCallee());
  auto [_, inserted] = manualComputationNames.insert(funcOp.getName());
  if (!inserted) {
    funcOp = funcOp.clone();
  }
  funcOp->walk([&](CallOp callOp) {
    if (!callOp.getCallee().contains(kManualComputationFuncName)) {
      return;
    }
    if (auto [funcOp, cloned] = cloneManualComputationsRecursively(
            callOp, symbolTable, manualComputationNames);
        cloned) {
      callOp.setCallee(symbolTable.insert(funcOp));
    }
  });
  return {funcOp, !inserted};
}

void flattenManualComputations(ModuleOp module, SymbolTable& symbolTable) {
  llvm::SmallDenseSet<StringRef> manualComputationNames;
  module->walk([&](FuncOp funcOp) {
    if (funcOp.getName().contains(kManualComputationFuncName)) {
      return;
    }
    funcOp->walk([&](CallOp callOp) {
      if (!callOp.getCallee().contains(kManualComputationFuncName)) {
        return;
      }
      if (auto [funcOp, cloned] = cloneManualComputationsRecursively(
              callOp, symbolTable, manualComputationNames);
          cloned) {
        callOp.setCallee(symbolTable.insert(funcOp));
      }
    });
  });
}

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
    mlir::IRRewriter rewriter(module);

    flattenManualComputations(module, symbolTable);

    llvm::SmallDenseSet<StringRef> manualComputationCalleeNames;
    mlir::CallGraph callGraph(module);
    llvm::ReversePostOrderTraversal<const mlir::CallGraph*> rpo(&callGraph);
    for (mlir::CallGraphNode* node : llvm::reverse(rpo)) {
      if (node->isExternal()) continue;
      if (node->getCallableRegion()
              ->walk([&](CallOp callOp) {
                if (!callOp.getCallee().contains(kManualComputationFuncName)) {
                  return mlir::WalkResult::advance();
                }
                manualComputationCalleeNames.insert(callOp.getCallee());
                rewriter.setInsertionPoint(callOp);
                if (mlir::failed(rewriteManualComputation(callOp, rewriter,
                                                          symbolTable))) {
                  callOp.emitError(
                      "failed to rewrite func.call to manual computation");
                  return mlir::WalkResult::interrupt();
                }
                return mlir::WalkResult::advance();
              })
              .wasInterrupted()) {
        return signalPassFailure();
      }
    }

    // Erase all `xla.sdy.GlobalToLocalShape` and `xla.sdy.LocalToGlobalShape`
    // custom calls.
    //
    // NOTE: In addition to the ones that were used by calls, at this point,
    // there may be stray `xla.sdy.GlobalToLocalShape` and
    // `xla.sdy.LocalToGlobalShape`, if the `@xla.sdy.manual_computation_body`
    // call was eliminated through DCE and the custom call uses were replaced
    // with constants as they had 0 elements, then it's safe to erase.
    module->walk([](CustomCallOp op) {
      if (op.getCallTargetName() == kGlobalToLocalShapeCallTargetName ||
          op.getCallTargetName() == kLocalToGlobalShapeCallTargetName) {
        CHECK(op.use_empty());
        op.erase();
      }
    });

    // Erase all manual computation func ops that now have no call ops.
    for (StringRef calleeName : manualComputationCalleeNames) {
      symbolTable.erase(symbolTable.lookup(calleeName));
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-import";
  }

  StringRef getDescription() const override {
    return "converts a CallOp calling a @xla.sdy.manual_computation_body func "
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

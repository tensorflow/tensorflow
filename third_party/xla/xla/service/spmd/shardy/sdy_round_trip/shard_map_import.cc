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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
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
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::WalkOrder;
using ::mlir::WalkResult;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;

// Reconstructs the global tensor sharding for an input of
// `sdy.manual_computation` from its local sharding on
// `xla.sdy.GlobalToLocalShape` and the manual axes.
sdy::TensorShardingAttr reconstructGlobalInSharding(
    sdy::TensorShardingAttr localSharding, mlir::Type globalType,
    mlir::Type localType, ArrayRef<StringAttr> manualAxes,
    const mlir::SymbolTable& symbolTable) {
  if (!localSharding || manualAxes.empty()) {
    return localSharding;
  }
  sdy::MeshAttr meshAttr = localSharding.getMesh(symbolTable);
  if (!meshAttr) {
    return localSharding;
  }
  auto globalTensorType =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(globalType);
  auto localTensorType =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(localType);
  if (!globalTensorType || !localTensorType) {
    return localSharding;
  }

  int64_t rank = globalTensorType.getRank();
  SmallVector<sdy::DimensionShardingAttr> dimShardings =
      llvm::to_vector(localSharding.getDimShardings());
  if (static_cast<int64_t>(dimShardings.size()) != rank) {
    return localSharding;
  }

  llvm::DenseSet<StringRef> usedManualAxes;
  for (int64_t d = 0; d < rank; ++d) {
    int64_t globalDim = globalTensorType.getDimSize(d);
    int64_t localDim = localTensorType.getDimSize(d);
    if (localDim <= 0 || globalDim == localDim) {
      continue;
    }
    int64_t ratio = globalDim / localDim;
    SmallVector<sdy::AxisRefAttr> axesForDim;
    for (StringAttr manualAxis : manualAxes) {
      if (usedManualAxes.contains(manualAxis.getValue())) {
        continue;
      }
      int64_t axisSize = meshAttr.getAxisSize(manualAxis.getValue());
      if (axisSize > 1 && (ratio % axisSize == 0)) {
        axesForDim.push_back(sdy::AxisRefAttr::get(manualAxis.getContext(),
                                                   manualAxis.getValue()));
        usedManualAxes.insert(manualAxis.getValue());
        ratio /= axisSize;
      }
    }
    sdy::DimensionShardingAttr existingDim = dimShardings[d];
    SmallVector<sdy::AxisRefAttr> newDimAxes = std::move(axesForDim);
    llvm::append_range(newDimAxes, existingDim.getAxes());
    dimShardings[d] = sdy::DimensionShardingAttr::get(
        existingDim.getContext(), newDimAxes, existingDim.getIsClosed(),
        existingDim.getPriority());
  }

  SmallVector<sdy::AxisRefAttr> replicatedAxes =
      llvm::to_vector(localSharding.getReplicatedAxes());
  for (StringAttr manualAxis : manualAxes) {
    if (!usedManualAxes.contains(manualAxis.getValue())) {
      replicatedAxes.push_back(sdy::AxisRefAttr::get(manualAxis.getContext(),
                                                     manualAxis.getValue()));
    }
  }

  return sdy::TensorShardingAttr::get(
      localSharding.getContext(), localSharding.getMeshOrRef(), dimShardings,
      replicatedAxes, localSharding.getUnreducedAxes());
}

// Infers manual axes from output shape expansion ratios on LocalToGlobalShape
// when GlobalToLocalShape is absent (input-free manual computation).
sdy::ManualAxesAttr inferManualAxesFromOutputs(
    CustomCallOp localToGlobalShape,
    sdy::TensorShardingPerValueAttr outShardings,
    const mlir::SymbolTable& symbolTable, MLIRContext* context) {
  if (!localToGlobalShape || !outShardings) {
    return sdy::ManualAxesAttr::get(context, {});
  }
  SmallVector<StringAttr> manualAxes;
  llvm::SmallDenseSet<StringRef> seenAxes;
  mlir::ValueRange localOperands = localToGlobalShape.getOperands();
  mlir::ResultRange globalResults = localToGlobalShape.getResults();
  for (auto [i, sharding] : llvm::enumerate(outShardings.getShardings())) {
    if (i >= localOperands.size() || i >= globalResults.size()) break;
    auto globalType =
        mlir::dyn_cast<mlir::RankedTensorType>(globalResults[i].getType());
    auto localType =
        mlir::dyn_cast<mlir::RankedTensorType>(localOperands[i].getType());
    if (!globalType || !localType) continue;
    sdy::MeshAttr mesh = sharding.getMesh(symbolTable);
    if (!mesh) continue;
    for (int64_t d = 0; d < globalType.getRank(); ++d) {
      int64_t globalDim = globalType.getDimSize(d);
      int64_t localDim = localType.getDimSize(d);
      if (localDim <= 0 || globalDim <= localDim) continue;
      int64_t ratio = globalDim / localDim;
      for (sdy::AxisRefAttr axisRef : sharding.getDimShardings()[d].getAxes()) {
        if (seenAxes.contains(axisRef.getName())) continue;
        int64_t size = mesh.getAxisSize(axisRef.getName());
        if (size > 1 && (ratio % size == 0)) {
          manualAxes.push_back(StringAttr::get(context, axisRef.getName()));
          seenAxes.insert(axisRef.getName());
          ratio /= size;
        }
      }
    }
  }
  return sdy::ManualAxesAttr::get(context, manualAxes);
}

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
      if (hasKey(frontendAttrs, shardingAttrName)) {
        shardings = parseStringAttr<sdy::TensorShardingPerValueAttr>(
            frontendAttrs, shardingAttrName);
      }
      if (manualAxes.empty() && hasKey(frontendAttrs, kManualAxes)) {
        manualAxes =
            parseStringAttr<sdy::ManualAxesAttr>(frontendAttrs, kManualAxes);
      }
    }
  };

  setShardingAttrs(globalToLocalShape, inShardings, kInShardings);
  setShardingAttrs(localToGlobalShape, outShardings, kOutShardings);

  auto getDirectSharding =
      [&](CustomCallOp op) -> sdy::TensorShardingPerValueAttr {
    if (!op) return nullptr;
    if (auto sharding = op->getAttrOfType<sdy::TensorShardingPerValueAttr>(
            sdy::kShardingAttr)) {
      return sharding;
    }
    if (auto single =
            op->getAttrOfType<sdy::TensorShardingAttr>(sdy::kShardingAttr)) {
      return sdy::TensorShardingPerValueAttr::get(context, single);
    }
    return nullptr;
  };

  if ((!outShardings || outShardings.empty()) && localToGlobalShape) {
    outShardings = getDirectSharding(localToGlobalShape);
  }

  // If frontend attributes were not present, read directly from custom call
  // attributes (HloShardingV3 round-trip).
  if (manualAxes.empty()) {
    for (CustomCallOp op : {globalToLocalShape, localToGlobalShape}) {
      if (op &&
          (manualAxes = op->getAttrOfType<sdy::ManualAxesAttr>(kManualAxes))) {
        break;
      }
    }
    if ((!manualAxes || manualAxes.empty()) && localToGlobalShape) {
      manualAxes = inferManualAxesFromOutputs(localToGlobalShape, outShardings,
                                              symbolTable, context);
    }
    if (!manualAxes) {
      manualAxes = sdy::ManualAxesAttr::get(context, {});
    }
  }

  if ((!inShardings || inShardings.empty()) && globalToLocalShape) {
    if (sdy::TensorShardingPerValueAttr localInShardings =
            getDirectSharding(globalToLocalShape)) {
      SmallVector<sdy::TensorShardingAttr> reconstructedShardings;
      mlir::ValueRange globalOperands = globalToLocalShape.getOperands();
      mlir::ResultRange localResults = globalToLocalShape.getResults();
      for (auto [i, localSharding] :
           llvm::enumerate(localInShardings.getShardings())) {
        mlir::Type globalType =
            i < globalOperands.size() ? globalOperands[i].getType() : nullptr;
        mlir::Type localType =
            i < localResults.size() ? localResults[i].getType() : nullptr;
        reconstructedShardings.push_back(
            reconstructGlobalInSharding(localSharding, globalType, localType,
                                        manualAxes.getValue(), symbolTable));
      }
      inShardings =
          sdy::TensorShardingPerValueAttr::get(context, reconstructedShardings);
    }
  }

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

SmallVector<StringAttr> getManualAxesList(
    ArrayRef<ArrayRef<StringAttr>> manualAxesStack) {
  SmallVector<StringAttr> manualAxesList;
  for (ArrayRef<StringAttr> manualAxesRefs : manualAxesStack) {
    for (StringAttr manualAxes : manualAxesRefs) {
      manualAxesList.push_back(manualAxes);
    }
  }
  return manualAxesList;
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

    if (!mlir::sdy::walkCalls(module, [&](CallOp callOp) {
          if (isManualComputation(callOp)) {
            rewriter.setInsertionPoint(callOp);
            if (mlir::failed(
                    rewriteManualComputation(callOp, rewriter, symbolTable))) {
              // TODO(enver): Return callOp.emitError direcly here and
              // elsewhere.
              callOp.emitError(
                  "failed to rewrite func.call to manual computation");
              return mlir::WalkResult::interrupt();
            }
          }
          return mlir::WalkResult::advance();
        })) {
      return signalPassFailure();
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

    // Erase all manual computation func ops as now they have no call ops.
    for (FuncOp funcOp : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
      StringRef funcName = funcOp.getName();
      if (isManualComputationOnName(funcName)) {
        symbolTable.erase(symbolTable.lookup(funcName));
      }
    }

    // Set func manual axes.
    sdy::iterateFuncs(
        module,
        [&](FuncOp funcOp) {
          SmallVector<ArrayRef<StringAttr>> manualAxesStack;
          if (auto funcManualAxes = funcOp->getAttrOfType<sdy::ManualAxesAttr>(
                  sdy::kFuncManualAxes)) {
            manualAxesStack.push_back(funcManualAxes.getValue());
          }

          funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (auto manualComputationOp =
                    mlir::dyn_cast<sdy::ManualComputationOp>(op)) {
              manualAxesStack.push_back(manualComputationOp.getManualAxes());
            } else if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
              if (!manualAxesStack.empty()) {
                FuncOp calledFuncOp =
                    sdy::getFuncOpOrDie(callOp.getCallee(), symbolTable);
                calledFuncOp->setAttr(
                    sdy::kFuncManualAxes,
                    sdy::ManualAxesAttr::get(
                        op->getContext(), getManualAxesList(manualAxesStack)));
              }
            } else if (op->hasTrait<mlir::OpTrait::IsTerminator>() &&
                       mlir::isa<sdy::ManualComputationOp>(op->getParentOp())) {
              manualAxesStack.pop_back();
            }
          });
        },
        /*preOrder=*/true);
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

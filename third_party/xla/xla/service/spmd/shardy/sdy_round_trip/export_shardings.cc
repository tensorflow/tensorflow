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

#include "xla/service/spmd/shardy/sdy_round_trip/export_shardings.h"

#include <cstdint>
#include <memory>

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
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::DictionaryAttr;
using ::mlir::LogicalResult;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::Value;
using ::mlir::func::FuncOp;

using ::mlir::mhlo::CustomCallOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Saves `shardingPerValueAttr` including any existing `frontendAttributes` on
// the `op`.
void saveOpShardingPerValueAttr(Operation* op,
                                TensorShardingPerValueAttr shardingPerValueAttr,
                                OpBuilder& builder) {
  addFrontendAttribute(op, kShardingRoundTripAttr, shardingPerValueAttr);
}

// Converts the shardings from `kShardingAttr` into
// `kShardingRoundTripStringAttr`.
LogicalResult exportFunc(FuncOp funcOp, OpBuilder& builder) {
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    if (auto oldSharding = funcOp.getArgAttrOfType<TensorShardingAttr>(
            argNum, kShardingAttr)) {
      addFrontendAttribute(funcOp, kShardingRoundTripAttr, oldSharding, argNum);
      funcOp.removeArgAttr(argNum, kShardingAttr);
    }
  }

  for (mlir::OpOperand& returnOperand :
       mlir::sdy::getBodyTerminatorOpOperands(funcOp)) {
    int64_t resultNum = returnOperand.getOperandNumber();
    if (auto sharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
            resultNum, kShardingAttr)) {
      // We cannot save the result shardings as frontend attributes. MHLO->HLO
      // conversion converts `mhlo.sharding`s on the results to a tuple
      // sharding on the ROOT instruction, but it discards the frontend
      // attributes on the MLIR results. Thus, instead of making the conversion
      // handle that, push the shardings to a temporary custom op for each
      // FuncOp result with a sharding. Then during import, we will copy the
      // Op's sharding to the FuncOp's result and delete te temporary custom
      // call.
      Value returnValue = returnOperand.get();
      builder.setInsertionPoint(returnOperand.getOwner());
      auto customCallOp = builder.create<CustomCallOp>(
          returnValue.getLoc(), returnValue.getType(), returnValue);
      customCallOp.setCallTargetName(kFuncResultShardingTargetName);
      // Want to prevent the canonicalizer from de-duplicating func sharding
      // custom calls which actually may have different sharding attributes.
      customCallOp.setHasSideEffect(true);
      saveOpShardingPerValueAttr(
          customCallOp,
          TensorShardingPerValueAttr::get(customCallOp.getContext(), sharding),
          builder);
      returnOperand.set(customCallOp.getResult(0));
      funcOp.removeResultAttr(resultNum, builder.getStringAttr(kShardingAttr));
    }
  }

  funcOp.front().walk([&](Operation* op) {
    if (auto oldShardingPerValue =
            op->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)) {
      saveOpShardingPerValueAttr(op, oldShardingPerValue, builder);
      op->removeAttr(kShardingAttr);
    }
  });

  return mlir::success();
}

class ExportShardonnayShardingsPass
    : public PassWrapper<ExportShardonnayShardingsPass,
                         mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportShardonnayShardingsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext* context = moduleOp.getContext();
    auto builder = OpBuilder(context);

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (mlir::failed(exportFunc(funcOp, builder))) {
        signalPassFailure();
      }
    }

    SmallVector<NamedAttribute> mhloMeshes;
    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);
    // Saves the MeshOps for MHLO<->HLO round-trip and removes them from the
    // ModuleOp.
    for (MeshOp meshOp :
         llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      mhloMeshes.emplace_back(
          meshOp.getSymNameAttr(),
          getStringAttribute(meshOp.getMeshAttr(), builder));
      symbolTable.erase(meshOp);
    }
    addFrontendAttribute(moduleOp, kMeshesRoundTripAttr,
                         DictionaryAttr::get(context, mhloMeshes));
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-export-shardings";
  }

  StringRef getDescription() const override {
    return "Converts the shardings from kShardingAttr to "
           "kShardingRoundTripAttr in the HLO frontend attributes and saves "
           "the mesh symbols as kMeshesRoundTripAttr in the module frontend "
           "attributes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect, mlir::mhlo::MhloDialect>();
  }
};

}  // namespace

void registerSdyRoundTripExportShardingsPass() {
  mlir::registerPass(createSdyRoundTripExportShardingsPass);
}

std::unique_ptr<Pass> createSdyRoundTripExportShardingsPass() {
  return std::make_unique<ExportShardonnayShardingsPass>();
}

}  // namespace sdy
}  // namespace xla

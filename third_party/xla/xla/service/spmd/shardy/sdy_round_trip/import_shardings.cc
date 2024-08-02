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

#include "xla/service/spmd/shardy/sdy_round_trip/import_shardings.h"

#include <cassert>
#include <cstdint>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::Attribute;
using ::mlir::DictionaryAttr;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::SymbolTableCollection;
using ::mlir::func::FuncOp;

using ::mlir::mhlo::CustomCallOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Parses `stringAttr` to an attribute of type `AttrTy`.
//
// NOTE: assumes `stringAttr` is of type `StringAttr`.
template <typename AttrTy>
AttrTy parseStringAttr(Attribute stringAttr) {
  return mlir::cast<AttrTy>(mlir::parseAttribute(
      mlir::cast<StringAttr>(stringAttr), stringAttr.getContext()));
}

// Parses `attrName` from `dictAttr` to an attribute of type `AttrTy`.
template <typename AttrTy>
AttrTy parseStringAttr(DictionaryAttr dictAttr, llvm::StringRef attrName) {
  return parseStringAttr<AttrTy>(dictAttr.get(attrName));
}

// Builds the shardings coming from Shardy previously. This means
// the module was exported from Shardy and we are now round-tripping back.
// This should happen after the meshes were created from the `ModuleOp` attrs
// (see `SdyRoundTripImportShardingsPass`).
void convertShardings(FuncOp funcOp) {
  // Copy over the argument shardings, but not the result shardings yet.
  // We need to wait until after we've converted all the Operations before
  // copying the result shardings.
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    funcOp.removeArgAttr(argNum, kXlaShardingAttr);
    // Attempt to extract the TensorShardingAttr from the frontend attributes of
    // the function argument/result.
    if (DictionaryAttr dictAttr = getFuncArgFrontendAttrs(funcOp, argNum)) {
      funcOp.setArgAttr(argNum, kShardingAttr,
                        parseStringAttr<TensorShardingAttr>(
                            dictAttr, kShardingRoundTripAttr));
      removeFrontendAttribute(funcOp, kShardingRoundTripAttr, argNum);
    }
  }

  // Due to `SdyRoundTripExportShardingsPass` keeping `mhlo.sharding`s, remove
  // them purely for cleanliness of the module.
  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    funcOp.removeResultAttr(
        resNum, StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
  }

  // Extract the round-tripped SDY shardings from the operations.
  funcOp.front().walk([&](Operation* op) {
    op->removeAttr(kXlaShardingAttr);
    if (DictionaryAttr dictAttr = getFrontendAttrs(op)) {
      // NOTE: we are only setting the sharding on known custom-calls. For any
      // other op that has a `kShardingRoundTripAttr` we discard it. XLA
      // sometimes creates new instructions, copying over the operand's frontend
      // attrs, which may mean the shapes are wrong when the new instruction is
      // a reshape for example. This does mean we can't fully round-trip b/w HLO
      // and MLIR after SDY propagation.
      if (auto customCallOp = mlir::dyn_cast<CustomCallOp>(op)) {
        StringRef targetName = customCallOp.getCallTargetName();
        if (targetName == kFuncResultShardingTargetName) {
          // This is a temporary CustomCallOp that holds the sharding from a
          // func result. When importing we want to move that sharding to the
          // func result and delete the CustomCallOp.
          auto shardingPerValueAttr =
              parseStringAttr<TensorShardingPerValueAttr>(
                  dictAttr, kShardingRoundTripAttr);
          for (mlir::OpOperand& use :
               llvm::make_early_inc_range(customCallOp->getUses())) {
            int64_t resNum = use.getOperandNumber();
            funcOp.setResultAttr(resNum, kShardingAttr,
                                 shardingPerValueAttr.getSharding(0));
            mlir::sdy::getBodyTerminator(funcOp)->setOperand(
                resNum, customCallOp.getOperand(0));
          }
          customCallOp.erase();
          return;
        }
        if (targetName == kShardingCustomCallTargetName ||
            targetName == kSPMDFullToShardShapeCallTargetName ||
            targetName == kSPMDShardToFullShapeCallTargetName) {
          customCallOp->setAttr(kShardingAttr,
                                parseStringAttr<TensorShardingPerValueAttr>(
                                    dictAttr, kShardingRoundTripAttr));
        }
      }
      removeFrontendAttribute(op, kShardingRoundTripAttr);
    }
  });
}

class SdyRoundTripImportShardingsPass
    : public mlir::PassWrapper<SdyRoundTripImportShardingsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripImportShardingsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);
    // If there is a dictionary attribute `kFrontendAttributesAttr` and it
    // contains `kMeshesRoundTripAttr`, it means that the function was a
    // Shardy function and we are roundtripping back to Shardy. In that
    // case, we can use the saved string attributes to restore the original mesh
    // and value shardings with the original mesh axis names and priorities on
    // the sharding.
    DictionaryAttr moduleDictAttr = getFrontendAttrs(moduleOp);
    if (!moduleDictAttr) {
      moduleOp.emitError(
          "Expected an attribute `kFrontendAttributesAttr` on the module that "
          "contains the Shardy meshes.");
      signalPassFailure();
      return;
    }

    auto sdyMeshes =
        parseStringAttr<DictionaryAttr>(moduleDictAttr, kMeshesRoundTripAttr);
    mlir::OpBuilder builder(moduleOp);
    // Insert the meshes before any functions.
    builder.setInsertionPointToStart(moduleOp.getBody());
    for (NamedAttribute mesh : sdyMeshes) {
      auto meshAttr = parseStringAttr<MeshAttr>(mesh.getValue());
      symbolTable.insert(builder.create<mlir::sdy::MeshOp>(
          moduleOp.getLoc(), mesh.getName(), meshAttr));
    }
    removeFrontendAttribute(moduleOp, kMeshesRoundTripAttr);

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      convertShardings(funcOp);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-import-shardings";
  }

  StringRef getDescription() const override {
    return "Converts the shardings from strings in MHLO frontend attributes to "
           "SDY meshes and shardings.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripImportShardingsPass() {
  return std::make_unique<SdyRoundTripImportShardingsPass>();
}

void registerSdyRoundTripImportShardingsPass() {
  mlir::registerPass(createSdyRoundTripImportShardingsPass);
}

}  // namespace sdy
}  // namespace xla

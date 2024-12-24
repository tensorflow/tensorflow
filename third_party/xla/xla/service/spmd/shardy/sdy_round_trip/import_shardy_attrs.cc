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

#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

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
#include "mlir/IR/PatternMatch.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::Attribute;
using ::mlir::DictionaryAttr;
using ::mlir::IRRewriter;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;

using ::mlir::stablehlo::CustomCallOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::kShardingRuleAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::OpShardingRuleAttr;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Builds the shardy attributes coming from Shardy previously. This means
// the module was exported from Shardy and we are now round-tripping back.
// This should happen after the meshes were created from the `ModuleOp` attrs
// (see `SdyRoundTripImportShardyAttrsPass`).
void convertShardyAttrs(FuncOp funcOp, IRRewriter& rewriter) {
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

  // Extract the round-tripped SDY shardy attributes from the operations.
  funcOp.front().walk([&](Operation* op) {
    op->removeAttr(kXlaShardingAttr);
    DictionaryAttr dictAttr = getFrontendAttrs(op);
    if (!dictAttr) {
      return;
    }
    // NOTE: we are only setting the sharding on known custom-calls. For any
    // other op that has a `kShardingRoundTripAttr` we discard it. XLA sometimes
    // creates new instructions, copying over the operand's frontend attrs,
    // which may mean the shapes are wrong when the new instruction is a reshape
    // for example. This does mean we can't fully round-trip b/w HLO and MLIR
    // after SDY propagation.
    if (auto customCallOp = mlir::dyn_cast<CustomCallOp>(op)) {
      StringRef targetName = customCallOp.getCallTargetName();
      if (targetName == kFuncResultShardingTargetName) {
        // This is a temporary CustomCallOp that holds the sharding from a
        // func result. When importing we want to move that sharding to the
        // func result and delete the CustomCallOp.
        auto shardingPerValueAttr = parseStringAttr<TensorShardingPerValueAttr>(
            dictAttr, kShardingRoundTripAttr);
        for (mlir::OpOperand& use :
             llvm::make_early_inc_range(customCallOp->getUses())) {
          // We currently ignore users that are not the func return op.
          // This might happen due to inlined func ops that originally had
          // result shardings.
          // TODO(b/370984308): explore if we need to support this properly.
          if (mlir::isa<mlir::func::ReturnOp>(use.getOwner())) {
            funcOp.setResultAttr(use.getOperandNumber(), kShardingAttr,
                                 shardingPerValueAttr.getSharding(0));
            use.set(customCallOp.getOperand(0));
          }
        }
        rewriter.replaceOp(customCallOp, customCallOp.getOperand(0));
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

    // Import sharding rules.
    if (auto shardingRuleAttr = parseStringAttr<OpShardingRuleAttr>(
            dictAttr, kShardingRuleRoundTripAttr)) {
      op->setAttr(kShardingRuleAttr, shardingRuleAttr);
      removeFrontendAttribute(op, kShardingRuleRoundTripAttr);
    }
  });
}

class SdyRoundTripImportShardyAttrsPass
    : public mlir::PassWrapper<SdyRoundTripImportShardyAttrsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripImportShardyAttrsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // We can use the saved string attributes to restore the original mesh and
    // value shardings with the original mesh axis names and priorities on the
    // sharding. If there is no `kMeshesRoundTripAttr, there were no meshes in
    // the original Shardy model.
    std::optional<DictionaryAttr> meshesAttr =
        tryGetFrontendAttr<DictionaryAttr>(moduleOp, kMeshesRoundTripAttr);
    mlir::ArrayRef<NamedAttribute> sdyMeshes =
        meshesAttr.has_value() ? meshesAttr.value().getValue()
                               : mlir::ArrayRef<NamedAttribute>();

    IRRewriter rewriter(moduleOp);
    // Insert the meshes before any functions.
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    SymbolTable symbolTable(moduleOp);
    for (NamedAttribute mesh : sdyMeshes) {
      auto meshAttr = mlir::cast<MeshAttr>(mesh.getValue());
      symbolTable.insert(rewriter.create<mlir::sdy::MeshOp>(
          moduleOp.getLoc(), mesh.getName(), meshAttr));
    }
    removeFrontendAttribute(moduleOp, kMeshesRoundTripAttr);

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      convertShardyAttrs(funcOp, rewriter);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-import-shardy-attrs";
  }

  StringRef getDescription() const override {
    return "Converts the shardy attributes from strings in MHLO frontend "
           "attributes to SDY meshes, shardings and sharding rules.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripImportShardyAttrsPass() {
  return std::make_unique<SdyRoundTripImportShardyAttrsPass>();
}

void registerSdyRoundTripImportShardyAttrsPass() {
  mlir::registerPass(createSdyRoundTripImportShardyAttrsPass);
}

}  // namespace sdy
}  // namespace xla

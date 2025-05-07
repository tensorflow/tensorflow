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

#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"

#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/service/spmd/shardy/constants.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayAttr;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::NamedComputationOp;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Converts a `NamedComputationOp` into a `CallOp`.
class ExportNamedComputationsPass
    : public mlir::PassWrapper<ExportNamedComputationsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportNamedComputationsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    mlir::Block& moduleBlock = moduleOp.getRegion().front();
    getOperation()->walk([&](NamedComputationOp namedComputationOp) {
      mlir::IRRewriter rewriter(namedComputationOp);
      rewriter.setInsertionPointToEnd(&moduleBlock);
      auto funcOp = rewriter.create<FuncOp>(
          namedComputationOp.getLoc(), namedComputationOp.getName(),
          rewriter.getFunctionType(
              namedComputationOp.getBody().getArgumentTypes(),
              namedComputationOp.getResultTypes()),
          rewriter.getStringAttr("private"),
          /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());
      rewriter.setInsertionPointToStart(funcOp->getBlock());
      mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
          namedComputationOp.getBody(), funcOp.getBody());
      rewriter.setInsertionPoint(namedComputationOp);

      ManualAxesAttr manualAxesAttr =
          namedComputationOp->getAttrOfType<ManualAxesAttr>(kManualAxes);
      std::optional<TensorShardingPerValueAttr> inShardings =
          namedComputationOp.getInShardings();
      std::optional<TensorShardingPerValueAttr> outShardings =
          namedComputationOp.getOutShardings();

      if (manualAxesAttr) {
        CHECK(!manualAxesAttr.getValue().empty());
        CHECK(inShardings.has_value());
        CHECK(outShardings.has_value());
      }

      // Copy the input shardings to the func.
      if (inShardings.has_value()) {
        for (auto [i, sharding] :
             llvm::enumerate(inShardings->getShardings())) {
          funcOp.setArgAttr(i, kShardingAttr, sharding);
          if (manualAxesAttr) {
            funcOp.setArgAttr(i, kManualAxes, manualAxesAttr);
          }
        }
      }

      // Copy the output shardings to the func.
      if (outShardings.has_value()) {
        for (auto [i, sharding] :
             llvm::enumerate(outShardings->getShardings())) {
          funcOp.setResultAttr(i, kShardingAttr, sharding);
          if (manualAxesAttr) {
            funcOp.setResultAttr(i, kManualAxes, manualAxesAttr);
          }
        }
      }

      // Replace the `NamedComputationOp` with a `CallOp`.
      mlir::SmallVector<NamedAttribute> callOpAttrs(
          namedComputationOp->getDiscardableAttrs());
      auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
          namedComputationOp, namedComputationOp.getResultTypes(),
          symbolTable.insert(funcOp), namedComputationOp.getOperands());
      callOp->setAttrs(callOpAttrs);

      // Copy the output shardings to the call op.
      if (outShardings.has_value()) {
        mlir::sdy::setShardings(callOp, *outShardings);
        if (manualAxesAttr) {
          callOp->setAttr(kManualAxes, manualAxesAttr);
        }
      }
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-export-named-computations";
  }

  StringRef getDescription() const override {
    return "Converts a `NamedComputationOp` to a `CallOp` with a new private "
           "function called the `NamedComputationOp`'s `name`. The new "
           "`FuncOp` and `CallOp` have the same shardings as the original "
           "`NamedComputationOp`s operands/results.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExportNamedComputationsPass() {
  return std::make_unique<ExportNamedComputationsPass>();
}

void registerExportNamedComputationsPass() {
  mlir::registerPass(createExportNamedComputationsPass);
}

}  // namespace sdy
}  // namespace xla

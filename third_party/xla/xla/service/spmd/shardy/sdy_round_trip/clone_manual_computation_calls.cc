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

#include "xla/service/spmd/shardy/sdy_round_trip/clone_manual_computation_calls.h"

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
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

namespace stablehlo = ::mlir::stablehlo;
namespace sdy = ::mlir::sdy;

class SdyRoundTripCloneManualComputationCallsPass
    : public mlir::PassWrapper<SdyRoundTripCloneManualComputationCallsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripCloneManualComputationCallsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    // Clone multiple calls to the same function.
    llvm::SmallDenseSet<StringRef> seenCalleeNames;
    moduleOp->walk([&](CallOp op) {
      if (!op.getCallee().contains(kManualComputationBodyFuncName)) {
        return;
      }
      if (!seenCalleeNames.contains(op.getCallee())) {
        seenCalleeNames.insert(op.getCallee());
        return;
      }
      auto funcOp =
          cast<FuncOp>(symbolTable.lookup<FuncOp>(op.getCallee())->clone());
      op.setCallee(symbolTable.insert(funcOp));
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-clone-manual-computation-calls";
  }

  StringRef getDescription() const override { return ""; }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<stablehlo::StablehloDialect>();
  }
};

}  // namespace

void registerSdyRoundTripCloneManualComputationCallsPass() {
  mlir::registerPass(createSdyRoundTripCloneManualComputationCallsPass);
}

std::unique_ptr<mlir::Pass>
createSdyRoundTripCloneManualComputationCallsPass() {
  return std::make_unique<SdyRoundTripCloneManualComputationCallsPass>();
}

}  // namespace sdy
}  // namespace xla

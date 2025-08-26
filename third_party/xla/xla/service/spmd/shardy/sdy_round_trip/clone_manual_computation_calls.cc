/* Copyright 2025 The OpenXLA Authors.

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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

namespace stablehlo = ::mlir::stablehlo;

class SdyRoundTripCloneManualComputationCallsPass
    : public mlir::PassWrapper<SdyRoundTripCloneManualComputationCallsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripCloneManualComputationCallsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Clone multiple calls to the same function.
    llvm::DenseSet<StringRef> seenCalleeNames;
    moduleOp->walk([&](CallOp op) {
      if (!op.getCallee().contains(kManualComputationFuncName)) {
        return;
      }
      if (seenCalleeNames.insert(op.getCallee()).second) {
        return;
      }
      // TODO(b/430894772): Clone manual computations but with a body just calls
      // a newly created shared function with the body of the manual computation
      // bodyinstead of copying the body for each cloned manual computation, so
      // they can be potentially be deduplicated at the end.
      auto funcOp = symbolTable.lookup<FuncOp>(op.getCallee())->clone();
      op.setCallee(symbolTable.insert(funcOp));
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-clone-manual-computation-calls";
  }

  StringRef getDescription() const override {
    return "Clone xla.sdy.manual_computation_body functions so that each call "
           "has its own unique function.";
  }
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

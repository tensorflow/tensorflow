/* Copyright 2026 The OpenXLA Authors.
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
#include "xla/service/spmd/shardy/sdy_round_trip/flatten_call_graph.h"

#include <memory>

#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {
namespace {

using ::mlir::ModuleOp;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingPerValueAttr;

class SdyRoundTripFlattenCallGraphPass
    : public mlir::PassWrapper<SdyRoundTripFlattenCallGraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripFlattenCallGraphPass)
  SdyRoundTripFlattenCallGraphPass() {}
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    llvm::SmallDenseSet<StringRef> funcNames;
    mlir::CallGraph callGraph(moduleOp);
    llvm::ReversePostOrderTraversal<const mlir::CallGraph*> rpo(&callGraph);
    for (mlir::CallGraphNode* node : llvm::reverse(rpo)) {
      if (node->isExternal()) {
        continue;
      }
      // TODO(enver): Should we special handle loops and conditionals?
      node->getCallableRegion()->walk([&](CallOp callOp) {
        FuncOp funcOp = symbolTable.lookup<FuncOp>(callOp.getCallee());
        CHECK(funcOp) << "Failed to lookup function: "
                      << callOp.getCallee().str();
        TensorShardingPerValueAttr callOpResultShardings =
            mlir::sdy::getShardingPerValue(callOp);
        if (auto [_, inserted] = funcNames.insert(funcOp.getName()); inserted) {
          if (callOpResultShardings) {
            mlir::sdy::setFuncResultShardings(funcOp, callOpResultShardings);
          }
          return;
        }
        callOp.setCallee(symbolTable.insert(
            cloneFuncRecursively(funcOp, callOpResultShardings, symbolTable)));
      });
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-flatten-call-graph";
  }

  StringRef getDescription() const override {
    return "Flattens the call graph.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripFlattenCallGraphPass() {
  return std::make_unique<SdyRoundTripFlattenCallGraphPass>();
}

void registerSdyRoundTripFlattenCallGraphPass() {
  mlir::registerPass([]() { return createSdyRoundTripFlattenCallGraphPass(); });
}
}  // namespace sdy
}  // namespace xla

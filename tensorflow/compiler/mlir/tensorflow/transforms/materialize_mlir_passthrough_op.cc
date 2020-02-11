/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Parser.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "tf-materialize-passthrough-op"

namespace mlir {
namespace {

class MaterializePassthroughOpPass
    : public FunctionPass<MaterializePassthroughOpPass> {
 public:
  void runOnFunction() override;
};

void MaterializePassthroughOpPass::runOnFunction() {
  getFunction().walk([](Operation *op) {
    auto passthrough_op = dyn_cast<TF::MlirPassthroughOp>(op);
    if (!passthrough_op) return;
    std::string module_string(passthrough_op.mlir_module());
    // Parse the module.
    auto nested_module = parseSourceString(module_string, op->getContext());
    if (!nested_module) {
      op->emitError() << "could not parse attached MLIR module";
      return;
    }
    FuncOp main = dyn_cast<FuncOp>(nested_module->lookupSymbol("main"));
    if (!main) {
      op->emitError() << "MLIR Opaque Op expects a main() entry point\n";
      return;
    }
    if (main.getNumArguments() != op->getNumOperands()) {
      op->emitError() << "mismatch between MLIR Opaque Op number of operands ("
                      << op->getNumOperands()
                      << ") and main() entry point in the module ("
                      << main.getNumArguments() << " args)\n";
      return;
    }
    if (main.getType().getNumResults() != op->getNumResults()) {
      op->emitError() << "mismatch between MLIR Opaque Op number of results ("
                      << op->getNumResults()
                      << ") and main() entry point in the module ("
                      << main.getType().getNumResults() << " results)\n";
      return;
    }
    Region &body = main.getBody();
    if (body.getBlocks().size() != 1) {
      op->emitError() << "MLIR Opaque Op expects a main() entry point with a "
                         "single block\n";
      return;
    }
    Block &block = body.front();
    for (const auto &arg_mapping :
         llvm::zip(block.getArguments(), op->getOperands())) {
      std::get<0>(arg_mapping).replaceAllUsesWith(std::get<1>(arg_mapping));
    }
    op->getBlock()->getOperations().splice(op->getIterator(),
                                           block.getOperations(), block.begin(),
                                           std::prev(block.end()));
    Operation &return_op = block.front();
    for (auto ret_mapping :
         llvm::zip(op->getResults(), return_op.getOperands())) {
      std::get<0>(ret_mapping).replaceAllUsesWith(std::get<1>(ret_mapping));
    }
    op->erase();
  });
}

}  // namespace

namespace TF {
std::unique_ptr<OpPassBase<FuncOp>> CreateMaterializePassthroughOpPass() {
  return std::make_unique<MaterializePassthroughOpPass>();
}
}  // namespace TF

static PassRegistration<MaterializePassthroughOpPass> pass(
    "tf-materialize-passthrough-op",
    "Materialize the MlirPassthroughOp by replacing it with the MLIR module "
    "attached as an attribute");

}  // namespace mlir

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"

namespace mlir {
namespace quant {
namespace {

class InsertQuantizedFunctionsPass
    : public mlir::PassWrapper<InsertQuantizedFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-insert-quantized-functions";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Insert quantized functions into the module";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect, StandardOpsDialect>();
  }

 private:
  void runOnOperation() override;
};

static PassRegistration<InsertQuantizedFunctionsPass> pass;

void InsertQuantizedFunctionsPass::runOnOperation() {
  tensorflow::Env* env = tensorflow::Env::Default();

  std::string input_file = tensorflow::io::JoinPath(
      env->GetRunfilesDir(), "tensorflow", "compiler", "mlir", "quantization",
      "tensorflow", "passes", "quantized_function_library.mlir");

  ModuleOp module = getOperation();

  std::string error_message;
  std::unique_ptr<llvm::MemoryBuffer> file =
      openInputFile(input_file, &error_message);
  if (!file) {
    module.emitError() << "couldn't find the quantized function library.";
    return signalPassFailure();
  }

  SymbolTable symbol_table(module);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<mlir::ModuleOp> module_ref =
      parseSourceFile(source_mgr, module.getContext());

  // Copy all functions used by this signature to the final MLIR module.
  for (FuncOp func : module_ref->getOps<FuncOp>()) {
    // The insert here is a NO-OP if the function already exists.
    symbol_table.insert(func.clone());
  }
}

}  // namespace

// Creates an instance of the pass for inserting quantized functions.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass() {
  return std::make_unique<InsertQuantizedFunctionsPass>();
}

}  // namespace quant
}  // namespace mlir

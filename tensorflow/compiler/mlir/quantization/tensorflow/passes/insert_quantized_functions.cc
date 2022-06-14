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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

// NOLINTNEXTLINE
llvm::cl::opt<mlir::quant::QuantizationMethod> quantization_method_opt(
    "quant-insert-library-quantization-method",
    llvm::cl::init(mlir::quant::QuantizationMethod::kPostTrainingQuantization),
    llvm::cl::desc("Insert library for the quantization method."),
    llvm::cl::values(
        clEnumValN(mlir::quant::QuantizationMethod::kPostTrainingQuantization,
                   "ptq", "Post-training static-range quantization"),
        clEnumValN(mlir::quant::QuantizationMethod::kDynamicRangeQuantization,
                   "drq", "Post-training dynamic-range quantizaiton")));

namespace mlir {
namespace quant {
namespace {

class InsertQuantizedFunctionsPass
    : public mlir::PassWrapper<InsertQuantizedFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertQuantizedFunctionsPass)

  explicit InsertQuantizedFunctionsPass() {
    quantization_method_ = quantization_method_opt;
  }
  explicit InsertQuantizedFunctionsPass(QuantizationMethod quantization_method)
      : quantization_method_(quantization_method) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-insert-quantized-functions";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Insert quantized functions into the module";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, func::FuncDialect>();
  }

 private:
  void runOnOperation() override;

  QuantizationMethod quantization_method_ =
      QuantizationMethod::kPostTrainingQuantization;
};

static PassRegistration<InsertQuantizedFunctionsPass> pass;

void InsertQuantizedFunctionsPass::runOnOperation() {
  std::unique_ptr<llvm::MemoryBuffer> mem_buffer;
  if (quantization_method_ == QuantizationMethod::kDynamicRangeQuantization) {
    mem_buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(kQuantizedFunctionLibraryInMLIR_DRQ),
        /*BufferName=*/"",
        /*RequiresNullTerminator=*/false);
  } else {
    mem_buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(kQuantizedFunctionLibraryInMLIR), /*BufferName=*/"",
        /*RequiresNullTerminator=*/false);
  }

  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(mem_buffer), llvm::SMLoc());
  OwningOpRef<mlir::ModuleOp> module_ref =
      parseSourceFile<mlir::ModuleOp>(source_mgr, module.getContext());
  // Inline and optimize loaded functions.
  MLIRContext* context = &getContext();
  PassManager pm(context);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  StatusScopedDiagnosticHandler diagnostic_handler(context);
  if (failed(pm.run(*module_ref))) {
    mlir::emitError(module.getLoc())
        << "failed to apply the optimization: "
        << diagnostic_handler.ConsumeStatus().error_message();
    signalPassFailure();
    return;
  }

  // Copy all functions used by this signature to the final MLIR module.
  for (func::FuncOp func : module_ref->getOps<func::FuncOp>()) {
    // Do nothing if the function already exists.
    if (symbol_table.lookup(func.getSymName()) != nullptr) continue;

    // Set the function to private and insert to the module.
    func::FuncOp new_func = func.clone();
    new_func.setPrivate();
    symbol_table.insert(new_func);
  }
}

}  // namespace

// Creates an instance of the pass for inserting quantized functions.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass(
    QuantizationMethod quantization_method) {
  return std::make_unique<InsertQuantizedFunctionsPass>(quantization_method);
}

}  // namespace quant
}  // namespace mlir

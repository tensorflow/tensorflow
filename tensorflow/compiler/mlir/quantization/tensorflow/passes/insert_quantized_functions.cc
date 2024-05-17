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
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace quant {
namespace {

using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;
using ::tensorflow::quantization::OpSet;

class InsertQuantizedFunctionsPass
    : public PassWrapper<InsertQuantizedFunctionsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertQuantizedFunctionsPass)

  explicit InsertQuantizedFunctionsPass() = default;
  explicit InsertQuantizedFunctionsPass(QuantMethod quantization_method,
                                        OpSet op_set) {
    quantization_method_ = quantization_method;
    op_set_ = op_set;
  }
  InsertQuantizedFunctionsPass(const InsertQuantizedFunctionsPass& other) {
    quantization_method_ = other.quantization_method_;
    op_set_ = other.op_set_;
  }

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

  // Returns the function library for the given quantization method and opset
  // pair.
  llvm::StringRef GetFunctionLibrary(QuantMethod quantization_method,
                                     OpSet op_set);

  Option<QuantMethod> quantization_method_{
      *this, "quantization-method",
      llvm::cl::init(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8),
      llvm::cl::desc("Choose quantization method."),
      llvm::cl::values(
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8,
                     "ptq", "Post-training static-range quantization"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8,
                     "drq", "Post-training dynamic-range quantizaiton"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8,
                     "weight_only", "Post-training weight_only quantizaiton"))};

  Option<OpSet> op_set_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};
};

llvm::StringRef InsertQuantizedFunctionsPass::GetFunctionLibrary(
    QuantMethod quantization_method, OpSet op_set) {
  absl::flat_hash_map<OpSet, llvm::StringRef> function_library_map;
  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    function_library_map = {
        {OpSet::TF, kQuantizedFunctionLibraryInMLIR_TF_DRQ},
        {OpSet::UNIFORM_QUANTIZED,
         kQuantizedFunctionLibraryInMLIR_UNIFORM_QUANTIZED_DRQ},
        {OpSet::XLA, kQuantizedFunctionLibraryInMLIR_TF_DRQ}};
  } else if (quantization_method ==
             tensorflow::quantization::QuantizationMethod::
                 METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8) {
    // Uniform quantized opset is not supported for weight-only as inputs for
    // weight quantization are floats. And only dequantize_i8 is used from the
    // quantized function library.
    function_library_map = {
        {OpSet::TF, kQuantizedFunctionLibraryInMLIR},
        {OpSet::XLA, kQuantizedFunctionLibraryInMLIR_XLA_WEIGHT_ONLY}};
  } else {
    function_library_map = {{OpSet::TF, kQuantizedFunctionLibraryInMLIR},
                            {OpSet::UNIFORM_QUANTIZED,
                             kQuantizedFunctionLibraryInMLIR_UNIFORM_QUANTIZED},
                            {OpSet::XLA, kQuantizedFunctionLibraryInMLIR}};
  }

  auto it = function_library_map.find(op_set);
  if (it != function_library_map.end()) {
    return it->second;
  }
  return llvm::StringRef();
}

static PassRegistration<InsertQuantizedFunctionsPass> pass;

void InsertQuantizedFunctionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);

  std::unique_ptr<llvm::MemoryBuffer> mem_buffer;
  llvm::StringRef quantized_function_library =
      GetFunctionLibrary(quantization_method_, op_set_);

  if (quantized_function_library.empty()) {
    emitError(module.getLoc())
        << "Failed to get function library for the opset.";
    signalPassFailure();
    return;
  }

  mem_buffer =
      llvm::MemoryBuffer::getMemBuffer(quantized_function_library,
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(mem_buffer), llvm::SMLoc());
  OwningOpRef<ModuleOp> module_ref =
      parseSourceFile<ModuleOp>(source_mgr, module.getContext());
  // Inline and optimize loaded functions.
  MLIRContext* context = &getContext();
  PassManager pm(context);
  pm.addPass(createInlinerPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  StatusScopedDiagnosticHandler diagnostic_handler(context);
  if (failed(pm.run(*module_ref))) {
    emitError(module.getLoc()) << "failed to apply the optimization: "
                               << diagnostic_handler.ConsumeStatus().message();
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

    // For consistency, we require all quantized composite function to have
    // the "tf_quant.quantized_ops" attribute.
    if (!new_func.getSymName().starts_with("quantized_")) continue;
    if (!new_func->hasAttrOfType<ArrayAttr>("tf_quant.quantized_ops")) {
      new_func->emitError() << "Missing \"tf_quant.quantized_ops\" "
                               "attribute in the quantized composite function.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the pass for inserting quantized functions.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass(
    QuantMethod quantization_method, OpSet target_opset) {
  return std::make_unique<InsertQuantizedFunctionsPass>(quantization_method,
                                                        target_opset);
}

}  // namespace quant
}  // namespace mlir

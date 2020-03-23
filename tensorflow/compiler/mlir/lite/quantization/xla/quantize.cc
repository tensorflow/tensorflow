/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/quantization/xla/quantize.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

namespace mlir {
namespace xla_hlo {

static void RegisterDialects() {
  static bool init_once = []() {
    mlir::registerDialect<mlir::xla_hlo::XlaHloDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
    return true;
  }();
  (void)init_once;
}

// Quantizes the model in the computation.
tensorflow::Status XlaQuantize(const tensorflow::tf2xla::Config& config,
                               xla::XlaComputation* computation) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloSnapshot> snapshot,
                      computation->Snapshot());

  RegisterDialects();
  MLIRContext context;
  OwningModuleRef module = ModuleOp::create(UnknownLoc::get(&context));
  auto status = xla::ConvertHloToMlirHlo(
      module.get(), snapshot->mutable_hlo()->mutable_hlo_module());
  if (!status.ok()) {
    LOG(ERROR) << "Hlo module import failed: " << status;
    return status;
  }

  PassManager pm(&context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  mlir::StatusScopedDiagnosticHandler diag_handler(&context);
  LogicalResult result = pm.run(module.get());
  (void)result;

  module->dump();

  return tensorflow::Status::OK();
}

}  // namespace xla_hlo
}  // namespace mlir

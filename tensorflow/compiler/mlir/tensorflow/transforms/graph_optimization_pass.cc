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

#include "tensorflow/compiler/mlir/tensorflow/transforms/graph_optimization_pass.h"

#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TF {
namespace {
using Status = ::tensorflow::Status;
using ConfigProto = ::tensorflow::ConfigProto;
}  // namespace

Status MlirGraphOptimizationPass::Run(const ConfigProto& config_proto,
                                      ModuleOp module) {
  if (!config_proto.experimental().enable_mlir_graph_optimization()) {
    VLOG(1) << "Skipping MLIR Graph Optimization Pass"
            << ", session flag not enabled";
    return Status::OK();
  }

  VLOG(1) << "Run MLIR Graph Optimization Passes";
  PassManager pm(module.getContext());

  // Run island coarsening before shape inference to allow more exact shape
  // inference using constant folding within islands.
  pm.addNestedPass<FuncOp>(tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(CreateTFShapeInferencePass());

  // Assign optimal data layout to layout sensitive operations and delete
  // redundant transposes from the IR.
  LayoutOptimizationPipelineOptions layout_optimization_options;
  CreateLayoutOptimizationPipeline(pm, layout_optimization_options);

  // Prepare IR for exporting.
  pm.addNestedPass<FuncOp>(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to the
  // MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = pm.run(module);
  (void)result;
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir

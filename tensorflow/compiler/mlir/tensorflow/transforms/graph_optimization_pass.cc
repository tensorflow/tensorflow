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

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace mlir {
namespace TF {
namespace {
using Status = absl::Status;
using ConfigProto = ::tensorflow::ConfigProto;
using Graph = ::tensorflow::Graph;
}  // namespace

Status MlirGraphOptimizationPass::Run(
    const std::string& function_name, const ConfigProto& config_proto,
    ModuleOp module, const Graph& graph,
    const tensorflow::FunctionLibraryDefinition& function_library) {
  if (GetPassState(/*device_set=*/nullptr, config_proto, graph,
                   function_library) ==
      ::tensorflow::MlirOptimizationPassState::Disabled) {
    VLOG(1) << "Skipping MLIR Graph Optimization Pass"
            << ", session flag not enabled";
    return absl::OkStatus();
  }

  VLOG(1) << "Run MLIR Graph Optimization Passes";
  PassManager pm(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(pm);

  // Run island coarsening before shape inference to allow more exact shape
  // inference using constant folding within islands.
  pm.addNestedPass<func::FuncOp>(
      tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(CreateTFShapeInferencePass());

  // Assign optimal data layout to layout sensitive operations and delete
  // redundant transposes from the IR.
  LayoutOptimizationPipelineOptions layout_optimization_options;
  CreateLayoutOptimizationPipeline(pm.nest<func::FuncOp>(),
                                   layout_optimization_options);

  // Prepare IR for exporting.
  pm.addPass(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to the
  // MLIRContext into a tensorflow::Status.
  StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = pm.run(module);
  (void)result;
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir

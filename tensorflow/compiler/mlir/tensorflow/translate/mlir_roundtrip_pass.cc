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

#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_pass.h"

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {

using mlir::MLIRContext;

static StatusOr<mlir::OwningModuleRef> Import(
    const GraphOptimizationPassOptions& options, const Graph& graph,
    MLIRContext* context) {
  // TODO(fengliuai): get debug info at runtime.
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  TF_ASSIGN_OR_RETURN(
      auto module,
      ConvertGraphToMlir(graph, debug_info, *options.flib_def, specs, context));
  mlir::StatusScopedDiagnosticHandler status_handler(context);
  if (failed(mlir::verify(*module))) {
    if (VLOG_IS_ON(1)) module->dump();
    return status_handler.ConsumeStatus();
  }
  return module;
}

static Status Export(mlir::OwningModuleRef module,
                     const GraphOptimizationPassOptions& options,
                     std::unique_ptr<Graph>* graph) {
  GraphExportConfig confs;
  FunctionLibraryDefinition exported_function_library(OpRegistry::Global(), {});
  TF_RETURN_IF_ERROR(
      ConvertMlirToGraph(*module, confs, graph, &exported_function_library));
  return options.flib_def->AddLibrary(exported_function_library);
}

static Status Roundtrip(const GraphOptimizationPassOptions& options,
                        std::unique_ptr<Graph>* graph, MLIRContext* context) {
  TF_ASSIGN_OR_RETURN(auto module, Import(options, **graph, context));

  {
    // The TF runtime doesn't like an optimization pipeline
    // to change library functions in-place, i.e. create different functions
    // that have the same names as the functions in the original function
    // library. Some of this constraint come from the fact that Session can
    // extend its function library with the output function library of the
    // bridge and equality checks of FunctionDef's are based on exact contents
    // which is not guaranteed by the TF importer/exporter.
    //
    // Therefore, we rename all these function to new names to avoid any
    // failures in Session::Extend.
    mlir::PassManager pm(context);
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::TF::CreateRenamePrivateFunctionPass());

    mlir::StatusScopedDiagnosticHandler status_handler(context);
    if (mlir::failed(pm.run(module.get()))) {
      return status_handler.ConsumeStatus();
    }
  }
  return Export(std::move(module), options, graph);
}

Status MlirRoundtripPass::Run(const GraphOptimizationPassOptions& options) {
  MLIRContext context;
  if (options.graph) return Roundtrip(options, options.graph, &context);

  // If the graph is partitioned, then try and round trip them individually.
  for (auto& it : *options.partition_graphs) {
    VLOG(1) << "Roundtripping: " << it.first;
    // TODO(jpienaar): Roundtrip results in different failures, investigate.
    TF_RETURN_IF_ERROR(Import(options, *it.second, &context).status());
  }
  return Status::OK();
}

Status MlirImportPass::Run(const GraphOptimizationPassOptions& options) {
  MLIRContext context;
  if (options.graph) {
    if (!Import(options, **options.graph, &context).ok()) {
      metrics::IncrementMLIRImportFailureCount();
    }
  }
  return Status::OK();
}

}  // namespace tensorflow

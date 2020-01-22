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

#include "mlir/Analysis/Verifier.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/graph/graph_constructor.h"
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
  return ConvertMlirToGraph(*module, confs, graph, options.flib_def);
}

static Status Roundtrip(const GraphOptimizationPassOptions& options,
                        std::unique_ptr<Graph>* graph, MLIRContext* context) {
  TF_ASSIGN_OR_RETURN(auto module, Import(options, **graph, context));
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

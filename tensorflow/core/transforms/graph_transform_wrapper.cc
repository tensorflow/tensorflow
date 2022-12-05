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

#include "tensorflow/core/transforms/graph_transform_wrapper.h"

#include <initializer_list>

#include "absl/memory/memory.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {

tensorflow::Status RunTransformOnGraph(
    tensorflow::Graph* graph,
    const std::initializer_list<
        llvm::function_ref<std::unique_ptr<mlir::Pass>()>>& passes,
    const tensorflow::GraphDebugInfo& debug_info) {
  // We are running only a set of Module passes on a Modul, so disable threading
  // to avoid overhead of creating threadpool that won't be used.
  MLIRContext context(MLIRContext::Threading::DISABLED);
  TF_ASSIGN_OR_RETURN(OwningOpRef<ModuleOp> module,
                      ImportGraphAndFunctionsToMlir(&context, debug_info,
                                                    *graph, graph->flib_def()));

  PassManager pm(&context, mlir::PassManager::Nesting::Explicit);
  // Construct passes.
  for (auto& pass : passes) pm.addPass(pass());
  mlir::StatusScopedDiagnosticHandler error_handler(&context);
  if (failed(pm.run(*module)))
    return error_handler.Combine(
        tensorflow::errors::InvalidArgument("MLIR Graph Optimizer failed: "));

  // Export and replace Graph.
  tensorflow::GraphDef graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertToGraphDef(*module, &graphdef),
                                  "when exporting MLIR module to GraphDef");
  graph->Clear();
  graph->mutable_flib_def()->Clear();
  tensorflow::GraphConstructorOptions opts;
  return ConvertGraphDefToGraph(opts, graphdef, graph);
}

}  // namespace tfg
}  // namespace mlir

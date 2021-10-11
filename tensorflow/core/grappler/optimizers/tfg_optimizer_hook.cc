/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/tfg_optimizer_hook.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

TfgGrapplerOptimizer::TfgGrapplerOptimizer(const std::string& pass_pipeline)
    : pass_pipeline_(pass_pipeline) {}

Status TfgGrapplerOptimizer::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::GraphDef* optimized_graph) {
  // The grappler passes operate on a single graph, there is no point in
  // having MLIR threading. Also creating and destroying thread on every
  // Grappler invocation has a non-trivial cost.
  MLIRContext context(MLIRContext::Threading::DISABLED);
#ifndef NDEBUG
  if (VLOG_IS_ON(5))
    fprintf(stderr, "Before Graph: %s\n", item.graph.DebugString().c_str());
#endif  // NDEBUG

  tensorflow::GraphDebugInfo debug_info;

  tensorflow::metrics::ScopedCounter<2> metrics(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {"TfgOptimizer", "convert_graphdef_to_tfg"});
  auto error_or_module = ImportGraphDefToMlir(&context, debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    ::tensorflow::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
#ifndef NDEBUG
    fprintf(
        stderr,
        "Error: %s\n\n=========\n=========\n=========\n=========\n=========\n%s"
        "=========\n=========\n=========\n=========\n",
        status.ToString().c_str(), item.graph.DebugString().c_str());
#endif
    return status;
  }
  metrics.ReportAndStop();

  ModuleOp module = (*error_or_module).get();

  auto graph_op = dyn_cast<GraphOp>(module.getBody()->front());
  if (!graph_op)
    return InvalidArgument("Invariant broken, missing graph op in Module");

  if (!pass_pipeline_.empty()) {
    // Parse the pipeline and run it on the graph.
    PassManager pm(&context, GraphOp::getOperationName());
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(parsePassPipeline(pass_pipeline_, pm, error_stream)))
      return InvalidArgument("Invalid pass_pipeline: ", error_stream.str());

    StatusScopedDiagnosticHandler error_handler(&context);
    if (failed(pm.run(graph_op)))
      return error_handler.Combine(
          InvalidArgument("MLIR Graph Optimizer failed: "));
  }

  tensorflow::GraphDef graphdef;
  *graphdef.mutable_library() = item.graph.library();
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::ExportMlirToGraphdef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  metrics.ReportAndStop();
  *optimized_graph = std::move(graphdef);
#ifndef NDEBUG
  if (VLOG_IS_ON(5)) {
    fprintf(stderr, "After Graph: %s\nMlir Module:\n",
            optimized_graph->DebugString().c_str());
    module.dump();
    fprintf(stderr, "======\n");
  }
#endif  // NDEBUG
  return Status::OK();
}

}  // end namespace tfg
}  // end namespace mlir

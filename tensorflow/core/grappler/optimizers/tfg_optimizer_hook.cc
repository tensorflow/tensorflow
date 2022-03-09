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

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
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
#include "tensorflow/core/transforms/pass_registration.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// The default pipeline is empty.
void DefaultGrapplerPipeline(PassManager& mgr) {}

// The implementation of the TFG optimizer. It holds the MLIR context and the
// pass manager.
class TFGGrapplerOptimizer::Impl {
 public:
  // Builds the pass pipeline. The context is initialized without threading.
  // Creating and destroying the threadpool each time Grappler is invoked is
  // prohibitively expensive.
  // TODO(jeffniu): Some passes may run in parallel on functions. Find a way to
  // hold and re-use a threadpool.
  explicit Impl(TFGPassPipelineBuilder builder)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    builder(mgr_);
  }

  // Runs the pass manager.
  LogicalResult RunPipeline(ModuleOp module) { return mgr_.run(module); }

  // Get the context.
  MLIRContext* GetContext() { return &ctx_; }

  // Convert the pass pipeline to a textual string.
  std::string GetPipelineString() {
    std::string pipeline;
    llvm::raw_string_ostream os(pipeline);
    mgr_.printAsTextualPipeline(os);
    return os.str();
  }

 private:
  // The MLIR context.
  MLIRContext ctx_;
  // The pass manager containing the loaded TFG pass pipeline.
  PassManager mgr_;
};

TFGGrapplerOptimizer::TFGGrapplerOptimizer(TFGPassPipelineBuilder builder)
    : impl_(std::make_unique<Impl>(std::move(builder))) {}

TFGGrapplerOptimizer::~TFGGrapplerOptimizer() = default;

std::string TFGGrapplerOptimizer::name() const {
  return absl::StrCat("tfg_optimizer{", impl_->GetPipelineString(), "}");
}

Status TFGGrapplerOptimizer::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::GraphDef* optimized_graph) {
  VLOG(5) << "TFG Before Graph: \n" << item.graph.DebugString();

  // Import the GraphDef to TFG.
  tensorflow::GraphDebugInfo debug_info;
  tensorflow::metrics::ScopedCounter<2> metrics(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {"TfgOptimizer", "convert_graphdef_to_tfg"});
  auto error_or_module =
      ImportGraphDefToMlir(impl_->GetContext(), debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    tensorflow::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    VLOG(4) << "GraphDef import error: " << status.ToString();
    return status;
  }
  metrics.ReportAndStop();

  // Run the pipeline on the graph.
  ModuleOp module = (*error_or_module).get();
  StatusScopedDiagnosticHandler error_handler(impl_->GetContext());
  if (failed(impl_->RunPipeline(module)))
    return error_handler.Combine(
        InvalidArgument("MLIR Graph Optimizer failed: "));

  // Export the TFG module to GraphDef.
  tensorflow::GraphDef graphdef;
  *graphdef.mutable_library() = item.graph.library();
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::ExportMlirToGraphdef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  metrics.ReportAndStop();
  *optimized_graph = std::move(graphdef);

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "TFG After Graph: \n"
            << optimized_graph->DebugString() << "\nMLIR module: \n";
    module.dump();
  }

  return Status::OK();
}

}  // end namespace tfg
}  // end namespace mlir

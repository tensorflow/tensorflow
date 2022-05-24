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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
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
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/transforms/func_to_graph/func_to_graph.h"
#include "tensorflow/core/transforms/graph_to_func/graph_to_func.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// Lift the graph into a function to embed the semantic of feeds, fetches and
// nodes-to-preserve nodes in GrapplerItem.
static Status LiftGraphToFunc(GraphOp graph,
                              const tensorflow::grappler::GrapplerItem& item) {
  SmallVector<std::string> feed = llvm::to_vector<4>(llvm::map_range(
      item.feed, [](const std::pair<std::string, tensorflow::Tensor>& item) {
        return item.first;
      }));
  SmallVector<std::string> control_rets =
      llvm::to_vector<4>(item.NodesToPreserve());

  Status status = GraphToFunc(graph, feed, item.fetch, control_rets);
  if (!status.ok()) {
    return InvalidArgument(
        "MLIR Graph optimizer failed: Can't lift graph into function: ",
        status.error_message());
  }
  return Status::OK();
}

// Lower the lifted graph function back to the graph.
static Status LowerFuncToGraph(ModuleOp module) {
  auto* dialect = module->getContext()->getLoadedDialect<TFGraphDialect>();
  StringAttr lifted_graph_func_name =
      dialect->getLiftedGraphFuncNameAttrIdentifier();
  GraphFuncOp lifted_graph_func;
  for (auto func : module.getOps<GraphFuncOp>()) {
    if (func.sym_name() == lifted_graph_func_name) {
      lifted_graph_func = func;
      break;
    }
  }

  if (!lifted_graph_func) {
    return InvalidArgument(
        "MLIR Graph Optimizer failed: module is missing lifted graph func op");
  }

  Status status = FuncToGraph(lifted_graph_func);
  if (!status.ok()) {
    return InvalidArgument(
        "MLIR Graph Optimizer failed: Can't lower function into graph: ",
        status.error_message());
  }
  return Status::OK();
}

// The implementation of the TFG optimizer. It holds the MLIR context and the
// pass manager.
class TFGGrapplerOptimizer::Impl {
 public:
  // Builds the pass pipeline. The context is initialized with threading
  // disabled. If the user specifies to run the optimizer with more than zero
  // threads, a threadpool is initialized and passed to the MLIR context.
  explicit Impl(TFGPassPipelineBuilder builder, unsigned num_tfg_threads)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    DialectRegistry registry;
    // Register the TF op registry interface so that passes can query it.
    registry.addExtension(+[](MLIRContext* ctx, TFGraphDialect* dialect) {
      dialect->addInterfaces<TensorFlowOpRegistryInterface>();
    });
    ctx_.appendDialectRegistry(registry);
    builder(mgr_);
    if (num_tfg_threads) {
      llvm::ThreadPoolStrategy strategy;
      strategy.ThreadsRequested = num_tfg_threads;
      threadpool_ = std::make_unique<llvm::ThreadPool>(strategy);
      ctx_.setThreadPool(*threadpool_);
    }
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
  // An optional threadpool for running MLIR with threading. Use an external
  // threadpool so the number of threads can be controlled.
  std::unique_ptr<llvm::ThreadPool> threadpool_;
  // The MLIR context.
  MLIRContext ctx_;
  // The pass manager containing the loaded TFG pass pipeline.
  PassManager mgr_;
};

TFGGrapplerOptimizer::TFGGrapplerOptimizer(TFGPassPipelineBuilder builder,
                                           unsigned num_tfg_threads)
    : impl_(std::make_unique<Impl>(std::move(builder), num_tfg_threads)) {}

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
      ImportGraphDef(impl_->GetContext(), debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    tensorflow::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    // Import errors are not fatal. Log the error here and return `Aborted` so
    // the meta optimizer knows to swallow the error.
    LOG(ERROR) << name() << " failed: " << status.ToString();
    return tensorflow::errors::Aborted(status.error_message());
  }
  metrics.ReportAndStop();

  ModuleOp module = (*error_or_module).get();

  // Lift the graph to a func to embed the semantic of feeds, fetches and
  // nodes-to-preserve nodes in GrapplerItem.
  TF_RETURN_IF_ERROR(LiftGraphToFunc(*module.getOps<GraphOp>().begin(), item));

  // TODO(chiahungduan): There was a StatusScopedDiagnosticHandler here to
  // collect the diagnostics emitted from the pass pipeline. Given that even a
  // successful pass execution may have error diagnostics emitted in between
  // execution and those logs are not useful for debugging. Besides, there's an
  // issue (b/36186527) which relates to the handler. Remove this to temporary
  // bypass the problem. Find a better way to collect the pipeline failure
  // message here.
  if (failed(impl_->RunPipeline(module))) {
    return InvalidArgument("MLIR Graph Optimizer failed: ");
  }

  // Convert the lifted graph function back to the graph.
  TF_RETURN_IF_ERROR(LowerFuncToGraph(module));

  // Export the TFG module to GraphDef.
  tensorflow::GraphDef graphdef;
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertToGraphDef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  // Ensure that an empty library is instantiated.
  (void)graphdef.mutable_library();
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

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

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

auto* shadow_run_success =
    monitoring::Counter<0>::New("/tensorflow/core/mlir_shadow_run_success",
                                "Success count of MLIR shadow runs");

auto* shadow_run_failure = monitoring::Counter<2>::New(
    "/tensorflow/core/mlir_shadow_run_failure",
    "Failure count of MLIR shadow runs", "kind", "name");

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

// Dumps the MLIR module to disk.
// This require the TF_DUMP_GRAPH_PREFIX to be set to a path that exist (or can
// be created).
static void DumpModule(mlir::ModuleOp module, std::string file_prefix) {
  std::string prefix = GetDumpDirFromEnvVar();
  if (prefix.empty()) return;

  auto* env = tensorflow::Env::Default();
  auto status = env->RecursivelyCreateDir(prefix);
  if (!status.ok()) {
    LOG(WARNING) << "cannot create directory '" + prefix +
                        "': " + status.error_message();
    return;
  }

  prefix += "/" + file_prefix;
  if (!tensorflow::Env::Default()->CreateUniqueFileName(&prefix, ".mlir")) {
    LOG(WARNING) << "cannot create unique filename, won't dump MLIR module.";
    return;
  }

  std::unique_ptr<WritableFile> file_writer;
  status = env->NewWritableFile(prefix, &file_writer);
  if (!status.ok()) {
    LOG(WARNING) << "cannot open file '" + prefix +
                        "': " + status.error_message();
    return;
  }

  // Print the module to a string before writing to the file.
  std::string txt_module;
  {
    llvm::raw_string_ostream os(txt_module);
    module.print(os);
  }

  status = file_writer->Append(txt_module);
  if (!status.ok()) {
    LOG(WARNING) << "error writing to file '" + prefix +
                        "': " + status.error_message();
    return;
  }
  (void)file_writer->Close();
  VLOG(1) << "Dumped MLIR module to " << prefix;
}

MlirOptimizationPassRegistry& MlirOptimizationPassRegistry::Global() {
  static auto* global = new MlirOptimizationPassRegistry();
  return *global;
}

static void RegisterDialects(mlir::DialectRegistry& registry) {
  // clang-format off
  registry.insert<mlir::StandardOpsDialect,
                  mlir::TF::TensorFlowDialect,
                  mlir::shape::ShapeDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect>();
  // clang-format on
}

Status MlirFunctionOptimizationPass::Run(
    const DeviceSet& device_set, const ConfigProto& config_proto,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    std::vector<std::string>* control_ret_node_names,
    bool* control_rets_updated) {
  // Skip conversion from Graph to MLIR if none of the passes are enabled.
  const bool is_enabled =
      llvm::any_of(registry_->passes(), [&](auto& pass_registration) -> bool {
        return pass_registration.pass->IsEnabled(config_proto, **graph);
      });

  if (!is_enabled) {
    LOG_FIRST_N(INFO, 1)
        << "None of the MLIR optimization passes are enabled "
        << "(registered " << registry_->passes().size() << ")";
    return Status::OK();
  }

  LOG_FIRST_N(INFO, 1) << "Running MLIR Graph Optimization Passes "
                          << "(registered " << registry_->passes().size()
                          << " passes)";

  // For scenarios when the new bridge is enabled by analysis we need to make
  // sure that MLIR transformations are executed in a shadow mode.
  // In this case, no changes should be done to the original `graph`
  // and no failures propagated to the user.
  bool enabled_by_analysis =
      mlir_rollout_policy_(**graph, config_proto) ==
      MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis;
  if (enabled_by_analysis) {
    LOG_FIRST_N(INFO, 1) << "Shadow run of MLIR enabled after graph analysis";
  }

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  RegisterDialects(context.getDialectRegistry());
  GraphImportConfig import_config;
  import_config.graph_as_function = true;
  import_config.control_outputs = *control_ret_node_names;
  import_config.upgrade_legacy = true;
  // Disable shape inference during import as some TensorFlow op fails during
  // shape inference with dynamic shaped operands. This in turn causes the
  // import to fail. Shape inference during import is going to be removed and
  // the shape inference pass is run early in the pass pipeline, shape inference
  // during import is not necessary.
  import_config.enable_shape_inference = false;

  auto module_ref_status = ConvertGraphToMlir(**graph, debug_info, *flib_def,
                                              import_config, &context);
  if (!module_ref_status.ok()) {
    if (enabled_by_analysis) {
      shadow_run_failure->GetCell("graph_to_mlir", "")->IncrementBy(1);

      // Do not fail, let the old bridge to run on the original `graph`.
      return Status::OK();
    }

    return module_ref_status.status();
  }

  auto module_ref = std::move(module_ref_status.ValueOrDie());
  AddDevicesToOp(*module_ref, &device_set);

  for (auto& pass_registration : registry_->passes()) {
    llvm::StringRef name = pass_registration.pass->name();
    VLOG(2) << "Run MLIR graph optimization pass: " << StringRefToView(name);

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_before_", name));
    }

    auto pass_status =
        pass_registration.pass->Run(config_proto, *module_ref, **graph);
    if (!pass_status.ok()) {
      if (enabled_by_analysis) {
        shadow_run_failure->GetCell("pass", name.str())->IncrementBy(1);
        // Do not fail, let the old bridge to run on the original `graph`.
        return Status::OK();
      }

      return pass_status;
    }

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_after_", name));
    }
  }

  GraphExportConfig export_config;
  absl::flat_hash_set<Node*> control_ret_nodes;

  // In case MLIR is enabled by analysis, verify that MLIR could be converted
  // back to TF graph. Original `graph` must stay the same.
  if (enabled_by_analysis) {
    auto empty_graph = std::make_unique<Graph>(OpRegistry::Global());
    FunctionLibraryDefinition empty_flib = empty_graph->flib_def();

    auto mlir_to_graph_status =
        ConvertMlirToGraph(*module_ref, export_config, &empty_graph,
                           &empty_flib, &control_ret_nodes);
    if (mlir_to_graph_status.ok()) {
      shadow_run_success->GetCell()->IncrementBy(1);
    } else {
      shadow_run_failure->GetCell("mlir_to_graph", "")->IncrementBy(1);
    }

    return Status::OK();
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(*module_ref, export_config, graph, flib_def,
                         &control_ret_nodes),
      "Error converting MLIR module back to graph");

  control_ret_node_names->clear();
  control_ret_node_names->reserve(control_ret_nodes.size());
  for (const auto* node : control_ret_nodes)
    control_ret_node_names->push_back(node->name());

  *control_rets_updated = true;

  return Status::OK();
}

MlirV1CompatOptimizationPassRegistry&
MlirV1CompatOptimizationPassRegistry::Global() {
  static auto* global = new MlirV1CompatOptimizationPassRegistry();
  return *global;
}

Status MlirV1CompatGraphOptimizationPass::Run(
    const GraphOptimizationPassOptions& options) {
  // Skip function graphs as MlirOptimizationPassRegistry_ will be used instead.
  if (options.is_function_graph) return Status::OK();

  // Skip conversion from Graph to MLIR if none of the passes are enabled.
  const bool is_enabled =
      absl::c_any_of(registry_->passes(), [&](auto& pass_registration) -> bool {
        return pass_registration.pass->IsEnabled(
            options.session_options->config, **options.graph);
      });

  if (!is_enabled) {
    LOG_FIRST_N(INFO, 1)
        << "None of the MLIR optimization passes are enabled "
        << "(registered " << registry_->passes().size() << " passes)";
    return Status::OK();
  }

  LOG_FIRST_N(INFO, 1) << "Running MLIR Graph Optimization V1 Compat Passes "
                          << "(registered " << registry_->passes().size()
                          << " passes)";

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  RegisterDialects(context.getDialectRegistry());
  GraphImportConfig import_config;
  import_config.upgrade_legacy = true;
  // Restrict functionalization to TPU nodes to avoid problems in v1 session
  // runtime.
  import_config.restrict_functionalization_to_tpu_nodes = true;
  TF_ASSIGN_OR_RETURN(
      auto module_ref,
      ConvertGraphToMlir(**options.graph, debug_info, *options.flib_def,
                         import_config, &context));

  AddDevicesToOp(*module_ref, options.device_set);

  for (auto& pass_registration : registry_->passes()) {
    llvm::StringRef name = pass_registration.pass->name();
    VLOG(2) << "Run MLIR graph optimization pass: " << StringRefToView(name);

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_before_", name));
    }

    TF_RETURN_IF_ERROR(pass_registration.pass->Run(options, *module_ref));

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_after_", name));
    }
  }

  GraphExportConfig export_config;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(*module_ref, export_config, options.graph,
                         options.flib_def),
      "Error converting MLIR module back to graph");

  return Status::OK();
}

}  // namespace tensorflow

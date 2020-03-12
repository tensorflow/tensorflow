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

#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

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

Status MlirFunctionOptimizationPass::Run(
    const DeviceSet& device_set, const ConfigProto& config_proto,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    std::vector<std::string>* control_ret_node_names,
    bool* control_rets_updated) {
  // Skip conversion from Graph to MLIR if none of the passes are enabled.
  const bool is_enabled =
      llvm::any_of(registry_->passes(), [&](auto& pass_registration) -> bool {
        return pass_registration.pass->IsEnabled(config_proto);
      });

  if (!is_enabled) {
    VLOG(1) << "None of the MLIR optimization passes are enabled "
            << "(registered " << registry_->passes().size() << ")";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Graph Optimization Passes "
          << "(registered " << registry_->passes().size() << " passes)";

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig import_config;
  import_config.graph_as_function = true;
  import_config.control_outputs = *control_ret_node_names;
  TF_ASSIGN_OR_RETURN(auto module_ref,
                      ConvertGraphToMlir(**graph, debug_info, *flib_def,
                                         import_config, &context));

  AddDevicesToOp(*module_ref, &device_set);

  for (auto& pass_registration : registry_->passes()) {
    llvm::StringRef name = pass_registration.pass->name();
    VLOG(2) << "Run MLIR graph optimization pass: " << StringRefToView(name);

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_before_", name));
    }

    TF_RETURN_IF_ERROR(pass_registration.pass->Run(config_proto, *module_ref));

    if (VLOG_IS_ON(1)) {
      DumpModule(*module_ref, llvm::formatv("mlir_{0}_after_", name));
    }
  }

  GraphExportConfig export_config;
  export_config.graph_as_function = true;
  absl::flat_hash_set<Node*> control_ret_nodes;
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
            options.session_options->config);
      });

  if (!is_enabled) {
    VLOG(1) << "None of the MLIR optimization passes are enabled "
            << "(registered" << registry_->passes().size() << " passes)";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Graph Optimization V1 Compat Passes "
          << "(registered" << registry_->passes().size() << " passes)";

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig import_config;
  import_config.upgrade_legacy = true;
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

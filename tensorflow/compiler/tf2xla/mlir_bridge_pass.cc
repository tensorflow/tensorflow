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

#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {

// Dumps the MLIR module to disk.
// This require the TF_DUMP_GRAPH_PREFIX to be set to a path that exist (or can
// be created).
static void DumpModule(mlir::ModuleOp module, llvm::StringRef file_prefix) {
  const char* prefix_env = getenv("TF_DUMP_GRAPH_PREFIX");
  if (!prefix_env) {
    LOG(WARNING)
        << "Failed to dump MLIR module because dump location is not "
        << " specified through TF_DUMP_GRAPH_PREFIX environment variable.";
    return;
  }
  std::string prefix = prefix_env;

  auto* env = tensorflow::Env::Default();
  auto status = env->RecursivelyCreateDir(prefix);
  if (!status.ok()) {
    LOG(WARNING) << "cannot create directory '" + prefix +
                        "': " + status.error_message();
    return;
  }
  prefix += "/" + file_prefix.str();
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

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const GraphOptimizationPassOptions& options) {
  if (!options.session_options->config.experimental().enable_mlir_bridge()) {
    VLOG(1) << "Skipping MLIR Bridge Pass, session flag not enabled";
    return Status::OK();
  }
  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig specs;
  GraphExportConfig confs;
  TF_ASSIGN_OR_RETURN(auto module,
                      ConvertGraphToMlir(**options.graph, debug_info,
                                         *options.flib_def, specs, &context));

  AddDevicesToOp(*module, options.device_set);

  if (VLOG_IS_ON(1)) DumpModule(*module, "mlir_bridge_before_");

  // Run the bridge now
  TF_RETURN_IF_ERROR(mlir::TFTPU::TPUBridge(*module));

  if (VLOG_IS_ON(1)) DumpModule(*module, "mlir_bridge_after_");

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(*module, confs, options.graph, options.flib_def),
      "Error converting MLIR module back to graph");

  return Status::OK();
}

}  // namespace tensorflow

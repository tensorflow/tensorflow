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

#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/transforms/pass_registration.h"
#include "tensorflow/tools/tfg_graph_transforms/export.h"
#include "tensorflow/tools/tfg_graph_transforms/import.h"

namespace {

llvm::cl::OptionCategory tfg_graph_transform_category(
    "TFG graph transform options");

// NOLINTNEXTLINE
llvm::cl::opt<std::string> input_file(
    llvm::cl::Positional, llvm::cl::desc("<Input model>"),
    llvm::cl::value_desc("Full path to the input model"),
    llvm::cl::cat(tfg_graph_transform_category), llvm::cl::Required);

// NOLINTNEXTLINE
llvm::cl::opt<std::string> output_file(
    "o", llvm::cl::desc("Output model"),
    llvm::cl::value_desc("Full path to the output model"),
    llvm::cl::cat(tfg_graph_transform_category), llvm::cl::Required);

enum class DataFormat { SavedModel = 0, GraphDef = 1 };

// NOLINTNEXTLINE
llvm::cl::opt<DataFormat> data_format(
    "data_format",
    llvm::cl::desc(
        "Data format for both input and output, e.g., SavedModel or GraphDef"),
    values(clEnumValN(DataFormat::SavedModel, "savedmodel",
                      "SavedModel format"),
           clEnumValN(DataFormat::GraphDef, "graphdef", "GraphDef format")),
    llvm::cl::init(DataFormat::SavedModel),
    llvm::cl::cat(tfg_graph_transform_category));

// Validate CL options and returns false in case of an error.
bool CheckCLParams() {
  if (input_file == output_file) {
    LOG(WARNING)
        << "Input and output files are set to the same location. "
           "The resulted model protobuf will overwrite the original one.\n";
  }
  if (!tensorflow::Env::Default()->FileExists(input_file).ok()) {
    LOG(ERROR) << "Provided file or directory does not exist: '" << input_file
               << "'\n";
    return false;
  }

  if (tensorflow::Env::Default()->IsDirectory(input_file).ok()) {
    LOG(ERROR)
        << "Expected full path to the model protobuf file, given directory: '"
        << input_file << "'\n";
    return false;
  }

  return true;
}

void RegisterDialects(mlir::DialectRegistry& registry) {
  // This potentially could be limited, for now keep all TF.
  mlir::RegisterAllTensorFlowDialects(registry);
}

tensorflow::Status RunOptimizationPasses(
    const mlir::PassPipelineCLParser& passPipeline, mlir::ModuleOp module,
    mlir::MLIRContext* context) {
  auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(module.getBody()->front());
  if (!graph_op) {
    return tensorflow::errors::InvalidArgument(
        "TFG MLIR module missing graph op");
  }

  mlir::PassManager pm(context, mlir::tfg::GraphOp::getOperationName());
  auto error_handler = [&](const llvm::Twine& msg) {
    emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
    return mlir::failure();
  };
  if (failed(passPipeline.addToPipeline(pm, error_handler))) {
    return tensorflow::errors::InvalidArgument(
        "Pipeline initialization failed");
  }

  mlir::StatusScopedDiagnosticHandler diagnostics_handler(context);
  if (failed(pm.run(graph_op))) {
    return diagnostics_handler.Combine(
        tensorflow::errors::InvalidArgument("MLIR Pass Manager failure: "));
  }

  return diagnostics_handler.ConsumeStatus();
}

// Import model to the TFG MLIR module.
tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportModel(
    DataFormat data_format, const std::string& input_file,
    mlir::MLIRContext* mlir_context) {
  switch (data_format) {
    case DataFormat::SavedModel:
      return mlir::tfg::graph_transforms::ImportSavedModel(mlir_context,
                                                           input_file);
    case DataFormat::GraphDef:
      return mlir::tfg::graph_transforms::ImportGraphDef(mlir_context,
                                                         input_file);
  }
}

tensorflow::Status ExportTFGModule(mlir::ModuleOp moduleOp,
                                   DataFormat data_format,
                                   const std::string& input_file,
                                   const std::string& output_file) {
  switch (data_format) {
    case DataFormat::SavedModel:
      return mlir::tfg::graph_transforms::ExportTFGToSavedModel(
          moduleOp, input_file, output_file);
    case DataFormat::GraphDef:
      return mlir::tfg::graph_transforms::ExportTFGToGraphDef(moduleOp,
                                                              output_file);
  }
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::tfg::registerTFGraphPasses();

  mlir::PassPipelineCLParser pass_pipeline("", "TFG passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "TFG optimization tool\n");

  if (!CheckCLParams()) {
    LOG(QFATAL) << "Command line parameters are invalid";
  }

  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);

  // Import model to the TFG MLIR module.
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_ref_status =
      ImportModel(data_format, input_file, &context);

  if (!module_ref_status.ok()) {
    LOG(QFATAL) << "Model import failed: "
                << module_ref_status.status().ToString();
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      std::move(module_ref_status.ValueOrDie());

  // Parse the optimization pipeline configuration and run requested graph
  // optimizations.
  tensorflow::Status pass_pipeline_status =
      RunOptimizationPasses(pass_pipeline, *module_ref, &context);
  if (!pass_pipeline_status.ok()) {
    LOG(QFATAL) << pass_pipeline_status.ToString() << "\n";
  }

  // Export MLIR TFG module to the resulting model proto.
  tensorflow::Status export_status =
      ExportTFGModule(*module_ref, data_format, input_file, output_file);

  if (!export_status.ok()) {
    LOG(QFATAL) << "Export of TFG module failed: " << export_status.ToString()
                << "\n";
  }

  return EXIT_SUCCESS;
}

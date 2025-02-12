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

#include <cstdlib>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/ir/importexport/savedmodel_export.h"
#include "tensorflow/core/ir/importexport/savedmodel_import.h"
#include "tensorflow/core/ir/tf_op_registry.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/transforms/pass_registration.h"
#include "tensorflow/tools/tfg_graph_transforms/utils.h"

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

// NOLINTNEXTLINE
llvm::cl::opt<bool> experimental_image_format(
    "experimental_image_format",
    llvm::cl::desc("Whether to expect use the experimental SavedModel image "
                   "format. Only applies to SavedModel inputs and outputs. "
                   "When enabled, the output filename may have a different "
                   "extension than the one provided."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<int> experimental_image_format_max_proto_size(
    "experimental_image_format_max_proto_size",
    llvm::cl::desc(
        "Sets the maximum chunk size in bytes allowed for protos (2GB by "
        "default). This flag should only be used for testing purposes and can "
        "be removed at any time."),
    llvm::cl::init(0));

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

  // Register the TF op registry interface so that passes can query it.
  registry.addExtension(
      +[](mlir::MLIRContext* ctx, mlir::tfg::TFGraphDialect* dialect) {
        dialect->addInterfaces<mlir::tfg::TensorFlowOpRegistryInterface>();
      });
}

absl::Status RunOptimizationPasses(
    const mlir::PassPipelineCLParser& passPipeline, mlir::ModuleOp module,
    mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  mlir::registerPassManagerCLOptions();
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    return tensorflow::errors::InvalidArgument(
        "Could not initialize MLIR pass manager CL options");
  }

  auto error_handler = [&](const llvm::Twine& msg) {
    emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
    return mlir::failure();
  };
  if (failed(passPipeline.addToPipeline(pm, error_handler))) {
    return tensorflow::errors::InvalidArgument(
        "Pipeline initialization failed");
  }

  mlir::StatusScopedDiagnosticHandler diagnostics_handler(context);
  if (failed(pm.run(module))) {
    return diagnostics_handler.Combine(
        tensorflow::errors::InvalidArgument("MLIR Pass Manager failure: "));
  }

  return diagnostics_handler.ConsumeStatus();
}

// Import model to the TFG MLIR module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportModel(
    DataFormat data_format, const std::string& input_file,
    bool experimental_image_format, mlir::MLIRContext* mlir_context) {
  tensorflow::GraphDebugInfo debug_info;

  switch (data_format) {
    case DataFormat::SavedModel: {
      tensorflow::SavedModel saved_model;
      if (experimental_image_format) {
        TF_RETURN_IF_ERROR(
            mlir::tfg::graph_transforms::ReadSavedModelImageFormat(
                input_file, saved_model));
      } else {
        TF_RETURN_IF_ERROR(
            mlir::tfg::graph_transforms::ReadModelProto<tensorflow::SavedModel>(
                input_file, saved_model));
      }
      return mlir::tfg::ImportSavedModelToMlir(mlir_context, debug_info,
                                               saved_model);
    }
    case DataFormat::GraphDef: {
      tensorflow::GraphDef graph_def;
      TF_RETURN_IF_ERROR(
          mlir::tfg::graph_transforms::ReadModelProto<tensorflow::GraphDef>(
              input_file, graph_def));
      return mlir::tfg::ImportGraphDef(mlir_context, debug_info, graph_def);
    }
  }
}

absl::Status ExportTFGModule(mlir::ModuleOp module_op, DataFormat data_format,
                             const std::string& input_file,
                             const std::string& output_file,
                             bool experimental_image_format,
                             int experimental_image_format_max_size) {
  switch (data_format) {
    case DataFormat::SavedModel: {
      tensorflow::SavedModel original_saved_model;
      if (experimental_image_format) {
        TF_RETURN_IF_ERROR(
            mlir::tfg::graph_transforms::ReadSavedModelImageFormat(
                input_file, original_saved_model));
      } else {
        TF_RETURN_IF_ERROR(
            mlir::tfg::graph_transforms::ReadModelProto<tensorflow::SavedModel>(
                input_file, original_saved_model));
      }

      tensorflow::SavedModel final_saved_model;

      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          mlir::tfg::ExportMlirToSavedModel(module_op, original_saved_model,
                                            &final_saved_model),
          "while converting TFG to SavedModel");

      if (experimental_image_format) {
        VLOG(1) << "Serializing resulting SavedModel to " << output_file
                << " (filename might not exactly match since "
                   "`experimental_image_format` has been enabled)";
        return mlir::tfg::graph_transforms::WriteSavedModelImageFormat(
            &final_saved_model, output_file,
            experimental_image_format_max_proto_size);
      } else {
        VLOG(1) << "Serializing resulting SavedModel to " << output_file;
        return mlir::tfg::graph_transforms::SerializeProto<
            tensorflow::SavedModel>(final_saved_model, output_file);
      }
    }
    case DataFormat::GraphDef: {
      tensorflow::GraphDef new_graphdef;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          mlir::tfg::ConvertToGraphDef(module_op, &new_graphdef),
          "while converting TFG to GraphDef");

      VLOG(1) << "Serializing resulting GraphDef to " << output_file;
      return mlir::tfg::graph_transforms::SerializeProto<tensorflow::GraphDef>(
          new_graphdef, output_file);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::tfg::registerTFGraphPasses();
  mlir::registerSymbolPrivatizePass();
  mlir::registerSymbolDCEPass();

  mlir::PassPipelineCLParser pass_pipeline("", "TFG passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "TFG optimization tool\n");

  if (!CheckCLParams()) {
    LOG(QFATAL) << "Command line parameters are invalid";
  }

  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);

  // Import model to the TFG MLIR module.
  auto module_ref_status =
      ImportModel(data_format, input_file, experimental_image_format, &context);

  if (!module_ref_status.ok()) {
    LOG(QFATAL) << "Model import failed: " << module_ref_status.status();
  }
  auto module_ref = std::move(module_ref_status.value());

  // Parse the optimization pipeline configuration and run requested graph
  // optimizations.
  absl::Status pass_pipeline_status =
      RunOptimizationPasses(pass_pipeline, *module_ref, &context);
  if (!pass_pipeline_status.ok()) {
    LOG(QFATAL) << pass_pipeline_status << "\n";
  }

  // Export MLIR TFG module to the resulting model proto.
  absl::Status export_status = ExportTFGModule(
      *module_ref, data_format, input_file, output_file,
      experimental_image_format, experimental_image_format_max_proto_size);

  if (!export_status.ok()) {
    LOG(QFATAL) << "Export of TFG module failed: " << export_status << "\n";
  }

  return EXIT_SUCCESS;
}

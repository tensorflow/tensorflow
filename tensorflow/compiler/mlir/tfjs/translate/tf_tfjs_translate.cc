
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

#include <iostream>
#include <string>

#include "absl/strings/str_split.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tfjs/tf_tfjs_passes.h"
#include "tensorflow/compiler/mlir/tfjs/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

using llvm::cl::opt;
using mlir::MLIRContext;
using stream_executor::port::StatusOr;

// NOLINTNEXTLINE
opt<std::string> input_file_name(llvm::cl::Positional,
                                 llvm::cl::desc("<input file>"),
                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_object_graph(
    "savedmodel-objectgraph-to-mlir",
    llvm::cl::desc("Import a saved model to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_signature_defs(
    "savedmodel-signaturedefs-to-mlir",
    llvm::cl::desc("Import a saved model V1 to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_file_name("o", llvm::cl::desc("<output file>"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));
// NOLINTNEXTLINE
opt<bool> input_mlir(
    "input-mlir",
    llvm::cl::desc("Take input TensorFlow model in textual MLIR instead of "
                   "GraphDef format"),
    llvm::cl::init(false), llvm::cl::Hidden);
// NOLINTNEXTLINE
opt<bool> output_mlir(
    "output-mlir",
    llvm::cl::desc("Output MLIR rather than JSON for the generated TFJS model"),
    llvm::cl::init(false));

// The following approach allows injecting opdefs in addition
// to those that are already part of the global TF registry  to be linked in
// prior to importing the graph. The primary goal is for support of custom ops.
// This is not intended to be a general solution for custom ops for the future
// but mainly for supporting older models like mobilenet_ssd. More appropriate
// mechanisms, such as op hints or using functions to represent composable ops
// like https://github.com/tensorflow/community/pull/113 should be encouraged
// going forward.
// NOLINTNEXTLINE
llvm::cl::list<std::string> custom_opdefs(
    "tf-custom-opdefs", llvm::cl::desc("List of custom opdefs when importing "
                                       "graphdef"));

// Debugging flag to print function mapping in the JSON.
// NOLINTNEXTLINE
static opt<bool> print_function_result_mapping(
    "print-function-result-mapping",
    llvm::cl::desc(
        "Print the mapping of function result to json output buffer"),
    llvm::cl::init(false));

enum TranslationStatus { kTrSuccess, kTrFailure };

static int PrintFunctionResultMapping(const std::string& result) {
  std::cout << result << std::endl;
  return kTrSuccess;
}

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF GraphDef to TFJS JSON converter\n");

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module;

  if (import_saved_model_object_graph || import_saved_model_signature_defs) {
    if (input_mlir)
      module = tensorflow::errors::InvalidArgument(
          "Importing saved model should not have input_mlir set");
    module = tensorflow::ImportSavedModel(
        import_saved_model_object_graph, import_saved_model_signature_defs,
        custom_opdefs, input_file_name, saved_model_tags,
        saved_model_exported_names, &context);
  } else {
    module = tensorflow::LoadFromGraphdefOrMlirSource(
        input_file_name, input_mlir, custom_opdefs, debug_info_file,
        input_arrays, input_dtypes, input_shapes, output_arrays,
        /*prune_unused_nodes=*/true, &source_mgr, &context);
  }

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  mlir::PassManager pm(&context);

  tensorflow::AddTFToTFJSConversionPasses(&pm);

  std::string result;
  auto status = tensorflow::ConvertTFOpsToTfjsJSON(module.ValueOrDie().get(),
                                                   output_mlir, &result, &pm);
  if (!status.ok()) return kTrFailure;

  std::string error_msg;
  auto output = mlir::openOutputFile(output_file_name, &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return kTrFailure;
  }
  output->os() << result;
  output->keep();

  // Print out debugging info related to function mapping.
  if (print_function_result_mapping) return PrintFunctionResultMapping(result);
  return kTrSuccess;
}

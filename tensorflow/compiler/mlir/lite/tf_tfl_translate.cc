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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using mlir::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using stream_executor::port::StatusOr;
using tensorflow::Status;

// NOLINTNEXTLINE
static llvm::cl::opt<bool> print_function_result_mapping(
    "print-function-result-mapping",
    llvm::cl::desc(
        "Print the mapping of function result to flatbuffer output buffer"),
    llvm::cl::init(false));

enum TranslationStatus { kTrSuccess, kTrFailure };

static int PrintFunctionResultMapping(const std::string &result,
                                      ModuleOp module) {
  // Build model from the resultant string to extract the return values from
  // their source of truth.
  auto model =
      tflite::FlatBufferModel::BuildFromBuffer(result.data(), result.size());
  if (!model) return kTrFailure;

  // Get an unknown location for where we don't have a terminator to get the
  // location of the return value from.
  auto unknown_loc = mlir::UnknownLoc::get(module.getContext());

  auto print_buffer = [&](const tflite::SubGraph &subgraph, int id, int buffer,
                          std::function<mlir::Location(int)> loc) {
    const auto &output_tensor = (*subgraph.tensors())[buffer];
    std::cout << "\tname: '"
              << (output_tensor->name() ? output_tensor->name()->str()
                                        : "<<unnamed>>")
              << "' buffer: " << buffer;
    if (loc) std::cout << llvm::formatv(" {0}", loc(id)).str();
    std::cout << '\n';
  };

  // For every subgraph print out the name (if available), each result's output
  // buffer number and location of the return value (if available).
  for (auto *subgraph : *(*model)->subgraphs()) {
    std::string subgraph_name =
        subgraph->name() ? subgraph->name()->str() : "<<unnamed subgraph>>";

    std::cout << '\'' << subgraph_name << "' inputs:\n";
    int i = 0;
    for (auto input : *subgraph->inputs())
      print_buffer(*subgraph, i++, input, nullptr);

    std::cout << '\'' << subgraph_name << "' outputs:\n";
    mlir::Operation *terminator = nullptr;
    if (subgraph->name()) {
      if (auto fn = module.lookupSymbol<FuncOp>(subgraph->name()->str()))
        terminator = fn.back().getTerminator();
    }
    i = 0;
    for (auto output : *subgraph->outputs()) {
      print_buffer(*subgraph, i, output, [&](int i) {
        return terminator ? terminator->getOperand(i)->getLoc() : unknown_loc;
      });
    }
  }
  return kTrSuccess;
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  // TODO(jpienaar): Revise the command line option parsing here.
  llvm::InitLLVM y(argc, argv);

  // TODO(antiagainst): We are pulling in multiple transformations as follows.
  // Each transformation has its own set of command-line options; options of one
  // transformation can essentially be aliases to another. For example, the
  // -tfl-annotate-inputs has -tfl-input-arrays, -tfl-input-data-types, and
  // -tfl-input-shapes, which are the same as -graphdef-to-mlir transformation's
  // -tf_input_arrays, -tf_input_data_types, and -tf_input_shapes, respectively.
  // We need to disable duplicated ones to provide a cleaner command-line option
  // interface. That also means we need to relay the value set in one option to
  // all its aliases.

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TF GraphDef to TFLite FlatBuffer converter\n");

  // TODO(ashwinm): Enable command line parsing for both sides.
  int fake_argc = 1;
  tensorflow::port::InitMain(argv[0], &fake_argc, &argv);

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningModuleRef> module =
      tensorflow::LoadFromGraphdefOrMlirSource(
          input_file_name, input_mlir, use_splatted_constant, extra_opdefs,
          debug_info_file, input_arrays, input_dtypes, input_shapes,
          output_arrays, inference_type, min_values, max_values,
          /*prune_unused_nodes=*/true, &source_mgr, &context);

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  std::string result;
  auto status = tensorflow::ConvertTFExecutorToTFLOrFlatbuffer(
      module.ValueOrDie().get(), output_mlir, emit_builtin_tflite_ops,
      emit_select_tf_ops, emit_custom_ops, emit_quant_adaptor_ops,
      lower_tensor_list_ops, &result);
  if (!status.ok()) return kTrFailure;

  auto output = mlir::openOutputFile(output_file_name);
  output->os() << result;
  output->keep();

  // Print out debugging info related to function mapping.
  if (print_function_result_mapping)
    return PrintFunctionResultMapping(result, module.ValueOrDie().get());
  return kTrSuccess;
}

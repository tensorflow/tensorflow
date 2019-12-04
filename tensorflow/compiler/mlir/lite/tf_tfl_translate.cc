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

#include "absl/strings/str_split.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate_flags.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using mlir::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using stream_executor::port::StatusOr;

// Debugging flag to print function mapping in the flatbuffer.
// NOLINTNEXTLINE
static llvm::cl::opt<bool> print_function_result_mapping(
    "print-function-result-mapping",
    llvm::cl::desc(
        "Print the mapping of function result to flatbuffer output buffer"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> weight_quantization(
    "weight_quantization",
    llvm::cl::desc("The type of the quantized weight buffer. Must be NONE, "
                   "INT8, FLOAT16."),
    llvm::cl::init("NONE"));

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
  // TODO(jpienaar): Revise the command line option parsing here.
  tensorflow::InitMlir y(&argc, &argv);

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

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningModuleRef> module =
      tensorflow::LoadFromGraphdefOrMlirSource(
          input_file_name, input_mlir, use_splatted_constant, custom_opdefs,
          debug_info_file, input_arrays, input_dtypes, input_shapes,
          output_arrays,
          /*prune_unused_nodes=*/true, &source_mgr, &context);

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  mlir::PassManager pm(&context);

  // Set the quantization specifications from the command line flags.
  mlir::TFL::QuantizationSpecs quant_specs;
  if (mlir::TFL::ParseInputNodeQuantSpecs(input_arrays, min_values, max_values,
                                          inference_type, &quant_specs)) {
    llvm::errs() << "Failed to get input quant spec.";
    return kTrFailure;
  }
  if (weight_quantization != "NONE") {
    quant_specs.weight_quantization = true;
    if (weight_quantization == "INT8") {
      quant_specs.inference_type = tensorflow::DT_QINT8;
    } else if (weight_quantization == "FLOAT16") {
      quant_specs.inference_type = tensorflow::DT_HALF;
    } else {
      llvm::errs() << "Unknown weight quantization " << weight_quantization;
      return kTrFailure;
    }
  }
  if (!emit_quant_adaptor_ops) {
    quant_specs.inference_input_type = quant_specs.inference_type;
  }

  if (!quant_stats_file_name.empty()) {
    std::string error_message;
    auto file = mlir::openInputFile(quant_stats_file_name, &error_message);
    if (!file) {
      llvm::errs() << "fail to open quant stats file: "
                   << quant_stats_file_name;
      return kTrFailure;
    }
    quant_specs.serialized_quant_stats = file->getBuffer().str();
  }

  mlir::TFL::PassConfig pass_config(quant_specs);
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.lower_tensor_list_ops = lower_tensor_list_ops;
  pass_config.inline_functions = inline_functions;

  tensorflow::AddTFToTFLConversionPasses(pass_config, &pm);

  std::string result;
  auto status = tensorflow::ConvertTFExecutorToTFLOrFlatbuffer(
      module.ValueOrDie().get(), output_mlir, emit_builtin_tflite_ops,
      emit_select_tf_ops, emit_custom_ops, quant_specs, &result, &pm);
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
  if (print_function_result_mapping)
    return PrintFunctionResultMapping(result, module.ValueOrDie().get());
  return kTrSuccess;
}

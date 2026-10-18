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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"

namespace tensorflow {
namespace {
static mlir::OwningOpRef<mlir::ModuleOp> FlatBufferFileToMlirTranslation(
    llvm::SourceMgr* source_mgr, mlir::MLIRContext* context) {
  const llvm::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc =
      mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, false, inputs, outputs, false);
}

}  // namespace

std::string FlatBufferFileToMlir(const std::string& model_file_or_buffer,
                                 bool input_is_filepath, bool bytecode,
                                 const std::vector<std::string>& cl_options) {
  ABSL_CONST_INIT static absl::Mutex cl_mutex(absl::kConstInit);
  absl::MutexLock lock(&cl_mutex);

  // Reset options from any previous invocation.
  llvm::cl::ResetAllOptionOccurrences();
  mlir::registerAsmPrinterCLOptions();

  if (!cl_options.empty()) {
    std::vector<const char*> argv;
    argv.reserve(cl_options.size() + 1);
    argv.push_back("flatbuffer_to_mlir");
    for (const auto& opt : cl_options) {
      argv.push_back(opt.c_str());
    }
    std::string cl_errors;
    llvm::raw_string_ostream cl_err_stream(cl_errors);
    if (!llvm::cl::ParseCommandLineOptions(
            argv.size(), argv.data(), "flatbuffer_to_mlir", &cl_err_stream)) {
      cl_err_stream.flush();
      if (!cl_errors.empty()) {
        llvm::errs() << "Failed to parse MLIR options: " << cl_errors << "\n";
      }
      return "";
    }
  }

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input;
  if (input_is_filepath) {
    input = mlir::openInputFile(model_file_or_buffer, &errorMessage);
    if (!input) {
      llvm::errs() << errorMessage << "\n";
      return "";
    }
  } else {
    input = llvm::MemoryBuffer::getMemBuffer(model_file_or_buffer, "flatbuffer",
                                             false);
    if (!input) {
      llvm::errs() << "Can't get llvm::MemoryBuffer\n";
      return "";
    }
  }

  mlir::MLIRContext context;
  context.printOpOnDiagnostic(true);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

  std::string diagnostic_str;
  llvm::raw_string_ostream diag_os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler diag_handler(sourceMgr, &context, diag_os);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      FlatBufferFileToMlirTranslation(&sourceMgr, &context);
  if (!module || failed(verify(*module))) {
    diag_os.flush();
    if (!diagnostic_str.empty()) {
      llvm::errs() << diagnostic_str << "\n";
    }
    return "";
  }

  std::string mlir_output;
  llvm::raw_string_ostream output_stream(mlir_output);
  if (bytecode) {
    if (mlir::failed(mlir::writeBytecodeToFile(*module, output_stream))) {
      llvm::errs() << "Failed to write MLIR bytecode.\n";
      return "";
    }
  } else {
    mlir::OpPrintingFlags flags;
    module->print(output_stream, flags);
  }
  output_stream.flush();
  return mlir_output;
}

std::string MlirToFlatBufferFile(const std::string& mlir_file_or_buffer,
                                 bool input_is_filepath,
                                 bool emit_builtin_tflite_ops,
                                 bool emit_select_tf_ops, bool emit_custom_ops,
                                 bool emit_stablehlo_ops) {
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input;
  if (input_is_filepath) {
    input = mlir::openInputFile(mlir_file_or_buffer, &errorMessage);
    if (!input) {
      llvm::errs() << errorMessage << "\n";
      return "";
    }
  } else {
    input =
        llvm::MemoryBuffer::getMemBuffer(mlir_file_or_buffer, "mlir", false);
    if (!input) {
      llvm::errs() << "Can't get llvm::MemoryBuffer\n";
      return "";
    }
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect,
                  mlir::TFL::TensorFlowLiteDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::vhlo::VhloDialect,
                  mlir::stablehlo::StablehloDialect>();
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);
  context.printOpOnDiagnostic(true);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

  std::string diagnostic_str;
  llvm::raw_string_ostream diag_os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler diag_handler(sourceMgr, &context, diag_os);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module || failed(verify(*module))) {
    diag_os.flush();
    if (!diagnostic_str.empty()) {
      llvm::errs() << diagnostic_str << "\n";
    } else {
      llvm::errs() << "Failed to parse MLIR source.\n";
    }
    return "";
  }

  std::string serialized_flatbuffer;
  tensorflow::OpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.converter_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.converter_flags.set_allow_custom_ops(emit_custom_ops);
  options.converter_flags.set_use_buffer_offset(true);
  options.op_or_arg_name_mapper = &op_or_arg_name_mapper;

  if (!tflite::MlirToFlatBufferTranslateFunction(
          *module, options, &serialized_flatbuffer, emit_stablehlo_ops)) {
    llvm::errs() << "MlirToFlatBufferTranslateFunction failed.\n";
    return "";
  }
  return serialized_flatbuffer;
}

}  // namespace tensorflow

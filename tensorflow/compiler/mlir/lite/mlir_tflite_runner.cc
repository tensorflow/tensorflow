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

// Tool to run a TFLite computation from a MLIR input using the TFLite
// interpreter.

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export_flags.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> input_filename(llvm::cl::Positional,
                                       desc("<input file>"), init("-"));

// NOLINTNEXTLINE
static opt<bool> dump_state("dump-interpreter-state",
                            desc("dump interpreter state post execution"),
                            init(false));

// TODO(jpienaar): Move these functions to some debug utils.
static std::string TfLiteTensorDimString(const TfLiteTensor& tensor) {
  auto begin = tensor.dims ? tensor.dims->data : nullptr;
  auto end = tensor.dims ? tensor.dims->data + tensor.dims->size : nullptr;
  return absl::StrJoin(begin, end, ", ");
}

template <typename T>
static std::string TfLiteTypedTensorString(const TfLiteTensor& tensor) {
  const T* data = reinterpret_cast<T*>(tensor.data.raw);
  if (!data) return "<null>";
  int count = tensor.bytes / sizeof(T);
  return absl::StrJoin(data, data + count, ", ");
}

// TODO(jpienaar): This really feels like something that should exist already.
static std::string TfLiteTensorString(const TfLiteTensor& tensor) {
  switch (tensor.type) {
    case kTfLiteInt32:
      return TfLiteTypedTensorString<int32_t>(tensor);
    case kTfLiteUInt32:
      return TfLiteTypedTensorString<uint32_t>(tensor);
    case kTfLiteInt64:
      return TfLiteTypedTensorString<int64_t>(tensor);
    case kTfLiteFloat32:
      return TfLiteTypedTensorString<float>(tensor);
    default:
      LOG(QFATAL) << "Unsupported type: " << TfLiteTypeGetName(tensor.type);
  }
}

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TFLite runner\n");

  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename.c_str());
  if (std::error_code error = file_or_err.getError()) {
    LOG(ERROR) << argv[0] << ": could not open input file '" << input_filename
               << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::TF::TensorFlowDialect, mlir::TFL::TensorFlowLiteDialect,
                  mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context));
  if (!module) return 1;

  // TODO(jpienaar): Expand to support inputs.
  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  QCHECK(main) << "No 'main' function specified.";
  if (main.getFunctionType().getNumInputs() != 0)
    LOG(QFATAL) << "NYI: Only nullary functions supported.";

  // Convert to flatbuffer.
  std::string serialized_flatbuffer;
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.toco_flags.set_allow_custom_ops(emit_custom_ops);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &serialized_flatbuffer))
    return 1;

  // Create TFLite interpreter & invoke converted program.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(serialized_flatbuffer.c_str(),
                                               serialized_flatbuffer.size());
  tflite::ops::builtin::BuiltinOpResolver builtins;
  std::unique_ptr<tflite::Interpreter> interpreter;
  QCHECK(tflite::InterpreterBuilder(*model, builtins)(&interpreter) ==
         kTfLiteOk);
  QCHECK(interpreter->AllocateTensors() == kTfLiteOk);
  QCHECK(interpreter->Invoke() == kTfLiteOk);

  // Print the resulting outputs.
  // TODO(jpienaar): Allow specifying output stream/file.
  QCHECK(interpreter->outputs().size() ==
         main.getFunctionType().getNumResults());
  for (int index : interpreter->outputs()) {
    const auto& out = *interpreter->tensor(index);
    // Print name if named.
    if (out.name) fprintf(stdout, "%s: ", out.name);
    // Print tensor result.
    fprintf(stdout, "Tensor<type: %s, shape: %s, values: %s>\n",
            TfLiteTypeGetName(out.type), TfLiteTensorDimString(out).c_str(),
            TfLiteTensorString(out).c_str());
  }

  if (dump_state) tflite::PrintInterpreterState(interpreter.get());

  return 0;
}

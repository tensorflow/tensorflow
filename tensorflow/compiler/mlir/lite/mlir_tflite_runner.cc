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
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate_flags.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using llvm::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

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

  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    LOG(ERROR) << argv[0] << ": could not open input file '" << inputFileName
               << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::MLIRContext context;
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  if (!module) return 1;

  // TODO(jpienaar): Expand to support inputs.
  mlir::FuncOp main = module->lookupSymbol<mlir::FuncOp>("main");
  QCHECK(main) << "No 'main' function specified.";
  if (main.getType().getNumInputs() != 0)
    LOG(QFATAL) << "NYI: Only nullary functions supported.";

  // Convert to flatbuffer.
  std::string serialized_flatbuffer;
  if (tflite::MlirToFlatBufferTranslateFunction(
          module.get(), &serialized_flatbuffer, emit_builtin_tflite_ops,
          emit_select_tf_ops, emit_custom_ops))
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
  QCHECK(interpreter->outputs().size() == main.getType().getNumResults());
  for (int index : interpreter->outputs()) {
    const auto& out = *interpreter->tensor(index);
    // Print name if named.
    if (out.name) fprintf(stdout, "%s: ", out.name);
    // Print tensor result.
    fprintf(stdout, "Tensor<type: %s, shape: %s, values: %s>\n",
            TfLiteTypeGetName(out.type), TfLiteTensorDimString(out).c_str(),
            TfLiteTensorString(out).c_str());
  }

  return 0;
}

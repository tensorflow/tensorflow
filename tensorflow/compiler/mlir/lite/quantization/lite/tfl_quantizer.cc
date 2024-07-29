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

#include <iostream>

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

using llvm::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

namespace mlir {
namespace {

absl::Status QuantizeAnnotatedModel(llvm::StringRef buffer,
                                    std::string& output_buffer) {
  return mlir::lite::QuantizeModel(
      buffer, tflite::TensorType_INT8, tflite::TensorType_INT8,
      tflite::TensorType_INT8, {}, /*disable_per_channel=*/false,
      /*fully_quantize=*/true, output_buffer);
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }
  auto buffer = file_or_err->get();
  std::string output_buffer;
  if (auto status = mlir::QuantizeAnnotatedModel(buffer->getBuffer().str(),
                                                 output_buffer);
      !status.ok()) {
    llvm::errs() << status.message() << "\n";
    return 1;
  }

  std::cout << output_buffer << "\n";
  return 0;
}

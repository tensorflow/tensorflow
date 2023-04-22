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
#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

using llvm::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

namespace mlir {
namespace {
TfLiteStatus QuantizeAnnotatedModel(llvm::StringRef buffer,
                                    flatbuffers::FlatBufferBuilder* builder) {
  auto model_ptr = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer.data(), buffer.size());
  if (nullptr == model_ptr) {
    return TfLiteStatus::kTfLiteError;
  }
  std::unique_ptr<tflite::ModelT> model(model_ptr->GetModel()->UnPack());

  tflite::StderrReporter error_reporter;
  return mlir::lite::QuantizeModel(
      *model, tflite::TensorType_INT8, tflite::TensorType_INT8,
      tflite::TensorType_INT8, {},
      /*disable_per_channel=*/false,
      /*fully_quantize=*/true, builder, &error_reporter);
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
  flatbuffers::FlatBufferBuilder builder;
  auto status =
      mlir::QuantizeAnnotatedModel(buffer->getBuffer().str(), &builder);
  if (status != kTfLiteOk) {
    return 1;
  }

  std::cout << std::string(
                   reinterpret_cast<const char*>(builder.GetBufferPointer()),
                   builder.GetSize())
            << "\n";
  return 0;
}

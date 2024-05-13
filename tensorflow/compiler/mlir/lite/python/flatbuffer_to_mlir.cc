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

#include "absl/strings/string_view.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

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
                                 bool input_is_filepath) {
  // referred logic from mlir::mlirTranslateMain().

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

  mlir::OwningOpRef<mlir::ModuleOp> module =
      FlatBufferFileToMlirTranslation(&sourceMgr, &context);
  if (!module || failed(verify(*module))) return "";

  std::string mlir_output;
  llvm::raw_string_ostream output_stream(mlir_output);
  // Dump MLIR with eliding large elements.
  module->print(
      output_stream,
      mlir::OpPrintingFlags().useLocalScope().elideLargeElementsAttrs());
  return mlir_output;
}

}  // namespace tensorflow

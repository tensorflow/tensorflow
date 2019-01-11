//===- FileUtilities.cpp - utilities for working with files -----*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Definitions of common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

std::unique_ptr<llvm::MemoryBuffer>
mlir::openInputFile(StringRef inputFilename, std::string *errorMessage) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = fileOrErr.getError()) {
    if (errorMessage)
      *errorMessage = "cannot open input file '" + inputFilename.str() +
                      "': " + error.message();
    return nullptr;
  }

  return std::move(*fileOrErr);
}

std::unique_ptr<llvm::ToolOutputFile>
mlir::openOutputFile(StringRef outputFilename, std::string *errorMessage) {
  std::error_code error;
  auto result = llvm::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                        llvm::sys::fs::F_None);
  if (error) {
    if (errorMessage)
      *errorMessage = "cannot open output file '" + outputFilename.str() +
                      "': " + error.message();
    return nullptr;
  }

  return result;
}

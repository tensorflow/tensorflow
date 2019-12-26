//===- FileUtilities.cpp - utilities for working with files ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
  auto result = std::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                       llvm::sys::fs::F_None);
  if (error) {
    if (errorMessage)
      *errorMessage = "cannot open output file '" + outputFilename.str() +
                      "': " + error.message();
    return nullptr;
  }

  return result;
}

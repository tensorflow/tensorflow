//===- FileUtilities.h - utilities for working with files -------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_FILEUTILITIES_H_
#define MLIR_SUPPORT_FILEUTILITIES_H_

#include <memory>
#include <string>

namespace llvm {
class MemoryBuffer;
class ToolOutputFile;
class StringRef;
} // namespace llvm

namespace mlir {

/// Open the file specified by its name for reading. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<llvm::MemoryBuffer>
openInputFile(llvm::StringRef inputFilename,
              std::string *errorMessage = nullptr);

/// Open the file specified by its name for writing. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<llvm::ToolOutputFile>
openOutputFile(llvm::StringRef outputFilename,
               std::string *errorMessage = nullptr);

} // namespace mlir

#endif // MLIR_SUPPORT_FILEUTILITIES_H_

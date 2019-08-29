//===- FileUtilities.h - utilities for working with files -------*- C++ -*-===//
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

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

#include "mlir/Support/LLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>

namespace mlir {

/// Open the file specified by its name for writing.
std::unique_ptr<llvm::ToolOutputFile> openOutputFile(StringRef outputFilename);

} // namespace mlir

#endif // MLIR_SUPPORT_FILEUTILITIES_H_

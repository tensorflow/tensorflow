//===- ToolUtilities.h - MLIR Tool Utilities --------------------*- C++ -*-===//
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
// This file declares common utilities for implementing MLIR tools.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TOOLUTILITIES_H
#define MLIR_SUPPORT_TOOLUTILITIES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace llvm {
class MemoryBuffer;
}

namespace mlir {
struct LogicalResult;

using ChunkBufferHandler = function_ref<LogicalResult(
    std::unique_ptr<llvm::MemoryBuffer> chunkBuffer, raw_ostream &os)>;

/// Splits the specified buffer on a marker (`// -----`), processes each chunk
/// independently according to the normal `processChunkBuffer` logic, and writes
/// all results to `os`.
///
/// This is used to allow a large number of small independent tests to be put
/// into a single file.
LogicalResult
splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> originalBuffer,
                      ChunkBufferHandler processChunkBuffer, raw_ostream &os);
} // namespace mlir

#endif // MLIR_SUPPORT_TOOLUTILITIES_H

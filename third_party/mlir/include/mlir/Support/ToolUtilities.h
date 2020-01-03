//===- ToolUtilities.h - MLIR Tool Utilities --------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

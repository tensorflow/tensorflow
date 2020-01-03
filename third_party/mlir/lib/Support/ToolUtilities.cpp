//===- ToolUtilities.cpp - MLIR Tool Utilities ----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for implementing MLIR tools.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/ToolUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

LogicalResult
mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> originalBuffer,
                            ChunkBufferHandler processChunkBuffer,
                            raw_ostream &os) {
  const char splitMarker[] = "// -----";

  auto *origMemBuffer = originalBuffer.get();
  SmallVector<StringRef, 8> sourceBuffers;
  origMemBuffer->getBuffer().split(sourceBuffers, splitMarker);

  // Add the original buffer to the source manager.
  llvm::SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(originalBuffer), llvm::SMLoc());

  // Process each chunk in turn.
  bool hadFailure = false;
  for (auto &subBuffer : sourceBuffers) {
    auto splitLoc = llvm::SMLoc::getFromPointer(subBuffer.data());
    unsigned splitLine = fileSourceMgr.getLineAndColumn(splitLoc).first;
    auto subMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(
        subBuffer, origMemBuffer->getBufferIdentifier() +
                       Twine(" split at line #") + Twine(splitLine));
    if (failed(processChunkBuffer(std::move(subMemBuffer), os)))
      hadFailure = true;
  }

  // If any fails, then return a failure of the tool.
  return failure(hadFailure);
}

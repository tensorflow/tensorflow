//===- ToolUtilities.cpp - MLIR Tool Utilities ----------------------------===//
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

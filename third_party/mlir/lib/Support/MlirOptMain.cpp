//===- MlirOptMain.cpp - MLIR Optimizer Driver ----------------------------===//
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
// This is a utility that runs an optimization pass and prints the result back
// out. It is designed to support unit testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/MlirOptMain.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace llvm;
using llvm::SMLoc;

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult
performActions(raw_ostream &os, bool verifyDiagnostics, bool verifyPasses,
               SourceMgr &sourceMgr, MLIRContext *context,
               const std::vector<const mlir::PassRegistryEntry *> &passList) {
  OwningModuleRef module(parseSourceFile(sourceMgr, context));
  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(context, verifyPasses);
  applyPassManagerCLOptions(pm);

  // Run each of the passes that were selected.
  for (const auto *passEntry : passList)
    passEntry->addToPipeline(pm);

  // Run the pipeline.
  if (failed(pm.run(*module)))
    return failure();

  // Print the output.
  module->print(os);
  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<MemoryBuffer> ownedBuffer,
              bool verifyDiagnostics, bool verifyPasses,
              const std::vector<const mlir::PassRegistryEntry *> &passList) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Parse the input file.
  MLIRContext context;

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return performActions(os, verifyDiagnostics, verifyPasses, sourceMgr,
                          &context, passList);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  performActions(os, verifyDiagnostics, verifyPasses, sourceMgr, &context,
                 passList);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

/// Split the specified file on a marker and process each chunk independently
/// according to the normal processBuffer logic.  This is primarily used to
/// allow a large number of small independent parser tests to be put into a
/// single test, but could be used for other purposes as well.
static LogicalResult splitAndProcessFile(
    raw_ostream &os, std::unique_ptr<MemoryBuffer> originalBuffer,
    bool verifyDiagnostics, bool verifyPasses,
    const std::vector<const mlir::PassRegistryEntry *> &passList) {
  const char marker[] = "// -----";
  auto *origMemBuffer = originalBuffer.get();
  SmallVector<StringRef, 8> sourceBuffers;
  origMemBuffer->getBuffer().split(sourceBuffers, marker);

  // Add the original buffer to the source manager.
  SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(originalBuffer), SMLoc());

  bool hadUnexpectedResult = false;

  // Process each chunk in turn.  If any fails, then return a failure of the
  // tool.
  for (auto &subBuffer : sourceBuffers) {
    auto splitLoc = SMLoc::getFromPointer(subBuffer.data());
    unsigned splitLine = fileSourceMgr.getLineAndColumn(splitLoc).first;
    auto subMemBuffer = MemoryBuffer::getMemBufferCopy(
        subBuffer, origMemBuffer->getBufferIdentifier() +
                       Twine(" split at line #") + Twine(splitLine));
    if (failed(processBuffer(os, std::move(subMemBuffer), verifyDiagnostics,
                             verifyPasses, passList)))
      hadUnexpectedResult = true;
  }

  return failure(hadUnexpectedResult);
}

LogicalResult
mlir::MlirOptMain(raw_ostream &os, std::unique_ptr<MemoryBuffer> buffer,
                  const std::vector<const mlir::PassRegistryEntry *> &passList,
                  bool splitInputFile, bool verifyDiagnostics,
                  bool verifyPasses) {
  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessFile(os, std::move(buffer), verifyDiagnostics,
                               verifyPasses, passList);

  return processBuffer(os, std::move(buffer), verifyDiagnostics, verifyPasses,
                       passList);
}

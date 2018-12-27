//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
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
// This is a command line utility that parses an MLIR file, runs an optimization
// pass, then prints the result back out.  It is designed to support unit
// testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass.h"
#include "mlir/Support/PassNameParser.h"
#include "mlir/TensorFlow/ControlFlowOps.h"
#include "mlir/TensorFlow/Passes.h"
#include "mlir/TensorFlowLite/Passes.h"
#include "mlir/Transforms/CFGFunctionViewGraph.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/XLA/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;
using llvm::SMLoc;

static cl::opt<std::string>
inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
               cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static std::vector<const mlir::PassInfo *> *passList;

enum OptResult { OptSuccess, OptFailure };

/// Open the specified output file and return it, exiting if there is any I/O or
/// other errors.
static std::unique_ptr<ToolOutputFile> getOutputStream() {
  std::error_code error;
  auto result =
      llvm::make_unique<ToolOutputFile>(outputFilename, error, sys::fs::F_None);
  if (error) {
    llvm::errs() << error.message() << '\n';
    exit(1);
  }

  return result;
}

/// Given a MemoryBuffer along with a line and column within it, return the
/// location being referenced.
static SMLoc getLocFromLineAndCol(MemoryBuffer &membuf, unsigned lineNo,
                                  unsigned columnNo) {
  // TODO: This should really be upstreamed to be a method on llvm::SourceMgr.
  // Doing so would allow it to use the offset cache that is already maintained
  // by SrcBuffer, making this more efficient.

  // Scan for the correct line number.
  const char *position = membuf.getBufferStart();
  const char *end = membuf.getBufferEnd();

  // We start counting line and column numbers from 1.
  --lineNo;
  --columnNo;

  while (position < end && lineNo) {
    auto curChar = *position++;

    // Scan for newlines.  If this isn't one, ignore it.
    if (curChar != '\r' && curChar != '\n')
      continue;

    // We saw a line break, decrement our counter.
    --lineNo;

    // Check for \r\n and \n\r and treat it as a single escape.  We know that
    // looking past one character is safe because MemoryBuffer's are always nul
    // terminated.
    if (*position != curChar && (*position == '\r' || *position == '\n'))
      ++position;
  }

  // If the line/column counter was invalid, return a pointer to the start of
  // the buffer.
  if (lineNo || position + columnNo > end)
    return SMLoc::getFromPointer(membuf.getBufferStart());

  // Otherwise return the right pointer.
  return SMLoc::getFromPointer(position + columnNo);
}

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static OptResult performActions(SourceMgr &sourceMgr, MLIRContext *context) {
  std::unique_ptr<Module> module(parseSourceFile(sourceMgr, context));
  if (!module)
    return OptFailure;

  // Run each of the passes that were selected.
  for (const auto *passInfo : *passList) {
    std::unique_ptr<Pass> pass(passInfo->createPass());
    PassResult result = pass->runOnModule(module.get());
    if (result)
      return OptFailure;

    // Verify that the result of the pass is still valid.
    if (module->verify())
      return OptFailure;
  }

  // Print the output.
  auto output = getOutputStream();
  module->print(output->os());
  output->keep();
  return OptSuccess;
}

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagnosticKindString(MLIRContext::DiagnosticKind kind) {
  switch (kind) {
  case MLIRContext::DiagnosticKind::Note:
    return "note";
  case MLIRContext::DiagnosticKind::Warning:
    return "warning";
  case MLIRContext::DiagnosticKind::Error:
    return "error";
  }
}

/// Given a diagnostic kind, returns the LLVM DiagKind.
static llvm::SourceMgr::DiagKind getDiagKind(MLIRContext::DiagnosticKind kind) {
  switch (kind) {
  case MLIRContext::DiagnosticKind::Note:
    return llvm::SourceMgr::DK_Note;
  case MLIRContext::DiagnosticKind::Warning:
    return llvm::SourceMgr::DK_Warning;
  case MLIRContext::DiagnosticKind::Error:
    return llvm::SourceMgr::DK_Error;
  }
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static OptResult processFile(std::unique_ptr<MemoryBuffer> ownedBuffer) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  auto &buffer = *ownedBuffer;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Parse the input file.
  MLIRContext context;

  // If we are in verify mode then we have a lot of work to do, otherwise just
  // perform the actions without worrying about it.
  if (!verifyDiagnostics) {

    // Register a simple diagnostic handler that prints out info with context.
    context.registerDiagnosticHandler([&](Location location, StringRef message,
                                          MLIRContext::DiagnosticKind kind) {
      unsigned line = 1, column = 1;
      SMLoc loc;
      if (auto fileLoc = location.dyn_cast<FileLineColLoc>()) {
        line = fileLoc->getLine();
        column = fileLoc->getColumn();
        loc = getLocFromLineAndCol(buffer, line, column);
      }

      sourceMgr.PrintMessage(loc, getDiagKind(kind), message);
    });

    // Run the test actions.
    return performActions(sourceMgr, &context);
  }

  // Keep track of the result of this file processing.  If there are no issues,
  // then we succeed.
  auto result = OptSuccess;

  // Record the expected diagnostic's position, substring and whether it was
  // seen.
  struct ExpectedDiag {
    MLIRContext::DiagnosticKind kind;
    unsigned lineNo;
    StringRef substring;
    SMLoc fileLoc;
    bool matched = false;
  };
  SmallVector<ExpectedDiag, 2> expectedDiags;

  // Error checker that verifies reported error was expected.
  auto checker = [&](Location location, StringRef message,
                     MLIRContext::DiagnosticKind kind) {
    unsigned line = 1, column = 1;
    if (auto fileLoc = location.dyn_cast<FileLineColLoc>()) {
      line = fileLoc->getLine();
      column = fileLoc->getColumn();
    }

    // If we find something that is close then emit a more specific error.
    ExpectedDiag *nearMiss = nullptr;

    // If this was an expected error, remember that we saw it and return.
    for (auto &e : expectedDiags) {
      if (line == e.lineNo && message.contains(e.substring)) {
        if (e.kind == kind) {
          e.matched = true;
          return;
        }

        // If this only differs based on the diagnostic kind, then consider it
        // to be a near miss.
        nearMiss = &e;
      }
    }

    // If there was a near miss, emit a specific diagnostic.
    if (nearMiss) {
      sourceMgr.PrintMessage(nearMiss->fileLoc, SourceMgr::DK_Error,
                             "'" + getDiagnosticKindString(kind) +
                                 "' diagnostic emitted when expecting a '" +
                                 getDiagnosticKindString(nearMiss->kind) + "'");
      result = OptFailure;
      return;
    }

    // If this error wasn't expected, produce an error out of mlir-opt saying
    // so.
    auto unexpectedLoc = getLocFromLineAndCol(buffer, line, column);
    sourceMgr.PrintMessage(unexpectedLoc, SourceMgr::DK_Error,
                           "unexpected error: " + Twine(message));
    result = OptFailure;
  };

  // Scan the file for expected-* designators and register a callback for the
  // error handler.
  // Extract the expected errors from the file.
  llvm::Regex expected(
      "expected-(error|note|warning) *(@[+-][0-9]+)? *{{(.*)}}");
  SmallVector<StringRef, 100> lines;
  buffer.getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 3> matches;
    if (expected.match(lines[lineNo], &matches)) {
      // Point to the start of expected-*.
      SMLoc expectedStart = SMLoc::getFromPointer(matches[0].data());

      MLIRContext::DiagnosticKind kind;
      if (matches[1] == "error")
        kind = MLIRContext::DiagnosticKind::Error;
      else if (matches[1] == "warning")
        kind = MLIRContext::DiagnosticKind::Warning;
      else {
        assert(matches[1] == "note");
        kind = MLIRContext::DiagnosticKind::Note;
      }

      ExpectedDiag record{kind, lineNo + 1, matches[3], expectedStart, false};
      auto offsetMatch = matches[2];
      if (!offsetMatch.empty()) {
        int offset;
        // Get the integer value without the @ and +/- prefix.
        if (!offsetMatch.drop_front(2).getAsInteger(0, offset)) {
          if (offsetMatch[1] == '+')
            record.lineNo += offset;
          else
            record.lineNo -= offset;
        }
      }
      expectedDiags.push_back(record);
    }
  }

  // Finally, register the error handler to capture them.
  context.registerDiagnosticHandler(checker);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  performActions(sourceMgr, &context);

  // Verify that all expected errors were seen.
  for (auto &err : expectedDiags) {
    if (!err.matched) {
      SMRange range(err.fileLoc,
                    SMLoc::getFromPointer(err.fileLoc.getPointer() +
                                          err.substring.size()));
      auto kind = getDiagnosticKindString(err.kind);
      sourceMgr.PrintMessage(err.fileLoc, SourceMgr::DK_Error,
                             "expected " + kind + " \"" + err.substring +
                                 "\" was not produced",
                             range);
      result = OptFailure;
    }
  }

  return result;
}

/// Split the specified file on a marker and process each chunk independently
/// according to the normal processFile logic.  This is primarily used to
/// allow a large number of small independent parser tests to be put into a
/// single test, but could be used for other purposes as well.
static OptResult
splitAndProcessFile(std::unique_ptr<MemoryBuffer> originalBuffer) {
  const char marker[] = "// -----\n";
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
    if (processFile(std::move(subMemBuffer)))
      hadUnexpectedResult = true;
  }

  return hadUnexpectedResult ? OptFailure : OptSuccess;
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  InitLLVM y(argc, argv);

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::list<const mlir::PassInfo *, bool, mlir::PassNameParser> passList(
      "", llvm::cl::desc("Compiler passes to run"));
  ::passList = &passList;
  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  // Set up the input file.
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessFile(std::move(*fileOrErr));

  return processFile(std::move(*fileOrErr));
}

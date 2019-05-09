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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
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

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static std::vector<const mlir::PassRegistryEntry *> *passList;

enum OptResult { OptSuccess, OptFailure };

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
  PassManager pm(verifyPasses);
  for (const auto *passEntry : *passList)
    passEntry->addToPipeline(pm);

  // Apply any pass manager command line options.
  applyPassManagerCLOptions(pm);

  // Run the pipeline.
  if (failed(pm.run(module.get())))
    return OptFailure;

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Print the output.
  module->print(output->os());
  output->keep();
  return OptSuccess;
}

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagnosticKindString(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
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
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // If we are in verify mode then we have a lot of work to do, otherwise just
  // perform the actions without worrying about it.
  if (!verifyDiagnostics) {
    // Run the test actions.
    return performActions(sourceMgr, &context);
  }

  // Keep track of the result of this file processing.  If there are no issues,
  // then we succeed.
  auto result = OptSuccess;

  // Record the expected diagnostic's position, substring and whether it was
  // seen.
  struct ExpectedDiag {
    DiagnosticSeverity kind;
    unsigned lineNo;
    StringRef substring;
    SMLoc fileLoc;
    bool matched;
  };
  SmallVector<ExpectedDiag, 2> expectedDiags;

  // Error checker that verifies reported error was expected.
  auto checker = [&](Location location, StringRef message,
                     DiagnosticSeverity kind) {
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
    sourceMgrHandler.emitDiagnostic(location, "unexpected error: " + message,
                                    DiagnosticSeverity::Error);
    result = OptFailure;
  };

  // Scan the file for expected-* designators and register a callback for the
  // error handler.
  // Extract the expected errors from the file.
  llvm::Regex expected(
      "expected-(error|note|remark|warning) *(@[+-][0-9]+)? *{{(.*)}}");
  SmallVector<StringRef, 100> lines;
  buffer.getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 3> matches;
    if (expected.match(lines[lineNo], &matches)) {
      // Point to the start of expected-*.
      SMLoc expectedStart = SMLoc::getFromPointer(matches[0].data());

      DiagnosticSeverity kind;
      if (matches[1] == "error")
        kind = DiagnosticSeverity::Error;
      else if (matches[1] == "warning")
        kind = DiagnosticSeverity::Warning;
      else if (matches[1] == "remark")
        kind = DiagnosticSeverity::Remark;
      else {
        assert(matches[1] == "note");
        kind = DiagnosticSeverity::Note;
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
  context.getDiagEngine().setHandler(checker);

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

  // Register any pass manager command line options.
  registerPassManagerCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::list<const mlir::PassRegistryEntry *, bool, PassNameParser>
      passList("", llvm::cl::desc("Compiler passes to run"));
  ::passList = &passList;
  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return OptFailure;
  }

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessFile(std::move(file));

  return processFile(std::move(file));
}

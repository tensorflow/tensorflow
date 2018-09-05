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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/TensorFlow/ControlFlowOps.h"
#include "mlir/TensorFlow/Passes.h"
#include "mlir/Transforms/CFGFunctionViewGraph.h"
#include "mlir/Transforms/Pass.h"
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

enum Passes {
  ConvertToCFG,
  LoopUnroll,
  LoopUnrollAndJam,
  PrintCFGGraph,
  SimplifyAffineExpr,
  TFRaiseControlFlow,
};

static cl::list<Passes> passList(
    "", cl::desc("Compiler passes to run"),
    cl::values(clEnumValN(ConvertToCFG, "convert-to-cfg",
                          "Convert all ML functions in the module to CFG ones"),
               clEnumValN(LoopUnroll, "loop-unroll", "Unroll loops"),
               clEnumValN(LoopUnrollAndJam, "loop-unroll-jam",
                          "Unroll and jam loops"),
               clEnumValN(PrintCFGGraph, "print-cfg-graph",
                          "Print CFG graph per function"),
               clEnumValN(SimplifyAffineExpr, "simplify-affine-expr",
                          "Simplify affine expressions"),
               clEnumValN(TFRaiseControlFlow, "tf-raise-control-flow",
                          "Dynamic TensorFlow Switch/Match nodes to a CFG")));

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

// The function to initialize the MLIRContext for different ops is defined in
// another compilation unit to allow different tests to link in different
// context initializations (e.g., op registrations).
extern void initializeMLIRContext(MLIRContext *ctx);

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
  for (auto passKind : passList) {
    Pass *pass = nullptr;
    switch (passKind) {
    case ConvertToCFG:
      pass = createConvertToCFGPass();
      break;
    case LoopUnroll:
      pass = createLoopUnrollPass();
      break;
    case LoopUnrollAndJam:
      pass = createLoopUnrollAndJamPass();
      break;
    case PrintCFGGraph:
      pass = createPrintCFGGraphPass();
      break;
    case SimplifyAffineExpr:
      pass = createSimplifyAffineExprPass();
      break;
    case TFRaiseControlFlow:
      pass = createRaiseTFControlFlowPass();
      break;
    }

    pass->runOnModule(module.get());
    delete pass;
    module->verify();
  }

  // Print the output.
  auto output = getOutputStream();
  module->print(output->os());
  output->keep();
  return OptSuccess;
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
  initializeMLIRContext(&context);

  // If we are in verify mode then we have a lot of work to do, otherwise just
  // perform the actions without worrying about it.
  if (!verifyDiagnostics) {

    // Register a simple diagnostic handler that prints out info with context.
    context.registerDiagnosticHandler([&](Location *location, StringRef message,
                                          MLIRContext::DiagnosticKind kind) {
      unsigned line = 1, column = 1;
      if (auto fileLoc = dyn_cast<FileLineColLoc>(location)) {
        line = fileLoc->getLine();
        column = fileLoc->getColumn();
      }

      auto unexpectedLoc = getLocFromLineAndCol(buffer, line, column);
      sourceMgr.PrintMessage(unexpectedLoc, SourceMgr::DK_Error, message);
    });

    // Run the test actions.
    return performActions(sourceMgr, &context);
  }

  // Keep track of the result of this file processing.  If there are no issues,
  // then we succeed.
  auto result = OptSuccess;

  // Record the expected error's position, substring and whether it was seen.
  struct ExpectedError {
    unsigned lineNo;
    StringRef substring;
    SMLoc fileLoc;
    bool matched = false;
  };
  SmallVector<ExpectedError, 2> expectedErrors;

  // Error checker that verifies reported error was expected.
  auto checker = [&](Location *location, StringRef message,
                     MLIRContext::DiagnosticKind kind) {
    unsigned line = 1, column = 1;
    if (auto *fileLoc = dyn_cast<FileLineColLoc>(location)) {
      line = fileLoc->getLine();
      column = fileLoc->getColumn();
    }

    // If this was an expected error, remember that we saw it and return.
    for (auto &e : expectedErrors) {
      if (line == e.lineNo && message.contains(e.substring)) {
        e.matched = true;
        return;
      }
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
  llvm::Regex expected("expected-error *(@[+-][0-9]+)? *{{(.*)}}");
  SmallVector<StringRef, 100> lines;
  buffer.getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 3> matches;
    if (expected.match(lines[lineNo], &matches)) {
      // Point to the start of expected-error.
      SMLoc errorStart = SMLoc::getFromPointer(matches[0].data());
      ExpectedError expErr{lineNo + 1, matches[2], errorStart, false};
      auto offsetMatch = matches[1];
      if (!offsetMatch.empty()) {
        int offset;
        // Get the integer value without the @ and +/- prefix.
        if (!offsetMatch.drop_front(2).getAsInteger(0, offset)) {
          if (offsetMatch[1] == '+')
            expErr.lineNo += offset;
          else
            expErr.lineNo -= offset;
        }
      }
      expectedErrors.push_back(expErr);
    }
  }

  // Finally, register the error handler to capture them.
  context.registerDiagnosticHandler(checker);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  performActions(sourceMgr, &context);

  // Verify that all expected errors were seen.
  for (auto &err : expectedErrors) {
    if (!err.matched) {
      SMRange range(err.fileLoc,
                    SMLoc::getFromPointer(err.fileLoc.getPointer() +
                                          err.substring.size()));
      sourceMgr.PrintMessage(
          err.fileLoc, SourceMgr::DK_Error,
          "expected error \"" + err.substring + "\" was not produced", range);
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
  const char marker[] = "-----";
  SmallVector<StringRef, 8> sourceBuffers;
  originalBuffer->getBuffer().split(sourceBuffers, marker);

  // Add the original buffer to the source manager.
  SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(originalBuffer), SMLoc());

  bool hadUnexpectedResult = false;

  // Process each chunk in turn.  If any fails, then return a failure of the
  // tool.
  for (auto &subBuffer : sourceBuffers) {
    auto subMemBuffer = MemoryBuffer::getMemBufferCopy(subBuffer);
    if (processFile(std::move(subMemBuffer)))
      hadUnexpectedResult = true;
  }

  return hadUnexpectedResult ? OptFailure : OptSuccess;
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  InitLLVM y(argc, argv);

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

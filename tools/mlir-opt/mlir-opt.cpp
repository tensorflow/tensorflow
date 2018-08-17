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
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/TensorFlow/ControlFlowOps.h"
#include "mlir/TensorFlow/Passes.h"
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

static cl::opt<std::string>
inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
               cl::init("-"));

static cl::opt<bool>
checkParserErrors("check-parser-errors", cl::desc("Check for parser errors"),
                  cl::init(false));

enum Passes {
  ConvertToCFG,
  UnrollInnermostLoops,
  UnrollShortLoops,
  TFRaiseControlFlow,
};

static cl::list<Passes> passList(
    "", cl::desc("Compiler passes to run"),
    cl::values(clEnumValN(ConvertToCFG, "convert-to-cfg",
                          "Convert all ML functions in the module to CFG ones"),
               clEnumValN(UnrollInnermostLoops, "unroll-innermost-loops",
                          "Unroll innermost loops"),
               clEnumValN(UnrollShortLoops, "unroll-short-loops",
                          "Unroll loops of trip count <= 2"),
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

/// Parses the memory buffer and, if successfully parsed, prints the parsed
/// output. Optionally, convert ML functions into CFG functions.
/// TODO: pull parsing and printing into separate functions.
OptResult parseAndPrintMemoryBuffer(std::unique_ptr<MemoryBuffer> buffer) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Parse the input file.
  MLIRContext context;
  initializeMLIRContext(&context);
  std::unique_ptr<Module> module(parseSourceFile(sourceMgr, &context));
  if (!module)
    return OptFailure;

  // Run each of the passes that were selected.
  for (auto passKind : passList) {
    Pass *pass = nullptr;
    switch (passKind) {
    case ConvertToCFG:
      pass = createConvertToCFGPass();
      break;
    case UnrollInnermostLoops:
      pass = createLoopUnrollPass();
      break;
    case UnrollShortLoops:
      pass = createLoopUnrollPass(2);
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

/// Split the memory buffer into multiple buffers using the marker -----.
OptResult
splitMemoryBufferForErrorChecking(std::unique_ptr<MemoryBuffer> buffer) {
  const char marker[] = "-----";
  SmallVector<StringRef, 2> sourceBuffers;
  buffer->getBuffer().split(sourceBuffers, marker);

  // Error reporter that verifies error reports matches expected error
  // substring.
  // TODO: Only checking for error cases below. Could be expanded to other kinds
  // of diagnostics.
  // TODO: Enable specifying errors on different lines (@-1).
  // TODO: Currently only checking if substring matches, enable regex checking.
  OptResult opt_result = OptSuccess;
  SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Record the expected errors's position, substring and whether it was seen.
  struct ExpectedError {
    int lineNo;
    StringRef substring;
    SMLoc fileLoc;
    bool matched;
  };

  // Tracks offset of subbuffer into original buffer.
  const char *fileOffset =
      fileSourceMgr.getMemoryBuffer(fileSourceMgr.getMainFileID())
          ->getBufferStart();

  for (auto &subbuffer : sourceBuffers) {
    SourceMgr sourceMgr;
    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    auto bufferId = sourceMgr.AddNewSourceBuffer(
        MemoryBuffer::getMemBufferCopy(subbuffer), SMLoc());

    // Extract the expected errors.
    llvm::Regex expected("expected-error(@[+-][0-9]+)? *{{(.*)}}");
    SmallVector<ExpectedError, 2> expectedErrors;
    SmallVector<StringRef, 100> lines;
    subbuffer.split(lines, '\n');
    size_t bufOffset = 0;
    for (int lineNo = 0; lineNo < lines.size(); ++lineNo) {
      SmallVector<StringRef, 3> matches;
      if (expected.match(lines[lineNo], &matches)) {
        // Point to the start of expected-error.
        SMLoc errorStart =
            SMLoc::getFromPointer(fileOffset + bufOffset +
                                  lines[lineNo].size() - matches[2].size() - 2);
        ExpectedError expErr{lineNo + 1, matches[2], errorStart, false};
        int offset;
        if (!matches[1].empty() &&
            !matches[1].drop_front().getAsInteger(0, offset)) {
          expErr.lineNo += offset;
        }
        expectedErrors.push_back(expErr);
      }
      bufOffset += lines[lineNo].size() + 1;
    }

    // Error checker that verifies reported error was expected.
    auto checker = [&](const SMDiagnostic &err) {
      for (auto &e : expectedErrors) {
        if (err.getLineNo() == e.lineNo &&
            err.getMessage().contains(e.substring)) {
          e.matched = true;
          return;
        }
      }
      // Report error if no match found.
      const auto &sourceMgr = *err.getSourceMgr();
      const char *bufferStart =
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())
              ->getBufferStart();

      size_t offset = err.getLoc().getPointer() - bufferStart;
      SMLoc loc = SMLoc::getFromPointer(fileOffset + offset);
      fileSourceMgr.PrintMessage(loc, SourceMgr::DK_Error,
                                 "unexpected error: " + err.getMessage());
      opt_result = OptFailure;
    };

    // Parse the input file.
    MLIRContext context;
    initializeMLIRContext(&context);

    // TODO: refactor into initializeMLIRContext so the normal parser pass
    // gets to use this.
    context.registerDiagnosticHandler([&](Attribute *location,
                                          StringRef message,
                                          MLIRContext::DiagnosticKind kind) {
      auto offset = cast<IntegerAttr>(location)->getValue();
      auto ptr = sourceMgr.getMemoryBuffer(bufferId)->getBufferStart() + offset;
      SourceMgr::DiagKind diagKind;
      switch (kind) {
      case MLIRContext::DiagnosticKind::Error:
        diagKind = SourceMgr::DK_Error;
        break;
      case MLIRContext::DiagnosticKind::Warning:
        diagKind = SourceMgr::DK_Warning;
        break;
      case MLIRContext::DiagnosticKind::Note:
        diagKind = SourceMgr::DK_Note;
        break;
      }
      checker(
          sourceMgr.GetMessage(SMLoc::getFromPointer(ptr), diagKind, message));
    });

    std::unique_ptr<Module> module(
        parseSourceFile(sourceMgr, &context, checker));

    // Verify that all expected errors were seen.
    for (auto err : expectedErrors) {
      if (!err.matched) {
        SMRange range(err.fileLoc,
                      SMLoc::getFromPointer(err.fileLoc.getPointer() +
                                            err.substring.size()));
        fileSourceMgr.PrintMessage(
            err.fileLoc, SourceMgr::DK_Error,
            "expected error \"" + err.substring + "\" was not produced", range);
        opt_result = OptFailure;
      }
    }

    fileOffset += subbuffer.size() + strlen(marker);
  }

  return opt_result;
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

  if (checkParserErrors)
    return splitMemoryBufferForErrorChecking(std::move(*fileOrErr));

  return parseAndPrintMemoryBuffer(std::move(*fileOrErr));
}

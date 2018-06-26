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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
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

/// Open the specified output file and return it, exiting if there is any I/O or
/// other errors.
static std::unique_ptr<ToolOutputFile> getOutputStream() {
  std::error_code error;
  auto result = make_unique<ToolOutputFile>(outputFilename, error,
                                            sys::fs::F_None);
  if (error) {
    llvm::errs() << error.message() << '\n';
    exit(1);
  }

  return result;
}

/// Parses the memory buffer and, if successfully parsed, prints the parsed
/// output. Returns whether parsing succeeded.
bool parseAndPrintMemoryBuffer(std::unique_ptr<MemoryBuffer> buffer,
                               const SMDiagnosticHandlerTy& errorReporter) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Parse the input file.
  MLIRContext context;
  std::unique_ptr<Module> module(
      parseSourceFile(sourceMgr, &context, errorReporter));
  if (!module) return false;

  // Print the output.
  auto output = getOutputStream();
  module->print(output->os());
  output->keep();

  // Success.
  return true;
}

/// Split the memory buffer into multiple buffers using the marker -----.
bool splitMemoryBufferForErrorChecking(std::unique_ptr<MemoryBuffer> buffer) {
  const char marker[] = "-----";
  SmallVector<StringRef, 2> sourceBuffers;
  buffer->getBuffer().split(sourceBuffers, marker);

  // Error reporter that verifies error reports matches expected error
  // substring.
  // TODO: Only checking for error cases below. Could be expanded to other kinds
  // of diagnostics.
  // TODO: Enable specifying errors on different lines (@-1).
  // TODO: Currently only checking if substring matches, enable regex checking.
  bool failed = false;
  SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Tracks offset of subbuffer into original buffer.
  const char *fileOffset =
      fileSourceMgr.getMemoryBuffer(fileSourceMgr.getMainFileID())
          ->getBufferStart();

  // Create error checker that uses the helper function to relate the reported
  // error to the file being parsed.
  SMDiagnosticHandlerTy checker = [&](SMDiagnostic err) {
    const auto &sourceMgr = *err.getSourceMgr();
    const char *bufferStart =
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBufferStart();

    StringRef line = err.getLineContents();
    size_t offset = err.getLoc().getPointer() - bufferStart;
    SMLoc loc = SMLoc::getFromPointer(fileOffset + offset);

    // Extract expected substring using regex and check simple containment in
    // error message.
    llvm::Regex expected("expected-error {{(.*)}}");
    SmallVector<StringRef, 2> matches;
    bool matched = expected.match(line, &matches);
    if (matches.size() != 2) {
      fileSourceMgr.PrintMessage(
          loc, SourceMgr::DK_Error,
          "unexpected error: " + err.getMessage());
      failed = true;
      return;
    }

    matched = err.getMessage().contains(matches[1]);
    if (!matched) {
      const char checkPrefix[] = "expected-error {{";
      loc = SMLoc::getFromPointer(fileOffset + offset + line.find(checkPrefix) -
                                  err.getColumnNo() + strlen(checkPrefix));
      fileSourceMgr.PrintMessage(loc, SourceMgr::DK_Error,
                                 "\"" + err.getMessage() +
                                     "\" did not contain expected substring \"" +
                                     matches[1] + "\"");
      failed = true;
      return;
    }
  };

  for (auto& subbuffer : sourceBuffers) {
    SourceMgr sourceMgr;
    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    sourceMgr.AddNewSourceBuffer(MemoryBuffer::getMemBufferCopy(subbuffer),
                                 SMLoc());

    int expectedCount = subbuffer.count("expected-error");
    if (expectedCount > 1) {
      size_t expectedOffset = subbuffer.find("expected-error");
      expectedOffset = subbuffer.find("expected-error", expectedOffset);
      SMLoc loc = SMLoc::getFromPointer(fileOffset + expectedOffset);
      fileSourceMgr.PrintMessage(loc, SourceMgr::DK_Error,
                                 "too many errors expected: unable to verify "
                                 "more than one error per group");
      fileOffset += subbuffer.size() + strlen(marker);
      failed = true;
      continue;
    }

    // Parse the input file.
    MLIRContext context;
    std::unique_ptr<Module> module(
        parseSourceFile(sourceMgr, &context, checker));
    bool parsed = module != nullptr;
    if (parsed && expectedCount != 0) {
      llvm::Regex expected(".*expected-error {{(.*)}}");
      SmallVector<StringRef, 2> matches;
      expected.match(subbuffer, &matches);

      // Highlight expected-error clause of unexpectedly passing test case.
      size_t expectedOffset = subbuffer.find("expected-error");
      size_t endOffset = matches[0].size();
      SMLoc loc = SMLoc::getFromPointer(fileOffset + expectedOffset);
      SMRange range(loc, SMLoc::getFromPointer(fileOffset + endOffset));
      fileSourceMgr.PrintMessage(
          loc, SourceMgr::DK_Error,
          "expected error \"" + matches[1] + "\" was not produced", range);
      failed = true;
    }

    fileOffset += subbuffer.size() + strlen(marker);
  }

  return !failed;
}

int main(int argc, char **argv) {
  InitLLVM x(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  // Set up the input file.
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  if (checkParserErrors)
    return !splitMemoryBufferForErrorChecking(std::move(*fileOrErr));

  // Error reporter that simply prints the errors reported.
  SMDiagnosticHandlerTy errorReporter = [](llvm::SMDiagnostic err) {
    const auto& sourceMgr = *err.getSourceMgr();
    sourceMgr.PrintMessage(err.getLoc(), err.getKind(), err.getMessage());
  };
  return !parseAndPrintMemoryBuffer(std::move(*fileOrErr), errorReporter);
}

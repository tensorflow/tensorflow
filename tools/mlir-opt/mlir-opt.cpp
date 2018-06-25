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
bool parseAndPrintMemoryBuffer(std::unique_ptr<MemoryBuffer> buffer) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Parse the input file.
  MLIRContext context;
  // Error reporter that simply prints the errors reported.
  SMDiagnosticHandlerTy errorReporter = [&sourceMgr](llvm::SMDiagnostic err) {
    sourceMgr.PrintMessage(err.getLoc(), err.getKind(), err.getMessage());
  };
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
  for (auto& subbuffer : sourceBuffers)
    parseAndPrintMemoryBuffer(MemoryBuffer::getMemBufferCopy(subbuffer));

  // Ignore errors returned by parseAndPrintMemoryBuffer when checking parse
  // errors reported.
  return true;
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
  return !parseAndPrintMemoryBuffer(std::move(*fileOrErr));
}

//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
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
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

static cl::opt<bool> EnableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code EC = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return nullptr;
  }
  auto buffer = FileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), filename);
  Parser parser(lexer);
  return parser.ParseModule();
}

mlir::LogicalResult optimize(mlir::Module module) {
  mlir::PassManager pm;
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Apply any generic pass manager command line options.
  applyPassManagerCLOptions(pm);

  return pm.run(module);
}

int dumpMLIR() {
  // Register our Dialect with MLIR
  mlir::registerDialect<ToyDialect>();

  // Add generic options for the pass-manager (timing, print-after-all, ...)
  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  if (inputType == InputType::MLIR ||
      llvm::StringRef(inputFilename).endswith(".mlir")) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << EC.message() << "\n";
      return -1;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
    }
    if (failed(module->verify())) {
      llvm::errs() << "Error verifying MLIR module\n";
      return 4;
    }
  } else {
    auto moduleAST = parseInputFile(inputFilename);
    module = mlirGen(context, *moduleAST);
  }
  if (!module)
    return 1;
  if (EnableOpt) {
    if (failed(optimize(*module))) {
      llvm::errs() << "Module optimization failed\n";
      return 7;
    }
  }
  module->dump();
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}

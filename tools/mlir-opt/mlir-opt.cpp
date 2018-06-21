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

#include "mlir/IR/Module.h"
#include "llvm/Support/CommandLine.h"
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

int main(int argc, char **argv) {
  InitLLVM x(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  // Instantiate an IR object.
  Module m;
  m.functionList.push_back(new Function("foo"));
  m.functionList.push_back(new Function("bar"));

  // Print the output.
  auto output = getOutputStream();
  m.print(output->os());
  output->keep();
}

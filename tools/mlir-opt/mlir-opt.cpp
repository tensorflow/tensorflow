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

#include "mlir/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM x(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  // Instantiate an IR object.
  Function f;
  (void)f;
}

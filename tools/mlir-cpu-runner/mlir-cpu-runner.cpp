//===- mlir-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
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
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

namespace mlir {
class ModuleOp;
struct LogicalResult;
} // namespace mlir

// TODO(herhut) Factor out into an include file and proper library.
extern int run(int argc, char **argv,
               llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)>);

int main(int argc, char **argv) { return run(argc, argv, nullptr); }

//===- JitRunner.h - MLIR CPU Execution Driver Library ----------*- C++ -*-===//
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
// This is a library that provides a shared implementation for command line
// utilities that execute an MLIR file on the CPU by translating MLIR to LLVM
// IR before JIT-compiling and executing the latter.
//
// The translation can be customized by providing an MLIR to MLIR
// transformation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_JITRUNNER_H_
#define MLIR_SUPPORT_JITRUNNER_H_

#include "llvm/ADT/STLExtras.h"

namespace mlir {

class ModuleOp;
struct LogicalResult;

// Entry point for all CPU runners. Expects the common argc/argv arguments for
// standard C++ main functions and an mlirTransformer.
// The latter is applied after parsing the input into MLIR IR and before passing
// the MLIR module to the ExecutionEngine.
int JitRunnerMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer);

} // namespace mlir

#endif // MLIR_SUPPORT_JITRUNNER_H_

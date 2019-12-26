//===- JitRunner.h - MLIR CPU Execution Driver Library ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

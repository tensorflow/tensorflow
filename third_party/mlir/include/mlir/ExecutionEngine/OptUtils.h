//===- OptUtils.h - MLIR Execution Engine opt pass utilities ----*- C++ -*-===//
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
// This file declares the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_OPTUTILS_H_
#define MLIR_EXECUTIONENGINE_OPTUTILS_H_

#include "llvm/Pass.h"

#include <functional>
#include <string>

namespace llvm {
class Module;
class Error;
class TargetMachine;
} // namespace llvm

namespace mlir {

/// Initialize LLVM passes that can be when running MLIR code using
/// ExecutionEngine.
void initializeLLVMPasses();

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes corresponding to the given speed and size optimization
/// levels (e.g. -O2 or -Os). If not null, `targetMachine` is used to
/// initialize passes that provide target-specific information to the LLVM
/// optimizer. `targetMachine` must outlive the returned std::function.
std::function<llvm::Error(llvm::Module *)>
makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                          llvm::TargetMachine *targetMachine);

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes explicitly specified, plus an optional optimization level,
/// Any optimization passes, if present, will be inserted before the pass at
/// position optPassesInsertPos. If not null, `targetMachine` is used to
/// initialize passes that provide target-specific information to the LLVM
/// optimizer. `targetMachine` must outlive the returned std::function.
std::function<llvm::Error(llvm::Module *)>
makeLLVMPassesTransformer(llvm::ArrayRef<const llvm::PassInfo *> llvmPasses,
                          llvm::Optional<unsigned> mbOptLevel,
                          llvm::TargetMachine *targetMachine,
                          unsigned optPassesInsertPos = 0);

} // end namespace mlir

#endif // LIR_EXECUTIONENGINE_OPTUTILS_H_

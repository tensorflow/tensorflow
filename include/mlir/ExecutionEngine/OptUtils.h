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

#include <functional>
#include <string>

namespace llvm {
class Module;
class Error;
} // namespace llvm

namespace mlir {

/// Initialize LLVM passes that can be when running MLIR code using
/// ExecutionEngine.
void initializeLLVMPasses();

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes corresponding to the given speed and size optimization
/// levels (e.g. -O2 or -Os).
std::function<llvm::Error(llvm::Module *)>
makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel);

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes specified by the configuration string that uses the same
/// syntax as LLVM opt tool.  For example, "-loop-distribute -loop-vectorize"
/// will run the loop distribution pass followed by the loop vectorizer.
std::function<llvm::Error(llvm::Module *)>
makeLLVMPassesTransformer(std::string config);

} // end namespace mlir

#endif // LIR_EXECUTIONENGINE_OPTUTILS_H_

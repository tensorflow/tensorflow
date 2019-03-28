//===- Transforms.h - Pass Entrypoints --------------------------*- C++ -*-===//
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

#ifndef MLIR_LLVMIR_TRANSFORMS_H_
#define MLIR_LLVMIR_TRANSFORMS_H_

#include <memory>

namespace mlir {
class DialectConversion;
class Module;
class ModulePassBase;

/// Creates a pass to convert Standard dialects into the LLVMIR dialect.
ModulePassBase *createConvertToLLVMIRPass();

/// Creates a dialect converter from the standard dialect to the LLVM IR
/// dialect and transfers ownership to the caller.
std::unique_ptr<DialectConversion> createStdToLLVMConverter();

namespace LLVM {
/// Make argument-taking successors of each block distinct.  PHI nodes in LLVM
/// IR use the predecessor ID to identify which value to take.  They do not
/// support different values coming from the same predecessor.  If a block has
/// another block as a successor more than once with different values, insert
/// a new dummy block for LLVM PHI nodes to tell the sources apart.
void ensureDistinctSuccessors(Module *m);
} // namespace LLVM

} // namespace mlir

#endif // MLIR_LLVMIR_TRANSFORMS_H_

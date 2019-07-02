//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

#include <memory>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace mlir {
class DialectConversion;
class LLVMTypeConverter;
class Module;
class ModulePassBase;
class RewritePattern;
class Type;

using OwningRewritePatternList = std::vector<std::unique_ptr<RewritePattern>>;

/// Creates a pass to convert Standard dialects into the LLVMIR dialect.
ModulePassBase *createConvertToLLVMIRPass();

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

namespace LLVM {
/// Make argument-taking successors of each block distinct.  PHI nodes in LLVM
/// IR use the predecessor ID to identify which value to take.  They do not
/// support different values coming from the same predecessor.  If a block has
/// another block as a successor more than once with different values, insert
/// a new dummy block for LLVM PHI nodes to tell the sources apart.
void ensureDistinctSuccessors(Module m);
} // namespace LLVM

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

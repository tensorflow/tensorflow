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

#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace mlir {
class DialectConversion;
class FuncOp;
class LLVMTypeConverter;
struct LogicalResult;
class MLIRContext;
class ModuleOp;
template <typename T> class OpPassBase;
class RewritePattern;
class Type;

// Owning list of rewriting patterns.
class OwningRewritePatternList;

/// Type for a callback constructing the owning list of patterns for the
/// conversion to the LLVMIR dialect.  The callback is expected to append
/// patterns to the owning list provided as the second argument.
using LLVMPatternListFiller =
    std::function<void(LLVMTypeConverter &, OwningRewritePatternList &)>;

/// Type for a callback constructing the type converter for the conversion to
/// the LLVMIR dialect.  The callback is expected to return an instance of the
/// converter.
using LLVMTypeConverterMaker =
    std::function<std::unique_ptr<LLVMTypeConverter>(MLIRContext *)>;

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
std::unique_ptr<OpPassBase<ModuleOp>> createLowerToLLVMPass();

/// Creates a pass to convert operations to the LLVMIR dialect.  The conversion
/// is defined by a list of patterns and a type converter that will be obtained
/// during the pass using the provided callbacks.
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(LLVMPatternListFiller patternListFiller,
                      LLVMTypeConverterMaker typeConverterMaker);

/// Creates a pass to convert operations to the LLVMIR dialect.  The conversion
/// is defined by a list of patterns obtained during the pass using the provided
/// callback and an optional type conversion class, an instance is created
/// during the pass.
template <typename TypeConverter = LLVMTypeConverter>
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(LLVMPatternListFiller patternListFiller) {
  return createLowerToLLVMPass(patternListFiller, [](MLIRContext *context) {
    return std::make_unique<TypeConverter>(context);
  });
}

namespace LLVM {
/// Make argument-taking successors of each block distinct.  PHI nodes in LLVM
/// IR use the predecessor ID to identify which value to take.  They do not
/// support different values coming from the same predecessor.  If a block has
/// another block as a successor more than once with different values, insert
/// a new dummy block for LLVM PHI nodes to tell the sources apart.
void ensureDistinctSuccessors(ModuleOp m);
} // namespace LLVM

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

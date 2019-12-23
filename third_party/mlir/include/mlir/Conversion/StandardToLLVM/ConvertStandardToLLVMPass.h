//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding the memory-related
/// operations.
void populateStdToLLVMMemoryConversionPatters(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// By default stdlib malloc/free are used for allocating MemRef payloads.
/// Specifying `useAlloca-true` emits stack allocations instead. In the future
/// this may become an enum when we have concrete uses for other options.
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(bool useAlloca = false);

/// Creates a pass to convert operations to the LLVMIR dialect.  The conversion
/// is defined by a list of patterns and a type converter that will be obtained
/// during the pass using the provided callbacks.
/// By default stdlib malloc/free are used for allocating MemRef payloads.
/// Specifying `useAlloca-true` emits stack allocations instead. In the future
/// this may become an enum when we have concrete uses for other options.
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(LLVMPatternListFiller patternListFiller,
                      LLVMTypeConverterMaker typeConverterMaker,
                      bool useAlloca = false);

/// Creates a pass to convert operations to the LLVMIR dialect.  The conversion
/// is defined by a list of patterns obtained during the pass using the provided
/// callback and an optional type conversion class, an instance is created
/// during the pass.
/// By default stdlib malloc/free are used for allocating MemRef payloads.
/// Specifying `useAlloca-true` emits stack allocations instead. In the future
/// this may become an enum when we have concrete uses for other options.
template <typename TypeConverter = LLVMTypeConverter>
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(LLVMPatternListFiller patternListFiller,
                      bool useAlloca = false) {
  return createLowerToLLVMPass(
      patternListFiller,
      [](MLIRContext *context) {
        return std::make_unique<TypeConverter>(context);
      },
      useAlloca);
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

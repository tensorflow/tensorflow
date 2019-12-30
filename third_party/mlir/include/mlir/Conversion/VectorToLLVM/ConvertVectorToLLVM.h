//===- ConvertVectorToLLVM.h - Utils to convert from the vector dialect ---===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename T> class OpPassBase;

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

/// Create a pass to convert vector operations to the LLVMIR dialect.
OpPassBase<ModuleOp> *createLowerVectorToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_

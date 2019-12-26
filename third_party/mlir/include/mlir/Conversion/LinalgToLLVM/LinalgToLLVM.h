//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
#define MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;

class LinalgTypeConverter : public LLVMTypeConverter {
public:
  using LLVMTypeConverter::LLVMTypeConverter;
  Type convertType(Type t) override;
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
void populateLinalgToLLVMConversionPatterns(LinalgTypeConverter &converter,
                                            OwningRewritePatternList &patterns,
                                            MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_

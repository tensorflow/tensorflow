//===- ConvertStandardToSPIRV.h - Convert to SPIR-V dialect -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to lower StandardOps to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H
#define MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating StandardOps to
/// SPIR-V ops. Also adds the patterns legalize ops not directly translated to
/// SPIR-V dialect.
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns);

/// Appends to a pattern list patterns to legalize ops that are not directly
/// lowered to SPIR-V.
void populateStdLegalizationPatternsForSPIRVLowering(
    MLIRContext *context, OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H

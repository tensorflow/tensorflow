//===- ConvertStandardToSPIRV.h - Convert to SPIR-V dialect -----*- C++ -*-===//
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

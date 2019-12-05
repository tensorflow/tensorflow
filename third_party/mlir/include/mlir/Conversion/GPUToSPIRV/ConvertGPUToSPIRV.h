//===- ConvertGPUToSPIRV.h - GPU Ops to SPIR-V dialect patterns ----C++ -*-===//
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
// Provides patterns for lowering GPU Ops to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRV_H
#define MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;
/// Appends to a pattern list additional patterns for translating GPU Ops to
/// SPIR-V ops.
void populateGPUToSPIRVPatterns(MLIRContext *context,
                                SPIRVTypeConverter &typeConverter,
                                OwningRewritePatternList &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRV_H

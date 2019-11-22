//===- VectorConversions.h - Utils to convert from the vector dialect -----===//
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
#ifndef MLIR_CONVERSION_VECTORCONVERSIONS_VECTORCONVERSIONS_H_
#define MLIR_CONVERSION_VECTORCONVERSIONS_VECTORCONVERSIONS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OpPassBase;

/// Collect a set of patterns to convert from the Vector dialect to affine loops
/// surrounding ops in different dialects (vector, std etc).
/// This is the general place where we want to implement Vector -> Vector and
/// Vector -> Std legalizations.
void populateVectorToAffineLoopsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Collect a set of patterns to convert from the Vector dialect to itself.
/// Should be merged with populateVectorToAffineLoopsConversionPatterns.
void populateVectorToVectorConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> coarseVectorShape = {},
    ArrayRef<int64_t> fineVectorShape = {});

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

/// Create a pass to convert vector operations to the LLVMIR dialect.
OpPassBase<ModuleOp> *createLowerVectorToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORCONVERSIONS_VECTORCONVERSIONS_H_

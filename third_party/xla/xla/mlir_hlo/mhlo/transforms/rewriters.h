/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H
#define MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H

#include <functional>
#include <memory>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

// Collection of rewrite patterns for lowering a general dot product.
void populateGeneralDotOpLoweringPatterns(RewritePatternSet *patterns,
                                          MLIRContext *ctx);

// Collection of rewrite patterns for lowering complex operations to equivalent
// float operations.
void populateComplexLoweringPatterns(MLIRContext *context,
                                     RewritePatternSet *patterns);

void populateOptimizeMhloPatterns(MLIRContext *context,
                                  RewritePatternSet *patterns);

// Rewrite patterns for dot to equivalent dot_general legalization.
void populateDotToDotGeneralPatterns(mlir::MLIRContext *context,
                                     RewritePatternSet *patterns);

// Rewrite patterns for einsum to equivalent dot_general legalization.
void populateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        RewritePatternSet *patterns);

// Rewrite patterns for torch index select to equivalent gather legalization.
void populateTorchIndexSelectToGatherPatterns(mlir::MLIRContext *context,
                                              RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of HLO to arithmetic dialect.
void populateHloToArithmeticConversionPatterns(RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering pointwise HLO ops with scalar
// arguments to arithmetic dialect.
void populateScalarHloToArithmeticConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns,
    llvm::function_ref<bool(Operation *)> filterFn = nullptr);

// Collection of rewrite patterns for lowering of HLO to Linalg dialect.
void populateHloToLinalgConversionPattern(MLIRContext *context,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet *patterns,
                                          bool enablePrimitiveOps = false);

// Converter to signless intergers to be used with linalg conversion patterns.
std::unique_ptr<TypeConverter> createHloToLinalgTypeConverter();

// Sets up legality definitions for materializing broadcasts.
void setupMaterializeBroadcastsLegality(MLIRContext *context,
                                        ConversionTarget *conversionTarget);

// Populates a collection of rewrite patterns for materializing broadcast
// attributes to equivalent sequences of ops.
void populateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           RewritePatternSet *patterns);

// Populate a collection of conversion patterns for un-fusing
// batch_norm_inference into constituent HLO ops.
void populateUnfuseBatchNormInferencePattern(MLIRContext *context,
                                             RewritePatternSet *patterns);

// Populate a collection of conversion patterns for un-fusing
// batch_norm_training into constituent HLO ops.
void populateUnfuseBatchNormTrainingPattern(MLIRContext *context,
                                            RewritePatternSet *patterns);

// Populate a collection of conversion patterns for un-fusing
// // batch_norm_inference and batch_norm_training into constituent HLO ops.
inline void populateUnfuseBatchNormPatterns(MLIRContext *context,
                                            RewritePatternSet *patterns) {
  populateUnfuseBatchNormInferencePattern(context, patterns);
  populateUnfuseBatchNormTrainingPattern(context, patterns);
}

// Populates patterns that translate the trigonometric operations from the
// standard dialect to approximations that do not use intrinsics.
void populateTrigonometricToApproximationPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns);

// Populate patterns to prepare moving dynamic broadcasts up over element-wise
// operations and broadcast the operands rather than the result. This will
// eventually allow for larger fusions.
void populateMergeAssumingOpsPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

// Populate patterns to group reduction and parallel dimensions of reduction
// operations and realize them through equivalent 1D or 2D reductions.
void populateGroupReductionDimensionsPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns,
                                              bool preferColumnsReductions);
}  // namespace mhlo

namespace chlo {

// Populates direct translations between CHLO and MHLO ops for higher level
// MHLO ops like TopK and Erf.
void populateChloToHighLevelMhloOpPatterns(MLIRContext *context,
                                           RewritePatternSet *patterns);

// Populates direct translations between CHLO->MHLO high level ops
// and CHLO->StableHLO->MHLO patterns.
void populateChloToHloPatterns(MLIRContext *context,
                               TypeConverter *typeConverter,
                               RewritePatternSet *patterns);

}  // namespace chlo

namespace stablehlo {

// Populates MHLO ops to StableHLO ops rewriting patterns.
// Also see `stablehlo::registerFuncOpsForTypeConversion` for helper patterns
// which make sure `func.func`, `func.call` and `func.return` which involve
// illegal types also get converted.
void populateHloToStablehloPatterns(RewritePatternSet *patterns,
                                    TypeConverter *converter,
                                    MLIRContext *context,
                                    bool allowExperimentalFeatures);

// Populates StableHLO ops to MHLO ops rewriting patterns.
// Also see `stablehlo::registerFuncOpsForTypeConversion` for helper patterns
// which make sure `func.func`, `func.call` and `func.return` which involve
// illegal types also get converted.
void populateStablehloToHloPatterns(RewritePatternSet *patterns,
                                    TypeConverter *converter,
                                    MLIRContext *context);

}  // namespace stablehlo

}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H

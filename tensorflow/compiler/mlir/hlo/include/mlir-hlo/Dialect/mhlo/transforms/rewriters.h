/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_REWRITERS_H
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_REWRITERS_H

#include <functional>
#include <memory>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace bufferization {
class BufferizeTypeConverter;
}  // namespace bufferization
namespace mhlo {

// Collection of rewrite patterns for lowering a general dot product.
void PopulateGeneralDotOpLoweringPatterns(RewritePatternSet *patterns,
                                          MLIRContext *ctx);

// Collection of rewrite patterns for lowering complex operations to equivalent
// float operations.
void PopulateComplexLoweringPatterns(MLIRContext *context,
                                     RewritePatternSet *patterns);

void PopulateOptimizeMHLOPatterns(MLIRContext *context,
                                  RewritePatternSet *patterns);

// Rewrite patterns for einsum to equivalent dot_general legalization.
void PopulateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        RewritePatternSet *patterns);

// Rewrite patterns for gather to equivalent torch index select legalization.
void PopulateGatherToTorchIndexSelectPatterns(mlir::MLIRContext *context,
                                              RewritePatternSet *patterns);

void PopulateMhloToStdPatterns(RewritePatternSet *patterns, MLIRContext *ctx);

// Collection of rewrite patterns for lowering all mhlo ops to their
// lmhlo counterparts.
void populateDynamicHLOToLHLOConversionPattern(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of HLO to LHLO dialect.
void populateHLOToLHLOConversionPattern(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of HLO to arithmetic dialect.
void populateHLOToArithmeticConversionPatterns(RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of shape operations from the HLO
// dialect to the standard dialect.
void populateHLOShapeOpsToStandardConversionPattern(
    MLIRContext *context, TypeConverter &type_converter,
    RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of HLO to Linalg dialect.
void populateHLOToLinalgConversionPattern(MLIRContext *context,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet *patterns);

// Collection of rewrite patterns for lowering of HLO dim operations.
void populateShapeComputationPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

// Converter to signless intergers to be used with linalg conversion patterns.
std::unique_ptr<TypeConverter> createHloToLinalgSignedIntegerConverter();

// Sets up legality definitions for materializing broadcasts.
void SetupMaterializeBroadcastsLegality(MLIRContext *context,
                                        ConversionTarget *conversionTarget);

// Populates a collection of rewrite patterns for materializing broadcast
// attributes to equivalent sequences of ops.
void PopulateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           RewritePatternSet *patterns);

// Populates a collection of rewrite patterns to realize element-wise operations
// on ranked tensors where possible.
void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns);

void PopulateDynamicShapeFusionPatterns(MLIRContext *context,
                                        RewritePatternSet *patterns);

// Populate a collection of conversion patterns for un-fusing
// batch_norm_inference and batch_norm_training into constituent HLO ops.
void PopulateUnfuseBatchNormPatterns(MLIRContext *context,
                                     RewritePatternSet *patterns);

// Populates patterns that translate the trigonometric operations from the
// standard dialect to approximations that do not use intrinsics.
void PopulateTrigonometricToApproximationPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns);

// Populate patterns to prepare moving dynamic broadcasts up over element-wise
// operations and broadcast the operands rather than the result. This will
// eventually allow for larger fusions.
void PopulateMergeAssumingOpsPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

// Populate patterns for iterative shape reification.
void PopulateShapeReificationPatterns(MLIRContext *, RewritePatternSet *);

// Populate patterns to group reduction and parallel dimensions of reduction
// operations and realize them through equivalent 1D or 2D reductions.
void populateGroupReductionDimensionsPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns,
                                              bool prefer_columns_reductions);

/// Populate rank specialization clustering and lowering patterns.
void PopulateRankSpecializationClusterPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns);
void PopulateRankSpecializationToSCFPatterns(MLIRContext *context,
                                             RewritePatternSet *patterns,
                                             int64_t max_target_rank);

/// Populate sparse tensor specific rewriting patterns.
void PopulateSparseRewritingPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx);

}  // namespace mhlo

namespace chlo {

// Populates a collection of conversion patterns for legalizing broadcasting
// client-HLO to their non-broadcasting counterparts.
void PopulateChloBroadcastingPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

// Populates a collection of conversion patterns for legalizing client-HLO to
// HLO by decomposing client-operations to corresponding sequences of more
// primitive operations. This does not include the
// PopulateChloBroadcastingPatterns above.
void PopulateDecomposeChloPatterns(MLIRContext *context,
                                   RewritePatternSet *patterns);

}  // namespace chlo

}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_REWRITERS_H

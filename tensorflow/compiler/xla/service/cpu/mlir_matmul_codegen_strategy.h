/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_
#define MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Vector/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/Vector/VectorTransforms.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

// TODO(kramerb): Remove this once strategy is in mlir core.

namespace xla {
namespace cpu {
namespace mlir_strategy {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  virtual ~Transformation() = default;
  virtual mlir::OwningRewritePatternList buildRewritePatterns(
      mlir::MLIRContext *context, mlir::linalg::LinalgMarker m) = 0;
  mlir::linalg::LinalgMarker marker;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Tile<LinalgOpType>`with the appropriate `options`.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Tile : public Transformation {
  explicit Tile(mlir::linalg::LinalgTilingOptions options) : options(options) {}

  mlir::OwningRewritePatternList buildRewritePatterns(
      mlir::MLIRContext *context, mlir::linalg::LinalgMarker m) override {
    mlir::OwningRewritePatternList tiling_patterns;
    tiling_patterns.insert<mlir::linalg::LinalgTilingPattern<LinalgOpType>>(
        context, options, m);
    return tiling_patterns;
  }

 private:
  mlir::linalg::LinalgTilingOptions options;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Promote<LinalgOpType>`with the appropriate `options`.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Promote : public Transformation {
  explicit Promote(mlir::linalg::LinalgPromotionOptions options)
      : options(options) {}

  mlir::OwningRewritePatternList buildRewritePatterns(
      mlir::MLIRContext *context, mlir::linalg::LinalgMarker m) override {
    mlir::OwningRewritePatternList promotion_patterns;
    promotion_patterns
        .insert<mlir::linalg::LinalgPromotionPattern<LinalgOpType>>(context,
                                                                    options, m);
    return promotion_patterns;
  }

 private:
  mlir::linalg::LinalgPromotionOptions options;
};

/// Vectorization transformation enqueues a particular stage-1 pattern for
/// `LinalgVectorizationPattern<LinalgOpType>` as well as copy to vector
/// transfer rewrite forwarding patterns.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Vectorize : public Transformation {
  mlir::OwningRewritePatternList buildRewritePatterns(
      mlir::MLIRContext *context, mlir::linalg::LinalgMarker m) override {
    mlir::OwningRewritePatternList vectorization_patterns;
    // FillOp may interfere with forwarding patterns atm, so we bump up the
    // priority of LinalgCopyVTRForwardingPattern /
    // LinalgCopyVTWForwardingPattern.
    vectorization_patterns
        .insert<mlir::linalg::LinalgVectorizationPattern<LinalgOpType>>(context,
                                                                        m);
    vectorization_patterns.insert<mlir::linalg::LinalgCopyVTRForwardingPattern,
                                  mlir::linalg::LinalgCopyVTWForwardingPattern>(
        context,
        /*benefit=*/2);
    return vectorization_patterns;
  }
};

/// Matmul-specific strategy object controls how a linalg.matmul is
/// progressively lowered.
/// The strategy uses a 3-level staged patterns strategy which allows ordering
/// transformations by using the Linalg `applyStagedPatterns` function, where:
///   1. The first stage consists of the successive `tile`, `promote` and
///   `vectorize` patterns, applied sequentially.
///   2. The second stage consists of common local canonicalization patterns
///   that are applied eagerly after each stage-1 pattern.
///   3. the third stage consists of more global transformation, also applied
///   eagerly, after all stage-2 patterns. Such more global transformations
struct MatmulCodegenStrategy {
  /// Append a pattern to add a level of tiling for `LinalgOpType` with tiling
  /// `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &tile(mlir::linalg::LinalgTilingOptions options) {
    transformation_sequence.emplace_back(new Tile<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for `LinalgOpType`
  /// with tiling `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &tileIf(bool b,
                                mlir::linalg::LinalgTilingOptions options) {
    return b ? tile<LinalgOpType>(options) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &promote(mlir::linalg::LinalgPromotionOptions options) {
    transformation_sequence.emplace_back(new Promote<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &promoteIf(
      bool b, mlir::linalg::LinalgPromotionOptions options) {
    return b ? promote<LinalgOpType>(options) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &vectorize() {
    transformation_sequence.emplace_back(new Vectorize<LinalgOpType>());
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &vectorizeIf(bool b) {
    return b ? vectorize<LinalgOpType>() : *this;
    return *this;
  }
  /// Configure the post staged-patterns late vector transformations.
  MatmulCodegenStrategy &setVectorTransformsOptions(
      mlir::vector::VectorTransformsOptions options) {
    vector_transforms_options = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf conversion.
  MatmulCodegenStrategy &setVectorTransferToSCFOptions(
      mlir::VectorTransferToSCFOptions options) {
    vector_to_scf_options = options;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup transformations
  /// interleaved.
  void transform(mlir::FuncOp func) const;

 private:
  mlir::LogicalResult postPatternTransforms(mlir::Operation *func) const;

  mlir::vector::VectorTransformsOptions vector_transforms_options;
  mlir::VectorTransferToSCFOptions vector_to_scf_options;
  llvm::SmallVector<std::unique_ptr<Transformation>, 4> transformation_sequence;
};

}  // namespace mlir_strategy
}  // namespace cpu
}  // namespace xla

#endif  // MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_

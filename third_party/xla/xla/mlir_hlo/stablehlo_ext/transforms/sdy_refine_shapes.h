/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_MLIR_HLO_STABLEHLO_EXT_TRANSFORMS_SDY_REFINE_SHAPES_H_
#define XLA_MLIR_HLO_STABLEHLO_EXT_TRANSFORMS_SDY_REFINE_SHAPES_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace stablehlo_ext {

/// Populates extension patterns for refining shapes of Shardy ops.
void populateSdyShapeRefinementPatterns(MLIRContext* context,
                                        RewritePatternSet* patterns);

}  // namespace stablehlo_ext
}  // namespace mlir

#endif  // XLA_MLIR_HLO_STABLEHLO_EXT_TRANSFORMS_SDY_REFINE_SHAPES_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_REWRITERS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_REWRITERS_H_

#include <memory>

#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project

namespace mlir {
namespace xla_hlo {

// Collection of rewrite patterns for lowering a general dot product.
void PopulateGeneralDotOpLoweringPatterns(OwningRewritePatternList *patterns,
                                          MLIRContext *ctx);

// Collection of rewrite patterns for lowering complex operations to equivalent
// float operations.
void PopulateComplexLoweringPatterns(MLIRContext *context,
                                     OwningRewritePatternList *patterns);

void PopulateXlaToStdPatterns(OwningRewritePatternList *patterns,
                              MLIRContext *ctx);

// Collection of rewrite patterns for lowering of HLO to LHLO dialect.
void populateHLOToLHLOConversionPattern(MLIRContext *context,
                                        OwningRewritePatternList *patterns);

}  // namespace xla_hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_REWRITERS_H_

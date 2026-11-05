/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// MHLO version of the LiteRT HLO->TF legalization helpers, kept for the
// MHLO-based HLO->TF legalization (tf_legalize_hlo), which is still used by
// the v1 converter (prepare_tf) and by DarwiNN. The StableHLO versions used by
// the HLO->TFL legalization live in hlo_matchers.h and
// legalize_hlo_conversions/. Remove once tf_legalize_hlo moves to StableHLO.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGACY_MHLO_REDUCE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGACY_MHLO_REDUCE_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
namespace mhlo_legacy {

void PopulateReduceArgMinMaxTFPatterns(MLIRContext* ctx,
                                       RewritePatternSet& patterns);

void PopulateReducePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                            ConversionTarget& target);

}  // namespace mhlo_legacy
}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGACY_MHLO_REDUCE_H_

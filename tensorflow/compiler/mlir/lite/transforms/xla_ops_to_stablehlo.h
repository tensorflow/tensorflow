/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_XLA_OPS_TO_STABLEHLO_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_XLA_OPS_TO_STABLEHLO_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// Populates patterns that directly rewrite the TF XLA wrapper ops
//   * tf.XlaConvV2 -> stablehlo.convolution
//   * tf.XlaGather -> stablehlo.gather
// plus a stablehlo.gather -> stablehlo.slice simplification for gathers with a
// single constant index vector (equivalent to the mhlo.gather "GatherSlice"
// canonicalization). These are intended to be combined with
// odml::PopulateLegalizeHloToTfPatterns so that the resulting StableHLO ops are
// immediately converted back to regular TF ops, without going through tf2xla
// or MHLO.
void PopulateLegalizeTfXlaOpsToStablehloPatterns(MLIRContext* context,
                                                 RewritePatternSet& patterns);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_XLA_OPS_TO_STABLEHLO_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented using other TensorFlow operations.
// TODO(laurenzo): For some reason, TFLite uses this pass and has exact
// requirements on what it can do. This is fragile and should be fixed (at a
// minimum, names should clearly convey scope). In the mean time, for a real
// compiler, use PopulateTFLoweringBeforeHLOPatterns.
void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns);

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented by means of other TensorFlow operations.
// This pattern collection preserves those TensorFlow operations that will later
// be lowered to equivalent operations in CHLO or MHLO. This allows for
// HLO-specific lowerings.
void PopulateTFLoweringBeforeHLOPatterns(MLIRContext *context,
                                         OwningRewritePatternList *patterns);

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented using other TensorFlow operations.
// Patterns are from ops with some inputs or outputs that are quantized types
// only to ops that allow non-quantized types on all inputs and outputs.
void PopulateLoweringQuantizedPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_

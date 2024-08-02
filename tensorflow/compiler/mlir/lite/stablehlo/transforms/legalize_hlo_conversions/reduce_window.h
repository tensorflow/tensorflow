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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::odml {

// Patterns to legalize mhlo.reduce_window to TFL.
//
// Maps the following representations of AvgPool in MHLO into a tfl.avg_pool
// operation when they cleanly map to 2D or 3D average pool with VALID or SAME
// padding:
// * div(reduce_sum_window(x), constant(sizeof(window)))
// * div(reduce_sum_window(x), reduce_sum_window(constant(1)))
//
// Emits: tfl.average_pool2d
void PopulateLegalizeReduceWindowPatterns(MLIRContext* ctx,
                                          RewritePatternSet& patterns,
                                          ConversionTarget& target);

// Patterns to prepare mhlo.reduce_window for legalization.
// Transposes reduce_windows to be NHWC.
//
// Emits: tfl.transpose
void PopulatePrepareReduceWindowPatterns(MLIRContext* ctx,
                                         RewritePatternSet& patterns);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_

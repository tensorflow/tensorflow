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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_PAD_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_PAD_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

// Patterns to legalize mhlo.pad to TFL
//
// Prefers tfl.pad over tfl.padv2 when it can be asserted that the pad
// values are zero.
//
// Matches: mhlo.pad
//  padding_high/low: all positive or zero
//. interior_padding: all zero
//
// Emits: tfl.pad, tfl.padv2
void PopulatePadPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                         ConversionTarget& target);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_PAD_H_

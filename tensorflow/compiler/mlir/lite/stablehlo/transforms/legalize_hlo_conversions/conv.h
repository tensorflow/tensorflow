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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::odml {

// Prepares mhlo.convolutions and legalizes to the corresponding tfl op.
//
// Note: "tfl-native" layouts are as follows:
// 2D             : [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// 3D             : [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// 2D (depthwise) : [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]
//
// Matches: mhlo.convolution
//   layout:        any (will transpose to tfl-native)
//   padding:       any (will pull into explicit pad_op)
//   lhs_dilations: trivial (all 1)
//   rhs_dilations: any
//   strides:       any
//   feature_group: see decision tree below
//   batch_group:   trivial (1)
//   reversal:      trivial (all False)
//   shape:         static, rank 4 or 5
//
// This pattern emits TFL convs based on the following decision tree:
// if lhs_dilations are trivial && kernel_out_features == output_features
//   if feature_group == 1:
//      if rank == 5: tfl.conv_3D
//      if rank == 4: tfl.conv_2D
//   else if input_features == feature_group:
//      if rank == 4: tfl.depthwise_conv TODO: b/352954597 - Add support.
//   else:
//      if rank == 4: tfl.conv_2D
// else:
//   tfl.transpose_conv TODO: b/352954597 - Add support.
void PopulateLegalizeConvPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                                  ConversionTarget& target);

void PopulatePrepareConvPatterns(MLIRContext* ctx, RewritePatternSet& patterns);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_H_

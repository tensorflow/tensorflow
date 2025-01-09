/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// Legalize mhlo.dot_general to tflite.batch_matmul.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_DOT_GENERAL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_DOT_GENERAL_H_

#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
// Converts mhlo.dot_general to tfl.BatchMatMul. Reshape and Transpose ops will
// be inserted to convert to well-formed matrix multiply; i.e., mhlo.dot_general
// -> tfl.batch_matmul(mhlo.transpose(mhlo.reshape(operand)), ...).
// Note:
// 1) Reshape/transpose are inserted because tfl.BatchMatMul requires
// size(contracting_dimensions) = 1 and size(output_dim) = 1, whereas
// mhlo.dot_general has no such restriction.
// 2) Inserted mhlo.reshape/transpose will be legalized to tf.reshape/transpose
// in LegalizeHloToTf (then from tf to tfl later).
// 3) If the operands are dynamic shaped tensors, mhlo.DynamicReshapeOp is
// inserted instead of the regular reshape, and additional ops (e.g. Gather,
// Concat ) are inserted for shape inference purposes.
// 4) All the DotOp are converted to DotGeneral during the optimization pass
// (ConvertDotOp).
class LowerDotGeneralOp : public OpConversionPattern<mhlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_DOT_GENERAL_H_

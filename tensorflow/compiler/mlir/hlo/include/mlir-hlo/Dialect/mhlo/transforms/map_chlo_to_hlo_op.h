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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_CHLO_TO_MHLO_OP_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_CHLO_TO_MHLO_OP_H_

#include <type_traits>

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace chlo {

template <typename FromOpTy, typename ToOpTy>
struct HloNaryElementwiseAdaptor {
  static ToOpTy CreateOp(FromOpTy from_op, Type result_type,
                         ValueRange broadcasted_operands, OpBuilder &builder) {
    return builder.create<ToOpTy>(from_op.getLoc(), result_type,
                                  broadcasted_operands);
  }
};
struct HloCompareAdaptor {
  static mhlo::CompareOp CreateOp(BroadcastCompareOp from_op, Type result_type,
                                  ValueRange broadcasted_operands,
                                  OpBuilder &builder) {
    return builder.create<mhlo::CompareOp>(
        from_op.getLoc(), result_type, broadcasted_operands[0],
        broadcasted_operands[1], from_op.comparison_direction(),
        from_op.compare_typeAttr());
  }
};

// Populate a pattern for each Broadcasting CHlo op. This requires the pattern
// to take a ChloOpTy, NonBroadcastingOpTy, and an Adaptor as templated values.
template <template <typename, typename, typename> class Pattern,
          typename... ConstructorArgs>
void PopulateForBroadcastingBinaryOp(MLIRContext *context,
                                     OwningRewritePatternList *patterns,
                                     ConstructorArgs &&...args) {
#define POPULATE_BCAST(ChloOp, HloOp)                                    \
  patterns->insert<                                                      \
      Pattern<ChloOp, HloOp, HloNaryElementwiseAdaptor<ChloOp, HloOp>>>( \
      context, args...);

  POPULATE_BCAST(BroadcastAddOp, mhlo::AddOp);
  POPULATE_BCAST(BroadcastAndOp, mhlo::AndOp);
  POPULATE_BCAST(BroadcastAtan2Op, mhlo::Atan2Op);
  POPULATE_BCAST(BroadcastComplexOp, mhlo::ComplexOp);
  POPULATE_BCAST(BroadcastDivOp, mhlo::DivOp);
  POPULATE_BCAST(BroadcastMaxOp, mhlo::MaxOp);
  POPULATE_BCAST(BroadcastMinOp, mhlo::MinOp);
  POPULATE_BCAST(BroadcastMulOp, mhlo::MulOp);
  POPULATE_BCAST(BroadcastNextAfterOp, NextAfterOp);
  POPULATE_BCAST(BroadcastOrOp, mhlo::OrOp);
  POPULATE_BCAST(BroadcastPolygammaOp, PolygammaOp);
  POPULATE_BCAST(BroadcastPowOp, mhlo::PowOp);
  POPULATE_BCAST(BroadcastRemOp, mhlo::RemOp);
  POPULATE_BCAST(BroadcastShiftLeftOp, mhlo::ShiftLeftOp);
  POPULATE_BCAST(BroadcastShiftRightArithmeticOp, mhlo::ShiftRightArithmeticOp);
  POPULATE_BCAST(BroadcastShiftRightLogicalOp, mhlo::ShiftRightLogicalOp);
  POPULATE_BCAST(BroadcastSubOp, mhlo::SubOp);
  POPULATE_BCAST(BroadcastXorOp, mhlo::XorOp);
  POPULATE_BCAST(BroadcastZetaOp, ZetaOp);

  // Broadcasting ops requiring special construction.
  patterns
      ->insert<Pattern<BroadcastCompareOp, mhlo::CompareOp, HloCompareAdaptor>>(
          context, args...);

#undef POPULATE_BCAST
}

}  // namespace chlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_CHLO_TO_HLO_OP_H_

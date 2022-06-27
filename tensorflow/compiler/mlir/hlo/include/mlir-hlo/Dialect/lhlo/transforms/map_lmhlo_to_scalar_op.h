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

#ifndef MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H
#define MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H

#include "mlir-hlo/Dialect/lhlo/transforms/map_lhlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"

namespace mlir {
namespace lmhlo {

struct LhloOpToStdScalarOp {
  // Implementation for LHLO ops except lmhlo::CompareOp.
  template <typename LhloOpTy, typename MhloOpTy = lmhlo::LhloToHloOp<LhloOpTy>,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, lmhlo::CompareOp>::value &&
                !std::is_same<MhloOpTy, std::false_type>::value>>
  static Value map(LhloOpTy op, ArrayRef<Type> resultTypes, ValueRange args,
                   OpBuilder* b, int /*i*/ = 0) {
    return mlir::mhlo::impl::MapMhloOpToStdScalarOp<MhloOpTy>(
        op.getLoc(), resultTypes, llvm::to_vector<4>(op->getOperandTypes()),
        args, b);
  }

  // Implementation for lmhlo::CompareOp.
  template <typename LhloOpTy, typename = std::enable_if_t<std::is_same<
                                   LhloOpTy, lmhlo::CompareOp>::value>>
  static Value map(lmhlo::CompareOp op, ArrayRef<Type> resultTypes,
                   ValueRange args, OpBuilder* b) {
    auto comparisonDirection = op.getComparisonDirection();
    return mlir::mhlo::impl::MapCompareOpToStdScalarOp(
        op.getLoc(), comparisonDirection, resultTypes,
        llvm::to_vector<4>(op->getOperandTypes()), args, b);
  }

  // Implementation for LHLO ops except lmhlo::CompareOp.
  template <typename LhloOpTy, typename MhloOpTy = lmhlo::LhloToHloOp<LhloOpTy>,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, lmhlo::CompareOp>::value &&
                !std::is_same<MhloOpTy, std::false_type>::value>>
  static Value map(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder* b,
                   unsigned /*i*/ = 0) {
    return mlir::mhlo::impl::MapMhloOpToStdScalarOp<MhloOpTy>(
        loc, resultTypes, argTypes, args, b);
  }
};

}  // namespace lmhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H

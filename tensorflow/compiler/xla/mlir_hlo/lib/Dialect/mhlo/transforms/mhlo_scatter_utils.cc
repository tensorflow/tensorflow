/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements utilities for canonicalization of ScatterOp.

#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_scatter_utils.h"

namespace mlir {
namespace mhlo {

bool isCanonicalScatter(ScatterOp scatterOp) {
  if (llvm::any_of(scatterOp.getOperandTypes(), [](Type operandType) {
        return !operandType.isa<RankedTensorType>();
      }))
    return false;

  ScatterDimensionNumbersAttr dimsAttrs = scatterOp.scatter_dimension_numbers();
  auto indicesType =
      scatterOp.getScatterIndices().getType().cast<RankedTensorType>();
  auto operandType =
      scatterOp.getOperands().front().getType().cast<RankedTensorType>();

  return indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
         dimsAttrs.getInsertedWindowDims().empty() &&
         dimsAttrs.getUpdateWindowDims() ==
             makeArrayRef(
                 to_vector(llvm::seq<int64_t>(1, operandType.getRank() + 1))) &&
         dimsAttrs.getScatterDimsToOperandDims() ==
             makeArrayRef(
                 to_vector(llvm::seq<int64_t>(0, indicesType.getDimSize(1))));
}

}  // namespace mhlo
}  // namespace mlir

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

#ifndef MLIR_HLO_GML_ST_UTILS_TENSOR_UTILS_H
#define MLIR_HLO_GML_ST_UTILS_TENSOR_UTILS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace gml_st {

// TODO(vuson): maybe overload this function instead of templating it.
// Check if the reshape operation is only expanding into/collapsing of
// unit-dimension.
template <typename TensorReshapeOp>
bool isDegenerateReshapeOp(TensorReshapeOp reshapeOp) {
  constexpr bool isExpanding =
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value;
  llvm::ArrayRef<int64_t> expandedShape =
      (isExpanding ? reshapeOp.getResultType().getShape()
                   : reshapeOp.getSrcType().getShape());
  for (auto &indices : reshapeOp.getReassociationIndices()) {
    // For each reassociation indices, a degenerate reshape op only has at most
    // 1 non-unit-dimension, i.e. number of unit-dimensions is greater or equal
    // to the indices size - 1.
    if (static_cast<size_t>(
            llvm::count_if(indices, [&expandedShape](int64_t idx) {
              return expandedShape[idx] == 1;
            })) < indices.size() - 1)
      return false;
  }
  return true;
}

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_UTILS_TENSOR_UTILS_H

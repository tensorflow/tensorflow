/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/dialect/BroadcastUtils.h"

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

namespace mlir {
namespace hlo {

bool isLegalNumpyRankedBroadcast(Value lhs, Value rhs,
                                 DenseIntElementsAttr broadcastDims) {
  RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType) return false;
  if (lhsType.getRank() == rhsType.getRank()) return true;

  // Otherwise, verify that broadcast_dims strictly performs left-padding.
  auto smallerRank = std::min(lhsType.getRank(), rhsType.getRank());
  auto largerRank = std::max(lhsType.getRank(), rhsType.getRank());

  if (smallerRank != broadcastDims.getNumElements()) {
    return false;
  }
  auto expectedExtents =
      llvm::seq<int64_t>(largerRank - smallerRank, largerRank);
  return std::equal(expectedExtents.begin(), expectedExtents.end(),
                    broadcastDims.value_begin<APInt>());
}

Value computeBinaryElementwiseBroadcastingResultExtents(Location loc, Value lhs,
                                                        Value rhs,
                                                        OpBuilder& builder) {
  return computeNaryElementwiseBroadcastingResultExtents(
      loc, ValueRange{lhs, rhs}, builder);
}

Value computeNaryElementwiseBroadcastingResultExtents(Location loc,
                                                      ValueRange operands,
                                                      OpBuilder& builder) {
  auto shapes = llvm::to_vector<4>(llvm::map_range(operands, [&](Value v) {
    return builder.createOrFold<shape::ShapeOfOp>(loc, v);
  }));

  int64_t resultRank = 0;
  for (Value s : shapes) {
    auto ty = s.getType().cast<RankedTensorType>();
    assert(ty.getRank() == 1 && "expect extent tensor type");
    if (ty.isDynamicDim(0)) {
      resultRank = ShapedType::kDynamicSize;
      break;
    }
    resultRank = std::max(resultRank, ty.getDimSize(0));
  }
  Type extentTensorTy =
      shape::getExtentTensorType(builder.getContext(), resultRank);

  return builder.createOrFold<shape::BroadcastOp>(loc, extentTensorTy, shapes,
                                                  /*error=*/nullptr);
}

}  // namespace hlo
}  // namespace mlir

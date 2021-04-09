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

#include "mlir-hlo/utils/broadcast_utils.h"

#include <algorithm>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace hlo {

bool IsLegalNumpyRankedBroadcast(Value lhs, Value rhs,
                                 DenseIntElementsAttr broadcast_dims) {
  RankedTensorType lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
  RankedTensorType rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhs_type || !rhs_type) return false;
  if (lhs_type.getRank() == rhs_type.getRank()) return true;

  // Otherwise, verify that broadcast_dims strictly performs left-padding.
  auto smaller_rank = std::min(lhs_type.getRank(), rhs_type.getRank());
  auto larger_rank = std::max(lhs_type.getRank(), rhs_type.getRank());

  if (smaller_rank != broadcast_dims.getNumElements()) {
    return false;
  }
  auto expected_extents =
      llvm::seq<int64_t>(larger_rank - smaller_rank, larger_rank);
  return std::equal(expected_extents.begin(), expected_extents.end(),
                    broadcast_dims.getIntValues().begin());
}

Value ComputeBinaryElementwiseBroadcastingResultExtents(
    Location loc, Value lhs, Value rhs, OpBuilder& builder,
    bool unsafe_as_extent_tensor) {
  auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhs_type || !rhs_type) {
    emitError(loc) << "shape computation for broadcasting elementwise ops "
                   << "is only implemented for ranked tensors";
    return nullptr;
  }

  Value lhs_shape_v = builder.createOrFold<shape::ShapeOfOp>(loc, lhs);
  Value rhs_shape_v = builder.createOrFold<shape::ShapeOfOp>(loc, rhs);

  if (unsafe_as_extent_tensor) {
    int64_t result_rank = std::max(lhs_type.getRank(), rhs_type.getRank());
    Value result_shape_v = builder.createOrFold<shape::BroadcastOp>(
        loc, shape::getExtentTensorType(builder.getContext()), lhs_shape_v,
        rhs_shape_v, nullptr /* error */);
    return builder.createOrFold<tensor::CastOp>(
        loc, RankedTensorType::get({result_rank}, builder.getIndexType()),
        result_shape_v);
  }

  return builder.createOrFold<shape::BroadcastOp>(
      loc, builder.getType<shape::ShapeType>(), lhs_shape_v, rhs_shape_v,
      nullptr /* error */);
}

}  // namespace hlo
}  // namespace mlir

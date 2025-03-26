/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_tensor_helper.h"

#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TF {

class IdentityOp;
class IdentityNOp;

// Returns the RankedTensorType for the given operand. TensorFlow constant ops
// may have non-static shape because the shape is not propagated during constant
// folding. If the defining op for the given operand is a constant op, this
// routine uses the constant op's attribute to get the actual shape.
RankedTensorType GetRankedTensorTypeForOperand(Value operand) {
  DenseElementsAttr attr;
  if (matchPattern(operand, m_Constant(&attr))) {
    return mlir::dyn_cast<RankedTensorType>(attr.getType());
  }
  return mlir::dyn_cast<RankedTensorType>(operand.getType());
}

// Returns the tf.Equal/tf.NotEqual result type given `x` and `y` and inputs. If
// `incompatible_shape_error` is true, reports error if `x` and `y` has
// incompatible shapes. Otherwise, returns a tensor type with unknown rank.
Type DeduceEqualCmpOpType(Builder *builder, Location loc, Value x, Value y,
                          BoolAttr incompatible_shape_error) {
  auto result_type =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());
  if (!result_type) {
    if (incompatible_shape_error.getValue()) {
      mlir::emitError(loc, "non-broadcastable operands");
    } else {
      return UnrankedTensorType::get(builder->getI1Type());
    }
  }

  auto ranked_type = mlir::dyn_cast<RankedTensorType>(result_type);
  if (!ranked_type) return UnrankedTensorType::get(builder->getI1Type());

  return RankedTensorType::get(ranked_type.getShape(), builder->getI1Type());
}

Type InferReductionOpType(Value input, Value reduction_indices,
                          BoolAttr keep_dims) {
  Type input_ty = input.getType();
  Type element_ty = getElementTypeOrSelf(input_ty);

  // Output type is unranked if input type is not ranked.
  auto ranked_ty = mlir::dyn_cast<RankedTensorType>(input_ty);
  if (!ranked_ty) return UnrankedTensorType::get(element_ty);
  int64_t rank = ranked_ty.getRank();

  DenseIntElementsAttr indices;
  if (!matchPattern(reduction_indices, m_Constant(&indices))) {
    // Output type is unranked if reduction indices are not constant and reduced
    // dimensions are not kept.
    if (!keep_dims.getValue()) return UnrankedTensorType::get(element_ty);

    // Otherwise, output type has same rank as the input.
    return RankedTensorType::get(
        SmallVector<int64_t, 4>(rank, ShapedType::kDynamic), element_ty);
  }

  int64_t num_reduce_dim = 0;
  llvm::SmallVector<bool, 4> is_reduce_dim(rank, false);
  for (const APInt &index : indices.getValues<APInt>()) {
    int64_t dim = GetDimForAxis(index.getSExtValue(), rank);
    // Invalid input.
    if (dim < 0 || dim >= rank) return UnrankedTensorType::get(element_ty);

    if (!is_reduce_dim[dim]) {
      is_reduce_dim[dim] = true;
      num_reduce_dim++;
    }
  }

  ArrayRef<int64_t> shape = ranked_ty.getShape();
  SmallVector<int64_t, 4> out_shape;
  out_shape.reserve(rank - (keep_dims.getValue() ? 0 : num_reduce_dim));
  for (int64_t i = 0; i < rank; ++i) {
    if (!is_reduce_dim[i])
      out_shape.push_back(shape[i]);
    else if (keep_dims.getValue())
      out_shape.push_back(1);
  }
  return RankedTensorType::get(out_shape, element_ty);
}

// Verifies that the given types are cast compatible. If not, emits appropriate
// error for the given op. If mask_one_dim is set to true, then the types are
// allowed to have one mismatching dimension. Masking one of the dimensions is
// useful for ops like Concat that requires all ranked inputs to have the same
// rank and match dimension sizes for all but one of the dimensions.
LogicalResult VerifyTypesCompatibility(Operation::operand_type_range types,
                                       bool mask_one_dim, Operation *op) {
  int64_t common_rank = ShapedType::kDynamic;
  llvm::SmallVector<int64_t, 4> common_dims;
  int64_t dim_to_mask = ShapedType::kDynamic;

  // Initialize common_rank with rank of the first ranked type and verify that
  // following ranked types have the same rank.
  // Similarly, initialize each of the dimensions with the first type that has
  // the dimension size available and verify that all following types have the
  // same size for the dimension. However, if mask_one_dim is true, note down
  // the dimension index on the first mismatch and ignore dimension at that
  // index in following types.
  for (Type ty : types) {
    RankedTensorType ranked_ty = mlir::dyn_cast<RankedTensorType>(ty);
    if (!ranked_ty) continue;

    int64_t rank = ranked_ty.getRank();
    if (common_rank == ShapedType::kDynamic) {
      common_rank = rank;
      common_dims.resize(common_rank, ShapedType::kDynamic);
    } else if (common_rank != rank) {
      return op->emitError()
             << "operand type " << ranked_ty
             << " is not compatible with preceding operands; expected rank: "
             << common_rank;
    }

    for (int64_t i = 0, e = common_rank; i != e; i++) {
      if (i == dim_to_mask) continue;

      int64_t dim = ranked_ty.getDimSize(i);
      if (dim == ShapedType::kDynamic) continue;

      int64_t &common_dim = common_dims[i];
      if (common_dim == ShapedType::kDynamic) {
        common_dim = dim;
      } else if (common_dim != dim) {
        // If mask_one_dim is true, do not emit an error if this is the only
        // dimension with mismatches. Note down the dimension to mask it from
        // the following types.
        if (mask_one_dim && dim_to_mask == ShapedType::kDynamic) {
          dim_to_mask = i;
          continue;
        }

        return op->emitError() << "operand type " << ranked_ty
                               << " is not compatible with preceding operands; "
                                  "expected dimension at index "
                               << i << ": " << common_dim;
      }
    }
  }
  return success();
}

}  // namespace TF
}  // namespace mlir

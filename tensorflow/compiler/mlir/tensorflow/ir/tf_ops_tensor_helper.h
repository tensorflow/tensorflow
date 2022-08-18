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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_TENSOR_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_TENSOR_HELPER_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace mlir {

class Builder;

namespace TF {

// Returns the RankedTensorType for the given operand. TensorFlow constant ops
// may have non-static shape because the shape is not propagated during constant
// folding. If the defining op for the given operand is a constant op, this
// routine uses the constant op's attribute to get the actual shape.
RankedTensorType GetRankedTensorTypeForOperand(Value operand);

// Returns true if the given `value` is of ranked float tensor type with the
// given `rank`.
inline bool IsOfRankedFloatTensorType(RankedTensorType type, int rank) {
  return type && type.getRank() == rank &&
         type.getElementType().isa<FloatType>();
}

// Returns true if the given `value` has the specified rank or has unranked
// type.
inline bool IsOfRankOrUnranked(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() == rank;
}

// Returns true if the given `value` has at least the specified rank or has
// unranked type.
inline bool HasRankAtLeast(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() >= rank;
}

// Returns true if the given `value` has at most the specified rank or has
// unranked type.
inline bool HasRankAtMost(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() <= rank;
}

inline bool IsUnknownDimOrRank(int64_t dim_or_rank) {
  return dim_or_rank == -1;
}

// Returns dimension index for the given TensorFlow axis that supports negative
// indexing.
inline int64_t GetDimForAxis(int64_t axis, int64_t rank) {
  return axis >= 0 ? axis : axis + rank;
}

// Returns the tf.Equal/tf.NotEqual result type given `x` and `y` and inputs. If
// `incompatible_shape_error` is true, reports error if `x` and `y` has
// incompatible shapes. Otherwise, returns a tensor type with unknown rank.
Type DeduceEqualCmpOpType(Builder *builder, Location loc, Value x, Value y,
                          BoolAttr incompatible_shape_error);

Type InferReductionOpType(Value input, Value reduction_indices,
                          BoolAttr keep_dims);

// Verifies that the given types are cast compatible. If not, emits appropriate
// error for the given op. If mask_one_dim is set to true, then the types are
// allowed to have one mismatching dimension. Masking one of the dimensions is
// useful for ops like Concat that requires all ranked inputs to have the same
// rank and match dimension sizes for all but one of the dimensions.
LogicalResult VerifyTypesCompatibility(Operation::operand_type_range types,
                                       bool mask_one_dim, Operation *op);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_TENSOR_HELPER_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

struct PermutationAndShape {
  DenseIntElementsAttr permutation;
  ShapedType shape;
};

// Check that `arr` is an R1 iota with integer element type starting from
// `start` with `size` number of values.
bool IsIotaAttr(ArrayRef<int64_t> arr, int64_t size, int64_t start = 0);

// Returns a DenseIntElementsAttr for a permutation and the shape after
// applying the permutation to a given shape through a transpose.
PermutationAndShape GetPermutationAndTransposedShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter);

// Create a single const integer.
Value BuildIntConstOp(ImplicitLocOpBuilder& builder,
                      ConversionPatternRewriter& rewriter, int64_t const_value,
                      Type type);

// Create a const integer vector tensor (1-dim).
template <typename ConstOpT = TF::ConstOp>
Value BuildIntArrayConstOp(ImplicitLocOpBuilder& builder,
                           ConversionPatternRewriter& rewriter,
                           ArrayRef<int64_t> const_value, Type type) {
  DenseIntElementsAttr const_value_raw;
  if (type == rewriter.getI64Type()) {
    const_value_raw = rewriter.getI64TensorAttr(const_value);
  } else {
    // Convert I64 const array to I32.
    llvm::SmallVector<int32_t> const_i32_vec;
    for (auto element : const_value) {
      const_i32_vec.push_back(static_cast<int32_t>(element));
    }
    const_value_raw = rewriter.getI32TensorAttr(const_i32_vec);
  }
  Value result_const = builder.create<ConstOpT>(const_value_raw);
  return result_const;
}

// Returns the inverse permutation array for a permutation array.
llvm::SmallVector<int64_t> GetInversePermutationArray(
    llvm::ArrayRef<int64_t> permutation_array);

// Returns the DenseIntElementsAttr for an inverse permutation given a
// permutation_array.
DenseIntElementsAttr GetInversePermutation(
    llvm::ArrayRef<int64_t> permutation_array,
    ConversionPatternRewriter& rewriter);

// Returns a DenseIntElementsAttr for an inverse permutation and the shape after
// applying the inverse permutation to a given shape through a transpose.
PermutationAndShape GetInversePermutationAndShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter);

// Returns true if the op needs reformat.
bool NeedsReformatTypeAndPermutation(int batch_dim, int feature_dim,
                                     int spatial_dim_start,
                                     int default_batch_dim,
                                     int default_feature_dim,
                                     int default_spatial_dim_start);

// Gets reformat type and permutation attribute. Call this function only if
// NeedsReformatTypeAndPermutation returns true. If
// NeedsReformatTypeAndPermutation returns false, this function returns the pair
// of input type and no-op permutation.

std::pair<RankedTensorType, DenseIntElementsAttr> GetReformatTypeAndPermutation(
    int batch_dim, int feature_dim, int spatial_dim_start,
    int default_batch_dim, int default_feature_dim,
    int default_spatial_dim_start, int num_spatial_dims, RankedTensorType type,
    ConversionPatternRewriter& rewriter);

// Insert transpose so the input value is converted to the format specified by
// the default dims
Value InsertTranspose(Value value, int batch_dim, int feature_dim,
                      ArrayRef<int64_t> spatial_dimensions,
                      int default_batch_dim, int default_feature_dim,
                      int default_spatial_dim_start, int num_spatial_dims,
                      ConversionPatternRewriter& rewriter);

// If index_vector_dim == indices.rank() then insert the implicit extra
// dimension into indices to normalize everything to index_vector_dim ==
// indices.rank() - 1.
LogicalResult NormalizeIndexVector(Operation* parent_op, Value& indices,
                                   ShapedType& indices_type,
                                   int64_t index_vector_dim,
                                   ConversionPatternRewriter& rewriter);

// Checks if the specified region is a binary reduction function that takes 2
// inputs, passes it to an instance of the specified reduction op and then
// returns the result.
template <typename ReductionOp>
LogicalResult MatchBinaryReduceFunction(mlir::Region& function) {
  Block& body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();

  ReductionOp reduce_op = dyn_cast_or_null<ReductionOp>(
      return_op.getOperands().front().getDefiningOp());
  if (!reduce_op) return failure();
  if (reduce_op.getLhs() != body.getArgument(0) ||
      reduce_op.getRhs() != body.getArgument(1))
    return failure();

  return success();
}

// Check if the specified region is a binary reduction function that takes 2
// inputs and returns the second input. Functions like this are used by update
// scatter like ops.
template <>
LogicalResult MatchBinaryReduceFunction<void>(mlir::Region& function);

// Util that casts 'val' to Int32 by adding a tfl cast Op.
Value CreateCastToInt32(Value val, Location loc, PatternRewriter& rewriter);

// Replaces `region`'s terminator to TFL::Yield.
void ReplaceTerminatorWithYield(Region& region, PatternRewriter& rewriter);
}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_
